// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_
#define CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/db_row_interpolation.h"
#include "crypto/encryption.h"
#include "crypto/hint_computation.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// This class is used to perform all preprocessing operations on the database.
// This class takes in PIR parameters and a database, and outputs all necessary
// preprocessing material as one object that is then passed on to the PirServer.
template <typename DbValueType, typename CoeffType, typename MatCoeffType>
class PreprocessingServer {
 public:
  struct CrossColorBlockPreprocessed {
    std::vector<CrossColorDbSlice<DbValueType>> db_slices;
    PreprocessMatrixPackOutput<CoeffType, MatCoeffType> pack_output;
  };

  // Stage 1 slices of one row stripe (see RowStripe) of a cross-color response
  // block, and their partial hint sum_{r in stripe} D_r a_r (d x d). Partial
  // hints add up mod 2^64 across stripes to the block's hint.
  struct CrossColorStripe {
    std::vector<CrossColorDbSlice<DbValueType>> db_slices;
    std::vector<std::vector<CoeffType>> hint;
  };

  // Creates a new PreprocessingServer.
  static absl::StatusOr<std::unique_ptr<PreprocessingServer>> Create(
      const PirParams<CoeffType>* params) {
    ASSIGN_OR_RETURN(std::string prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return absl::WrapUnique(new PreprocessingServer(params, prng_seed));
  }

  static absl::StatusOr<std::unique_ptr<PreprocessingServer>> Create(
      const PirParams<CoeffType>* params, absl::string_view prng_seed) {
    return absl::WrapUnique(
        new PreprocessingServer(params, std::string(prng_seed)));
  }

  // Preprocesses one cross-color response block `block_index` covering global
  // LWE coefficients [block_index * d, min((block_index + 1) * d, P * d_row)),
  // computing only the `d_row` useful Stage 1 rows and LWE hints per color.
  // Colors may be taller than d. A row of `raw[slot]` holds at least d_row
  // words; d_row may exceed d (C = d_row / d full columns per color).
  // `centered_db`: see PreprocessCrossColorStripe.
  static absl::StatusOr<CrossColorBlockPreprocessed> PreprocessCrossColorBlock(
      const RlweParams<CoeffType>& rlwe_params,
      const GadgetParams& pack_gadget_params,
      const std::vector<int>& rows,
      const std::vector<std::vector<DbValueType>>& raw,
      const std::vector<std::string>& slot_selector_seeds,
      const std::vector<size_t>& slot_offsets,
      absl::string_view packing_seed,
      size_t d_row,
      size_t block_index,
      bool centered_db = false) {
    ASSIGN_OR_RETURN(
        CrossColorStripe stripe,
        PreprocessCrossColorStripe(rlwe_params, rows, raw, slot_selector_seeds,
                                   slot_offsets, d_row, block_index,
                                   /*stripe=*/0, /*num_stripes=*/1,
                                   centered_db));
    ASSIGN_OR_RETURN(auto pack_output,
                     PreprocessPacking(rlwe_params, pack_gadget_params,
                                       packing_seed, stripe.hint));
    return CrossColorBlockPreprocessed{
        .db_slices = std::move(stripe.db_slices),
        .pack_output = std::move(pack_output),
    };
  }

  // Row stripe `stripe` of `num_stripes` of PreprocessCrossColorBlock: slices
  // contract only their color's rows in the stripe (colors with none are
  // skipped). With `centered_db` (uint16_t only) Stage 1 contracts the signed
  // int16_t(D) = (D ^ 2^15) - 2^15, which is D mod 2^16, so the plaintext is
  // unchanged for p = 2^16 while |entries| <= 2^15 (4x less noise variance).
  static absl::StatusOr<CrossColorStripe> PreprocessCrossColorStripe(
      const RlweParams<CoeffType>& rlwe_params, const std::vector<int>& rows,
      const std::vector<std::vector<DbValueType>>& raw,
      const std::vector<std::string>& slot_selector_seeds,
      const std::vector<size_t>& slot_offsets, size_t d_row,
      size_t block_index, size_t stripe, size_t num_stripes,
      bool centered_db) {
    const size_t p = rows.size();
    const size_t d = static_cast<size_t>(rlwe_params.Degree());
    if (p == 0 || raw.size() != p || slot_selector_seeds.size() != p ||
        slot_offsets.size() != p || d_row == 0 || stripe >= num_stripes ||
        (centered_db && !std::is_same_v<DbValueType, uint16_t>)) {
      return absl::InvalidArgumentError(
          "Invalid arguments for PreprocessCrossColorStripe.");
    }
    const size_t total_u16 = p * d_row;
    const size_t g0 = block_index * d;
    const size_t g1 = std::min((block_index + 1) * d, total_u16);
    if (g0 >= g1) {
      return absl::InvalidArgumentError("Cross-color block_index out of range.");
    }

    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    CrossColorStripe out{.hint = std::vector<std::vector<CoeffType>>(
                             d, std::vector<CoeffType>(d, 0))};
    for (size_t pos = g0; pos < g1;) {
      const size_t slot = pos / d_row;
      const size_t u0 = pos % d_row;
      const size_t take = std::min(d_row - u0, g1 - pos);
      const size_t coeff0 = pos - g0;
      pos += take;
      const size_t m_rows = static_cast<size_t>(std::max(rows[slot], 0));
      if (m_rows == 0 || raw[slot].size() % m_rows != 0 ||
          raw[slot].size() / m_rows < d_row ||
          (slot_offsets[slot] > 0 && slot_offsets[slot] + m_rows > d)) {
        return absl::InvalidArgumentError("Invalid slot database dimensions.");
      }
      const auto [r0, r1] = RowStripe(m_rows, d, stripe, num_stripes);
      if (r0 == r1) continue;

      Matrix<DbValueType> db =
          TransposeRows(raw[slot].data(), raw[slot].size() / m_rows, r0, r1,
                        u0, take, centered_db ? 0x8000 : 0);
      ASSIGN_OR_RETURN(auto a_prng,
                       ::rlwe::ChaChaPrng::Create(slot_selector_seeds[slot]));
      ASSIGN_OR_RETURN(auto first_dim_a,
                       SampleAComponents(rlwe_params, (m_rows + d - 1) / d,
                                         *a_prng));
      if (slot_offsets[slot] > 0) {
        ASSIGN_OR_RETURN(
            first_dim_a[0],
            RotateGroupMaskForInterval(first_dim_a[0], slot_offsets[slot],
                                       rlwe_params.Modulus()));
      }
      const std::vector<Polynomial<CoeffType>> a(
          first_dim_a.begin() + r0 / d, first_dim_a.begin() + (r1 + d - 1) / d);
      ASSIGN_OR_RETURN(auto slice_hints,
                       (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                           db, a, *fft_ctx)));
      for (size_t u = 0; u < take; ++u) {
        auto row_start = slice_hints.Data().begin() + u * d;
        std::copy(row_start, row_start + d, out.hint[coeff0 + u].begin());
      }
      if (centered_db) {
        // The hint of the stored (D ^ 2^15) minus 2^15 * sum_r a_r, i.e. 2^15
        // times the hint of an all-ones row.
        Matrix<DbValueType> ones(std::vector<DbValueType>(r1 - r0, 1), 1,
                                 static_cast<int>(r1 - r0));
        ASSIGN_OR_RETURN(auto sum,
                         (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                             ones, a, *fft_ctx)));
        for (size_t u = 0; u < take; ++u) {
          for (size_t j = 0; j < d; ++j) {
            out.hint[coeff0 + u][j] -= CoeffType{0x8000} * sum.Data()[j];
          }
        }
      }
      const int padded_cols = db.Cols();
      RETURN_IF_ERROR(PadAndCondenseForSimd(db));
      out.db_slices.push_back(CrossColorDbSlice<DbValueType>{
          .color = slot,
          .padded_cols = padded_cols,
          .rows = static_cast<int>(take),
          .db_submatrix = std::move(db),
          .row0 = r0,
          .coeff0 = coeff0,
          .centered = centered_db,
      });
    }
    return out;
  }

  // Packing material for a d x d (summed) hint under the common packing mask
  // drawn from `packing_seed`.
  static absl::StatusOr<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
  PreprocessPacking(const RlweParams<CoeffType>& rlwe_params,
                    const GadgetParams& pack_gadget_params,
                    absl::string_view packing_seed,
                    const std::vector<std::vector<CoeffType>>& hint) {
    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(
            absl::bit_width(static_cast<uint32_t>(rlwe_params.Degree())) - 1));
    ASSIGN_OR_RETURN(auto pack_prng, ::rlwe::ChaChaPrng::Create(packing_seed));
    ASSIGN_OR_RETURN(auto pack_a,
                     SampleAComponents(rlwe_params,
                                       2 * pack_gadget_params.num_digits,
                                       *pack_prng));
    std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(), pack_a.begin() + pack_gadget_params.num_digits);
    std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + pack_gadget_params.num_digits, pack_a.end());
    return PreprocessMatrixPack<CoeffType, MatCoeffType>(
        rlwe_params, hint, w_vec_g, w_vec_h, pack_gadget_params, *fft_ctx);
  }

  absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
  Preprocess(const std::vector<DbValueType>& db_raw,
             absl::string_view packing_seed = "",
             size_t selector_offset = 0) const {
    ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(prng_seed_));
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();
    const int n = params_->NumEntries();
    const int entry_size_multiple = params_->EntrySizeMultiple();

    if (selector_offset > 0 &&
        (t != 1 ||
         selector_offset + static_cast<size_t>(n) > static_cast<size_t>(d))) {
      return absl::InvalidArgumentError(
          "Grouped selector interval [selector_offset, selector_offset + n) "
          "must fit within ring degree d at t=1.");
    }

    if (db_raw.size() != static_cast<size_t>(n) * d * entry_size_multiple) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<Matrix<DbValueType>> db_matrices;
    db_matrices.reserve(entry_size_multiple);

    for (size_t k = 0; k < entry_size_multiple; ++k) {
      if (t == 1) {
        // No interpolation: transpose column k straight into Stage 1 layout.
        db_matrices.push_back(TransposeRows(db_raw.data(),
                                            static_cast<size_t>(d) *
                                                entry_size_multiple,
                                            0, n, k * d, d, 0));
        continue;
      }
      std::vector<DbValueType> db_shard_raw(static_cast<size_t>(n) * d);
      for (size_t idx = 0; idx < n; ++idx) {
        for (size_t offset = 0; offset < d; ++offset) {
          db_shard_raw[idx * static_cast<size_t>(d) + offset] =
              db_raw[idx * static_cast<size_t>(d) * entry_size_multiple +
                     k * static_cast<size_t>(d) + offset];
        }
      }
      absl::Time start_interp = absl::Now();
      ASSIGN_OR_RETURN(auto db_matrix, InterpolateAndTranspose(db_shard_raw));
      interpolate_time_ += absl::Now() - start_interp;
      db_matrices.push_back(std::move(db_matrix));
    }

    const int first_dim_a_samples = params_->NumFirstDimSamples();
    ASSIGN_OR_RETURN(auto first_dim_a,
                     SampleAComponents(params_->RlweParameters(),
                                       first_dim_a_samples, *a_prng));
    if (selector_offset > 0) {
      ASSIGN_OR_RETURN(
          first_dim_a[0],
          RotateGroupMaskForInterval(first_dim_a[0], selector_offset,
                                     params_->RlweParameters().Modulus()));
    }
    const int second_dim_a_samples =
        t > 1 ? 2 * params_->PolyEvalGadgetParams().num_digits : 0;
    std::vector<Polynomial<CoeffType>> second_dim_a;
    if (second_dim_a_samples > 0) {
      ASSIGN_OR_RETURN(second_dim_a,
                       SampleAComponents(params_->RlweParameters(),
                                         second_dim_a_samples, *a_prng));
    }
    // Batch requests use one common packing mask, distinct from every selector.
    if (!packing_seed.empty()) {
      if (params_->InterpolationDegree() != 1) {
        return absl::InvalidArgumentError("Shared packing seed requires t=1.");
      }
      ASSIGN_OR_RETURN(a_prng, ::rlwe::ChaChaPrng::Create(packing_seed));
    }
    const int pack_a_samples = 2 * params_->PackGadgetParams().num_digits;
    ASSIGN_OR_RETURN(auto pack_a, SampleAComponents(params_->RlweParameters(),
                                                    pack_a_samples, *a_prng));

    std::vector<
        std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
        preprocessed_outputs;
    preprocessed_outputs.reserve(entry_size_multiple);
    for (size_t k = 0; k < entry_size_multiple; ++k) {
      absl::Time start_comp = absl::Now();
      ASSIGN_OR_RETURN(
          auto preprocessed_output,
          ComputePreprocessedOutputs(db_matrices[k], first_dim_a, pack_a));
      compute_outputs_time_ += absl::Now() - start_comp;
      preprocessed_outputs.push_back(std::move(preprocessed_output));
    }

    for (size_t k = 0; k < entry_size_multiple; ++k) {
      RETURN_IF_ERROR(PadAndCondenseForSimd(db_matrices[k]));
    }

    return PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>{
        std::move(db_matrices),
        std::move(preprocessed_outputs),
        std::move(second_dim_a),
    };
  }

  // Returns PRNG seed.
  const std::string& GetPrngSeed() const { return prng_seed_; }

  absl::Duration GetInterpolateTime() const { return interpolate_time_; }
  absl::Duration GetComputeOutputsTime() const { return compute_outputs_time_; }
  absl::Duration GetComputeHintsTime() const { return compute_hints_time_; }
  absl::Duration GetPreprocessPackTime() const { return preprocess_pack_time_; }

 private:
  explicit PreprocessingServer(const PirParams<CoeffType>* params,
                               absl::string_view prng_seed)
      : params_(params), prng_seed_(prng_seed) {}

  static absl::Status PadAndCondenseForSimd(Matrix<DbValueType>& matrix) {
    if constexpr ((std::is_same_v<DbValueType, uint16_t> ||
                   std::is_same_v<DbValueType, uint8_t>) &&
                  std::is_same_v<CoeffType, uint64_t>) {
      const int cols = matrix.Cols();
      const int simd_multiple = std::is_same_v<DbValueType, uint8_t> ? 64 : 32;
      if (cols % simd_multiple != 0) {
        const int padded_simd_cols =
            ((cols + simd_multiple - 1) / simd_multiple) * simd_multiple;
        std::vector<std::vector<DbValueType>> padded_data(
            matrix.Rows(), std::vector<DbValueType>(padded_simd_cols, 0));
        for (int i = 0; i < matrix.Rows(); ++i) {
          for (int j = 0; j < cols; ++j) {
            padded_data[i][j] = matrix.Data()[i * cols + j];
          }
        }
        ASSIGN_OR_RETURN(matrix,
                         Matrix<DbValueType>::Create(std::move(padded_data)));
      }
      auto condensed_or = Matrix<DbValueType>::CreateCondensed(matrix);
      if (condensed_or.ok()) {
        matrix = std::move(*condensed_or);
      }
    }
    return absl::OkStatus();
  }

  // The take x round_up(row1 - row0, SIMD) Stage 1 matrix M[u][r - row0] =
  // src[r * stride + u0 + u] ^ flip for rows [row0, row1), zero-padded; tiled so
  // that 64 source rows stay cache-resident across all `take` words.
  static Matrix<DbValueType> TransposeRows(const DbValueType* src,
                                           size_t stride, size_t row0,
                                           size_t row1, size_t u0, size_t take,
                                           DbValueType flip) {
    constexpr size_t kSimd = std::is_same_v<DbValueType, uint8_t> ? 64 : 32;
    const size_t rows = row1 - row0;
    const size_t cols = (rows + kSimd - 1) / kSimd * kSimd;
    std::vector<DbValueType> data(take * cols, 0);
    for (size_t t0 = 0; t0 < rows; t0 += 64) {
      const size_t t1 = std::min(rows, t0 + 64);
      for (size_t u = 0; u < take; ++u) {
        for (size_t r = t0; r < t1; ++r) {
          data[u * cols + r] = src[(row0 + r) * stride + u0 + u] ^ flip;
        }
      }
    }
    return Matrix<DbValueType>(std::move(data), static_cast<int>(take),
                               static_cast<int>(cols));
  }

  // Stage 1: DB interpolation.
  absl::StatusOr<Matrix<DbValueType>> InterpolateAndTranspose(
      const std::vector<DbValueType>& db_raw) const {
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();
    const int n = params_->NumEntries();
    const int plaintext_modulus = params_->PlaintextModulus();

    if (db_raw.size() != static_cast<size_t>(n) * d) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<std::vector<DbValueType>> db_reshaped(
        n / t, std::vector<DbValueType>(static_cast<size_t>(t) * d));
    for (size_t i = 0; i < n / t; ++i) {
      auto start = db_raw.begin() + i * (static_cast<size_t>(t) * d);
      db_reshaped[i].assign(start, start + static_cast<size_t>(t) * d);
    }
    std::vector<std::vector<DbValueType>> db_interpolated(
        n / t, std::vector<DbValueType>(static_cast<size_t>(t) * d));
    for (size_t i = 0; i < n / t; ++i) {
      ASSIGN_OR_RETURN(
          db_interpolated[i],
          Interpolate(db_reshaped[i], d, t,
                      static_cast<DbValueType>(plaintext_modulus)));
    }
    std::vector<std::vector<DbValueType>> db_transposed(
        static_cast<size_t>(t) * d, std::vector<DbValueType>(n / t));
    for (size_t i = 0; i < static_cast<size_t>(t) * d; ++i) {
      for (size_t j = 0; j < n / t; ++j) {
        db_transposed[i][j] = db_interpolated[j][i];
      }
    }
    return Matrix<DbValueType>::Create(db_transposed);
  }

  // Stage 2: Compute preprocessed materials.
  absl::StatusOr<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
  ComputePreprocessedOutputs(
      const Matrix<DbValueType>& db_matrix,
      const std::vector<Polynomial<CoeffType>>& first_dim_a,
      const std::vector<Polynomial<CoeffType>>& pack_a) const {
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();

    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    absl::Time start_hint = absl::Now();
    ASSIGN_OR_RETURN(auto hint_matrix,
                     (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                         db_matrix, first_dim_a, *fft_ctx)));
    compute_hints_time_ += absl::Now() - start_hint;

    std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(),
        pack_a.begin() + params_->PackGadgetParams().num_digits);
    std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + params_->PackGadgetParams().num_digits, pack_a.end());
    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        preprocessed_outputs;
    preprocessed_outputs.reserve(t);
    for (size_t i = 0; i < t; ++i) {
      std::vector<std::vector<CoeffType>> block;
      block.reserve(d);
      for (size_t r = 0; r < d; ++r) {
        auto start = hint_matrix.Data().begin() +
                     static_cast<size_t>(i * d + r) * hint_matrix.Cols();
        block.emplace_back(start, start + d);
      }
      absl::Time start_pack = absl::Now();
      ASSIGN_OR_RETURN(auto pack_output,
                       (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                           params_->RlweParameters(), block, w_vec_g, w_vec_h,
                           params_->PackGadgetParams(), *fft_ctx)));
      preprocess_pack_time_ += absl::Now() - start_pack;
      preprocessed_outputs.push_back(std::move(pack_output));
    }
    return preprocessed_outputs;
  }

  const PirParams<CoeffType>* params_;
  std::string prng_seed_;
  mutable absl::Duration interpolate_time_ = absl::ZeroDuration();
  mutable absl::Duration compute_outputs_time_ = absl::ZeroDuration();
  mutable absl::Duration compute_hints_time_ = absl::ZeroDuration();
  mutable absl::Duration preprocess_pack_time_ = absl::ZeroDuration();
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_
