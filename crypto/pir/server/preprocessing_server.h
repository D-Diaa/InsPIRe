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

  // Stage 1 material for one row stripe (see RowStripe) of a t=1 table: the
  // condensed stripe database (one matrix per column for Preprocess tables, or
  // the slices of one cross-color response block) and its partial hints
  // sum_{r in stripe} D_r a_r (d x d per column or block). Partial hints add up
  // mod 2^64 across stripes to the hints that PreprocessPacking consumes.
  struct Stage1Stripe {
    std::vector<Matrix<DbValueType>> db_matrices;
    std::vector<CrossColorDbSlice<DbValueType>> db_slices;
    std::vector<std::vector<std::vector<CoeffType>>> hints;
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
        Stage1Stripe stripe,
        PreprocessCrossColorStripe(rlwe_params, rows, raw, slot_selector_seeds,
                                   slot_offsets, d_row, block_index,
                                   /*stripe=*/0, /*num_stripes=*/1,
                                   centered_db));
    ASSIGN_OR_RETURN(auto outputs,
                     PreprocessPacking(rlwe_params, pack_gadget_params,
                                       packing_seed, stripe.hints));
    return CrossColorBlockPreprocessed{
        .db_slices = std::move(stripe.db_slices),
        .pack_output = std::move(outputs[0]),
    };
  }

  // Stage 1 slices and partial hint of row stripe `stripe` of `num_stripes` of
  // cross-color response block `block_index` (see PreprocessCrossColorBlock).
  // Colors may be taller than d; a color with no rows in the stripe is skipped.
  static absl::StatusOr<Stage1Stripe> PreprocessCrossColorStripe(
      const RlweParams<CoeffType>& rlwe_params, const std::vector<int>& rows,
      const std::vector<std::vector<DbValueType>>& raw,
      const std::vector<std::string>& slot_selector_seeds,
      const std::vector<size_t>& slot_offsets, size_t d_row,
      size_t block_index, size_t stripe, size_t num_stripes,
      bool centered_db) {
    const size_t p = rows.size();
    const size_t d = static_cast<size_t>(rlwe_params.Degree());
    if (p == 0 || raw.size() != p || slot_selector_seeds.size() != p ||
        slot_offsets.size() != p || d_row == 0 || d_row > d ||
        stripe >= num_stripes) {
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
    Stage1Stripe out;
    out.hints.emplace_back(d, std::vector<CoeffType>(d, 0));
    auto& hint_block = out.hints[0];

    for (size_t pos = g0; pos < g1;) {
      const size_t slot = pos / d_row;
      const size_t u0 = pos % d_row;
      const size_t take = std::min(d_row - u0, g1 - pos);
      const size_t coeff0 = pos - g0;
      pos += take;
      const size_t m_rows = static_cast<size_t>(rows[slot]);
      if (rows[slot] <= 0 || raw[slot].size() < m_rows * d ||
          (slot_offsets[slot] > 0 && slot_offsets[slot] + m_rows > d)) {
        return absl::InvalidArgumentError("Invalid slot database dimensions.");
      }
      const auto [r0, r1] = RowStripe(m_rows, d, stripe, num_stripes);
      if (r0 == r1) continue;

      const size_t cols = PadToSimd(r1 - r0);
      std::vector<DbValueType> data(take * cols, 0);
      for (size_t r = r0; r < r1; ++r) {
        const DbValueType* row_ptr = raw[slot].data() + r * d + u0;
        for (size_t u = 0; u < take; ++u) data[u * cols + (r - r0)] = row_ptr[u];
      }
      Matrix<DbValueType> db(std::move(data), static_cast<int>(take),
                             static_cast<int>(cols));
      ASSIGN_OR_RETURN(
          auto a, StripeMasks(rlwe_params, slot_selector_seeds[slot],
                              (m_rows + d - 1) / d, slot_offsets[slot], r0, r1));
      ASSIGN_OR_RETURN(auto hints, (ComputeHintMatrix<CoeffType, DbValueType,
                                                      CoeffType>(db, a,
                                                                 *fft_ctx)));
      for (size_t u = 0; u < take; ++u) {
        auto row_start = hints.Data().begin() + u * d;
        std::copy(row_start, row_start + d, hint_block[coeff0 + u].begin());
      }
      if (centered_db) {
        RETURN_IF_ERROR(SubtractCenteredOffset(
            absl::MakeSpan(hint_block).subspan(coeff0, take), r1 - r0, a,
            *fft_ctx));
      }
      RETURN_IF_ERROR(PadAndCondenseForSimd(db));
      out.db_slices.push_back(CrossColorDbSlice<DbValueType>{
          .color = slot,
          .padded_cols = static_cast<int>(cols),
          .rows = static_cast<int>(take),
          .db_submatrix = std::move(db),
          .row0 = r0,
          .coeff0 = coeff0,
      });
    }
    return out;
  }

  // Packing material (one PreprocessMatrixPack per d x d hint) under the
  // common packing mask drawn from `packing_seed`.
  static absl::StatusOr<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
  PreprocessPacking(
      const RlweParams<CoeffType>& rlwe_params,
      const GadgetParams& pack_gadget_params, absl::string_view packing_seed,
      const std::vector<std::vector<std::vector<CoeffType>>>& hints) {
    ASSIGN_OR_RETURN(auto pack_prng, ::rlwe::ChaChaPrng::Create(packing_seed));
    return PackHints(rlwe_params, pack_gadget_params, *pack_prng, hints);
  }

  // Stage 1 material for rows [row0, row1) (row0 a multiple of d) of every
  // column of a t=1 database `db_raw`, transposed directly into SIMD-padded
  // Stage 1 matrices without interpolation copies.
  absl::StatusOr<Stage1Stripe> PreprocessStripe(
      const std::vector<DbValueType>& db_raw, size_t row0, size_t row1,
      size_t selector_offset = 0) const {
    const size_t d = params_->RlweParameters().Degree();
    const size_t n = params_->NumEntries();
    const size_t esm = params_->EntrySizeMultiple();
    if (params_->InterpolationDegree() != 1 || row0 % d != 0 || row0 >= row1 ||
        row1 > n || (selector_offset > 0 && selector_offset + n > d)) {
      return absl::InvalidArgumentError(
          "PreprocessStripe needs t=1, d-aligned rows within the table, and a "
          "grouped selector interval within d.");
    }
    if (db_raw.size() != n * d * esm) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }
    ASSIGN_OR_RETURN(auto fft_ctx,
                     FftContext::Create(absl::bit_width(d) - 1));
    ASSIGN_OR_RETURN(auto a, StripeMasks(params_->RlweParameters(), prng_seed_,
                                         params_->NumFirstDimSamples(),
                                         selector_offset, row0, row1));
    const size_t rows = row1 - row0;
    const size_t cols = PadToSimd(rows);
    Stage1Stripe out;
    for (size_t k = 0; k < esm; ++k) {
      // Tiled transpose: 64 source rows stay cache-resident across all d.
      std::vector<DbValueType> data(d * cols, 0);
      for (size_t t0 = 0; t0 < rows; t0 += 64) {
        const size_t t1 = std::min(rows, t0 + 64);
        for (size_t u = 0; u < d; ++u) {
          for (size_t r = t0; r < t1; ++r) {
            data[u * cols + r] = db_raw[((row0 + r) * esm + k) * d + u];
          }
        }
      }
      Matrix<DbValueType> db(std::move(data), static_cast<int>(d),
                             static_cast<int>(cols));
      absl::Time start_hint = absl::Now();
      ASSIGN_OR_RETURN(auto hint_matrix,
                       (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                           db, a, *fft_ctx)));
      std::vector<std::vector<CoeffType>> hint(d);
      for (size_t u = 0; u < d; ++u) {
        auto row_start = hint_matrix.Data().begin() + u * d;
        hint[u].assign(row_start, row_start + d);
      }
      if (params_->CenteredDb()) {
        RETURN_IF_ERROR(
            SubtractCenteredOffset(absl::MakeSpan(hint), rows, a, *fft_ctx));
      }
      compute_hints_time_ += absl::Now() - start_hint;
      out.hints.push_back(std::move(hint));
      RETURN_IF_ERROR(PadAndCondenseForSimd(db));
      out.db_matrices.push_back(std::move(db));
    }
    return out;
  }

  absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
  Preprocess(const std::vector<DbValueType>& db_raw,
             absl::string_view packing_seed = "",
             size_t selector_offset = 0) const {
    const int t = params_->InterpolationDegree();
    if (t == 1) {
      ASSIGN_OR_RETURN(Stage1Stripe stripe,
                       PreprocessStripe(db_raw, 0, params_->NumEntries(),
                                        selector_offset));
      // Batch requests use one common packing mask; otherwise the packing
      // mask follows the selector masks in this server's own stream.
      ASSIGN_OR_RETURN(auto a_prng,
                       ::rlwe::ChaChaPrng::Create(packing_seed.empty()
                                                      ? absl::string_view(prng_seed_)
                                                      : packing_seed));
      if (packing_seed.empty()) {
        RETURN_IF_ERROR(SampleAComponents(params_->RlweParameters(),
                                          params_->NumFirstDimSamples(),
                                          *a_prng)
                            .status());
      }
      absl::Time start_pack = absl::Now();
      ASSIGN_OR_RETURN(auto outputs,
                       PackHints(params_->RlweParameters(),
                                 params_->PackGadgetParams(), *a_prng,
                                 stripe.hints));
      preprocess_pack_time_ += absl::Now() - start_pack;
      std::vector<
          std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
          preprocessed_outputs;
      for (auto& output : outputs) {
        preprocessed_outputs.emplace_back();
        preprocessed_outputs.back().push_back(std::move(output));
      }
      return PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>{
          std::move(stripe.db_matrices), std::move(preprocessed_outputs), {}};
    }

    ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(prng_seed_));
    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();
    const int entry_size_multiple = params_->EntrySizeMultiple();

    if (selector_offset > 0 || !packing_seed.empty()) {
      return absl::InvalidArgumentError(
          "Grouped selectors and shared packing seeds require t=1.");
    }

    if (db_raw.size() != static_cast<size_t>(n) * d * entry_size_multiple) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<Matrix<DbValueType>> db_matrices;
    db_matrices.reserve(entry_size_multiple);

    for (size_t k = 0; k < entry_size_multiple; ++k) {
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
    const int second_dim_a_samples =
        2 * params_->PolyEvalGadgetParams().num_digits;
    ASSIGN_OR_RETURN(auto second_dim_a,
                     SampleAComponents(params_->RlweParameters(),
                                       second_dim_a_samples, *a_prng));
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

  static size_t PadToSimd(size_t cols) {
    constexpr size_t kSimd = std::is_same_v<DbValueType, uint8_t> ? 64 : 32;
    return (cols + kSimd - 1) / kSimd * kSimd;
  }

  // The selector masks of rows [row0, row1) (row0 a multiple of d) of a table
  // whose `num_samples` masks are drawn from `seed`, the first rotated by
  // X^{-offset} for a grouped selector interval at `offset`.
  static absl::StatusOr<std::vector<Polynomial<CoeffType>>> StripeMasks(
      const RlweParams<CoeffType>& rlwe_params, absl::string_view seed,
      size_t num_samples, size_t offset, size_t row0, size_t row1) {
    const size_t d = rlwe_params.Degree();
    ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(seed));
    ASSIGN_OR_RETURN(auto a, SampleAComponents(rlwe_params, num_samples,
                                               *a_prng));
    if (offset > 0) {
      ASSIGN_OR_RETURN(a[0], RotateGroupMaskForInterval(a[0], offset,
                                                        rlwe_params.Modulus()));
    }
    return std::vector<Polynomial<CoeffType>>(a.begin() + row0 / d,
                                              a.begin() + (row1 + d - 1) / d);
  }

  // Turns hints of unsigned entries D into hints of D - kCenteredDbOffset:
  // subtracts kCenteredDbOffset * sum_{r < rows} a_r, the hint of an all-ones
  // row of `rows` entries, from every row of `hint`.
  static absl::Status SubtractCenteredOffset(
      absl::Span<std::vector<CoeffType>> hint, size_t rows,
      const std::vector<Polynomial<CoeffType>>& a, FftContext& ctx) {
    if constexpr (!std::is_same_v<DbValueType, uint16_t>) {
      return absl::InvalidArgumentError("centered_db requires uint16_t data.");
    }
    Matrix<DbValueType> ones(std::vector<DbValueType>(rows, 1), 1,
                             static_cast<int>(rows));
    ASSIGN_OR_RETURN(auto sum, (ComputeHintMatrix<CoeffType, DbValueType,
                                                  CoeffType>(ones, a, ctx)));
    for (auto& row : hint) {
      for (size_t j = 0; j < row.size(); ++j) {
        row[j] -= static_cast<CoeffType>(kCenteredDbOffset) * sum.Data()[j];
      }
    }
    return absl::OkStatus();
  }

  // PreprocessMatrixPack of every hint, with packing masks drawn from `prng`.
  static absl::StatusOr<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
  PackHints(const RlweParams<CoeffType>& rlwe_params,
            const GadgetParams& pack_gadget_params, ::rlwe::SecurePrng& prng,
            const std::vector<std::vector<std::vector<CoeffType>>>& hints) {
    ASSIGN_OR_RETURN(auto pack_a,
                     SampleAComponents(rlwe_params,
                                       2 * pack_gadget_params.num_digits, prng));
    const std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(), pack_a.begin() + pack_gadget_params.num_digits);
    const std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + pack_gadget_params.num_digits, pack_a.end());
    ASSIGN_OR_RETURN(auto fft_ctx,
                     FftContext::Create(absl::bit_width(static_cast<uint32_t>(
                                            rlwe_params.Degree())) -
                                        1));
    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>> outputs;
    outputs.reserve(hints.size());
    for (const auto& hint : hints) {
      ASSIGN_OR_RETURN(auto output,
                       (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                           rlwe_params, hint, w_vec_g, w_vec_h,
                           pack_gadget_params, *fft_ctx)));
      outputs.push_back(std::move(output));
    }
    return outputs;
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
