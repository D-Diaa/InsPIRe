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

#include "crypto/pir/server/pir_server.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/external_product.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "crypto/status_macros.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
    const PirParams<CoeffType>& params,
    PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
        preprocessed_data) {
  const int t = params.InterpolationDegree();
  const int d = params.RlweParameters().Degree();
  const int r = params.InterpolatedRows();
  const int entry_size_multiple = params.EntrySizeMultiple();
  if (preprocessed_data.db_matrices.size() != entry_size_multiple) {
    return absl::InvalidArgumentError("db_matrices size mismatch.");
  }
  const int db_cols = preprocessed_data.db_matrices[0].Cols();
  if (db_cols < r) {
    return absl::InvalidArgumentError(
        "DB matrix must have at least N/t columns.");
  }
  for (int k = 0; k < entry_size_multiple; ++k) {
    if (preprocessed_data.db_matrices[k].Rows() != t * d) {
      return absl::InvalidArgumentError("DB matrix must have td rows.");
    }
    if (preprocessed_data.db_matrices[k].Cols() != db_cols) {
      return absl::InvalidArgumentError(
          "All DB matrices must have the same number of columns.");
    }
  }

  if (preprocessed_data.preprocessed_outputs.size() != entry_size_multiple) {
    return absl::InvalidArgumentError("preprocessed_outputs size mismatch.");
  }
  for (int k = 0; k < entry_size_multiple; ++k) {
    if (preprocessed_data.preprocessed_outputs[k].size() != t) {
      return absl::InvalidArgumentError(
          "Preprocessed outputs must be of length t.");
    }
  }

  if (t > 1 && preprocessed_data.second_dim_a.size() !=
                   2 * params.PolyEvalGadgetParams().num_digits) {
    return absl::InvalidArgumentError(
        "second_dim_query_a must have length 2 * "
        "PolyEvalGadgetParams().num_digits.");
  }

  // Combine db_matrices
  ASSIGN_OR_RETURN(
      auto combined_db_matrix,
      Matrix<DbDataType>::VerticalConcatenate(preprocessed_data.db_matrices));

  // Combine pack matrices
  std::vector<Matrix<MatCoeffType>> pack_matrices;
  pack_matrices.reserve(entry_size_multiple * t);
  for (int k = 0; k < entry_size_multiple; ++k) {
    for (int i = 0; i < t; ++i) {
      pack_matrices.push_back(
          std::move(preprocessed_data.preprocessed_outputs[k][i].matrix));
      // Replace with empty matrix to avoid duplication
      preprocessed_data.preprocessed_outputs[k][i].matrix =
          Matrix<MatCoeffType>();
    }
  }
  ASSIGN_OR_RETURN(auto combined_pack_matrix,
                   Matrix<MatCoeffType>::VerticalConcatenate(pack_matrices));

  return CreateFromComponents(params, std::move(combined_db_matrix),
                              std::move(combined_pack_matrix),
                              std::move(preprocessed_data.preprocessed_outputs),
                              std::move(preprocessed_data.second_dim_a));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::CreateCrossColor(
    const PirParams<CoeffType>& params,
    std::vector<CrossColorDbSlice<DbDataType>> db_slices,
    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        pack_outputs) {
  if (params.InterpolationDegree() != 1) {
    return absl::InvalidArgumentError(
        "CreateCrossColor requires interpolation_degree == 1.");
  }
  const int entry_size_multiple = params.EntrySizeMultiple();
  if (db_slices.empty() ||
      pack_outputs.size() != static_cast<size_t>(entry_size_multiple)) {
    return absl::InvalidArgumentError(
        "CreateCrossColor slice or pack_outputs size mismatch.");
  }
  std::vector<Matrix<MatCoeffType>> pack_matrices;
  pack_matrices.reserve(entry_size_multiple);
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs(entry_size_multiple);
  for (int k = 0; k < entry_size_multiple; ++k) {
    pack_matrices.push_back(std::move(pack_outputs[k].matrix));
    pack_outputs[k].matrix = Matrix<MatCoeffType>();
    preprocessed_outputs[k].push_back(std::move(pack_outputs[k]));
  }
  ASSIGN_OR_RETURN(auto combined_pack_matrix,
                   Matrix<MatCoeffType>::VerticalConcatenate(pack_matrices));
  return CreateFromComponents(params, Matrix<DbDataType>(),
                              std::move(combined_pack_matrix),
                              std::move(preprocessed_outputs),
                              /*second_dim_query_a=*/{}, std::move(db_slices));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::CreateFromComponents(
    const PirParams<CoeffType>& params, Matrix<DbDataType> combined_db_matrix,
    Matrix<MatCoeffType> combined_pack_matrix,
    std::vector<
        std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
        preprocessed_outputs,
    std::vector<Polynomial<CoeffType>> second_dim_query_a,
    std::vector<CrossColorDbSlice<DbDataType>> db_slices) {
  const int d = params.RlweParameters().Degree();
  ASSIGN_OR_RETURN(
      auto fft_ctx,
      FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  ASSIGN_OR_RETURN(auto t_vec_h_ffts,
                   PrecomputeTVecHFfts(params, preprocessed_outputs, *fft_ctx));

  std::vector<Polynomial<CoeffType>> a_mod_switched_t1;
  std::vector<std::string> a_mod_switched_t1_packed;
  if (params.InterpolationDegree() == 1) {
    a_mod_switched_t1.reserve(preprocessed_outputs.size());
    a_mod_switched_t1_packed.reserve(preprocessed_outputs.size());
    const int log_q1 = params.RlweParameters().LogModulus1AfterSwitch();
    for (size_t k = 0; k < preprocessed_outputs.size(); ++k) {
      ASSIGN_OR_RETURN(
          auto a_ms,
          preprocessed_outputs[k][0].a_tilde_agg.Rescale(
              params.RlweParameters().LogModulus(), log_q1));
      a_mod_switched_t1_packed.push_back(
          Polynomial<CoeffType>::PackCoefficients(a_ms.Coeffs(), log_q1));
      a_mod_switched_t1.push_back(std::move(a_ms));
    }
  }

  return absl::WrapUnique(new PirServer<DbDataType, CoeffType, MatCoeffType>(
      params, std::move(combined_db_matrix), std::move(combined_pack_matrix),
      std::move(preprocessed_outputs), std::move(t_vec_h_ffts),
      std::move(a_mod_switched_t1), std::move(a_mod_switched_t1_packed),
      std::move(second_dim_query_a), std::move(fft_ctx),
      std::move(db_slices)));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::vector<std::vector<ChunkedFft>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::PrecomputeTVecHFfts(
    const PirParams<CoeffType>& params,
    const std::vector<
        std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>&
        preprocessed_outputs,
    FftContext& fft_ctx) {
  std::vector<std::vector<ChunkedFft>> t_vec_h_ffts(
      preprocessed_outputs.size());
  for (size_t k = 0; k < preprocessed_outputs.size(); ++k) {
    t_vec_h_ffts[k].reserve(preprocessed_outputs[k].size());
    for (const auto& output : preprocessed_outputs[k]) {
      ASSIGN_OR_RETURN(
          ChunkedFft fft,
          TVecHFft(output.t_vec_h, params.PackGadgetParams(), fft_ctx));
      t_vec_h_ffts[k].push_back(std::move(fft));
    }
  }
  return t_vec_h_ffts;
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>>
PirServer<DbDataType, CoeffType, MatCoeffType>::EvalPoly(
    const std::vector<RlweCiphertext<CoeffType>>& coeffs,
    const RgswCiphertext<CoeffType>& eval_point, FftContext& ctx) const {
  if (coeffs.empty()) {
    return absl::InvalidArgumentError("coeffs vector must not be empty");
  }

  // Implementing Horner's Method:
  // p(x) = c_0 + x * (c_1 + x * (... + x * c_{t-1} ))
  RlweCiphertext<CoeffType> result = coeffs.back();

  const auto& params = params_.RlweParameters();
  for (int i = coeffs.size() - 2; i >= 0; --i) {
    ASSIGN_OR_RETURN(RlweCiphertext<CoeffType> product,
                     ExternalProduct(params, params_.PolyEvalGadgetParams(),
                                     result, eval_point, ctx));
    ASSIGN_OR_RETURN(Polynomial<CoeffType> next_a, product.a.Add(coeffs[i].a));
    ASSIGN_OR_RETURN(Polynomial<CoeffType> next_b, product.b.Add(coeffs[i].b));
    result = {std::move(next_a), std::move(next_b)};
  }

  return result;
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<PirResponse<CoeffType>>
PirServer<DbDataType, CoeffType, MatCoeffType>::ProcessResponse(
    const PirRequest<CoeffType>& request, const ChunkedFft* y_vec_h_fft,
    bool skip_a_t1) const {
  const int d = params_.RlweParameters().Degree();
  const int t = params_.InterpolationDegree();
  const int entry_size_multiple = params_.EntrySizeMultiple();

  // Extract y_vec_g and y_vec_h from the request packing key.
  const int num_digits = params_.PackGadgetParams().num_digits;
  if (request.packing_key.size() != 2 * num_digits) {
    return absl::InvalidArgumentError(
        "Packing key must contain 2 * num_digits polynomials.");
  }
  auto pack_begin = request.packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                             pack_begin + num_digits);
  std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                             pack_begin + 2 * num_digits);

  // Stage 1: First dimension query processing via matrix-vector multiplication.
  const int db_cols = combined_db_matrix_.Cols();
  if (db_slices_.empty()) {
    if (request.first_dimension_query.size() <
            static_cast<size_t>(params_.InterpolatedRows()) ||
        request.first_dimension_query.size() > static_cast<size_t>(db_cols)) {
      return absl::InvalidArgumentError(
          absl::StrCat("First dimension query size (",
                       request.first_dimension_query.size(),
                       ") is invalid for DB columns (", db_cols, ")."));
    }
  } else if (request.color_queries == nullptr) {
    return absl::InvalidArgumentError(
        "Cross-color server requires request.color_queries.");
  }

  // Stage 3 (setup): Second dimension query processing via polynomial
  // evaluation.
  RgswCiphertext<CoeffType> rgsw_ct;
  if (t > 1) {
    const int eval_num_digits = params_.PolyEvalGadgetParams().num_digits;
    if (request.second_dimension_query.size() != 2 * eval_num_digits) {
      return absl::InvalidArgumentError(
          "second_dimension_query size mismatch. Must be 2 * num_digits.");
    }
    rgsw_ct.u.reserve(eval_num_digits);
    rgsw_ct.v.reserve(eval_num_digits);
    for (int i = 0; i < eval_num_digits; ++i) {
      rgsw_ct.u.push_back(
          {second_dim_query_a_[i], request.second_dimension_query[i]});
    }
    for (int i = eval_num_digits; i < 2 * eval_num_digits; ++i) {
      rgsw_ct.v.push_back(
          {second_dim_query_a_[i], request.second_dimension_query[i]});
    }
  }

  FftContext* fft_ctx = fft_ctx_.get();

  std::vector<RlweCiphertext<CoeffType>> shard_responses;
  shard_responses.reserve(entry_size_multiple);

  // ONE large DB multiplication (or per-slice multiplication for cross-color).
  auto start_db = absl::Now();
  std::vector<CoeffType> b_prime_all;
  if (db_slices_.empty()) {
    if (request.first_dimension_query.size() == static_cast<size_t>(db_cols)) {
      ASSIGN_OR_RETURN(b_prime_all,
                       combined_db_matrix_.template Multiply<CoeffType>(
                           request.first_dimension_query));
    } else {
      std::vector<CoeffType> padded_first_dim(db_cols, 0);
      std::copy(request.first_dimension_query.begin(),
                request.first_dimension_query.end(), padded_first_dim.begin());
      ASSIGN_OR_RETURN(b_prime_all,
                       combined_db_matrix_.template Multiply<CoeffType>(
                           padded_first_dim));
    }
  } else {
    b_prime_all.assign(static_cast<size_t>(entry_size_multiple * t * d), 0);
    size_t out_off = 0;
    for (const auto& sl : db_slices_) {
      if (sl.color >= request.color_queries->size()) {
        return absl::InvalidArgumentError(
            "color_queries index out of bounds.");
      }
      const auto& q_vec = (*request.color_queries)[sl.color];
      if (q_vec.empty() ||
          q_vec.size() > static_cast<size_t>(sl.padded_cols)) {
        return absl::InvalidArgumentError(
            "Invalid color query length for cross-color DB slice.");
      }
      std::vector<CoeffType> part;
      if (q_vec.size() == static_cast<size_t>(sl.padded_cols)) {
        ASSIGN_OR_RETURN(part,
                         sl.db_submatrix.template Multiply<CoeffType>(q_vec));
      } else {
        std::vector<CoeffType> padded_q(sl.padded_cols, 0);
        std::copy(q_vec.begin(), q_vec.end(), padded_q.begin());
        ASSIGN_OR_RETURN(
            part, sl.db_submatrix.template Multiply<CoeffType>(padded_q));
      }
      std::copy(part.begin(), part.end(), b_prime_all.begin() + out_off);
      out_off += static_cast<size_t>(sl.rows);
    }
  }
  db_mult_time_ += (absl::Now() - start_db);

  if (b_prime_all.size() != static_cast<size_t>(entry_size_multiple * t * d)) {
    return absl::InternalError(
        "DB multiplication resulted in incorrect vector size.");
  }

  // ONE large packing multiplication
  auto start_pack = absl::Now();
  const int k_gadget = y_vec_g.size();
  std::vector<CoeffType> y_vec_g_vec(k_gadget * d);
  for (int l = 0; l < k_gadget; ++l) {
    for (int m = 0; m < d; ++m) {
      y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
    }
  }

  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_agg_partial_all,
      combined_pack_matrix_.template Multiply<CoeffType>(y_vec_g_vec));

  if (b_agg_partial_all.size() !=
      static_cast<size_t>(entry_size_multiple * t * d)) {
    return absl::InternalError(
        "Packing multiplication resulted in incorrect vector size.");
  }
  const auto matrix_elapsed = absl::Now() - start_pack;
  packing_time_ += matrix_elapsed;
  pack_matrix_time_ += matrix_elapsed;

  // The packing key's y_vec_h is shared by every shard and chunk below.
  ChunkedFft own_y_vec_h_fft;
  if (y_vec_h_fft == nullptr) {
    start_pack = absl::Now();
    ASSIGN_OR_RETURN(own_y_vec_h_fft,
                     YVecHFft(params_.RlweParameters(), y_vec_h, *fft_ctx));
    const auto y_fft_elapsed = absl::Now() - start_pack;
    packing_time_ += y_fft_elapsed;
    finalize_time_ += y_fft_elapsed;
    y_vec_h_fft = &own_y_vec_h_fft;
  }

  const int log_q = params_.RlweParameters().LogModulus();
  const int log_q1 = params_.RlweParameters().LogModulus1AfterSwitch();
  const int log_q2 = params_.RlweParameters().LogModulus2AfterSwitch();
  const CoeffType q_mask =
      (log_q <= 0 || log_q >= static_cast<int>(8 * sizeof(CoeffType)))
          ? ~CoeffType{0}
          : ((CoeffType{1} << log_q) - 1);

  for (int k = 0; k < entry_size_multiple; ++k) {
    start_pack = absl::Now();
    std::vector<RlweCiphertext<CoeffType>> packed_ciphertexts;
    packed_ciphertexts.reserve(t);
    for (int i = 0; i < t; ++i) {
      const size_t offset = static_cast<size_t>(k * t + i) * d;
      ASSIGN_OR_RETURN(
          Polynomial<CoeffType> b_agg,
          Polynomial<CoeffType>::InnerProductFft(t_vec_h_ffts_[k][i],
                                                 *y_vec_h_fft, *fft_ctx));
      CoeffType* __restrict b_ptr = b_agg.MutableCoeffs().data();
      const CoeffType* __restrict bp_ptr = b_prime_all.data() + offset;
      const CoeffType* __restrict ba_ptr = b_agg_partial_all.data() + offset;
      for (int m = 0; m < d; ++m) {
        b_ptr[m] = (b_ptr[m] + bp_ptr[m] + ba_ptr[m]) & q_mask;
      }
      Polynomial<CoeffType> a_tilde;
      if (t > 1) {
        a_tilde = preprocessed_outputs_[k][i].a_tilde_agg;
      }
      packed_ciphertexts.push_back({std::move(a_tilde), std::move(b_agg)});
    }
    const auto finalize_elapsed = absl::Now() - start_pack;
    packing_time_ += finalize_elapsed;
    finalize_time_ += finalize_elapsed;

    RlweCiphertext<CoeffType> final_ciphertext;
    if (t == 1) {
      final_ciphertext = std::move(packed_ciphertexts[0]);
    } else {
      auto start_eval = absl::Now();
      ASSIGN_OR_RETURN(final_ciphertext,
                       EvalPoly(packed_ciphertexts, rgsw_ct, *fft_ctx));
      poly_eval_time_ += (absl::Now() - start_eval);
    }

    const auto start_modswitch = absl::Now();
    Polynomial<CoeffType> a_mod_switched;
    if (t == 1) {
      if (!skip_a_t1 && static_cast<size_t>(k) < a_mod_switched_t1_.size()) {
        a_mod_switched = a_mod_switched_t1_[k];
      }
    } else {
      ASSIGN_OR_RETURN(a_mod_switched,
                       final_ciphertext.a.Rescale(log_q, log_q1));
    }

    CoeffType* __restrict b_ptr = final_ciphertext.b.MutableCoeffs().data();
    if (log_q > log_q2) {
      const int shift = log_q - log_q2;
      const CoeffType half = CoeffType{1} << (shift - 1);
      for (int m = 0; m < d; ++m) {
        b_ptr[m] = static_cast<CoeffType>(
            (static_cast<absl::uint128>(b_ptr[m]) + half) >> shift);
      }
    } else if (log_q < log_q2) {
      const int shift = log_q2 - log_q;
      for (int m = 0; m < d; ++m) {
        b_ptr[m] <<= shift;
      }
    }

    shard_responses.push_back(
        {std::move(a_mod_switched), std::move(final_ciphertext.b)});
    modswitch_time_ += absl::Now() - start_modswitch;
  }

  return PirResponse<CoeffType>{std::move(shard_responses)};
}

template class PirServer<uint16_t, uint32_t, uint32_t>;
template class PirServer<uint16_t, uint64_t, uint64_t>;
template class PirServer<uint16_t, uint32_t, int16_t>;
template class PirServer<uint16_t, uint64_t, int16_t>;
template class PirServer<uint16_t, uint32_t, int32_t>;
template class PirServer<uint16_t, uint64_t, int32_t>;

template class PirServer<uint8_t, uint32_t, uint32_t>;
template class PirServer<uint8_t, uint64_t, uint64_t>;
template class PirServer<uint8_t, uint32_t, int16_t>;
template class PirServer<uint8_t, uint64_t, int16_t>;
template class PirServer<uint8_t, uint32_t, int32_t>;
template class PirServer<uint8_t, uint64_t, int32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
