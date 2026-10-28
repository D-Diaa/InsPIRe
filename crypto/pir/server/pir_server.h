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

#ifndef CRYPTO_PIR_SERVER_PIR_SERVER_H_
#define CRYPTO_PIR_SERVER_PIR_SERVER_H_

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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// PirServer represents the server side of a Private Information Retrieval (PIR)
// protocol.
template <typename DbDataType, typename CoeffType, typename MatCoeffType>
class PirServer {
 public:
  // Creates a PirServer instance.
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  Create(const PirParams<CoeffType>& params,
         PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
             preprocessed_data);

  // Creates a t=1 server holding only the Stage 1 database `db_matrices` (one
  // per column) of selector rows [row0, row0 + cols) of a taller table: it
  // answers ComputeStage1 only, and its outputs are summed mod 2^64 with the
  // other row stripes' before PackFromStage1 on the server owning row 0.
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  CreateStage1Only(const PirParams<CoeffType>& params,
                   std::vector<Matrix<DbDataType>> db_matrices, size_t row0);

  // Creates a t=1 PirServer instance that packs LWE outputs concatenated
  // across multiple color slices `db_slices` into `pack_outputs.size()` RLWE
  // response ciphertexts. With empty `pack_outputs` the server answers
  // ComputeStage1 only (one row stripe of the slices' colors).
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  CreateCrossColor(
      const PirParams<CoeffType>& params,
      std::vector<CrossColorDbSlice<DbDataType>> db_slices,
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
          pack_outputs);

  // Serializes the server's preprocessed data and database to a binary file on
  // disk.
  absl::Status SaveToFile(absl::string_view file_path) const;

  // Loads the server's preprocessed data and database from a byte buffer.
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  LoadFromBuffer(const PirParams<CoeffType>& params, const uint8_t* buffer,
                 size_t size);

  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  LoadFromFile(const PirParams<CoeffType>& params, absl::string_view file_path);

  // Processes a PirRequest to produce a PirResponse. `y_vec_h_fft`, if given,
  // must be YVecHFft of the request's y_vec_h (the last num_digits packing key
  // polynomials), so that a caller answering one request with several servers
  // transforms the shared key only once.
  // Equivalent to PackFromStage1(request, ComputeStage1(request), ...).
  absl::StatusOr<PirResponse<CoeffType>> ProcessResponse(
      const PirRequest<CoeffType>& request,
      const ChunkedFft* y_vec_h_fft = nullptr,
      bool skip_a_t1 = false) const;

  // Stage 1: this server's (row stripe's) LWE "b" values, entry_size_multiple
  // * t * d of them, correct modulo q.
  absl::StatusOr<std::vector<CoeffType>> ComputeStage1(
      const PirRequest<CoeffType>& request) const;

  // Packing, polynomial evaluation and modulus switching of `b_prime`, the sum
  // mod 2^64 of ComputeStage1 over every row stripe of this server's table.
  absl::StatusOr<PirResponse<CoeffType>> PackFromStage1(
      const PirRequest<CoeffType>& request, std::vector<CoeffType> b_prime,
      const ChunkedFft* y_vec_h_fft = nullptr, bool skip_a_t1 = false) const;

  const std::string& GetPrepackedAModSwitched(int k) const {
    return a_mod_switched_t1_packed_[k];
  }

  absl::Duration GetDbMultiplyTime() const { return db_mult_time_; }
  absl::Duration GetPackingTime() const { return packing_time_; }
  absl::Duration GetPackMatrixTime() const { return pack_matrix_time_; }
  absl::Duration GetFinalizeTime() const { return finalize_time_; }
  absl::Duration GetModswitchTime() const { return modswitch_time_; }
  int DbCols() const { return combined_db_matrix_.Cols(); }
  int DbRows() const { return combined_db_matrix_.Rows(); }
  int PackCols() const { return combined_pack_matrix_.Cols(); }
  int PackRows() const { return combined_pack_matrix_.Rows(); }
  absl::Duration GetPolyEvalTime() const { return poly_eval_time_; }

 private:
  PirServer(
      const PirParams<CoeffType>& params, Matrix<DbDataType> combined_db_matrix,
      Matrix<MatCoeffType> combined_pack_matrix,
      std::vector<
          std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
          preprocessed_outputs,
      std::vector<std::vector<ChunkedFft>> t_vec_h_ffts,
      std::vector<Polynomial<CoeffType>> a_mod_switched_t1,
      std::vector<std::string> a_mod_switched_t1_packed,
      std::vector<Polynomial<CoeffType>> second_dim_query_a,
      std::unique_ptr<FftContext> fft_ctx,
      std::vector<CrossColorDbSlice<DbDataType>> db_slices, size_t row0)
      : params_(params),
        combined_db_matrix_(std::move(combined_db_matrix)),
        combined_pack_matrix_(std::move(combined_pack_matrix)),
        preprocessed_outputs_(std::move(preprocessed_outputs)),
        t_vec_h_ffts_(std::move(t_vec_h_ffts)),
        a_mod_switched_t1_(std::move(a_mod_switched_t1)),
        a_mod_switched_t1_packed_(std::move(a_mod_switched_t1_packed)),
        second_dim_query_a_(std::move(second_dim_query_a)),
        fft_ctx_(std::move(fft_ctx)),
        db_slices_(std::move(db_slices)),
        row0_(row0),
        db_mult_time_(absl::ZeroDuration()),
        packing_time_(absl::ZeroDuration()),
        poly_eval_time_(absl::ZeroDuration()) {}

  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  CreateFromComponents(
      const PirParams<CoeffType>& params, Matrix<DbDataType> combined_db_matrix,
      Matrix<MatCoeffType> combined_pack_matrix,
      std::vector<
          std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
          preprocessed_outputs,
      std::vector<Polynomial<CoeffType>> second_dim_query_a,
      std::vector<CrossColorDbSlice<DbDataType>> db_slices = {},
      size_t row0 = 0);

  // Stage 1 product of `mat` with selector entries [row0, row0 + mat.Cols())
  // of `query` (clipped and zero-padded), correct modulo q.
  absl::StatusOr<std::vector<CoeffType>> MultiplyStripe(
      const Matrix<DbDataType>& mat, const std::vector<CoeffType>& query,
      size_t row0) const;

  // Forward FFTs of every preprocessed t_vec_h, indexed like
  // `preprocessed_outputs`, computed once so that ProcessResponse does not
  // transform them per request.
  static absl::StatusOr<std::vector<std::vector<ChunkedFft>>>
  PrecomputeTVecHFfts(
      const PirParams<CoeffType>& params,
      const std::vector<
          std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>&
          preprocessed_outputs,
      FftContext& fft_ctx);

  // Evaluates a polynomial where coefficients are given by `coeffs` using
  // Horner's method at the point x = `eval_point`.
  // Here, `coeffs` encrypts the coefficients c_0, c_1, ..., c_{t-1}.
  absl::StatusOr<RlweCiphertext<CoeffType>> EvalPoly(
      const std::vector<RlweCiphertext<CoeffType>>& coeffs,
      const RgswCiphertext<CoeffType>& eval_point, FftContext& ctx) const;

  const PirParams<CoeffType> params_;
  const Matrix<DbDataType> combined_db_matrix_;
  const Matrix<MatCoeffType> combined_pack_matrix_;
  const std::vector<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs_;
  const std::vector<std::vector<ChunkedFft>> t_vec_h_ffts_;
  const std::vector<Polynomial<CoeffType>> a_mod_switched_t1_;
  const std::vector<std::string> a_mod_switched_t1_packed_;
  const std::vector<Polynomial<CoeffType>> second_dim_query_a_;
  std::unique_ptr<FftContext> fft_ctx_;
  const std::vector<CrossColorDbSlice<DbDataType>> db_slices_;
  // First selector row contracted by combined_db_matrix_ (row stripes).
  const size_t row0_;

  mutable absl::Duration pack_matrix_time_ = absl::ZeroDuration();
  mutable absl::Duration finalize_time_ = absl::ZeroDuration();
  mutable absl::Duration modswitch_time_ = absl::ZeroDuration();
  mutable absl::Duration db_mult_time_;
  mutable absl::Duration packing_time_;
  mutable absl::Duration poly_eval_time_;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_SERVER_PIR_SERVER_H_
