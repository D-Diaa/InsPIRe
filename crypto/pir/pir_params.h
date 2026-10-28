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

#ifndef CRYPTO_PIR_PIR_PARAMS_H_
#define CRYPTO_PIR_PIR_PARAMS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/polynomial.h"
#include "crypto/proto/pir.pb.h"
#include "absl/numeric/bits.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Parameters for PIR.
//
// We treat each database entry as Z_p^d where d is the ring degree and p is the
// plaintext modulus.
template <typename CoeffType>
class PirParams {
 public:
  // Creates and validates a new PirParams instance.
  // entry_size_multiple is the number of RLWE plaintexts needed to encode each
  // database entry.
  // Returns InvalidArgumentError if:
  // - `num_entries` <= 0
  // - `interpolation_degree` is not a power of 2
  // - `interpolation_degree` does not divide num_entries
  // - `interpolation_degree` > 2 * rlwe_params.Degree()
  static absl::StatusOr<PirParams<CoeffType>> Create(
      const RlweParams<CoeffType>& rlwe_params, CoeffType plaintext_modulus,
      const GadgetParams& pack_gadget_params,
      const GadgetParams& poly_eval_gadget_params, int num_entries,
      int interpolation_degree, int entry_size_multiple = 1);

  // Accessors
  const RlweParams<CoeffType>& RlweParameters() const { return rlwe_params_; }

  CoeffType PlaintextModulus() const { return plaintext_modulus_; }

  const GadgetParams& PackGadgetParams() const { return pack_gadget_params_; }

  const GadgetParams& PolyEvalGadgetParams() const {
    return poly_eval_gadget_params_;
  }

  int NumEntries() const { return num_entries_; }

  int InterpolationDegree() const { return interpolation_degree_; }

  int EntrySizeMultiple() const { return entry_size_multiple_; }

  // Number of rows in the interpolated database (columns of the transposed DB).
  int InterpolatedRows() const { return num_entries_ / interpolation_degree_; }

  // Number of RLWE ciphertext samples needed for the first dimension query.
  int NumFirstDimSamples() const {
    return (InterpolatedRows() + rlwe_params_.Degree() - 1) /
           rlwe_params_.Degree();
  }

  // Number of columns padded to a multiple of ring degree for offline hint
  // computation.
  int PaddedCols() const {
    return NumFirstDimSamples() * rlwe_params_.Degree();
  }

 private:
  PirParams(const RlweParams<CoeffType>& rlwe_params,
            CoeffType plaintext_modulus, const GadgetParams& pack_gadget_params,
            const GadgetParams& poly_eval_gadget_params, int num_entries,
            int interpolation_degree, int entry_size_multiple)
      : rlwe_params_(rlwe_params),
        plaintext_modulus_(plaintext_modulus),
        pack_gadget_params_(pack_gadget_params),
        poly_eval_gadget_params_(poly_eval_gadget_params),
        num_entries_(num_entries),
        interpolation_degree_(interpolation_degree),
        entry_size_multiple_(entry_size_multiple) {}

  const RlweParams<CoeffType> rlwe_params_;
  const CoeffType plaintext_modulus_;
  const GadgetParams pack_gadget_params_;
  const GadgetParams poly_eval_gadget_params_;
  const int num_entries_;
  const int interpolation_degree_;
  const int entry_size_multiple_;
};

// Rows [first, last) of row stripe `stripe` out of `num_stripes` for a table of
// `rows` rows, splitting its ceil(rows / d) d-row blocks as evenly as possible
// (stripe 0 is never empty). Empty when first == last.
inline std::pair<size_t, size_t> RowStripe(size_t rows, size_t d, size_t stripe,
                                           size_t num_stripes) {
  const size_t blocks = (rows + d - 1) / d;
  auto edge = [&](size_t s) {
    return std::min(rows, (s * blocks + num_stripes - 1) / num_stripes * d);
  };
  return {edge(stripe), edge(stripe + 1)};
}

extern template class PirParams<uint32_t>;
extern template class PirParams<uint64_t>;

template <typename CoeffType>
absl::StatusOr<proto::PirParams> SerializePirParams(
    const PirParams<CoeffType>& params,
    proto::PirParams::DbValueType db_value_type,
    proto::PirParams::MatCoeffType mat_coeff_type) {
  proto::PirParams proto;

  // Serialize RlweParams
  auto* rlwe_proto = proto.mutable_rlwe_params();
  const auto& rlwe_params = params.RlweParameters();
  rlwe_proto->set_log_degree(
      absl::bit_width(static_cast<uint32_t>(rlwe_params.Degree() - 1)));
  rlwe_proto->set_log_modulus(rlwe_params.LogModulus());
  rlwe_proto->set_log_modulus1_after_switch(
      rlwe_params.LogModulus1AfterSwitch());
  rlwe_proto->set_log_modulus2_after_switch(
      rlwe_params.LogModulus2AfterSwitch());
  rlwe_proto->set_variance(rlwe_params.Variance());

  proto.set_plaintext_modulus(static_cast<int64_t>(params.PlaintextModulus()));

  // Serialize GadgetParams
  auto* pack_proto = proto.mutable_pack_gadget_params();
  pack_proto->set_log_digit(params.PackGadgetParams().log_digit);
  pack_proto->set_num_digits(params.PackGadgetParams().num_digits);

  auto* eval_proto = proto.mutable_poly_eval_gadget_params();
  eval_proto->set_log_digit(params.PolyEvalGadgetParams().log_digit);
  eval_proto->set_num_digits(params.PolyEvalGadgetParams().num_digits);

  proto.set_num_entries(params.NumEntries());
  proto.set_interpolation_degree(params.InterpolationDegree());
  proto.set_entry_size_multiple(params.EntrySizeMultiple());

  // Set type parameters
  proto.set_db_value_type(db_value_type);
  if constexpr (std::is_same_v<CoeffType, uint32_t>) {
    proto.set_coeff_type(proto::PirParams::COEFF_TYPE_UINT32);
  } else if constexpr (std::is_same_v<CoeffType, uint64_t>) {
    proto.set_coeff_type(proto::PirParams::COEFF_TYPE_UINT64);
  }
  proto.set_mat_coeff_type(mat_coeff_type);

  return proto;
}

template <typename CoeffType>
absl::StatusOr<PirParams<CoeffType>> DeserializePirParams(
    const proto::PirParams& proto) {
  // Reconstruct RlweParams
  const auto& rlwe_proto = proto.rlwe_params();
  int degree = 1 << rlwe_proto.log_degree();
  CoeffType modulus = rlwe_proto.log_modulus() == 8 * sizeof(CoeffType)
                          ? 0
                          : (CoeffType{1} << rlwe_proto.log_modulus());
  CoeffType modulus1 =
      rlwe_proto.log_modulus1_after_switch() == 8 * sizeof(CoeffType)
          ? 0
          : (CoeffType{1} << rlwe_proto.log_modulus1_after_switch());
  CoeffType modulus2 =
      rlwe_proto.log_modulus2_after_switch() == 8 * sizeof(CoeffType)
          ? 0
          : (CoeffType{1} << rlwe_proto.log_modulus2_after_switch());

  ASSIGN_OR_RETURN(auto rlwe_params, RlweParams<CoeffType>::Create(
                                         degree, modulus, modulus1, modulus2,
                                         rlwe_proto.variance()));

  GadgetParams pack_gadget_params{
      .log_digit = proto.pack_gadget_params().log_digit(),
      .num_digits = proto.pack_gadget_params().num_digits(),
  };

  GadgetParams poly_eval_gadget_params{
      .log_digit = proto.poly_eval_gadget_params().log_digit(),
      .num_digits = proto.poly_eval_gadget_params().num_digits(),
  };

  return PirParams<CoeffType>::Create(
      rlwe_params, static_cast<CoeffType>(proto.plaintext_modulus()),
      pack_gadget_params, poly_eval_gadget_params, proto.num_entries(),
      proto.interpolation_degree(), proto.entry_size_multiple());
}

// A request object generated by the client, consisting of "b" components of the
// RLWE ciphertexts.
template <typename CoeffType>
struct PirRequest {
  // The part of PIR query corresponding to the first dimension of database
  // items.
  std::vector<CoeffType> first_dimension_query;

  // The part of PIR query corresponding to the second dimension of database
  // items.
  std::vector<Polynomial<CoeffType>> second_dimension_query;

  // The key used for packing..
  std::vector<Polynomial<CoeffType>> packing_key;

  // Optional pointer to all colors' Stage 1 query vectors for cross-color
  // response servers.
  const std::vector<std::vector<CoeffType>>* color_queries = nullptr;
};

// A response object containing the result of the PIR query.
template <typename CoeffType>
struct PirResponse {
  // The ciphertexts containing the queried item shards, encrypted under
  // client's secret key.
  std::vector<RlweCiphertext<CoeffType>> ciphertexts;
};

// One color's disjoint coefficient interval [offset, offset + rows) inside a
// degree-d grouped selector encryption.
struct SelectorSliceSpec {
  size_t slot = 0;
  size_t offset = 0;
  size_t rows = 0;
};

// A group of whole colors packed into one degree-d selector encryption, or a
// single color taller than d spread over `num_samples` = ceil(rows / d) RLWE
// samples (offset 0) under the same secret.
struct SelectorGroupPlan {
  size_t group_index = 0;
  size_t total_rows = 0;
  std::vector<SelectorSliceSpec> slices;
  size_t num_samples = 1;
};

// Greedily partitions consecutive colors with row counts `rows_per_slot` into
// whole-color groups of total height <= `ring_degree`; a color taller than
// `ring_degree` gets a multi-sample group of its own.
inline absl::StatusOr<std::vector<SelectorGroupPlan>> BuildSelectorGroupPlans(
    const std::vector<int>& rows_per_slot, size_t ring_degree = 2048) {
  if (rows_per_slot.empty() || ring_degree == 0) {
    return absl::InvalidArgumentError(
        "rows_per_slot and ring_degree must be positive.");
  }
  std::vector<SelectorGroupPlan> groups;
  SelectorGroupPlan current;
  size_t offset = 0;
  auto flush = [&] {
    current.group_index = groups.size();
    current.total_rows = offset;
    groups.push_back(std::move(current));
    current = SelectorGroupPlan{};
    offset = 0;
  };
  for (size_t slot = 0; slot < rows_per_slot.size(); ++slot) {
    if (rows_per_slot[slot] <= 0) {
      return absl::InvalidArgumentError("Each slot row count must be positive.");
    }
    const size_t r = static_cast<size_t>(rows_per_slot[slot]);
    const bool tall = r > ring_degree;
    if (!current.slices.empty() && (tall || offset + r > ring_degree)) flush();
    current.slices.push_back(SelectorSliceSpec{slot, offset, r});
    offset += r;
    if (tall) {
      current.num_samples = (r + ring_degree - 1) / ring_degree;
      flush();
    }
  }
  if (!current.slices.empty()) flush();
  return groups;
}

// Selector bits per physical row by default: 40-bit (5 B) selectors stay
// noise-safe to about 10k rows with signed entries, 42-bit ones (same 3-digit
// Stage 1 kernel) to about 168k.
inline int AutoSelectorBits(size_t max_rows) { return max_rows > 8192 ? 42 : 40; }

// LWE outputs a response block spends on a row of `row_u16` words: rows of at
// most d / 2 words share blocks across colors, wider rows take whole blocks.
inline size_t PackedRowWords(size_t row_u16, size_t d = 2048) {
  return row_u16 <= d / 2 ? row_u16 : (row_u16 + d - 1) / d * d;
}

// Plaintext row bundling: `bundle` consecutive logical rows of `row_u16` words
// form one physical PIR row of bundle * row_u16 words, the trailing partial
// bundle zero-padded. The client selects physical row target / bundle and keeps
// its words [(target % bundle) * row_u16, +row_u16). Logical rows are
// `src_stride` words apart, physical rows `dst_stride`.
inline std::vector<uint16_t> BundleRows(const std::vector<uint16_t>& logical,
                                        size_t rows, size_t src_stride,
                                        size_t row_u16, size_t bundle,
                                        size_t dst_stride) {
  std::vector<uint16_t> phys((rows + bundle - 1) / bundle * dst_stride, 0);
  for (size_t r = 0; r < rows; ++r) {
    std::copy_n(logical.begin() + r * src_stride, row_u16,
                phys.begin() + r / bundle * dst_stride + r % bundle * row_u16);
  }
  return phys;
}

// Exact wire bytes of one batch request (four 52-bit packing key polynomials,
// one selector word per physical row, one 11,008-byte response per block) over
// colors of `rows` logical rows of `row_u16` words bundled by `bundle`, with
// `selector_bits` (0: AutoSelectorBits of the tallest physical color).
inline size_t BundledWireBytes(const std::vector<int>& rows, size_t row_u16,
                               size_t bundle, int selector_bits,
                               size_t* responses = nullptr) {
  constexpr size_t kD = 2048, kKeyBytes = 4 * (kD * 52 / 8);
  constexpr size_t kResponseBytes = (kD * 25 + kD * 18) / 8;
  size_t max_phys = 0;
  for (int r : rows) max_phys = std::max<size_t>(max_phys, (r + bundle - 1) / bundle);
  const size_t bits = selector_bits ? selector_bits : AutoSelectorBits(max_phys);
  size_t upload = kKeyBytes;
  for (int r : rows) upload += ((r + bundle - 1) / bundle * bits + 7) / 8;
  const size_t n_resp =
      (rows.size() * PackedRowWords(bundle * row_u16, kD) + kD - 1) / kD;
  if (responses != nullptr) *responses = n_resp;
  return upload + n_resp * kResponseBytes;
}

// The bundle factor in [1, 64] minimizing BundledWireBytes, ties broken toward
// fewer responses and then smaller bundles; 1 when no color exceeds 2048 rows.
inline size_t ChooseBundleFactor(const std::vector<int>& rows, size_t row_u16,
                                 int selector_bits = 0) {
  const size_t max_rows = *std::max_element(rows.begin(), rows.end());
  if (max_rows <= 2048) return 1;
  size_t best = 1, best_bytes = SIZE_MAX, best_resp = SIZE_MAX;
  for (size_t bundle = 1; bundle <= std::min<size_t>(64, max_rows); ++bundle) {
    size_t resp;
    const size_t bytes =
        BundledWireBytes(rows, row_u16, bundle, selector_bits, &resp);
    if (bytes < best_bytes || (bytes == best_bytes && resp < best_resp)) {
      best = bundle, best_bytes = bytes, best_resp = resp;
    }
  }
  return best;
}

// Computes the signed negacyclic rotation X^{-offset} * group_mask(X) mod
// (X^d + 1), whose first R_i negacyclic-matrix rows (for offset + R_i <= d)
// equal the extraction rows [offset, offset + R_i) of `group_mask`'s
// negacyclic matrix.
template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> RotateGroupMaskForInterval(
    const Polynomial<CoeffType>& group_mask, size_t offset,
    CoeffType modulus) {
  const auto& src = group_mask.Coeffs();
  const size_t d = src.size();
  if (offset >= d) {
    return absl::InvalidArgumentError("Interval offset must be < ring degree.");
  }
  if (offset == 0) {
    return group_mask;
  }
  std::vector<CoeffType> rotated(d, 0);
  for (size_t r = 0; r < d - offset; ++r) {
    rotated[r] = src[r + offset];
  }
  for (size_t r = d - offset; r < d; ++r) {
    const CoeffType val = src[r + offset - d];
    rotated[r] = (val == 0) ? 0 : (modulus - val);
  }
  return Polynomial<CoeffType>::Create(std::move(rotated));
}

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_PIR_PARAMS_H_
