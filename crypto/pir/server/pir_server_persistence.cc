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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/file_util.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/polynomial.h"
#include "crypto/status_macros.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::Status PirServer<DbDataType, CoeffType, MatCoeffType>::SaveToFile(
    absl::string_view file_path) const {
  std::ofstream out(std::string(file_path), std::ios::binary);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for writing: ", file_path));
  }

  // 1. Write combined_db_matrix_
  int32_t db_rows = combined_db_matrix_.Rows();
  int32_t db_cols = combined_db_matrix_.PhysicalCols();
  uint8_t db_condensed = combined_db_matrix_.IsCondensed() ? 1 : 0;
  int32_t db_orig_cols = combined_db_matrix_.OriginalCols();
  out.write(reinterpret_cast<const char*>(&db_rows), sizeof(db_rows));
  out.write(reinterpret_cast<const char*>(&db_cols), sizeof(db_cols));
  out.write(reinterpret_cast<const char*>(&db_condensed), sizeof(db_condensed));
  out.write(reinterpret_cast<const char*>(&db_orig_cols), sizeof(db_orig_cols));

  if (combined_db_matrix_.IsCondensed()) {
    const auto& cdata = combined_db_matrix_.CondensedData();
    uint64_t num_elems = cdata.size();
    out.write(reinterpret_cast<const char*>(&num_elems), sizeof(num_elems));
    if (num_elems > 0) {
      out.write(reinterpret_cast<const char*>(cdata.data()),
                num_elems * sizeof(uint64_t));
    }
  } else {
    const auto& data = combined_db_matrix_.Data();
    uint64_t num_elems = data.size();
    out.write(reinterpret_cast<const char*>(&num_elems), sizeof(num_elems));
    if (num_elems > 0) {
      out.write(reinterpret_cast<const char*>(data.data()),
                num_elems * sizeof(DbDataType));
    }
  }

  // 2. Write combined_pack_matrix_ in bit-packed format (pack_log_digit +
  // log2(d) bits per value).
  int32_t pack_rows = combined_pack_matrix_.Rows();
  int32_t pack_cols = combined_pack_matrix_.Cols();
  int32_t pack_phys_cols = combined_pack_matrix_.PhysicalCols();
  uint8_t pack_condensed = combined_pack_matrix_.IsCondensed() ? 1 : 0;
  int32_t pack_orig_cols = combined_pack_matrix_.OriginalCols();
  int32_t log2_d = absl::bit_width(
      static_cast<uint32_t>(params_.RlweParameters().Degree() - 1));
  int32_t pack_bits_per_entry = params_.PackGadgetParams().log_digit + log2_d;

  out.write(reinterpret_cast<const char*>(&pack_rows), sizeof(pack_rows));
  out.write(reinterpret_cast<const char*>(&pack_cols), sizeof(pack_cols));
  out.write(reinterpret_cast<const char*>(&pack_phys_cols),
            sizeof(pack_phys_cols));
  out.write(reinterpret_cast<const char*>(&pack_condensed),
            sizeof(pack_condensed));
  out.write(reinterpret_cast<const char*>(&pack_orig_cols),
            sizeof(pack_orig_cols));
  out.write(reinterpret_cast<const char*>(&pack_bits_per_entry),
            sizeof(pack_bits_per_entry));

  std::vector<MatCoeffType> pack_entries;
  pack_entries.reserve(static_cast<size_t>(pack_rows) * pack_cols);
  if (combined_pack_matrix_.IsCondensed()) {
    const auto& cdata = combined_pack_matrix_.CondensedData();
    for (int r = 0; r < pack_rows; ++r) {
      for (int c = 0; c < pack_cols; ++c) {
        if constexpr (std::is_same_v<MatCoeffType, int32_t>) {
          uint64_t w = cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)];
          int32_t val = (c % 2 == 0)
                            ? static_cast<int32_t>(w & 0xFFFFFFFFULL)
                            : static_cast<int32_t>((w >> 32) & 0xFFFFFFFFULL);
          pack_entries.push_back(val);
        } else if constexpr (std::is_same_v<MatCoeffType, int16_t>) {
          uint64_t w = cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 4)];
          int16_t val = static_cast<int16_t>((w >> ((c % 4) * 16)) & 0xFFFFULL);
          pack_entries.push_back(val);
        } else {
          pack_entries.push_back(0);
        }
      }
    }
  } else {
    pack_entries = combined_pack_matrix_.Data();
  }

  std::string packed_matrix_bytes =
      PackIntegers<MatCoeffType>(pack_entries, pack_bits_per_entry);
  uint64_t packed_matrix_len = packed_matrix_bytes.size();
  out.write(reinterpret_cast<const char*>(&packed_matrix_len),
            sizeof(packed_matrix_len));
  if (packed_matrix_len > 0) {
    out.write(packed_matrix_bytes.data(), packed_matrix_len);
  }

  // 3. Write preprocessed_outputs_ and second_dim_query_a_ in bit-packed
  // format.
  int32_t log_modulus = params_.RlweParameters().LogModulus();
  out.write(reinterpret_cast<const char*>(&log_modulus), sizeof(log_modulus));

  auto write_poly = [&](const Polynomial<CoeffType>& poly) {
    const auto& coeffs = poly.Coeffs();
    uint64_t len = coeffs.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    std::string packed =
        Polynomial<CoeffType>::PackCoefficients(coeffs, log_modulus);
    uint64_t packed_len = packed.size();
    out.write(reinterpret_cast<const char*>(&packed_len), sizeof(packed_len));
    if (packed_len > 0) {
      out.write(packed.data(), packed_len);
    }
  };

  int32_t num_shards = static_cast<int32_t>(preprocessed_outputs_.size());
  out.write(reinterpret_cast<const char*>(&num_shards), sizeof(num_shards));
  for (int32_t k = 0; k < num_shards; ++k) {
    int32_t num_slices = static_cast<int32_t>(preprocessed_outputs_[k].size());
    out.write(reinterpret_cast<const char*>(&num_slices), sizeof(num_slices));
    for (int32_t i = 0; i < num_slices; ++i) {
      write_poly(preprocessed_outputs_[k][i].a_tilde_agg);
      const auto& t_vec = preprocessed_outputs_[k][i].t_vec_h;
      uint64_t t_vec_len = t_vec.size();
      out.write(reinterpret_cast<const char*>(&t_vec_len), sizeof(t_vec_len));
      for (const auto& poly : t_vec) {
        write_poly(poly);
      }
    }
  }

  uint64_t num_second_dim_a = second_dim_query_a_.size();
  out.write(reinterpret_cast<const char*>(&num_second_dim_a),
            sizeof(num_second_dim_a));
  for (const auto& poly : second_dim_query_a_) {
    write_poly(poly);
  }

  out.flush();
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Error writing to file: ", file_path));
  }
  return absl::OkStatus();
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromBuffer(
    const PirParams<CoeffType>& params, const uint8_t* buffer, size_t size) {
  class BufferReader {
   public:
    BufferReader(const uint8_t* buffer, size_t size)
        : buffer_(buffer), size_(size), offset_(0) {}
    void read(char* dest, size_t n) {
      if (offset_ + n > size_) {
        ok_ = false;
        return;
      }
      std::memcpy(dest, buffer_ + offset_, n);
      offset_ += n;
    }
    bool good() const { return ok_; }
    bool eof() const { return offset_ >= size_; }

   private:
    const uint8_t* buffer_;
    size_t size_;
    size_t offset_;
    bool ok_ = true;
  };

  BufferReader in(buffer, size);

  // 1. Read combined_db_matrix_
  int32_t db_rows = 0, db_cols = 0, db_orig_cols = 0;
  uint8_t db_condensed = 0;
  in.read(reinterpret_cast<char*>(&db_rows), sizeof(db_rows));
  in.read(reinterpret_cast<char*>(&db_cols), sizeof(db_cols));
  in.read(reinterpret_cast<char*>(&db_condensed), sizeof(db_condensed));
  in.read(reinterpret_cast<char*>(&db_orig_cols), sizeof(db_orig_cols));

  uint64_t db_num_elems = 0;
  in.read(reinterpret_cast<char*>(&db_num_elems), sizeof(db_num_elems));

  Matrix<DbDataType> combined_db_matrix;
  if (db_condensed != 0) {
    std::vector<uint64_t> cdata(db_num_elems);
    if (db_num_elems > 0) {
      in.read(reinterpret_cast<char*>(cdata.data()),
              db_num_elems * sizeof(uint64_t));
    }
    combined_db_matrix = Matrix<DbDataType>(
        std::vector<DbDataType>(), std::move(cdata), db_rows, db_cols,
        /*is_condensed=*/true, db_orig_cols);
  } else {
    std::vector<DbDataType> data(db_num_elems);
    if (db_num_elems > 0) {
      in.read(reinterpret_cast<char*>(data.data()),
              db_num_elems * sizeof(DbDataType));
    }
    combined_db_matrix = Matrix<DbDataType>(
        std::move(data), std::vector<uint64_t>(), db_rows, db_cols,
        /*is_condensed=*/false, db_orig_cols);
  }

  // 2. Read combined_pack_matrix_ from bit-packed format.
  int32_t pack_rows = 0, pack_cols = 0, pack_phys_cols = 0, pack_orig_cols = 0;
  uint8_t pack_condensed = 0;
  int32_t pack_bits_per_entry = 0;
  in.read(reinterpret_cast<char*>(&pack_rows), sizeof(pack_rows));
  in.read(reinterpret_cast<char*>(&pack_cols), sizeof(pack_cols));
  in.read(reinterpret_cast<char*>(&pack_phys_cols), sizeof(pack_phys_cols));
  in.read(reinterpret_cast<char*>(&pack_condensed), sizeof(pack_condensed));
  in.read(reinterpret_cast<char*>(&pack_orig_cols), sizeof(pack_orig_cols));
  in.read(reinterpret_cast<char*>(&pack_bits_per_entry),
          sizeof(pack_bits_per_entry));

  uint64_t packed_matrix_len = 0;
  in.read(reinterpret_cast<char*>(&packed_matrix_len),
          sizeof(packed_matrix_len));
  std::string packed_matrix_bytes(packed_matrix_len, '\0');
  if (packed_matrix_len > 0) {
    in.read(&packed_matrix_bytes[0], packed_matrix_len);
  }

  size_t total_pack_entries = static_cast<size_t>(pack_rows) * pack_cols;
  ASSIGN_OR_RETURN(std::vector<MatCoeffType> pack_entries,
                   UnpackIntegers<MatCoeffType>(packed_matrix_bytes,
                                                total_pack_entries,
                                                pack_bits_per_entry));

  Matrix<MatCoeffType> combined_pack_matrix;
  if (pack_condensed != 0) {
    std::vector<uint64_t> cdata(static_cast<size_t>(pack_rows) * pack_phys_cols,
                                0);
    for (int r = 0; r < pack_rows; ++r) {
      for (int c = 0; c < pack_cols; ++c) {
        MatCoeffType val = pack_entries[static_cast<size_t>(r) * pack_cols + c];
        if constexpr (std::is_same_v<MatCoeffType, int32_t>) {
          if (c % 2 == 0) {
            cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)] |=
                static_cast<uint64_t>(static_cast<uint32_t>(val));
          } else {
            cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)] |=
                (static_cast<uint64_t>(static_cast<uint32_t>(val)) << 32);
          }
        } else if constexpr (std::is_same_v<MatCoeffType, int16_t>) {
          cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 4)] |=
              (static_cast<uint64_t>(static_cast<uint16_t>(val))
               << ((c % 4) * 16));
        }
      }
    }
    combined_pack_matrix = Matrix<MatCoeffType>(
        std::vector<MatCoeffType>(), std::move(cdata), pack_rows, pack_cols,
        /*is_condensed=*/true, pack_orig_cols);
  } else {
    combined_pack_matrix = Matrix<MatCoeffType>(
        std::move(pack_entries), std::vector<uint64_t>(), pack_rows, pack_cols,
        /*is_condensed=*/false, pack_orig_cols);
  }

  // 3. Read preprocessed_outputs_ and second_dim_query_a_ in bit-packed format.
  int32_t log_modulus = 0;
  in.read(reinterpret_cast<char*>(&log_modulus), sizeof(log_modulus));

  auto read_poly = [&]() -> absl::StatusOr<Polynomial<CoeffType>> {
    uint64_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof(len));
    uint64_t packed_len = 0;
    in.read(reinterpret_cast<char*>(&packed_len), sizeof(packed_len));
    std::string packed(packed_len, '\0');
    if (packed_len > 0) {
      in.read(&packed[0], packed_len);
    }
    ASSIGN_OR_RETURN(
        std::vector<CoeffType> coeffs,
        Polynomial<CoeffType>::UnpackCoefficients(packed, len, log_modulus));
    return Polynomial<CoeffType>::Create(std::move(coeffs));
  };

  int32_t num_shards = 0;
  in.read(reinterpret_cast<char*>(&num_shards), sizeof(num_shards));
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs(num_shards);
  for (int32_t k = 0; k < num_shards; ++k) {
    int32_t num_slices = 0;
    in.read(reinterpret_cast<char*>(&num_slices), sizeof(num_slices));
    preprocessed_outputs[k].resize(num_slices);
    for (int32_t i = 0; i < num_slices; ++i) {
      ASSIGN_OR_RETURN(preprocessed_outputs[k][i].a_tilde_agg, read_poly());
      uint64_t t_vec_len = 0;
      in.read(reinterpret_cast<char*>(&t_vec_len), sizeof(t_vec_len));
      preprocessed_outputs[k][i].t_vec_h.resize(t_vec_len);
      for (uint64_t j = 0; j < t_vec_len; ++j) {
        ASSIGN_OR_RETURN(preprocessed_outputs[k][i].t_vec_h[j], read_poly());
      }
    }
  }

  uint64_t num_second_dim_a = 0;
  in.read(reinterpret_cast<char*>(&num_second_dim_a), sizeof(num_second_dim_a));
  std::vector<Polynomial<CoeffType>> second_dim_query_a(num_second_dim_a);
  for (uint64_t j = 0; j < num_second_dim_a; ++j) {
    ASSIGN_OR_RETURN(second_dim_query_a[j], read_poly());
  }

  if (!in.good() && !in.eof()) {
    return absl::InternalError("Error reading from buffer");
  }

  return CreateFromComponents(params, std::move(combined_db_matrix),
                              std::move(combined_pack_matrix),
                              std::move(preprocessed_outputs),
                              std::move(second_dim_query_a));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromFile(
    const PirParams<CoeffType>& params, absl::string_view file_path) {
  std::string buffer;
  if (!file_util::GetFileContents(file_path, &buffer).ok()) {
    return absl::InternalError("Failed to read file.");
  }
  return LoadFromBuffer(
      params, reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
}

#define INSTANTIATE_PIR_SERVER_PERSISTENCE(DbType, CoeffType, MatType)        \
  template absl::Status                                                       \
  PirServer<DbType, CoeffType, MatType>::SaveToFile(absl::string_view) const; \
  template absl::StatusOr<                                                    \
      std::unique_ptr<PirServer<DbType, CoeffType, MatType>>>                 \
  PirServer<DbType, CoeffType, MatType>::LoadFromBuffer(                      \
      const PirParams<CoeffType>&, const uint8_t*, size_t);                   \
  template absl::StatusOr<                                                    \
      std::unique_ptr<PirServer<DbType, CoeffType, MatType>>>                 \
  PirServer<DbType, CoeffType, MatType>::LoadFromFile(                        \
      const PirParams<CoeffType>&, absl::string_view)

INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint32_t, uint32_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint64_t, uint64_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint32_t, int16_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint64_t, int16_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint32_t, int32_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint16_t, uint64_t, int32_t);

INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint32_t, uint32_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint64_t, uint64_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint32_t, int16_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint64_t, int16_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint32_t, int32_t);
INSTANTIATE_PIR_SERVER_PERSISTENCE(uint8_t, uint64_t, int32_t);

#undef INSTANTIATE_PIR_SERVER_PERSISTENCE

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
