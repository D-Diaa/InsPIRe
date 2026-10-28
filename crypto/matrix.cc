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

#include "crypto/matrix.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/proto/pir_server.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "hwy/cache_control.h"
#include "hwy/highway.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::Create(
    std::vector<std::vector<CoeffType>> data) {
  const int rows = data.size();
  if (rows == 0) {
    return Matrix<CoeffType>(std::vector<CoeffType>(), 0, 0);
  }
  const int cols = data[0].size();
  std::vector<CoeffType> flat_data;
  flat_data.reserve(static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      return absl::InvalidArgumentError(
          "All rows must have the same number of columns.");
    }
    flat_data.insert(flat_data.end(), data[i].begin(), data[i].end());
  }
  return Matrix<CoeffType>(std::move(flat_data), rows, cols);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::Create(int rows,
                                                            int cols) {
  if (rows < 0 || cols < 0) {
    return absl::InvalidArgumentError(
        "Matrix dimensions must be non-negative.");
  }
  return Matrix<CoeffType>(
      std::vector<CoeffType>(static_cast<size_t>(rows) * cols, 0), rows, cols);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::CreateCondensed(
    std::vector<std::vector<CoeffType>> data) {
  if constexpr (std::is_same_v<CoeffType, int32_t>) {
    const int rows = data.size();
    if (rows == 0) {
      return Matrix<CoeffType>(std::vector<CoeffType>(), 0, 0,
                               /*is_condensed=*/true, /*original_cols=*/0);
    }
    const int cols = data[0].size();
    for (int i = 0; i < rows; ++i) {
      if (static_cast<int>(data[i].size()) != cols) {
        return absl::InvalidArgumentError(
            "All rows must have the same number of columns.");
      }
    }
    if (cols % 2 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 2 to condense int32_t.");
    }
    const int phys_cols = cols / 2;
    std::vector<uint64_t> condensed_flat(static_cast<size_t>(rows) * phys_cols,
                                         0);
    for (int i = 0; i < rows; ++i) {
      if (cols > 0) {
        std::memcpy(&condensed_flat[static_cast<size_t>(i) * phys_cols],
                    data[i].data(),
                    static_cast<size_t>(cols) * sizeof(int32_t));
      }
    }
    Matrix<CoeffType> matrix(std::vector<CoeffType>(), rows, phys_cols,
                             /*is_condensed=*/true, /*original_cols=*/cols);
    matrix.condensed_data_ = std::move(condensed_flat);
    return matrix;
  }
  auto normal_matrix_or = Create(std::move(data));
  if (!normal_matrix_or.ok()) {
    return normal_matrix_or.status();
  }
  return CreateCondensed(*normal_matrix_or);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::CreateCondensed(
    const Matrix<CoeffType>& normal_matrix) {
  if (normal_matrix.IsCondensed()) {
    return absl::InvalidArgumentError(
        "Cannot condense a matrix that is already condensed.");
  }

  const int rows = normal_matrix.Rows();
  const int cols = normal_matrix.Cols();
  const std::vector<CoeffType>& data = normal_matrix.Data();

  std::vector<uint64_t> condensed_flat;

  if constexpr (std::is_same_v<CoeffType, uint8_t>) {
    if (cols % 8 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 8 to condense uint8_t.");
    }
    condensed_flat.resize(static_cast<size_t>(rows) * (cols / 8), 0);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols / 8; ++j) {
        uint64_t condensed_val = 0;
        for (size_t k = 0; k < 8; ++k) {
          condensed_val |= static_cast<uint64_t>(
                               data[i * static_cast<size_t>(cols) + j * 8 + k])
                           << (k * 8);
        }
        condensed_flat[i * static_cast<size_t>(cols / 8) + j] = condensed_val;
      }
    }
  } else if constexpr (std::is_same_v<CoeffType, uint16_t>) {
    if (cols % 4 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 4 to condense uint16_t.");
    }
    const size_t phys_cols = static_cast<size_t>(cols / 4);
    condensed_flat.resize(static_cast<size_t>(rows) * phys_cols, 0);
    if (rows > 0 && cols > 0) {
      std::memcpy(condensed_flat.data(), data.data(),
                  static_cast<size_t>(rows) * cols * sizeof(uint16_t));
      for (uint64_t& w : condensed_flat) {
        w ^= 0x8000800080008000ULL;
      }
    }
  } else if constexpr (std::is_same_v<CoeffType, int32_t>) {
    if (cols % 2 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 2 to condense int32_t.");
    }
    const size_t phys_cols = static_cast<size_t>(cols / 2);
    condensed_flat.resize(static_cast<size_t>(rows) * phys_cols, 0);
    if (rows > 0 && cols > 0) {
      std::memcpy(condensed_flat.data(), data.data(),
                  static_cast<size_t>(rows) * cols * sizeof(int32_t));
    }
  } else {
    return absl::FailedPreconditionError(
        "CreateCondensed is only supported for uint8_t, uint16_t, or int32_t "
        "matrices.");
  }

  int physical_cols = 0;
  if constexpr (std::is_same_v<CoeffType, uint8_t>) {
    physical_cols = cols / 8;
  } else if constexpr (std::is_same_v<CoeffType, uint16_t>) {
    physical_cols = cols / 4;
  } else if constexpr (std::is_same_v<CoeffType, int32_t>) {
    physical_cols = cols / 2;
  }

  Matrix<CoeffType> matrix(std::vector<CoeffType>(), rows, physical_cols,
                           /*is_condensed=*/true, /*original_cols=*/cols);
  matrix.condensed_data_ = std::move(condensed_flat);
  return matrix;
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::VerticalConcatenate(
    const std::vector<Matrix<CoeffType>>& matrices) {
  if (matrices.empty()) {
    return Matrix<CoeffType>();
  }
  const int cols = matrices[0].Cols();
  const bool is_condensed = matrices[0].IsCondensed();
  const int physical_cols = matrices[0].cols_;
  int total_rows = 0;

  for (const auto& m : matrices) {
    if (m.Cols() != cols) {
      return absl::InvalidArgumentError(
          "All matrices must have the same number of columns.");
    }
    if (m.IsCondensed() != is_condensed) {
      return absl::InvalidArgumentError(
          "All matrices must have the same condensed state.");
    }
    total_rows += m.Rows();
  }

  std::vector<CoeffType> combined_data;
  std::vector<uint64_t> combined_condensed_data;

  for (const auto& m : matrices) {
    combined_data.insert(combined_data.end(), m.Data().begin(), m.Data().end());
    if (is_condensed) {
      combined_condensed_data.insert(combined_condensed_data.end(),
                                     m.CondensedData().begin(),
                                     m.CondensedData().end());
    }
  }

  if (is_condensed) {
    Matrix<CoeffType> matrix(std::vector<CoeffType>(), total_rows,
                             physical_cols, /*is_condensed=*/true,
                             /*original_cols=*/cols);
    matrix.condensed_data_ = std::move(combined_condensed_data);
    return matrix;
  } else {
    return Matrix<CoeffType>(std::move(combined_data), total_rows, cols);
  }
}

template <typename CoeffType>
absl::Status Matrix<CoeffType>::AddInPlace(const Matrix& that) {
  if (is_condensed_) {
    return absl::FailedPreconditionError(
        "AddInPlace is not supported on a condensed matrix.");
  }
  if (rows_ != that.Rows() || cols_ != that.Cols()) {
    return absl::InvalidArgumentError(
        "Matrix dimensions must match for addition.");
  }
  const std::vector<CoeffType>& that_data = that.Data();
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] += that_data[i];
  }
  return absl::OkStatus();
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::SubMatrix(
    int row_offset, int col_offset, int num_rows, int num_cols) const {
  if (is_condensed_) {
    return absl::FailedPreconditionError(
        "SubMatrix is not supported on a condensed matrix.");
  }
  if (row_offset < 0 || col_offset < 0 || num_rows < 0 || num_cols < 0) {
    return absl::InvalidArgumentError(
        "Offsets and dimensions must be non-negative.");
  }
  if (row_offset + num_rows > rows_ || col_offset + num_cols > cols_) {
    return absl::InvalidArgumentError("Submatrix is out of bounds.");
  }
  std::vector<CoeffType> sub_data(static_cast<size_t>(num_rows) * num_cols);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      size_t flat_idx =
          (static_cast<size_t>(row_offset) + i) * cols_ + (col_offset + j);
      sub_data[i * static_cast<size_t>(num_cols) + j] = data_[flat_idx];
    }
  }
  return Matrix<CoeffType>(std::move(sub_data), num_rows, num_cols);
}

namespace {

template <typename CoeffType>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply(
    const Matrix<CoeffType>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  return absl::FailedPreconditionError(
      "CondensedMultiply is only supported for uint8_t, uint16_t, and "
      "int32_t matrices.");
}

// z[i] = sum_j m[i][j] * y[j] mod 2^64 for a uint16 matrix and a uint64 vector.
//
// In CreateCondensed<uint16_t>, each unsigned database entry m[i][j] in
// [0, 65535] is centered to signed int16_t m'[i][j] = (m[i][j] ^ 0x8000) =
// m[i][j] - 32768 in [-32768, 32767]. Thus:
//   sum_j m[i][j] * y[j] == sum_j m'[i][j] * y[j] + 32768 * sum_j y[j] (mod 2^64)
// where y_bias = 32768 * sum_j y[j] is computed once per call.
//
// Each query element (y[j] >> kShift) is decomposed once per call into signed
// 14-bit digits d_k[j] in [-8191, +8192] at base 2^14:
//   (y[j] >> kShift) == sum_{k=0}^{kNumDigits-1} d_k[j] * 2^{14k}.
// Because m' in [-32768, +32767] and d_k in [-8191, +8192], each 16-lane
// _mm256_madd_epi16(m', d_k) pairwise product lies in [-2^29, +2^29 - 16384],
// so 4 consecutive vectors (64 columns) can be accumulated in int32_t without
// overflow before widening to int64_t.
// Entries of `vec` past its end, up to `cols`, are zero.
template <int kNumDigits, int kShift>
HWY_INLINE void PackQueryDigitsUint16(absl::Span<const uint64_t> vec,
                                      size_t cols,
                                      int16_t* HWY_RESTRICT y_packed) {
  const size_t num_vecs = cols / 16;
  for (size_t v_idx = 0; v_idx < num_vecs; ++v_idx) {
    int16_t* HWY_RESTRICT out = y_packed + v_idx * kNumDigits * 16;
    for (size_t lane = 0; lane < 16; ++lane) {
      const size_t j = v_idx * 16 + lane;
      uint64_t v = j < vec.size() ? vec[j] >> kShift : 0;
      HWY_UNROLL(kNumDigits)
      for (int d = 0; d < kNumDigits; ++d) {
        const uint64_t r = v & 0x3FFFULL;
        const uint64_t borrow = (r > 8192ULL) ? 1ULL : 0ULL;
        out[d * 16 + lane] = static_cast<int16_t>(
            static_cast<int64_t>(r) - static_cast<int64_t>(borrow << 14));
        v = (v >> 14) + borrow;
      }
    }
  }
}

#if defined(__AVX2__)
template <int kNumDigits>
HWY_INLINE void WidenAndAccumulate32To64(const __m256i (&acc32)[kNumDigits],
                                         __m256i (&acc64)[kNumDigits]) {
  HWY_UNROLL(kNumDigits)
  for (int d = 0; d < kNumDigits; ++d) {
    const __m256i lo64 =
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc32[d]));
    const __m256i hi64 =
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(acc32[d], 1));
    acc64[d] = _mm256_add_epi64(acc64[d], _mm256_add_epi64(lo64, hi64));
  }
}
#endif

template <int kNumDigits, int kShift>
HWY_INLINE uint64_t CondensedMultiplyUint16Row(
    const uint64_t* HWY_RESTRICT m_row, int condensed_cols,
    const int16_t* HWY_RESTRICT y_packed, uint64_t y_bias) {
#if defined(__AVX2__)
  const size_t next_row = static_cast<size_t>(condensed_cols);
  __m256i acc64[kNumDigits];
  for (int d = 0; d < kNumDigits; ++d) {
    acc64[d] = _mm256_setzero_si256();
  }

  const int16_t* HWY_RESTRICT yp = y_packed;
  size_t c = 0;
  for (; c + 16 <= static_cast<size_t>(condensed_cols);
       c += 16, yp += 4 * kNumDigits * 16) {
    __m256i acc32[kNumDigits];
    HWY_UNROLL(4)
    for (int s = 0; s < 4; ++s) {
      if (s == 0 || s == 2) {
        hwy::Prefetch(m_row + c + 4 * s + next_row);
      }
      const __m256i vm = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(m_row + c + 4 * s));
      HWY_UNROLL(kNumDigits)
      for (int d = 0; d < kNumDigits; ++d) {
        const __m256i vy = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(yp + (s * kNumDigits + d) * 16));
        const __m256i prod = _mm256_madd_epi16(vm, vy);
        acc32[d] = (s == 0) ? prod : _mm256_add_epi32(acc32[d], prod);
      }
    }
    WidenAndAccumulate32To64<kNumDigits>(acc32, acc64);
  }

  const size_t rem_vecs = (static_cast<size_t>(condensed_cols) - c) / 4;
  if (rem_vecs > 0) {
    __m256i acc32[kNumDigits];
    for (size_t s = 0; s < rem_vecs; ++s) {
      const __m256i vm = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(m_row + c + 4 * s));
      HWY_UNROLL(kNumDigits)
      for (int d = 0; d < kNumDigits; ++d) {
        const __m256i vy = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(yp + (s * kNumDigits + d) * 16));
        const __m256i prod = _mm256_madd_epi16(vm, vy);
        acc32[d] = (s == 0) ? prod : _mm256_add_epi32(acc32[d], prod);
      }
    }
    WidenAndAccumulate32To64<kNumDigits>(acc32, acc64);
  }

  __m256i total64 =
      (kShift == 0) ? acc64[0] : _mm256_slli_epi64(acc64[0], kShift);
  for (int d = 1; d < kNumDigits; ++d) {
    total64 = _mm256_add_epi64(total64,
                               _mm256_slli_epi64(acc64[d], kShift + 14 * d));
  }
  const __m128i sum128 = _mm_add_epi64(_mm256_castsi256_si128(total64),
                                       _mm256_extracti128_si256(total64, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  const uint64_t row_sum =
      static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_add_epi64(sum128, hi64)));
  return row_sum + y_bias;
#else
  uint64_t total = y_bias;
  const int16_t* m_i16 = reinterpret_cast<const int16_t*>(m_row);
  for (int v_idx = 0; v_idx < condensed_cols / 4; ++v_idx) {
    const int16_t* yp = y_packed + v_idx * kNumDigits * 16;
    for (int d = 0; d < kNumDigits; ++d) {
      int64_t sum_d = 0;
      for (int lane = 0; lane < 16; ++lane) {
        sum_d += static_cast<int64_t>(m_i16[v_idx * 16 + lane]) *
                 static_cast<int64_t>(yp[d * 16 + lane]);
      }
      total += static_cast<uint64_t>(sum_d) << (kShift + 14 * d);
    }
  }
  return total;
#endif
}

template <int kNumDigits, int kShift>
HWY_INLINE void RunCondensedMultiplyUint16(const uint64_t* data, int rows,
                                           int condensed_cols,
                                           absl::Span<const uint64_t> vec,
                                           uint64_t y_bias,
                                           uint64_t* HWY_RESTRICT result) {
  const size_t cols = static_cast<size_t>(condensed_cols) * 4;
  alignas(32) int16_t y_stack[2048 * kNumDigits];
  std::vector<int16_t> y_heap;
  int16_t* y_packed = y_stack;
  if (cols > 2048) {
    y_heap.resize(cols * kNumDigits);
    y_packed = y_heap.data();
  }
  PackQueryDigitsUint16<kNumDigits, kShift>(vec, cols, y_packed);

  for (int i = 0; i < rows; ++i) {
    const uint64_t* m_row = &data[static_cast<size_t>(i) * condensed_cols];
    result[i] = CondensedMultiplyUint16Row<kNumDigits, kShift>(
        m_row, condensed_cols, y_packed, y_bias);
  }
}

// Centered (no 2^15 bias) and mod-2^result_bits variant of
// CondensedMultiply<uint16_t>; see Matrix::MultiplyUint16.
absl::StatusOr<std::vector<uint64_t>> CondensedMultiplyUint16(
    const Matrix<uint16_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols, bool centered, int result_bits) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint64_t> d;
  const size_t N = hn::Lanes(d);

  if (original_cols % (4 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 4 * Lanes.");
  }
  if (vec.size() > static_cast<size_t>(original_cols)) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  const int rows = condensed_matrix.Rows();
  const int condensed_cols = original_cols / 4;
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  uint64_t or_all = 0;
  uint64_t y_sum = 0;
  for (size_t j = 0; j < vec.size(); ++j) {
    or_all |= vec[j];
    y_sum += vec[j];
  }
  const uint64_t y_bias = centered ? 0 : y_sum << 15;

  std::vector<uint64_t> result(rows, 0);
  if (((or_all & 0xFFFULL) == 0) && ((or_all >> 52) == 0)) {
    // 40-bit modulus-switched selector in bits [12, 52): 3 signed 14-bit digits.
    RunCondensedMultiplyUint16<3, 12>(data.data(), rows, condensed_cols, vec,
                                      y_bias, result.data());
  } else if (result_bits <= 52 && ((or_all & 0x3FFULL) == 0) &&
             ((or_all >> 52) == 0)) {
    // 42-bit selector in bits [10, 52): 3 signed 14-bit digits; the top digit's
    // dropped carry has weight 2^52, so results are exact modulo 2^52.
    RunCondensedMultiplyUint16<3, 10>(data.data(), rows, condensed_cols, vec,
                                      y_bias, result.data());
  } else if ((or_all >> 55) == 0) {
    // Up to 55-bit query (including 52-bit RLWE modulus): 4 signed 14-bit digits.
    RunCondensedMultiplyUint16<4, 0>(data.data(), rows, condensed_cols, vec,
                                     y_bias, result.data());
  } else {
    // General 64-bit vector: 5 signed 14-bit digits.
    RunCondensedMultiplyUint16<5, 0>(data.data(), rows, condensed_cols, vec,
                                     y_bias, result.data());
  }
  return result;
}

template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<uint16_t>(
    const Matrix<uint16_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  return CondensedMultiplyUint16(condensed_matrix, vec, original_cols,
                                 /*centered=*/false, /*result_bits=*/64);
}

// z[i] = sum_j m[i][j] * y[j] mod 2^64 for an int32 matrix and a uint64 vector.
//
// AVX2 has no 64-bit multiply, so a 64x64 MulAdd costs several instructions
// per lane and the loop is instruction-bound well below memory bandwidth.
// Split y instead, once per call:
//
//   y_lo = int32(y mod 2^32)  (sign-extended by the multiply)
//   y_hi = int32((y >> 32) + bit31(y))  (the +1 cancels that sign extension)
//
// so that y == y_lo + 2^32 * y_hi (mod 2^64) and
//
//   m * y == m * y_lo + 2^32 * (m * y_hi mod 2^32)  (mod 2^64).
//
// m * y_lo is one signed 32x32->64 product (MulEven); of m * y_hi only the low
// 32 bits survive the shift, one 32x32->32 product (Mul). A condensed lane
// holds (m[2j], m[2j+1]) as its low and high halves: MulEven reads the low
// halves directly and the high halves after a 32-bit shift, while Mul sees
// both as ordinary int32 lanes. Rows are processed kRowBlock at a time so
// each y vector is loaded once per block.
template <int kRows>
HWY_INLINE void CondensedMultiplyInt32Rows(
    const uint64_t* const* m_rows, int condensed_cols,
    const int32_t* HWY_RESTRICT y_packed, uint64_t* HWY_RESTRICT result) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint64_t> d64;
  const hn::Repartition<int64_t, decltype(d64)> d64s;
  const hn::Repartition<int32_t, decltype(d64)> d32;
  const hn::Repartition<uint32_t, decltype(d64)> du32;
  const size_t N = hn::Lanes(d64);

  hn::Vec<decltype(d64s)> lo[kRows];
  hn::Vec<decltype(du32)> hi[kRows];
  for (int r = 0; r < kRows; ++r) {
    lo[r] = hn::Zero(d64s);
    hi[r] = hn::Zero(du32);
  }

  const int32_t* HWY_RESTRICT y_ptr = y_packed;
  for (size_t c = 0; c < static_cast<size_t>(condensed_cols);
       c += N, y_ptr += 4 * N) {
    const auto yl = hn::LoadU(d32, y_ptr);
    const auto yl_odd =
        hn::BitCast(d32, hn::ShiftRight<32>(hn::BitCast(d64, yl)));
    const auto yh = hn::LoadU(d32, y_ptr + 2 * N);
    HWY_UNROLL(kRows)
    for (int r = 0; r < kRows; ++r) {
      const auto m = hn::LoadU(d64, m_rows[r] + c);
      const auto m_even = hn::BitCast(d32, m);
      const auto m_odd = hn::BitCast(d32, hn::ShiftRight<32>(m));
      lo[r] = hn::Add(
          lo[r], hn::Add(hn::MulEven(m_even, yl), hn::MulEven(m_odd, yl_odd)));
      hi[r] = hn::Add(hi[r], hn::BitCast(du32, hn::Mul(m_even, yh)));
    }
  }

  for (int r = 0; r < kRows; ++r) {
    result[r] = static_cast<uint64_t>(hn::ReduceSum(d64s, lo[r])) +
                (static_cast<uint64_t>(hn::ReduceSum(du32, hi[r])) << 32);
  }
}

template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<int32_t>(
    const Matrix<int32_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<uint64_t> d64;
  const size_t N = hn::Lanes(d64);

  if (original_cols % (2 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 2 * Lanes.");
  }
  if (vec.size() != static_cast<size_t>(original_cols)) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  const int rows = condensed_matrix.Rows();
  const int condensed_cols = original_cols / 2;
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  const size_t step_elems = 2 * N;
  std::vector<int32_t> y_packed(2 * static_cast<size_t>(original_cols));
  int32_t* HWY_RESTRICT yp = y_packed.data();
  for (size_t j = 0; j < static_cast<size_t>(original_cols);
       j += step_elems, yp += 2 * step_elems) {
    for (size_t l = 0; l < step_elems; ++l) {
      const uint64_t v = vec[j + l];
      yp[l] = static_cast<int32_t>(static_cast<uint32_t>(v));
      yp[step_elems + l] =
          static_cast<int32_t>(static_cast<uint32_t>(v >> 32) +
                               static_cast<uint32_t>((v >> 31) & 1));
    }
  }

  constexpr int kRowBlock = 6;
  std::vector<uint64_t> result(rows, 0);
  int i = 0;
  for (; i + kRowBlock <= rows; i += kRowBlock) {
    const uint64_t* m_rows[kRowBlock];
    for (int r = 0; r < kRowBlock; ++r) {
      m_rows[r] = &data[static_cast<size_t>(i + r) * condensed_cols];
    }
    CondensedMultiplyInt32Rows<kRowBlock>(m_rows, condensed_cols,
                                          y_packed.data(), &result[i]);
  }
  for (; i + 2 <= rows; i += 2) {
    const uint64_t* m_rows[2] = {
        &data[static_cast<size_t>(i) * condensed_cols],
        &data[static_cast<size_t>(i + 1) * condensed_cols]};
    CondensedMultiplyInt32Rows<2>(m_rows, condensed_cols, y_packed.data(),
                                  &result[i]);
  }
  for (; i < rows; ++i) {
    const uint64_t* m_row = &data[static_cast<size_t>(i) * condensed_cols];
    CondensedMultiplyInt32Rows<1>(&m_row, condensed_cols, y_packed.data(),
                                  &result[i]);
  }
  return result;
}


template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<uint8_t>(
    const Matrix<uint8_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  namespace hn = hwy::HWY_NAMESPACE;
  hn::ScalableTag<uint64_t> d;
  const size_t N = hn::Lanes(d);

  if (original_cols % (8 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 8 * Lanes.");
  }
  if (vec.size() != original_cols) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  int rows = condensed_matrix.Rows();
  int condensed_cols = original_cols / 8;

  std::vector<uint64_t> result(rows, 0);
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  int num_blocks = original_cols / (8 * N);

  // Rearrange the vector as suggested.
  std::vector<uint64_t> rearranged_vec(original_cols);
  for (int b = 0; b < num_blocks; ++b) {
    for (int k = 0; k < 8; ++k) {
      for (size_t l = 0; l < N; ++l) {
        rearranged_vec[b * 8 * N + k * N + l] = vec[(b * N + l) * 8 + k];
      }
    }
  }

  auto mask = hn::Set(d, 0xFF);

  for (int i = 0; i < rows; ++i) {
    auto accum = hn::Zero(d);

    for (int b = 0; b < num_blocks; ++b) {
      auto m_condensed = hn::LoadU(d, &data[i * condensed_cols + b * N]);

      // k = 0
      auto m0 = hn::And(m_condensed, mask);
      auto V_0 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 0 * N]);
      accum = hn::MulAdd(m0, V_0, accum);

      // k = 1
      auto m1 = hn::And(hn::ShiftRight<8>(m_condensed), mask);
      auto V_1 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 1 * N]);
      accum = hn::MulAdd(m1, V_1, accum);

      // k = 2
      auto m2 = hn::And(hn::ShiftRight<16>(m_condensed), mask);
      auto V_2 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 2 * N]);
      accum = hn::MulAdd(m2, V_2, accum);

      // k = 3
      auto m3 = hn::And(hn::ShiftRight<24>(m_condensed), mask);
      auto V_3 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 3 * N]);
      accum = hn::MulAdd(m3, V_3, accum);

      // k = 4
      auto m4 = hn::And(hn::ShiftRight<32>(m_condensed), mask);
      auto V_4 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 4 * N]);
      accum = hn::MulAdd(m4, V_4, accum);

      // k = 5
      auto m5 = hn::And(hn::ShiftRight<40>(m_condensed), mask);
      auto V_5 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 5 * N]);
      accum = hn::MulAdd(m5, V_5, accum);

      // k = 6
      auto m6 = hn::And(hn::ShiftRight<48>(m_condensed), mask);
      auto V_6 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 6 * N]);
      accum = hn::MulAdd(m6, V_6, accum);

      // k = 7
      auto m7 = hn::ShiftRight<56>(m_condensed);
      auto V_7 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 7 * N]);
      accum = hn::MulAdd(m7, V_7, accum);
    }

    result[i] = hn::ReduceSum(d, accum);
  }

  return result;
}

}  // namespace

template <typename CoeffType>
absl::StatusOr<std::vector<uint64_t>> Matrix<CoeffType>::CondensedMultiply(
    const std::vector<uint64_t>& vec) const {
  if (!is_condensed_) {
    return absl::FailedPreconditionError(
        "CondensedMultiply can only be called on a condensed matrix.");
  }

  return ::private_membership::rlwe::v2::CondensedMultiply<CoeffType>(
      *this, vec, original_cols_);
}

template <typename CoeffType>
absl::StatusOr<std::vector<uint64_t>> Matrix<CoeffType>::MultiplyUint16(
    absl::Span<const uint64_t> vec, bool centered, int result_bits) const {
  if constexpr (std::is_same_v<CoeffType, uint16_t>) {
    if (is_condensed_) {
      return CondensedMultiplyUint16(*this, vec, original_cols_, centered,
                                     result_bits);
    }
  }
  return absl::FailedPreconditionError(
      "MultiplyUint16 requires a condensed uint16_t matrix.");
}

template <typename CoeffType>
absl::StatusOr<proto::Matrix> Matrix<CoeffType>::ToProto() const {
  proto::Matrix proto;
  proto.set_rows(rows_);
  proto.set_cols(cols_);
  proto.set_is_condensed(is_condensed_);
  proto.set_original_cols(original_cols_);
  if (!data_.empty()) {
    proto.set_data(std::string(reinterpret_cast<const char*>(data_.data()),
                               data_.size() * sizeof(CoeffType)));
  }
  if (!condensed_data_.empty()) {
    proto.set_condensed_data(
        std::string(reinterpret_cast<const char*>(condensed_data_.data()),
                    condensed_data_.size() * sizeof(uint64_t)));
  }
  return proto;
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::FromProto(
    const proto::Matrix& proto) {
  if (proto.has_data() && proto.has_condensed_data()) {
    return absl::InvalidArgumentError(
        "Matrix cannot have both data and condensed_data set.");
  }
  if (proto.is_condensed() && proto.has_data()) {
    return absl::InvalidArgumentError(
        "Condensed matrix cannot have normal data set.");
  }
  if (!proto.is_condensed() && proto.has_condensed_data()) {
    return absl::InvalidArgumentError(
        "Normal matrix cannot have condensed data set.");
  }

  std::vector<CoeffType> data;
  if (!proto.data().empty()) {
    if (proto.data().size() % sizeof(CoeffType) != 0) {
      return absl::InvalidArgumentError("Invalid serialized data size.");
    }
    data.resize(proto.data().size() / sizeof(CoeffType));
    std::memcpy(data.data(), proto.data().data(), proto.data().size());
  }
  std::vector<uint64_t> condensed_data;
  if (!proto.condensed_data().empty()) {
    if (proto.condensed_data().size() % sizeof(uint64_t) != 0) {
      return absl::InvalidArgumentError(
          "Invalid serialized condensed_data size.");
    }
    condensed_data.resize(proto.condensed_data().size() / sizeof(uint64_t));
    std::memcpy(condensed_data.data(), proto.condensed_data().data(),
                proto.condensed_data().size());
  }
  return Matrix(std::move(data), std::move(condensed_data), proto.rows(),
                proto.cols(), proto.is_condensed(), proto.original_cols());
}

template class Matrix<uint8_t>;
template class Matrix<uint16_t>;
template class Matrix<uint32_t>;
template class Matrix<uint64_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
