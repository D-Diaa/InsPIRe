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

#ifndef CRYPTO_FFT_H_
#define CRYPTO_FFT_H_

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace security::fft {

enum class Normalization {
  kNone,
};

template <typename T, size_t N>
class FftPlan;

template <>
class FftPlan<double, 1> {
 public:
  explicit FftPlan(std::array<size_t, 1> dims,
                   Normalization norm = Normalization::kNone)
      : size_(dims[0]) {
    bit_rev_.resize(size_);
    bit_rev32_.resize(size_);
    size_t log_n = 0;
    while ((size_t(1) << log_n) < size_) {
      ++log_n;
    }
    for (size_t i = 0; i < size_; ++i) {
      size_t rev = 0;
      for (size_t b = 0; b < log_n; ++b) {
        if (i & (size_t(1) << b)) {
          rev |= (size_t(1) << (log_n - 1 - b));
        }
      }
      bit_rev_[i] = rev;
      bit_rev32_[i] = static_cast<uint32_t>(rev);
    }

    twiddles_forward_.resize(size_ / 2);
    twiddles_backward_.resize(size_ / 2);
    const double pi = std::acos(-1.0);
    for (size_t i = 0; i < size_ / 2; ++i) {
      double angle = -2.0 * pi * i / size_;
      twiddles_forward_[i] =
          std::complex<double>(std::cos(angle), std::sin(angle));
      twiddles_backward_[i] =
          std::complex<double>(std::cos(-angle), std::sin(-angle));
    }

    if (size_ >= 2) {
      stage_twiddles_forward_.reserve(size_ - 1);
      stage_twiddles_backward_.reserve(size_ - 1);
      for (size_t len = 2; len <= size_; len <<= 1) {
        size_t half_len = len >> 1;
        size_t step = size_ / len;
        for (size_t j = 0; j < half_len; ++j) {
          stage_twiddles_forward_.push_back(twiddles_forward_[j * step]);
          stage_twiddles_backward_.push_back(twiddles_backward_[j * step]);
        }
      }
    }
  }

  ~FftPlan() = default;

  FftPlan(const FftPlan&) = default;
  FftPlan& operator=(const FftPlan&) = default;
  FftPlan(FftPlan&&) = default;
  FftPlan& operator=(FftPlan&&) = default;

  size_t size() const { return size_; }
  const std::vector<size_t>& bit_rev() const { return bit_rev_; }
  const std::vector<uint32_t>& bit_rev32() const { return bit_rev32_; }
  const std::vector<std::complex<double>>& twiddles_forward() const {
    return twiddles_forward_;
  }
  const std::vector<std::complex<double>>& twiddles_backward() const {
    return twiddles_backward_;
  }
  const std::vector<std::complex<double>>& stage_twiddles_forward() const {
    return stage_twiddles_forward_;
  }
  const std::vector<std::complex<double>>& stage_twiddles_backward() const {
    return stage_twiddles_backward_;
  }

 private:
  size_t size_;
  std::vector<size_t> bit_rev_;
  std::vector<uint32_t> bit_rev32_;
  std::vector<std::complex<double>> twiddles_forward_;
  std::vector<std::complex<double>> twiddles_backward_;
  std::vector<std::complex<double>> stage_twiddles_forward_;
  std::vector<std::complex<double>> stage_twiddles_backward_;
};

inline void ComputeFftWithStageTwiddles(
    absl::Span<const std::complex<double>> in,
    absl::Span<std::complex<double>> out,
    const FftPlan<double, 1>& plan,
    const std::vector<std::complex<double>>& stage_twiddles) {
  const size_t n = plan.size();
  const uint32_t* __restrict br = plan.bit_rev32().data();
  const std::complex<double>* __restrict in_ptr = in.data();
  std::complex<double>* __restrict out_ptr = out.data();
  for (size_t i = 0; i < n; ++i) {
    out_ptr[br[i]] = in_ptr[i];
  }
  if (n < 2) return;

#if defined(__AVX2__) && defined(__FMA__)
  // Stage len = 2: twiddle is (1, 0), no complex multiplications needed.
  for (size_t i = 0; i < n; i += 2) {
    __m128d u = _mm_loadu_pd(reinterpret_cast<const double*>(&out_ptr[i]));
    __m128d v = _mm_loadu_pd(reinterpret_cast<const double*>(&out_ptr[i + 1]));
    _mm_storeu_pd(reinterpret_cast<double*>(&out_ptr[i]), _mm_add_pd(u, v));
    _mm_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + 1]), _mm_sub_pd(u, v));
  }

  size_t tw_offset = 1;
  for (size_t len = 4; len <= n; len <<= 1) {
    const size_t half_len = len >> 1;
    const std::complex<double>* __restrict stw =
        stage_twiddles.data() + tw_offset;
    for (size_t i = 0; i < n; i += len) {
      size_t j = 0;
      for (; j + 4 <= half_len; j += 4) {
        __m256d u0 =
            _mm256_loadu_pd(reinterpret_cast<const double*>(&out_ptr[i + j]));
        __m256d b0 = _mm256_loadu_pd(
            reinterpret_cast<const double*>(&out_ptr[i + j + half_len]));
        __m256d w0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&stw[j]));
        __m256d b0_re = _mm256_movedup_pd(b0);
        __m256d b0_im = _mm256_unpackhi_pd(b0, b0);
        __m256d w0_sw = _mm256_permute_pd(w0, 0x5);
        __m256d v0 =
            _mm256_fmaddsub_pd(b0_re, w0, _mm256_mul_pd(b0_im, w0_sw));

        __m256d u1 = _mm256_loadu_pd(
            reinterpret_cast<const double*>(&out_ptr[i + j + 2]));
        __m256d b1 = _mm256_loadu_pd(
            reinterpret_cast<const double*>(&out_ptr[i + j + 2 + half_len]));
        __m256d w1 =
            _mm256_loadu_pd(reinterpret_cast<const double*>(&stw[j + 2]));
        __m256d b1_re = _mm256_movedup_pd(b1);
        __m256d b1_im = _mm256_unpackhi_pd(b1, b1);
        __m256d w1_sw = _mm256_permute_pd(w1, 0x5);
        __m256d v1 =
            _mm256_fmaddsub_pd(b1_re, w1, _mm256_mul_pd(b1_im, w1_sw));

        _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + j]),
                         _mm256_add_pd(u0, v0));
        _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + j + half_len]),
                         _mm256_sub_pd(u0, v0));
        _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + j + 2]),
                         _mm256_add_pd(u1, v1));
        _mm256_storeu_pd(
            reinterpret_cast<double*>(&out_ptr[i + j + 2 + half_len]),
            _mm256_sub_pd(u1, v1));
      }
      for (; j < half_len; j += 2) {
        __m256d u0 =
            _mm256_loadu_pd(reinterpret_cast<const double*>(&out_ptr[i + j]));
        __m256d b0 = _mm256_loadu_pd(
            reinterpret_cast<const double*>(&out_ptr[i + j + half_len]));
        __m256d w0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&stw[j]));
        __m256d b0_re = _mm256_movedup_pd(b0);
        __m256d b0_im = _mm256_unpackhi_pd(b0, b0);
        __m256d w0_sw = _mm256_permute_pd(w0, 0x5);
        __m256d v0 =
            _mm256_fmaddsub_pd(b0_re, w0, _mm256_mul_pd(b0_im, w0_sw));
        _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + j]),
                         _mm256_add_pd(u0, v0));
        _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[i + j + half_len]),
                         _mm256_sub_pd(u0, v0));
      }
    }
    tw_offset += half_len;
  }
#else
  size_t tw_offset = 0;
  for (size_t len = 2; len <= n; len <<= 1) {
    size_t half_len = len >> 1;
    const std::complex<double>* __restrict stw =
        stage_twiddles.data() + tw_offset;
    for (size_t i = 0; i < n; i += len) {
      for (size_t j = 0; j < half_len; ++j) {
        const double u_re = out_ptr[i + j].real();
        const double u_im = out_ptr[i + j].imag();
        const double b_re = out_ptr[i + j + half_len].real();
        const double b_im = out_ptr[i + j + half_len].imag();
        const double w_re = stw[j].real();
        const double w_im = stw[j].imag();
        const double v_re = b_re * w_re - b_im * w_im;
        const double v_im = b_re * w_im + b_im * w_re;
        out_ptr[i + j] = std::complex<double>(u_re + v_re, u_im + v_im);
        out_ptr[i + j + half_len] =
            std::complex<double>(u_re - v_re, u_im - v_im);
      }
    }
    tw_offset += half_len;
  }
#endif
}

inline void ComputeFftInternal(
    absl::Span<const std::complex<double>> in,
    absl::Span<std::complex<double>> out,
    const FftPlan<double, 1>& plan,
    const std::vector<std::complex<double>>& twiddles) {
  if (&twiddles == &plan.twiddles_forward()) {
    ComputeFftWithStageTwiddles(in, out, plan, plan.stage_twiddles_forward());
    return;
  }
  if (&twiddles == &plan.twiddles_backward()) {
    ComputeFftWithStageTwiddles(in, out, plan, plan.stage_twiddles_backward());
    return;
  }
  const size_t n = plan.size();
  const auto& bit_rev = plan.bit_rev();
  for (size_t i = 0; i < n; ++i) {
    out[bit_rev[i]] = in[i];
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    size_t half_len = len >> 1;
    size_t step = n / len;
    for (size_t i = 0; i < n; i += len) {
      for (size_t j = 0; j < half_len; ++j) {
        std::complex<double> u = out[i + j];
        std::complex<double> v = out[i + j + half_len] * twiddles[j * step];
        out[i + j] = u + v;
        out[i + j + half_len] = u - v;
      }
    }
  }
}

inline absl::Status Fft(absl::Span<const std::complex<double>> in,
                        absl::Span<std::complex<double>> out,
                        const FftPlan<double, 1>& plan) {
  if (in.size() != plan.size() || out.size() != plan.size()) {
    return absl::InvalidArgumentError(
        "Input/output size mismatch with FFT plan.");
  }
  ComputeFftWithStageTwiddles(in, out, plan, plan.stage_twiddles_forward());
  return absl::OkStatus();
}

inline absl::Status Ifft(absl::Span<const std::complex<double>> in,
                         absl::Span<std::complex<double>> out,
                         const FftPlan<double, 1>& plan) {
  if (in.size() != plan.size() || out.size() != plan.size()) {
    return absl::InvalidArgumentError(
        "Input/output size mismatch with FFT plan.");
  }
  ComputeFftWithStageTwiddles(in, out, plan, plan.stage_twiddles_backward());
  return absl::OkStatus();
}

}  // namespace security::fft

#endif  // CRYPTO_FFT_H_
