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

#ifndef CRYPTO_POLYNOMIAL_FFT_H_
#define CRYPTO_POLYNOMIAL_FFT_H_

#include <complex>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/polynomial.h"
#include "crypto/fft.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Exactness budget of the chunked f64 negacyclic products in
// Polynomial::MultFft(const FftPolynomial&, ...): with chunk magnitudes at
// most 2^u and 2^v, every coefficient of a chunk product is an integer bounded
// by ||x||_2 ||y||_2 <= 2^(u + v + log2 d). The f64 round trip (twist, two
// forward FFTs, pointwise product, inverse FFT, untwist) perturbs it by at most
// c * 2^-53 * ||x||_2 ||y||_2 with c < 2^8: Percival's bound for a radix-2 FFT
// convolution with accurately rounded twiddles is c ~ 14 log2(d) + 15, i.e.
// ~170 at d = 2^11 and < 256 up to d = 2^16. Rounding therefore recovers the
// exact integer whenever
//   u + v + log2(d) <= 53 - 8 - 1 = kFftExactProductBits.
inline constexpr int kFftExactProductBits = 44;

// FftContext holds pre-allocated buffers and precomputed FFT plans needed
// for optimized FFT-based polynomial multiplication.
struct FftContext {
  static absl::StatusOr<std::unique_ptr<FftContext>> Create(int log_d);

  ~FftContext() = default;

  // Non-copyable and non-movable.
  FftContext(const FftContext&) = delete;
  FftContext& operator=(const FftContext&) = delete;

  int d;
  // Scratch buffers, 2d long; MultFft(const FftPolynomial&, ...) uses their
  // first d entries.
  std::vector<std::complex<double>> forward_in;
  std::vector<std::complex<double>> forward_out;
  std::vector<std::complex<double>> backward_in;
  std::vector<std::complex<double>> backward_out;

  // 2d-point plan for the zero-padded products of InnerProductFft.
  std::unique_ptr<::security::fft::FftPlan<double, 1>> plan;

  // d-point plan and twist tables w^j = exp(i*pi*j/d), j < d, for the
  // negacyclic products of MultFft(const FftPolynomial&, ...): the cyclic
  // convolution of x_j w^j and y_j w^j equals w^k (x * y mod X^d + 1)_k.
  std::unique_ptr<::security::fft::FftPlan<double, 1>> negacyclic_plan;
  std::vector<std::complex<double>> twist;
  std::vector<std::complex<double>> untwist;

  // Reusable scratch buffers for InnerProductFft to avoid per-call heap
  // allocations.
  std::vector<std::vector<std::complex<double>>> accum_scratch;
  std::vector<std::vector<std::complex<double>>> u_ffts_scratch;
  std::vector<std::vector<std::complex<double>>> v_ffts_scratch;

 private:
  FftContext() = default;
};

// Forward FFTs of a vector of polynomials split into base-2^chunk_bits digits:
// the form in which Polynomial::InnerProductFft consumes its operands. Produced
// by Polynomial::ToChunkedFft so that an operand reused across many inner
// products is transformed only once.
struct ChunkedFft {
  int chunk_bits = 0;
  int max_chunk_magnitude_bits = 0;
  // ffts[i][c] is the length-2d FFT of digit c of polynomial i, zero-padded.
  std::vector<std::vector<std::vector<std::complex<double>>>> ffts;
};

// A polynomial with small coefficients (e.g. a ternary secret key) held in
// the negacyclic FFT domain of an FftContext, so that it can be multiplied
// by many polynomials at the cost of the forward transforms of the other
// operand only. The coefficients are read as `bits`-bit values, two's
// complement if `is_signed` (a ternary key storing -1 as q - 1 is `bits` = 2,
// `is_signed` = true); higher bits are ignored.
class FftPolynomial {
 public:
  template <typename CoeffType>
  static absl::StatusOr<FftPolynomial> Create(const Polynomial<CoeffType>& poly,
                                              FftContext& ctx, int bits,
                                              bool is_signed);

  int Len() const { return fft_.size(); }

  // Coefficient magnitudes are at most 2^MagnitudeBits().
  int MagnitudeBits() const { return magnitude_bits_; }

  const std::vector<std::complex<double>>& Fft() const { return fft_; }

 private:
  FftPolynomial(std::vector<std::complex<double>> fft, int magnitude_bits)
      : fft_(std::move(fft)), magnitude_bits_(magnitude_bits) {}

  std::vector<std::complex<double>> fft_;
  int magnitude_bits_;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_POLYNOMIAL_FFT_H_
