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

#include "crypto/polynomial_fft.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/polynomial.h"
#include "crypto/fft.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

namespace fft = ::security::fft;

absl::StatusOr<std::unique_ptr<FftContext>> FftContext::Create(int log_d) {
  if (log_d < 0 || log_d > 30) {
    return absl::InvalidArgumentError("Invalid log_d.");
  }
  int d = 1 << log_d;
  int len2 = 2 * d;
  std::unique_ptr<FftContext> ctx(new FftContext());
  ctx->d = d;

  ctx->forward_in.resize(len2);
  ctx->forward_out.resize(len2);
  ctx->backward_in.resize(len2);
  ctx->backward_out.resize(len2);

  ctx->plan = std::make_unique<fft::FftPlan<double, 1>>(
      std::array<size_t, 1>{static_cast<size_t>(len2)},
      fft::Normalization::kNone);

  ctx->negacyclic_plan = std::make_unique<fft::FftPlan<double, 1>>(
      std::array<size_t, 1>{static_cast<size_t>(d)},
      fft::Normalization::kNone);
  ctx->twist.resize(d);
  ctx->untwist.resize(d);
  const double pi = std::acos(-1.0);
  for (int j = 0; j < d; ++j) {
    const double angle = pi * j / d;
    ctx->twist[j] = std::complex<double>(std::cos(angle), std::sin(angle));
    ctx->untwist[j] = std::conj(ctx->twist[j]);
  }

  return ctx;
}

namespace {

using Fft = std::vector<std::complex<double>>;

// Returns bits [shift, shift + width) of `coeff` as a double, read as a
// `signed_width`-bit two's complement value when `signed_width` > 0.
double ChunkValue(uint64_t coeff, int shift, int width, int signed_width) {
  const uint64_t mask = width >= 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
  const uint64_t chunk = (coeff >> shift) & mask;
  if (signed_width > 0) {
    const int sign_shift = 64 - signed_width;
    return static_cast<double>(static_cast<int64_t>(chunk << sign_shift) >>
                               sign_shift);
  }
  return static_cast<double>(chunk);
}

// Number of base-2^chunk_bits digits of a `bits`-bit operand, capped at the
// digits that fit in CoeffType.
template <typename CoeffType>
int NumChunks(int bits, int chunk_bits) {
  const int max_chunks = (8 * sizeof(CoeffType) + chunk_bits - 1) / chunk_bits;
  return std::min((bits + chunk_bits - 1) / chunk_bits, max_chunks);
}

// Forward FFT (length 2d, zero-padded) of digit `chunk` of `coeffs`. The top
// digit of a signed operand is sign-extended.
template <typename CoeffType>
absl::Status DigitFft(const std::vector<CoeffType>& coeffs, int chunk,
                      int chunk_bits, int num_chunks, int bits, bool is_signed,
                      FftContext& ctx, Fft& out) {
  const int d = coeffs.size();
  const int len2 = 2 * d;
  const int signed_width =
      (is_signed && chunk == num_chunks - 1) ? bits - chunk_bits * chunk : 0;
  for (int j = 0; j < len2; ++j) {
    const double val =
        j < d ? ChunkValue(coeffs[j], chunk_bits * chunk, chunk_bits,
                           signed_width)
              : 0.0;
    ctx.forward_in[j] = std::complex<double>(val, 0.0);
  }
  return fft::Fft(absl::MakeConstSpan(ctx.forward_in), absl::MakeSpan(out),
                  *ctx.plan);
}

// accum[a * v.size() + b] += u[a] * v[b] pointwise, for every digit pair whose
// weight 2^(chunk_bits * (a + b)) is below the coefficient modulus.
template <typename CoeffType>
void AccumulateDigitProducts(const std::vector<Fft>& u,
                             const std::vector<Fft>& v, int chunk_bits,
                             std::vector<Fft>& accum) {
  const int v_num_chunks = v.size();
  for (int a = 0; a < static_cast<int>(u.size()); ++a) {
    for (int b = 0; b < v_num_chunks; ++b) {
      if (chunk_bits * (a + b) >= 8 * sizeof(CoeffType)) {
        continue;
      }
      Fft& acc = accum[a * v_num_chunks + b];
      for (size_t j = 0; j < acc.size(); ++j) {
        acc[j] += u[a][j] * v[b][j];
      }
    }
  }
}

// Inverse-transforms each digit-pair accumulator, rounds it to integers and
// folds the result modulo X^d + 1 at weight 2^(chunk_bits * (a + b)).
template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> RecombineDigitProducts(
    const std::vector<Fft>& accum, int u_num_chunks, int v_num_chunks,
    int chunk_bits, FftContext& ctx) {
  const int d = ctx.d;
  const int len2 = 2 * d;
  std::vector<CoeffType> result(d, 0);
  for (int a = 0; a < u_num_chunks; ++a) {
    for (int b = 0; b < v_num_chunks; ++b) {
      if (chunk_bits * (a + b) >= 8 * sizeof(CoeffType)) {
        continue;
      }
      RETURN_IF_ERROR(
          fft::Ifft(absl::MakeConstSpan(accum[a * v_num_chunks + b]),
                    absl::MakeSpan(ctx.backward_out), *ctx.plan));
      for (int j = 0; j < len2; ++j) {
        double val_double = ctx.backward_out[j].real() / len2;
        int64_t val64 = static_cast<int64_t>(std::round(val_double));
        CoeffType val = static_cast<CoeffType>(static_cast<uint64_t>(val64));
        if (j < d) {
          result[j] += (val << (chunk_bits * (a + b)));
        } else {
          result[j - d] -= (val << (chunk_bits * (a + b)));
        }
      }
    }
  }
  return Polynomial<CoeffType>::Create(std::move(result));
}

}  // namespace

template <typename CoeffType>
absl::StatusOr<FftPolynomial> FftPolynomial::Create(
    const Polynomial<CoeffType>& poly, FftContext& ctx, int bits,
    bool is_signed) {
  const int d = poly.Len();
  if (d != ctx.d || ctx.negacyclic_plan == nullptr) {
    return absl::InvalidArgumentError(
        "Polynomial length or context mismatch.");
  }
  if (bits <= 0 || bits > static_cast<int>(8 * sizeof(CoeffType))) {
    return absl::InvalidArgumentError("Invalid bits.");
  }
  const int signed_width = is_signed ? bits : 0;
  for (int j = 0; j < d; ++j) {
    ctx.forward_in[j] =
        ChunkValue(poly.Coeffs()[j], 0, bits, signed_width) * ctx.twist[j];
  }
  std::vector<std::complex<double>> result(d);
  RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(ctx.forward_in.data(), d),
                           absl::MakeSpan(result), *ctx.negacyclic_plan));
  return FftPolynomial(std::move(result), is_signed ? bits - 1 : bits);
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::MultFft(
    const FftPolynomial& that, FftContext& ctx, int this_bits) const {
  const int d = Len();
  if (d != ctx.d || that.Len() != d || ctx.negacyclic_plan == nullptr) {
    return absl::InvalidArgumentError(
        "Polynomial lengths or context mismatch.");
  }
  if (this_bits <= 0 ||
      this_bits > static_cast<int>(8 * sizeof(CoeffType))) {
    return absl::InvalidArgumentError("Invalid this_bits.");
  }
  const int log_d = absl::bit_width(static_cast<uint32_t>(d)) - 1;
  const int chunk_bits = std::min(
      this_bits, kFftExactProductBits - log_d - that.MagnitudeBits());
  if (chunk_bits <= 0) {
    return absl::InvalidArgumentError(
        "Operands too large for an exact FFT product.");
  }
  const int num_chunks = (this_bits + chunk_bits - 1) / chunk_bits;

  // Each chunk product is an exact integer polynomial (see
  // kFftExactProductBits); the chunks are recombined modulo 2^(8 * sizeof
  // CoeffType) by the shifts.
  std::vector<CoeffType> result(d, 0);
  for (int c = 0; c < num_chunks; ++c) {
    const int shift = c * chunk_bits;
    const int width = std::min(chunk_bits, this_bits - shift);
    for (int j = 0; j < d; ++j) {
      ctx.forward_in[j] =
          ChunkValue(coeffs_[j], shift, width, 0) * ctx.twist[j];
    }
    RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(ctx.forward_in.data(), d),
                             absl::MakeSpan(ctx.forward_out.data(), d),
                             *ctx.negacyclic_plan));
    for (int j = 0; j < d; ++j) {
      ctx.backward_in[j] = ctx.forward_out[j] * that.Fft()[j];
    }
    RETURN_IF_ERROR(fft::Ifft(absl::MakeConstSpan(ctx.backward_in.data(), d),
                              absl::MakeSpan(ctx.backward_out.data(), d),
                              *ctx.negacyclic_plan));
    for (int j = 0; j < d; ++j) {
      const double val = (ctx.backward_out[j] * ctx.untwist[j]).real() / d;
      const int64_t val64 = static_cast<int64_t>(std::round(val));
      result[j] += static_cast<CoeffType>(static_cast<uint64_t>(val64))
                   << shift;
    }
  }

  return Polynomial<CoeffType>::Create(std::move(result));
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::MultFft(
    const Polynomial& that, FftContext& ctx, int this_bits, int that_bits,
    int chunk_bits) const {
  return InnerProductFft({*this}, {that}, ctx, this_bits, that_bits,
                         chunk_bits);
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::InnerProductFft(
    const std::vector<Polynomial>& u, const std::vector<Polynomial>& v,
    FftContext& ctx, int u_bits, int v_bits, int chunk_bits, bool u_is_signed) {
  if (u.size() != v.size()) {
    return absl::InvalidArgumentError(
        "Polynomial vectors must have the same size.");
  }
  if (u.empty()) {
    return absl::InvalidArgumentError("Polynomial vectors cannot be empty.");
  }
  int d = u[0].Len();
  int len2 = 2 * d;
  for (size_t i = 0; i < u.size(); ++i) {
    if (u[i].Len() != d || v[i].Len() != d || d != ctx.d) {
      return absl::InvalidArgumentError(
          "Polynomial lengths or context mismatch.");
    }
  }
  if (ctx.plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  if (u_bits <= 0 || v_bits <= 0 || chunk_bits <= 0 || chunk_bits > 63) {
    return absl::InvalidArgumentError("Invalid bits or chunk_bits.");
  }
  if (std::log2(u.size()) + std::log2(d) + 2.0 * chunk_bits > 53.0) {
    return absl::InvalidArgumentError(
        "Potential precision overflow in accumulation.");
  }

  const int u_num_chunks = NumChunks<CoeffType>(u_bits, chunk_bits);
  const int v_num_chunks = NumChunks<CoeffType>(v_bits, chunk_bits);

  std::vector<Fft> accum(u_num_chunks * v_num_chunks, Fft(len2, {0.0, 0.0}));
  std::vector<Fft> u_ffts(u_num_chunks, Fft(len2));
  std::vector<Fft> v_ffts(v_num_chunks, Fft(len2));

  for (size_t i = 0; i < u.size(); ++i) {
    for (int c = 0; c < u_num_chunks; ++c) {
      RETURN_IF_ERROR(DigitFft(u[i].coeffs_, c, chunk_bits, u_num_chunks,
                               u_bits, u_is_signed, ctx, u_ffts[c]));
    }
    for (int c = 0; c < v_num_chunks; ++c) {
      RETURN_IF_ERROR(DigitFft(v[i].coeffs_, c, chunk_bits, v_num_chunks,
                               v_bits, /*is_signed=*/false, ctx, v_ffts[c]));
    }
    AccumulateDigitProducts<CoeffType>(u_ffts, v_ffts, chunk_bits, accum);
  }

  return RecombineDigitProducts<CoeffType>(accum, u_num_chunks, v_num_chunks,
                                           chunk_bits, ctx);
}

template <typename CoeffType>
absl::StatusOr<ChunkedFft> Polynomial<CoeffType>::ToChunkedFft(
    const std::vector<Polynomial>& u, FftContext& ctx, int u_bits,
    int chunk_bits, bool u_is_signed) {
  if (u.empty()) {
    return absl::InvalidArgumentError("Polynomial vector cannot be empty.");
  }
  for (const Polynomial& poly : u) {
    if (poly.Len() != ctx.d) {
      return absl::InvalidArgumentError(
          "Polynomial lengths or context mismatch.");
    }
  }
  if (ctx.plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  if (u_bits <= 0 || chunk_bits <= 0 || chunk_bits > 63) {
    return absl::InvalidArgumentError("Invalid bits or chunk_bits.");
  }

  const int num_chunks = NumChunks<CoeffType>(u_bits, chunk_bits);
  ChunkedFft result;
  result.chunk_bits = chunk_bits;
  result.ffts.assign(u.size(), std::vector<Fft>(num_chunks, Fft(2 * ctx.d)));
  for (size_t i = 0; i < u.size(); ++i) {
    for (int c = 0; c < num_chunks; ++c) {
      RETURN_IF_ERROR(DigitFft(u[i].coeffs_, c, chunk_bits, num_chunks, u_bits,
                               u_is_signed, ctx, result.ffts[i][c]));
    }
  }
  return result;
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::InnerProductFft(
    const ChunkedFft& u, const ChunkedFft& v, FftContext& ctx) {
  if (u.ffts.size() != v.ffts.size()) {
    return absl::InvalidArgumentError(
        "Polynomial vectors must have the same size.");
  }
  if (u.ffts.empty()) {
    return absl::InvalidArgumentError("Polynomial vectors cannot be empty.");
  }
  if (u.chunk_bits != v.chunk_bits || u.chunk_bits <= 0 || u.chunk_bits > 63) {
    return absl::InvalidArgumentError("Invalid or mismatched chunk_bits.");
  }
  if (ctx.plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  const int chunk_bits = u.chunk_bits;
  const int d = ctx.d;
  const size_t len2 = 2 * d;
  const int u_num_chunks = u.ffts[0].size();
  const int v_num_chunks = v.ffts[0].size();
  for (size_t i = 0; i < u.ffts.size(); ++i) {
    if (u.ffts[i].size() != static_cast<size_t>(u_num_chunks) ||
        v.ffts[i].size() != static_cast<size_t>(v_num_chunks)) {
      return absl::InvalidArgumentError("Inconsistent number of digits.");
    }
    for (const Fft& fft : u.ffts[i]) {
      if (fft.size() != len2) {
        return absl::InvalidArgumentError("FFT length and context mismatch.");
      }
    }
    for (const Fft& fft : v.ffts[i]) {
      if (fft.size() != len2) {
        return absl::InvalidArgumentError("FFT length and context mismatch.");
      }
    }
  }
  if (std::log2(u.ffts.size()) + std::log2(d) + 2.0 * chunk_bits > 53.0) {
    return absl::InvalidArgumentError(
        "Potential precision overflow in accumulation.");
  }

  std::vector<Fft> accum(u_num_chunks * v_num_chunks, Fft(len2, {0.0, 0.0}));
  for (size_t i = 0; i < u.ffts.size(); ++i) {
    AccumulateDigitProducts<CoeffType>(u.ffts[i], v.ffts[i], chunk_bits,
                                       accum);
  }
  return RecombineDigitProducts<CoeffType>(accum, u_num_chunks, v_num_chunks,
                                           chunk_bits, ctx);
}

template absl::StatusOr<FftPolynomial> FftPolynomial::Create(
    const Polynomial<uint32_t>&, FftContext&, int, bool);
template absl::StatusOr<FftPolynomial> FftPolynomial::Create(
    const Polynomial<uint64_t>&, FftContext&, int, bool);

template class Polynomial<uint32_t>;
template class Polynomial<uint64_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
