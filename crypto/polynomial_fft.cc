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
  const int d = 1 << log_d;
  std::unique_ptr<FftContext> ctx(new FftContext());
  ctx->d = d;

  ctx->forward_in.resize(d);
  ctx->forward_out.resize(d);
  ctx->backward_in.resize(d);
  ctx->backward_out.resize(d);

  auto init_tables = [d](std::unique_ptr<fft::FftPlan<double, 1>>& plan,
                         std::vector<std::complex<double>>& twist,
                         std::vector<std::complex<double>>& untwist) {
    plan = std::make_unique<fft::FftPlan<double, 1>>(
        std::array<size_t, 1>{static_cast<size_t>(d)},
        fft::Normalization::kNone);
    twist.resize(d);
    untwist.resize(d);
    const double pi = std::acos(-1.0);
    for (int j = 0; j < d; ++j) {
      const double angle = pi * j / d;
      twist[j] = std::complex<double>(std::cos(angle), std::sin(angle));
      untwist[j] = std::conj(twist[j]);
    }
  };

  if (log_d <= 16) {
    struct CachedPlans {
      int log_d = -1;
      std::vector<std::complex<double>> twist;
      std::vector<std::complex<double>> untwist;
      std::unique_ptr<fft::FftPlan<double, 1>> negacyclic_plan;
    };
    thread_local CachedPlans cache;
    if (cache.log_d != log_d) {
      cache.log_d = log_d;
      init_tables(cache.negacyclic_plan, cache.twist, cache.untwist);
    }
    ctx->negacyclic_plan =
        std::make_unique<fft::FftPlan<double, 1>>(*cache.negacyclic_plan);
    ctx->twist = cache.twist;
    ctx->untwist = cache.untwist;
    return ctx;
  }

  init_tables(ctx->negacyclic_plan, ctx->twist, ctx->untwist);
  return ctx;
}

namespace {

using Fft = std::vector<std::complex<double>>;

// Returns bits [shift, shift + width) of `coeff` as a double, read as a
// `signed_width`-bit two's complement value when `signed_width` > 0.
inline double ChunkValue(uint64_t coeff, int shift, int width,
                         int signed_width) {
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

// Forward twisted negacyclic FFT (length d) of digit `chunk` of `coeffs`. The
// top digit of a signed operand is sign-extended.
template <typename CoeffType>
absl::Status DigitFft(const std::vector<CoeffType>& coeffs, int chunk,
                      int chunk_bits, int num_chunks, int bits, bool is_signed,
                      FftContext& ctx, Fft& out) {
  const int d = coeffs.size();
  const int shift = chunk_bits * chunk;
  const int signed_width =
      (is_signed && chunk == num_chunks - 1) ? bits - shift : 0;
  const std::complex<double>* __restrict tw = ctx.twist.data();
  std::complex<double>* __restrict fin = ctx.forward_in.data();
  for (int j = 0; j < d; ++j) {
    const double val = ChunkValue(coeffs[j], shift, chunk_bits, signed_width);
    fin[j] = std::complex<double>(val * tw[j].real(), val * tw[j].imag());
  }
  return fft::Fft(absl::MakeConstSpan(fin, d), absl::MakeSpan(out.data(), d),
                  *ctx.negacyclic_plan);
}

// Simultaneous forward twisted negacyclic FFT (length d) of two real digit
// polynomials via a single complex FFT z[j] = (x[j] + i * y[j]) * w^j,
// followed by conjugate-symmetry unpacking with kp = (d + 1 - k) mod d:
//   X[k] = 0.5 * (Z[k] + conj(Z[kp]))
//   Y[k] = -0.5i * (Z[k] - conj(Z[kp]))
template <typename CoeffType>
absl::Status TwoDigitsFft(const std::vector<CoeffType>& coeffs0, int chunk0,
                          const std::vector<CoeffType>& coeffs1, int chunk1,
                          int chunk_bits, int num_chunks, int bits,
                          bool is_signed, FftContext& ctx, Fft& out0,
                          Fft& out1) {
  const int d = coeffs0.size();
  const int shift0 = chunk_bits * chunk0;
  const int shift1 = chunk_bits * chunk1;
  const int signed_width0 =
      (is_signed && chunk0 == num_chunks - 1) ? bits - shift0 : 0;
  const int signed_width1 =
      (is_signed && chunk1 == num_chunks - 1) ? bits - shift1 : 0;
  const std::complex<double>* __restrict tw = ctx.twist.data();
  std::complex<double>* __restrict fin = ctx.forward_in.data();
  std::complex<double>* __restrict fout = ctx.forward_out.data();
  for (int j = 0; j < d; ++j) {
    const double v0 =
        ChunkValue(coeffs0[j], shift0, chunk_bits, signed_width0);
    const double v1 =
        ChunkValue(coeffs1[j], shift1, chunk_bits, signed_width1);
    const double tw_r = tw[j].real();
    const double tw_i = tw[j].imag();
    fin[j] = std::complex<double>(v0 * tw_r - v1 * tw_i, v0 * tw_i + v1 * tw_r);
  }
  RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(fin, d),
                           absl::MakeSpan(fout, d), *ctx.negacyclic_plan));
  const std::complex<double>* __restrict z = fout;
  if (d == 1) {
    out0[0] = std::complex<double>(z[0].real(), 0.0);
    out1[0] = std::complex<double>(z[0].imag(), 0.0);
    return absl::OkStatus();
  }
  auto unpack_pair = [&](int k, int kp) {
    const double ar = z[k].real();
    const double ai = z[k].imag();
    const double br = z[kp].real();
    const double bi = z[kp].imag();
    const double x_re = 0.5 * (ar + br);
    const double x_im = 0.5 * (ai - bi);
    const double y_re = 0.5 * (ai + bi);
    const double y_im = 0.5 * (br - ar);
    out0[k] = std::complex<double>(x_re, x_im);
    out0[kp] = std::complex<double>(x_re, -x_im);
    out1[k] = std::complex<double>(y_re, y_im);
    out1[kp] = std::complex<double>(y_re, -y_im);
  };
  unpack_pair(0, 1);
  for (int k = 2; k <= d / 2; ++k) {
    unpack_pair(k, d + 1 - k);
  }
  return absl::OkStatus();
}

inline void PointwiseComplexMultiply(
    const std::complex<double>* __restrict u_ptr,
    const std::complex<double>* __restrict v_ptr,
    std::complex<double>* __restrict out_ptr, size_t n, bool accumulate) {
#if defined(__AVX2__) && defined(__FMA__)
  size_t j = 0;
  if (!accumulate) {
    for (; j + 2 <= n; j += 2) {
      __m256d u_vec =
          _mm256_loadu_pd(reinterpret_cast<const double*>(&u_ptr[j]));
      __m256d v_vec =
          _mm256_loadu_pd(reinterpret_cast<const double*>(&v_ptr[j]));
      __m256d u_re = _mm256_movedup_pd(u_vec);
      __m256d u_im = _mm256_unpackhi_pd(u_vec, u_vec);
      __m256d v_sw = _mm256_permute_pd(v_vec, 0x5);
      __m256d prod =
          _mm256_fmaddsub_pd(u_re, v_vec, _mm256_mul_pd(u_im, v_sw));
      _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[j]), prod);
    }
  } else {
    for (; j + 2 <= n; j += 2) {
      __m256d acc_vec =
          _mm256_loadu_pd(reinterpret_cast<const double*>(&out_ptr[j]));
      __m256d u_vec =
          _mm256_loadu_pd(reinterpret_cast<const double*>(&u_ptr[j]));
      __m256d v_vec =
          _mm256_loadu_pd(reinterpret_cast<const double*>(&v_ptr[j]));
      __m256d u_re = _mm256_movedup_pd(u_vec);
      __m256d u_im = _mm256_unpackhi_pd(u_vec, u_vec);
      __m256d v_sw = _mm256_permute_pd(v_vec, 0x5);
      __m256d prod =
          _mm256_fmaddsub_pd(u_re, v_vec, _mm256_mul_pd(u_im, v_sw));
      _mm256_storeu_pd(reinterpret_cast<double*>(&out_ptr[j]),
                       _mm256_add_pd(acc_vec, prod));
    }
  }
  for (; j < n; ++j) {
    const double ur = u_ptr[j].real();
    const double ui = u_ptr[j].imag();
    const double vr = v_ptr[j].real();
    const double vi = v_ptr[j].imag();
    const std::complex<double> p(ur * vr - ui * vi, ur * vi + ui * vr);
    out_ptr[j] = accumulate ? (out_ptr[j] + p) : p;
  }
#else
  for (size_t j = 0; j < n; ++j) {
    const double ur = u_ptr[j].real();
    const double ui = u_ptr[j].imag();
    const double vr = v_ptr[j].real();
    const double vi = v_ptr[j].imag();
    const std::complex<double> p(ur * vr - ui * vi, ur * vi + ui * vr);
    out_ptr[j] = accumulate ? (out_ptr[j] + p) : p;
  }
#endif
}

template <typename CoeffType>
absl::Status TransformPolysDigits(const Polynomial<CoeffType>* polys,
                                  int num_polys, int num_chunks, int chunk_bits,
                                  int bits, bool is_signed, bool allow_paired,
                                  FftContext& ctx, Fft* out_ffts) {
  const int total = num_polys * num_chunks;
  int idx = 0;
  if (allow_paired) {
    for (; idx + 2 <= total; idx += 2) {
      const int p0 = idx / num_chunks;
      const int c0 = idx % num_chunks;
      const int p1 = (idx + 1) / num_chunks;
      const int c1 = (idx + 1) % num_chunks;
      RETURN_IF_ERROR(TwoDigitsFft(polys[p0].Coeffs(), c0, polys[p1].Coeffs(),
                                   c1, chunk_bits, num_chunks, bits, is_signed,
                                   ctx, out_ffts[idx], out_ffts[idx + 1]));
    }
  }
  for (; idx < total; ++idx) {
    const int p0 = idx / num_chunks;
    const int c0 = idx % num_chunks;
    RETURN_IF_ERROR(DigitFft(polys[p0].Coeffs(), c0, chunk_bits, num_chunks,
                             bits, is_signed, ctx, out_ffts[idx]));
  }
  return absl::OkStatus();
}

// accum[a * v.size() + b] += u[a] * v[b] pointwise (or = when first_term is
// true), for every digit pair whose weight 2^(chunk_bits * (a + b)) is below
// the coefficient modulus.
template <typename CoeffType>
void AccumulateDigitProducts(const Fft* u, int u_num_chunks, const Fft* v,
                             int v_num_chunks, int chunk_bits, bool first_term,
                             std::vector<Fft>& accum) {
  for (int a = 0; a < u_num_chunks; ++a) {
    for (int b = 0; b < v_num_chunks; ++b) {
      if (chunk_bits * (a + b) >= static_cast<int>(8 * sizeof(CoeffType))) {
        continue;
      }
      Fft& acc = accum[a * v_num_chunks + b];
      PointwiseComplexMultiply(u[a].data(), v[b].data(), acc.data(), acc.size(),
                               /*accumulate=*/!first_term);
    }
  }
}

// Inverse-transforms each twisted negacyclic digit-pair accumulator, untwists,
// rounds to integers and accumulates at weight 2^(chunk_bits * (a + b)).
// Because each untwisted inverse transform is a real-valued sequence, pairs of
// accumulators (A_0, A_1) are inverted simultaneously via a single length-d
// complex IFFT of A_0 + i * A_1 whenever the exactness budget has headroom for
// the sqrt(2) norm increase.
template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> RecombineDigitProducts(
    const std::vector<Fft>& accum, int u_num_chunks, int v_num_chunks,
    int chunk_bits, bool allow_paired_ifft, FftContext& ctx) {
  const int d = ctx.d;
  const double inv_d = 1.0 / static_cast<double>(d);
  std::vector<CoeffType> result(d, 0);

  struct ActiveTerm {
    int idx;
    int shift;
  };
  std::vector<ActiveTerm> active;
  active.reserve(u_num_chunks * v_num_chunks);
  for (int a = 0; a < u_num_chunks; ++a) {
    for (int b = 0; b < v_num_chunks; ++b) {
      const int shift = chunk_bits * (a + b);
      if (shift < static_cast<int>(8 * sizeof(CoeffType))) {
        active.push_back({a * v_num_chunks + b, shift});
      }
    }
  }

  const std::complex<double>* __restrict untw = ctx.untwist.data();
  std::complex<double>* __restrict bin = ctx.backward_in.data();
  std::complex<double>* __restrict bout = ctx.backward_out.data();

  size_t p = 0;
  if (allow_paired_ifft) {
    for (; p + 2 <= active.size(); p += 2) {
      const Fft& acc0 = accum[active[p].idx];
      const Fft& acc1 = accum[active[p + 1].idx];
      const int shift0 = active[p].shift;
      const int shift1 = active[p + 1].shift;
      for (int j = 0; j < d; ++j) {
        bin[j] = std::complex<double>(
            acc0[j].real() - acc1[j].imag(), acc0[j].imag() + acc1[j].real());
      }
      RETURN_IF_ERROR(fft::Ifft(absl::MakeConstSpan(bin, d),
                                absl::MakeSpan(bout, d),
                                *ctx.negacyclic_plan));
      for (int j = 0; j < d; ++j) {
        const double br = bout[j].real();
        const double bi = bout[j].imag();
        const double ur = untw[j].real();
        const double ui = untw[j].imag();
        const double re0 = br * ur - bi * ui;
        const double re1 = br * ui + bi * ur;
        const int64_t v0 = static_cast<int64_t>(std::nearbyint(re0 * inv_d));
        const int64_t v1 = static_cast<int64_t>(std::nearbyint(re1 * inv_d));
        const CoeffType c0 = static_cast<CoeffType>(static_cast<uint64_t>(v0));
        const CoeffType c1 = static_cast<CoeffType>(static_cast<uint64_t>(v1));
        result[j] += (c0 << shift0) + (c1 << shift1);
      }
    }
  }

  for (; p < active.size(); ++p) {
    const Fft& acc0 = accum[active[p].idx];
    const int shift0 = active[p].shift;
    RETURN_IF_ERROR(fft::Ifft(absl::MakeConstSpan(acc0.data(), d),
                              absl::MakeSpan(bout, d), *ctx.negacyclic_plan));
    for (int j = 0; j < d; ++j) {
      const double re0 =
          bout[j].real() * untw[j].real() - bout[j].imag() * untw[j].imag();
      const int64_t v0 = static_cast<int64_t>(std::nearbyint(re0 * inv_d));
      const CoeffType c0 = static_cast<CoeffType>(static_cast<uint64_t>(v0));
      result[j] += (c0 << shift0);
    }
  }

  return Polynomial<CoeffType>::Create(std::move(result));
}

void EnsureScratchSize(std::vector<Fft>& scratch, size_t count, size_t len2) {
  if (scratch.size() < count) {
    scratch.resize(count);
  }
  for (size_t i = 0; i < count; ++i) {
    if (scratch[i].size() != len2) {
      scratch[i].resize(len2);
    }
  }
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
  const CoeffType* __restrict coeffs = poly.Coeffs().data();
  const std::complex<double>* __restrict tw = ctx.twist.data();
  std::complex<double>* __restrict fin = ctx.forward_in.data();
  for (int j = 0; j < d; ++j) {
    const double val = ChunkValue(coeffs[j], 0, bits, signed_width);
    fin[j] = std::complex<double>(val * tw[j].real(), val * tw[j].imag());
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
  const double inv_d = 1.0 / static_cast<double>(d);

  // Each chunk product is an exact integer polynomial (see
  // kFftExactProductBits); the chunks are recombined modulo 2^(8 * sizeof
  // CoeffType) by the shifts.
  std::vector<CoeffType> result(d, 0);
  const std::complex<double>* __restrict tw = ctx.twist.data();
  const std::complex<double>* __restrict untw = ctx.untwist.data();
  const std::complex<double>* __restrict that_fft = that.Fft().data();
  std::complex<double>* __restrict fin = ctx.forward_in.data();
  std::complex<double>* __restrict fout = ctx.forward_out.data();
  std::complex<double>* __restrict bin = ctx.backward_in.data();
  std::complex<double>* __restrict bout = ctx.backward_out.data();

  for (int c = 0; c < num_chunks; ++c) {
    const int shift = c * chunk_bits;
    const int width = std::min(chunk_bits, this_bits - shift);
    const uint64_t mask =
        width >= 64 ? ~uint64_t{0} : (uint64_t{1} << width) - 1;
    for (int j = 0; j < d; ++j) {
      const double val =
          static_cast<double>((static_cast<uint64_t>(coeffs_[j]) >> shift) & mask);
      fin[j] = std::complex<double>(val * tw[j].real(), val * tw[j].imag());
    }
    RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(fin, d),
                             absl::MakeSpan(fout, d), *ctx.negacyclic_plan));
    PointwiseComplexMultiply(fout, that_fft, bin, d, /*accumulate=*/false);
    RETURN_IF_ERROR(fft::Ifft(absl::MakeConstSpan(bin, d),
                              absl::MakeSpan(bout, d), *ctx.negacyclic_plan));
    for (int j = 0; j < d; ++j) {
      const double re =
          bout[j].real() * untw[j].real() - bout[j].imag() * untw[j].imag();
      const int64_t val64 = static_cast<int64_t>(std::nearbyint(re * inv_d));
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
  for (size_t i = 0; i < u.size(); ++i) {
    if (u[i].Len() != d || v[i].Len() != d || d != ctx.d) {
      return absl::InvalidArgumentError(
          "Polynomial lengths or context mismatch.");
    }
  }
  if (ctx.negacyclic_plan == nullptr) {
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
  const int eff_u_bits =
      u_is_signed ? std::min(chunk_bits, u_bits - 1) : std::min(chunk_bits, u_bits);
  const int eff_v_bits = std::min(chunk_bits, v_bits);
  const bool allow_paired =
      (std::log2(u.size()) + std::log2(d) + eff_u_bits + eff_v_bits <= 50.0);

  EnsureScratchSize(ctx.accum_scratch, u_num_chunks * v_num_chunks, d);
  EnsureScratchSize(ctx.u_ffts_scratch, 2 * u_num_chunks, d);
  EnsureScratchSize(ctx.v_ffts_scratch, 2 * v_num_chunks, d);

  size_t i = 0;
  if (allow_paired) {
    for (; i + 2 <= u.size(); i += 2) {
      RETURN_IF_ERROR(TransformPolysDigits(
          &u[i], 2, u_num_chunks, chunk_bits, u_bits, u_is_signed,
          /*allow_paired=*/true, ctx, ctx.u_ffts_scratch.data()));
      RETURN_IF_ERROR(TransformPolysDigits(
          &v[i], 2, v_num_chunks, chunk_bits, v_bits, /*is_signed=*/false,
          /*allow_paired=*/true, ctx, ctx.v_ffts_scratch.data()));

      AccumulateDigitProducts<CoeffType>(&ctx.u_ffts_scratch[0], u_num_chunks,
                                         &ctx.v_ffts_scratch[0], v_num_chunks,
                                         chunk_bits, /*first_term=*/(i == 0),
                                         ctx.accum_scratch);
      AccumulateDigitProducts<CoeffType>(
          &ctx.u_ffts_scratch[u_num_chunks], u_num_chunks,
          &ctx.v_ffts_scratch[v_num_chunks], v_num_chunks, chunk_bits,
          /*first_term=*/false, ctx.accum_scratch);
    }
  }

  for (; i < u.size(); ++i) {
    RETURN_IF_ERROR(TransformPolysDigits(&u[i], 1, u_num_chunks, chunk_bits,
                                         u_bits, u_is_signed, allow_paired, ctx,
                                         ctx.u_ffts_scratch.data()));
    RETURN_IF_ERROR(TransformPolysDigits(&v[i], 1, v_num_chunks, chunk_bits,
                                         v_bits, /*is_signed=*/false,
                                         allow_paired, ctx,
                                         ctx.v_ffts_scratch.data()));

    AccumulateDigitProducts<CoeffType>(&ctx.u_ffts_scratch[0], u_num_chunks,
                                       &ctx.v_ffts_scratch[0], v_num_chunks,
                                       chunk_bits, /*first_term=*/(i == 0),
                                       ctx.accum_scratch);
  }

  return RecombineDigitProducts<CoeffType>(
      ctx.accum_scratch, u_num_chunks, v_num_chunks, chunk_bits, allow_paired,
      ctx);
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
  if (ctx.negacyclic_plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  if (u_bits <= 0 || chunk_bits <= 0 || chunk_bits > 63) {
    return absl::InvalidArgumentError("Invalid bits or chunk_bits.");
  }

  const int num_chunks = NumChunks<CoeffType>(u_bits, chunk_bits);
  ChunkedFft result;
  result.chunk_bits = chunk_bits;
  result.max_chunk_magnitude_bits =
      u_is_signed ? std::min(chunk_bits, u_bits - 1)
                  : std::min(chunk_bits, u_bits);
  result.ffts.assign(u.size(), std::vector<Fft>(num_chunks, Fft(ctx.d)));
  const size_t total = u.size() * static_cast<size_t>(num_chunks);
  size_t idx = 0;
  for (; idx + 2 <= total; idx += 2) {
    const size_t i0 = idx / num_chunks;
    const int c0 = idx % num_chunks;
    const size_t i1 = (idx + 1) / num_chunks;
    const int c1 = (idx + 1) % num_chunks;
    RETURN_IF_ERROR(TwoDigitsFft(u[i0].coeffs_, c0, u[i1].coeffs_, c1,
                                 chunk_bits, num_chunks, u_bits, u_is_signed,
                                 ctx, result.ffts[i0][c0],
                                 result.ffts[i1][c1]));
  }
  if (idx < total) {
    const size_t i0 = idx / num_chunks;
    const int c0 = idx % num_chunks;
    RETURN_IF_ERROR(DigitFft(u[i0].coeffs_, c0, chunk_bits, num_chunks, u_bits,
                             u_is_signed, ctx, result.ffts[i0][c0]));
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
  if (ctx.negacyclic_plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  const int chunk_bits = u.chunk_bits;
  const int d = ctx.d;
  const size_t len_d = d;
  const int u_num_chunks = u.ffts[0].size();
  const int v_num_chunks = v.ffts[0].size();
  for (size_t i = 0; i < u.ffts.size(); ++i) {
    if (u.ffts[i].size() != static_cast<size_t>(u_num_chunks) ||
        v.ffts[i].size() != static_cast<size_t>(v_num_chunks)) {
      return absl::InvalidArgumentError("Inconsistent number of digits.");
    }
    for (const Fft& fft : u.ffts[i]) {
      if (fft.size() != len_d) {
        return absl::InvalidArgumentError("FFT length and context mismatch.");
      }
    }
    for (const Fft& fft : v.ffts[i]) {
      if (fft.size() != len_d) {
        return absl::InvalidArgumentError("FFT length and context mismatch.");
      }
    }
  }
  if (std::log2(u.ffts.size()) + std::log2(d) + 2.0 * chunk_bits > 53.0) {
    return absl::InvalidArgumentError(
        "Potential precision overflow in accumulation.");
  }

  const int u_mag = u.max_chunk_magnitude_bits > 0 ? u.max_chunk_magnitude_bits
                                                   : chunk_bits;
  const int v_mag = v.max_chunk_magnitude_bits > 0 ? v.max_chunk_magnitude_bits
                                                   : chunk_bits;
  const bool allow_paired =
      (std::log2(u.ffts.size()) + std::log2(d) + u_mag + v_mag <= 50.0);

  EnsureScratchSize(ctx.accum_scratch, u_num_chunks * v_num_chunks, len_d);
  for (size_t i = 0; i < u.ffts.size(); ++i) {
    AccumulateDigitProducts<CoeffType>(
        u.ffts[i].data(), u_num_chunks, v.ffts[i].data(), v_num_chunks,
        chunk_bits, /*first_term=*/(i == 0), ctx.accum_scratch);
  }
  return RecombineDigitProducts<CoeffType>(
      ctx.accum_scratch, u_num_chunks, v_num_chunks, chunk_bits, allow_paired,
      ctx);
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
