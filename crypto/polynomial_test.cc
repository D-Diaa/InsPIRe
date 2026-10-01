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

#include "crypto/polynomial.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "crypto/polynomial_fft.h"
#include "crypto/proto/rlwe.pb.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::rlwe::testing::StatusIs;

TEST(PolynomialTest, CreateFailsIfCoeffsLengthIsNotPowerOfTwo) {
  EXPECT_THAT(Polynomial<uint32_t>::Create({1, 2, 3}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a non-zero power of two")));
}

TEST(PolynomialTest, CreateFailsIfCoeffsIsEmpty) {
  EXPECT_THAT(Polynomial<uint32_t>::Create({}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a non-zero power of two")));
}

TEST(PolynomialTest, CreateZeroFailsIfLengthIsNotPowerOfTwo) {
  EXPECT_THAT(Polynomial<uint32_t>::CreateZero(3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a non-zero power of two")));
}

TEST(PolynomialTest, CreateZeroFailsIfLengthIsZero) {
  EXPECT_THAT(Polynomial<uint32_t>::CreateZero(0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a non-zero power of two")));
}

TEST(PolynomialTest, CreateZero) {
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::CreateZero(4));
  EXPECT_THAT(p.Coeffs(), ElementsAre(0, 0, 0, 0));
}

TEST(PolynomialTest, Add) {
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create({2, 3, 4, 5}));
  ASSERT_OK_AND_ASSIGN(auto p3, p1.Add(p2));
  EXPECT_THAT(p3.Coeffs(), ElementsAre(3, 5, 7, 9));
}

TEST(PolynomialTest, AddOverflowsModulo) {
  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  ASSERT_OK_AND_ASSIGN(auto p1,
                       Polynomial<uint32_t>::Create({max_val, 2, 0, 0}));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create({2, 0, 0, 0}));
  ASSERT_OK_AND_ASSIGN(auto p3, p1.Add(p2));
  EXPECT_THAT(p3.Coeffs(), ElementsAre(1, 2, 0, 0));
}

TEST(PolynomialTest, AddFailsIfLengthsMismatch) {
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  std::vector<uint32_t> coeffs2(8, 0);
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(coeffs2));
  EXPECT_THAT(p1.Add(p2), StatusIs(absl::StatusCode::kInvalidArgument,
                                   HasSubstr("Polynomial lengths must match")));
}

TEST(PolynomialTest, Negate) {
  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  ASSERT_OK_AND_ASSIGN(auto p1,
                       Polynomial<uint32_t>::Create({1, 5, 0, max_val}));
  ASSERT_OK_AND_ASSIGN(auto p2, p1.Negate());
  EXPECT_THAT(p2.Coeffs(), ElementsAre(max_val, max_val - 4, 0, 1));
}

TEST(PolynomialTest, Automorph) {
  // Automorph of P(X) = a0 + a1 X + a2 X^2 + a3 X^3 with power 3
  // P(X^3) = a0 + a1 X^3 + a2 X^6 + a3 X^9
  // Modulo X^4 + 1:
  // X^6 = X^4 * X^2 = -X^2
  // X^9 = (X^4)^2 * X = X
  // Result: a0 + a3 X - a2 X^2 + a1 X^3
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto p_auto, p.Automorph(3));
  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  EXPECT_THAT(p_auto.Coeffs(), ElementsAre(1, 4, max_val - 2, 2));
}

TEST(PolynomialTest, AutomorphFailsIfPowerIsInvalid) {
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  EXPECT_THAT(
      p.Automorph(-1),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Automorphism power must be a non-negative odd integer")));
  EXPECT_THAT(
      p.Automorph(2),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Automorphism power must be a non-negative odd integer")));
  EXPECT_THAT(
      p.Automorph(8),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Automorphism power must be a non-negative odd integer")));
  EXPECT_THAT(
      p.Automorph(9),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Automorphism power must be a non-negative odd integer")));
}

TEST(PolynomialTest, RightShift) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  ASSERT_OK_AND_ASSIGN(auto p_shifted, p.RightShift(4));
  EXPECT_THAT(p_shifted.Coeffs(), ElementsAre(0x01234567, 0x0FFFFFFF, 0, 0));

  ASSERT_OK_AND_ASSIGN(auto p_shifted_more, p.RightShift(32));
  EXPECT_THAT(p_shifted_more.Coeffs(), ElementsAre(0, 0, 0, 0));

  ASSERT_OK_AND_ASSIGN(auto p_shifted_even_more, p.RightShift(40));
  EXPECT_THAT(p_shifted_even_more.Coeffs(), ElementsAre(0, 0, 0, 0));
}

TEST(PolynomialTest, RightShiftFailsIfAmountIsNegative) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  EXPECT_THAT(p.RightShift(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shift amount cannot be negative")));
}

TEST(PolynomialTest, LowBits) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  ASSERT_OK_AND_ASSIGN(auto p_low, p.LowBits(4));
  EXPECT_THAT(p_low.Coeffs(), ElementsAre(0x8, 0xF, 0, 1));

  ASSERT_OK_AND_ASSIGN(auto p_low_more, p.LowBits(32));
  EXPECT_THAT(p_low_more.Coeffs(), ElementsAre(0x12345678, 0xFFFFFFFF, 0, 1));

  ASSERT_OK_AND_ASSIGN(auto p_low_even_more, p.LowBits(40));
  EXPECT_THAT(p_low_even_more.Coeffs(),
              ElementsAre(0x12345678, 0xFFFFFFFF, 0, 1));
}

TEST(PolynomialTest, LowBitsFailsIfAmountIsNegative) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  EXPECT_THAT(p.LowBits(-1), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("Amount cannot be negative")));
}

TEST(PolynomialTest, GadgetInv) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));

  ASSERT_OK_AND_ASSIGN(auto gadget, p.GadgetInv(0, 8, 4));
  ASSERT_EQ(gadget.size(), 4);
  EXPECT_THAT(gadget[0].Coeffs(), ElementsAre(0x78, 0xFF, 0, 1));
  EXPECT_THAT(gadget[1].Coeffs(), ElementsAre(0x56, 0xFF, 0, 0));
  EXPECT_THAT(gadget[2].Coeffs(), ElementsAre(0x34, 0xFF, 0, 0));
  EXPECT_THAT(gadget[3].Coeffs(), ElementsAre(0x12, 0xFF, 0, 0));

  ASSERT_OK_AND_ASSIGN(auto gadget2, p.GadgetInv(0, 16, 3));
  ASSERT_EQ(gadget2.size(), 3);
  EXPECT_THAT(gadget2[0].Coeffs(), ElementsAre(0x5678, 0xFFFF, 0, 1));
  EXPECT_THAT(gadget2[1].Coeffs(), ElementsAre(0x1234, 0xFFFF, 0, 0));
  EXPECT_THAT(gadget2[2].Coeffs(), ElementsAre(0, 0, 0, 0));
}

TEST(PolynomialTest, ApproximateGadgetInv) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1,
                                            0x80000000, 0x7FFFFFFF, 0, 0}));

  ASSERT_OK_AND_ASSIGN(auto gadget1, p.GadgetInv(0, 8, 2));
  ASSERT_EQ(gadget1.size(), 2);
  EXPECT_THAT(gadget1[0].Coeffs(), ElementsAre(0x34, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_THAT(gadget1[1].Coeffs(),
              ElementsAre(0x12, 0, 0, 0, 0x80, 0x80, 0, 0));
}

TEST(PolynomialTest, SignedGadgetInv) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345698, 0xFFFFFFFF, 0, 1}));

  ASSERT_OK_AND_ASSIGN(auto gadget, p.SignedGadgetInv(0, 8, 4));
  ASSERT_EQ(gadget.size(), 4);
  EXPECT_THAT(gadget[0].Coeffs(), ElementsAre(0xFFFFFF98, 0xFFFFFFFF, 0, 1));
  EXPECT_THAT(gadget[1].Coeffs(), ElementsAre(0x57, 0, 0, 0));
  EXPECT_THAT(gadget[2].Coeffs(), ElementsAre(0x34, 0, 0, 0));
  EXPECT_THAT(gadget[3].Coeffs(), ElementsAre(0x12, 0, 0, 0));

  // Verify that summing the gadget components yields back the original
  // polynomial modulo 2^32.
  std::vector<uint32_t> reconstructed(4, 0);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      reconstructed[j] += gadget[i].Coeffs()[j] * (1 << (8 * i));
    }
  }
  EXPECT_THAT(reconstructed, ElementsAre(0x12345698, 0xFFFFFFFF, 0, 1));
}

TEST(PolynomialTest, ApproximateSignedGadgetInv) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0x129B9856, 0, 1}));

  ASSERT_OK_AND_ASSIGN(auto gadget, p.SignedGadgetInv(0, 8, 2));
  ASSERT_EQ(gadget.size(), 2);
  EXPECT_THAT(gadget[0].Coeffs(), ElementsAre(0x34, 0xFFFFFF9C, 0, 0));
  EXPECT_THAT(gadget[1].Coeffs(), ElementsAre(0x12, 0x13, 0, 0));
}

TEST(PolynomialTest, SignedGadgetInvFailsIfParamsInvalid) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  EXPECT_THAT(p.SignedGadgetInv(0, 0, 4),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.SignedGadgetInv(0, -1, 4),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.SignedGadgetInv(0, 7, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.SignedGadgetInv(0, 7, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
}

TEST(PolynomialTest, GadgetInvFailsIfParamsInvalid) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint32_t>::Create({0x12345678, 0xFFFFFFFF, 0, 1}));
  EXPECT_THAT(p.GadgetInv(0, 0, 4),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.GadgetInv(0, -1, 4),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.GadgetInv(0, 7, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
  EXPECT_THAT(p.GadgetInv(0, 7, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("log_digit and num_digits must be positive")));
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> NaiveMult(
    const Polynomial<CoeffType>& p1, const Polynomial<CoeffType>& p2) {
  int n = p1.Len();
  std::vector<CoeffType> result(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int index = (i + j) % n;
      int sign = ((i + j) >= n) ? -1 : 1;
      if (sign == 1) {
        result[index] += p1.Coeffs()[i] * p2.Coeffs()[j];
      } else {
        result[index] -= p1.Coeffs()[i] * p2.Coeffs()[j];
      }
    }
  }
  return Polynomial<CoeffType>::Create(std::move(result));
}

TEST(PolynomialTest, Mult) {
  // Test degree 4 multiplication
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(2));
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create({5, 6, 7, 8}));
  ASSERT_OK_AND_ASSIGN(auto p3, p1.Mult(p2, ctx));
  EXPECT_THAT(p3.Coeffs(), testing::ElementsAreArray({
                               4294967240U,  // -56 mod 2^32
                               4294967260U,  // -36 mod 2^32
                               2U,           // 2   mod 2^32
                               60U           // 60  mod 2^32
                           }));
}

TEST(PolynomialTest, MultFailsIfLengthsMismatch) {
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  std::vector<uint32_t> coeffs2(8, 0);
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(coeffs2));
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(2));
  EXPECT_THAT(p1.Mult(p2, ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Polynomial lengths must match")));
}

TEST(PolynomialTest, MultLargeDegree) {
  int degree = 1024;
  std::vector<uint64_t> coeffs1(degree);
  std::vector<uint64_t> coeffs2(degree);
  // Generate some deterministic coefficients
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = i * 1000000000ULL;
    coeffs2[i] = (degree - i) * 1000000000ULL;
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto expected, NaiveMult(p1, p2));
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(10));
  ASSERT_OK_AND_ASSIGN(auto p3, p1.Mult(p2, ctx));
  EXPECT_EQ(p3.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, ToProtoAndCreateFromProto) {
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto proto, p.ToProto(3));
  ASSERT_OK_AND_ASSIGN(auto deserialized,
                       Polynomial<uint32_t>::CreateFromProto(proto, 4, 3));
  EXPECT_EQ(deserialized.Coeffs(), p.Coeffs());
}

TEST(PolynomialTest, ToProtoAndCreateFromProtoLogModulusLarge) {
  ASSERT_OK_AND_ASSIGN(
      auto p, Polynomial<uint64_t>::Create(
                  {0x1122334455667788ULL, 0x99AABBCCDDEEFF11ULL, 0, 1}));
  ASSERT_OK_AND_ASSIGN(auto proto, p.ToProto(60));
  ASSERT_OK_AND_ASSIGN(auto deserialized,
                       Polynomial<uint64_t>::CreateFromProto(proto, 4, 60));

  std::vector<uint64_t> expected = {0x1122334455667788ULL & ((1ULL << 60) - 1),
                                    0x99AABBCCDDEEFF11ULL & ((1ULL << 60) - 1),
                                    0, 1};
  EXPECT_EQ(deserialized.Coeffs(), expected);
}

TEST(PolynomialTest, CreateFromProtoFailsWithInvalidInputs) {
  proto::Polynomial empty_proto;
  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(empty_proto, 4, 3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Empty serialized polynomial")));

  proto::Polynomial non_empty_proto;
  non_empty_proto.set_encoded_coeffs("somebytes");

  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(non_empty_proto, 0, 3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid length")));

  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(non_empty_proto, 3, 3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid length")));

  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(non_empty_proto, 4, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid log_modulus")));

  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(non_empty_proto, 4, 100),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid log_modulus")));

  EXPECT_THAT(Polynomial<uint32_t>::CreateFromProto(non_empty_proto, 4, 3),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected serialized size")));
}

TEST(PolynomialTest, ToProtoFailsWithInvalidLogModulus) {
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  EXPECT_THAT(p.ToProto(0), StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Invalid log_modulus")));
  EXPECT_THAT(p.ToProto(33), StatusIs(absl::StatusCode::kInvalidArgument,
                                      HasSubstr("Invalid log_modulus")));
}

TEST(PolynomialTest, MultFftMatchesMultDegree4Uint32) {
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(2));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create({5, 6, 7, 8}));

  ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));
  ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx));
  EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, MultFftMatchesMultDegree16Uint64) {
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(4));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(4));
  std::vector<uint64_t> c1(16), c2(16);
  for (int i = 0; i < 16; ++i) {
    c1[i] = 0x123456789ABCDEFULL * (i + 1);
    c2[i] = 0xFEDCBA987654321ULL * (16 - i);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));

  ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));
  ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx));
  EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, MultFftMatchesMultDegree1024Uint64) {
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(10));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(10));
  std::vector<uint64_t> c1(1024), c2(1024);
  for (int i = 0; i < 1024; ++i) {
    c1[i] = i * 1000000000ULL;
    c2[i] = (1024 - i) * 1000000000ULL;
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));

  ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));
  ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx));
  EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, MultFftMatchesMultRandom) {
  std::mt19937_64 gen(12345);
  std::uniform_int_distribution<uint64_t> dist(
      0, std::numeric_limits<uint64_t>::max());

  int degrees[] = {128, 256, 512, 1024, 2048};
  for (int degree : degrees) {
    // Test uint32_t
    {
      int log_degree = 0;
      while ((1 << log_degree) < degree) log_degree++;
      ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(log_degree));
      ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_degree));

      std::vector<uint32_t> c1(degree), c2(degree);
      for (int i = 0; i < degree; ++i) {
        c1[i] = static_cast<uint32_t>(dist(gen));
        c2[i] = static_cast<uint32_t>(dist(gen));
      }
      ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create(c1));
      ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(c2));

      ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));
      ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx));
      EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
    }

    // Test uint64_t
    {
      int log_degree = 0;
      while ((1 << log_degree) < degree) log_degree++;
      ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(log_degree));
      ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_degree));

      std::vector<uint64_t> c1(degree), c2(degree);
      for (int i = 0; i < degree; ++i) {
        c1[i] = dist(gen);
        c2[i] = dist(gen);
      }
      ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
      ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));

      ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));
      ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx));
      EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
    }
  }
}

TEST(PolynomialTest, MultFftWithExplicitBitsRandom) {
  std::mt19937_64 gen(12345);

  int degrees[] = {128, 256, 512, 1024, 2048};
  int chunk_bit_options[] = {12, 16, 20};
  for (int degree : degrees) {
    int log_degree = 0;
    while ((1 << log_degree) < degree) log_degree++;
    ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(log_degree));
    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_degree));

    // Pick random bit lengths between 1 and maximum bit length
    int this_bits = (gen() % 31) + 1;
    int that_bits = (gen() % 31) + 1;

    uint32_t this_mask =
        (this_bits == 32) ? 0xFFFFFFFF : ((1U << this_bits) - 1);
    uint32_t that_mask =
        (that_bits == 32) ? 0xFFFFFFFF : ((1U << that_bits) - 1);

    std::vector<uint32_t> c1(degree), c2(degree);
    for (int i = 0; i < degree; ++i) {
      c1[i] = static_cast<uint32_t>(gen()) & this_mask;
      c2[i] = static_cast<uint32_t>(gen()) & that_mask;
    }

    ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create(c1));
    ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(c2));

    ASSERT_OK_AND_ASSIGN(auto expected, p1.Mult(p2, ctx));

    for (int chunk_bits : chunk_bit_options) {
      ASSERT_OK_AND_ASSIGN(auto actual, p1.MultFft(p2, *fft_ctx, this_bits,
                                                   that_bits, chunk_bits));
      EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
    }
  }
}

TEST(PolynomialTest, InnerProductFftTest) {
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(2));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));

  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create({5, 6, 7, 8}));
  ASSERT_OK_AND_ASSIGN(auto p3, Polynomial<uint32_t>::Create({2, 3, 4, 5}));
  ASSERT_OK_AND_ASSIGN(auto p4, Polynomial<uint32_t>::Create({1, 1, 1, 1}));

  std::vector<Polynomial<uint32_t>> v1 = {p1, p3};
  std::vector<Polynomial<uint32_t>> v2 = {p2, p4};

  ASSERT_OK_AND_ASSIGN(auto m1, p1.Mult(p2, ctx));
  ASSERT_OK_AND_ASSIGN(auto m2, p3.Mult(p4, ctx));
  ASSERT_OK_AND_ASSIGN(auto expected, m1.Add(m2));

  ASSERT_OK_AND_ASSIGN(auto actual,
                       Polynomial<uint32_t>::InnerProductFft(v1, v2, *fft_ctx));
  EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, InnerProductFftTestRandom) {
  std::mt19937_64 gen(12345);
  int degrees[] = {128, 512, 2048};
  int vector_sizes[] = {1, 5, 10};
  int default_chunk_options[] = {12, 16, 20};

  for (int degree : degrees) {
    int log_degree = 0;
    while ((1 << log_degree) < degree) log_degree++;
    ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(log_degree));
    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_degree));

    for (int k : vector_sizes) {
      int this_bits = (gen() % 31) + 1;
      int that_bits = (gen() % 31) + 1;
      uint32_t this_mask =
          (this_bits == 32) ? 0xFFFFFFFF : ((1U << this_bits) - 1);
      uint32_t that_mask =
          (that_bits == 32) ? 0xFFFFFFFF : ((1U << that_bits) - 1);

      std::vector<Polynomial<uint32_t>> v1;
      std::vector<Polynomial<uint32_t>> v2;
      ASSERT_OK_AND_ASSIGN(auto expected,
                           Polynomial<uint32_t>::CreateZero(degree));

      for (int i = 0; i < k; ++i) {
        std::vector<uint32_t> c1(degree), c2(degree);
        for (int j = 0; j < degree; ++j) {
          c1[j] = static_cast<uint32_t>(gen()) & this_mask;
          c2[j] = static_cast<uint32_t>(gen()) & that_mask;
        }
        ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create(c1));
        ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(c2));

        ASSERT_OK_AND_ASSIGN(auto prod, p1.Mult(p2, ctx));
        ASSERT_OK_AND_ASSIGN(expected, expected.Add(prod));

        v1.push_back(std::move(p1));
        v2.push_back(std::move(p2));
      }

      std::vector<int> chunk_bit_options(std::begin(default_chunk_options),
                                         std::end(default_chunk_options));
      // Compute the extreme limit for chunk size that still won't overflow
      int boundary_chunk =
          static_cast<int>((53.0 - std::log2(k) - std::log2(degree)) / 2.0);
      if (boundary_chunk > 0) {
        chunk_bit_options.push_back(boundary_chunk);
      }

      for (int chunk_bits : chunk_bit_options) {
        if (std::log2(k) + std::log2(degree) + 2.0 * chunk_bits <= 53.0) {
          ASSERT_OK_AND_ASSIGN(
              auto actual,
              Polynomial<uint32_t>::InnerProductFft(v1, v2, *fft_ctx, this_bits,
                                                    that_bits, chunk_bits));
          EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
        }
      }
    }
  }
}

// Random `num_polys` pairs (u signed `u_bits`-bit digits, v unsigned `v_bits`
// bits) of degree 2^log_degree: the inner product through ToChunkedFft must
// equal InnerProductFft on the polynomials coefficient for coefficient, and
// both must equal the exact NTT product.
template <typename CoeffType>
void ExpectChunkedFftMatchesDirect(int log_degree, int num_polys, int u_bits,
                                   int v_bits, int chunk_bits,
                                   std::mt19937_64& gen) {
  const int degree = 1 << log_degree;
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(log_degree));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_degree));

  std::vector<Polynomial<CoeffType>> u;
  std::vector<Polynomial<CoeffType>> v;
  ASSERT_OK_AND_ASSIGN(auto expected,
                       Polynomial<CoeffType>::CreateZero(degree));
  for (int i = 0; i < num_polys; ++i) {
    std::vector<CoeffType> cu(degree), cv(degree);
    for (int j = 0; j < degree; ++j) {
      // Signed digit in [-2^(u_bits-1), 2^(u_bits-1)), two's complement.
      const int64_t digit = static_cast<int64_t>(gen() >> (64 - u_bits)) -
                            (int64_t{1} << (u_bits - 1));
      cu[j] = static_cast<CoeffType>(digit);
      cv[j] = static_cast<CoeffType>(gen() >> (64 - v_bits));
    }
    ASSERT_OK_AND_ASSIGN(auto pu, Polynomial<CoeffType>::Create(cu));
    ASSERT_OK_AND_ASSIGN(auto pv, Polynomial<CoeffType>::Create(cv));
    ASSERT_OK_AND_ASSIGN(auto prod, pu.Mult(pv, ctx));
    ASSERT_OK_AND_ASSIGN(expected, expected.Add(prod));
    u.push_back(std::move(pu));
    v.push_back(std::move(pv));
  }

  ASSERT_OK_AND_ASSIGN(auto direct, Polynomial<CoeffType>::InnerProductFft(
                                        u, v, *fft_ctx, u_bits, v_bits,
                                        chunk_bits, /*u_is_signed=*/true));
  ASSERT_OK_AND_ASSIGN(ChunkedFft u_fft, Polynomial<CoeffType>::ToChunkedFft(
                                             u, *fft_ctx, u_bits, chunk_bits,
                                             /*u_is_signed=*/true));
  ASSERT_OK_AND_ASSIGN(ChunkedFft v_fft, Polynomial<CoeffType>::ToChunkedFft(
                                             v, *fft_ctx, v_bits, chunk_bits,
                                             /*u_is_signed=*/false));
  EXPECT_EQ(u_fft.ffts[0].size(), (u_bits + chunk_bits - 1) / chunk_bits);
  EXPECT_EQ(v_fft.ffts[0].size(), (v_bits + chunk_bits - 1) / chunk_bits);
  ASSERT_OK_AND_ASSIGN(auto chunked, Polynomial<CoeffType>::InnerProductFft(
                                         u_fft, v_fft, *fft_ctx));
  EXPECT_EQ(chunked.Coeffs(), direct.Coeffs());
  EXPECT_EQ(chunked.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, InnerProductFftOnChunkedFftMatchesDirect) {
  std::mt19937_64 gen(2026);
  // The packing shape of the Tiptoe workload: d = 2048, two signed 19-bit
  // gadget digit polynomials against 52-bit key polynomials, 20-bit chunks.
  ExpectChunkedFftMatchesDirect<uint64_t>(/*log_degree=*/11, /*num_polys=*/2,
                                          /*u_bits=*/19, /*v_bits=*/52,
                                          /*chunk_bits=*/20, gen);
  ExpectChunkedFftMatchesDirect<uint32_t>(/*log_degree=*/6, /*num_polys=*/14,
                                          /*u_bits=*/4, /*v_bits=*/27,
                                          /*chunk_bits=*/20, gen);
}

TEST(PolynomialTest, InnerProductFftOnChunkedFftRejectsMismatch) {
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint32_t>::Create({1, 2, 3, 4}));
  ASSERT_OK_AND_ASSIGN(auto fft20, Polynomial<uint32_t>::ToChunkedFft(
                                       {p}, *fft_ctx, 32, 20, false));
  ASSERT_OK_AND_ASSIGN(auto fft16, Polynomial<uint32_t>::ToChunkedFft(
                                       {p}, *fft_ctx, 32, 16, false));
  EXPECT_THAT(Polynomial<uint32_t>::InnerProductFft(fft20, fft16, *fft_ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("chunk_bits")));
  ASSERT_OK_AND_ASSIGN(auto two, Polynomial<uint32_t>::ToChunkedFft(
                                     {p, p}, *fft_ctx, 32, 20, false));
  EXPECT_THAT(Polynomial<uint32_t>::InnerProductFft(fft20, two, *fft_ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same size")));
}

// Ternary key {0, 1, q - 1} as a Polynomial, with -1 stored as q - 1 (q = 0
// denotes 2^64). `pattern` selects a random key or one of the extremes.
enum class KeyPattern { kRandom, kAllMinusOne, kAlternating };

Polynomial<uint64_t> TernaryKey(int d, uint64_t q, KeyPattern pattern,
                                std::mt19937_64& gen) {
  std::vector<uint64_t> coeffs(d);
  for (int i = 0; i < d; ++i) {
    int t;
    switch (pattern) {
      case KeyPattern::kRandom:
        t = gen() % 3;
        break;
      case KeyPattern::kAllMinusOne:
        t = 2;
        break;
      case KeyPattern::kAlternating:
        t = (i % 2 == 0) ? 1 : 2;
        break;
    }
    coeffs[i] = (t == 2) ? q - 1 : t;
  }
  return Polynomial<uint64_t>::Create(std::move(coeffs)).value();
}

// The negacyclic FFT product against a cached ternary key must agree bit for
// bit (mod 2^64) with the NTT path on random and extreme operands at the
// workload parameters (d = 2048, log q = 52) and at full 64-bit width,
// including the widest single chunk (32 bits at d = 2048).
TEST(PolynomialTest, MultFftWithFftPolynomialMatchesNttOnRandomAndExtremes) {
  const int log_d = 11;
  const int d = 1 << log_d;
  std::mt19937_64 gen(2024);
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::CreateForTernary(log_d));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_d));

  struct Case {
    uint64_t q;
    int a_bits;
    bool a_extreme;  // all coefficients equal to 2^a_bits - 1
    KeyPattern key;
  };
  const Case cases[] = {
      {uint64_t{1} << 52, 52, false, KeyPattern::kRandom},
      {uint64_t{1} << 52, 52, true, KeyPattern::kAllMinusOne},
      {uint64_t{1} << 52, 52, true, KeyPattern::kAlternating},
      {uint64_t{1} << 52, 29, false, KeyPattern::kRandom},
      {uint64_t{1} << 52, 29, true, KeyPattern::kAllMinusOne},
      {uint64_t{1} << 52, 32, true, KeyPattern::kAllMinusOne},
      {uint64_t{1} << 52, 32, true, KeyPattern::kAlternating},
      {0, 64, false, KeyPattern::kRandom},
      {0, 64, true, KeyPattern::kAllMinusOne},
      {0, 64, true, KeyPattern::kAlternating},
  };
  for (const Case& c : cases) {
    const uint64_t a_mask =
        c.a_bits == 64 ? ~uint64_t{0} : (uint64_t{1} << c.a_bits) - 1;
    std::vector<uint64_t> a_coeffs(d);
    for (int i = 0; i < d; ++i) {
      a_coeffs[i] = c.a_extreme ? a_mask : (gen() & a_mask);
    }
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<uint64_t>::Create(a_coeffs));
    Polynomial<uint64_t> s = TernaryKey(d, c.q, c.key, gen);

    ASSERT_OK_AND_ASSIGN(auto s_ntt, s.ToNtt(ctx, /*is_ternary=*/true));
    ASSERT_OK_AND_ASSIGN(auto expected, a.Mult(s_ntt, ctx));

    ASSERT_OK_AND_ASSIGN(auto s_fft, FftPolynomial::Create(s, *fft_ctx,
                                                           /*bits=*/2,
                                                           /*is_signed=*/true));
    EXPECT_EQ(s_fft.MagnitudeBits(), 1);
    ASSERT_OK_AND_ASSIGN(auto actual, a.MultFft(s_fft, *fft_ctx, c.a_bits));
    EXPECT_EQ(actual.Coeffs(), expected.Coeffs())
        << "q=" << c.q << " a_bits=" << c.a_bits;
    // The full width must also agree when the high bits are zero.
    ASSERT_OK_AND_ASSIGN(auto actual_full, a.MultFft(s_fft, *fft_ctx));
    EXPECT_EQ(actual_full.Coeffs(), expected.Coeffs())
        << "q=" << c.q << " a_bits=" << c.a_bits;
  }
}

TEST(PolynomialTest, MultFftWithFftPolynomialMatchesNttUint32) {
  const int log_d = 10;
  const int d = 1 << log_d;
  std::mt19937_64 gen(77);
  ASSERT_OK_AND_ASSIGN(auto ctx, Context::CreateForTernary(log_d));
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(log_d));

  std::vector<uint32_t> a_coeffs(d), s_coeffs(d);
  for (int i = 0; i < d; ++i) {
    a_coeffs[i] = static_cast<uint32_t>(gen());
    const int t = gen() % 3;
    s_coeffs[i] = (t == 2) ? ~uint32_t{0} : t;
  }
  ASSERT_OK_AND_ASSIGN(auto a, Polynomial<uint32_t>::Create(a_coeffs));
  ASSERT_OK_AND_ASSIGN(auto s, Polynomial<uint32_t>::Create(s_coeffs));

  ASSERT_OK_AND_ASSIGN(auto s_ntt, s.ToNtt(ctx, /*is_ternary=*/true));
  ASSERT_OK_AND_ASSIGN(auto expected, a.Mult(s_ntt, ctx));
  ASSERT_OK_AND_ASSIGN(auto s_fft, FftPolynomial::Create(s, *fft_ctx,
                                                         /*bits=*/2,
                                                         /*is_signed=*/true));
  ASSERT_OK_AND_ASSIGN(auto actual, a.MultFft(s_fft, *fft_ctx));
  EXPECT_EQ(actual.Coeffs(), expected.Coeffs());
}

TEST(PolynomialTest, MultFftWithFftPolynomialRejectsInvalidArguments) {
  ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(11));
  ASSERT_OK_AND_ASSIGN(auto other_ctx, FftContext::Create(10));
  ASSERT_OK_AND_ASSIGN(auto p, Polynomial<uint64_t>::CreateZero(2048));
  ASSERT_OK_AND_ASSIGN(auto short_p, Polynomial<uint64_t>::CreateZero(1024));

  EXPECT_THAT(FftPolynomial::Create(short_p, *fft_ctx, 2, true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("context mismatch")));
  EXPECT_THAT(FftPolynomial::Create(p, *fft_ctx, 0, false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid bits")));

  ASSERT_OK_AND_ASSIGN(auto key, FftPolynomial::Create(p, *fft_ctx, 2, true));
  EXPECT_THAT(short_p.MultFft(key, *other_ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("context mismatch")));
  EXPECT_THAT(p.MultFft(key, *fft_ctx, 65),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid this_bits")));

  // A 40-bit unsigned operand leaves no room for an exact chunk at d = 2048:
  // 40 + 11 > kFftExactProductBits.
  ASSERT_OK_AND_ASSIGN(auto wide,
                       FftPolynomial::Create(p, *fft_ctx, 40, false));
  EXPECT_THAT(p.MultFft(wide, *fft_ctx),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("too large")));
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
