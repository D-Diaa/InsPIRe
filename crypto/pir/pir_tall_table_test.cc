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

// Tiptoe-parameter (d=2048, q=2^52, q1=2^25, q2=2^18, p=2^16, t=1) tests of
// color tables taller than d: multi-sample selectors, centered databases,
// 42-bit selectors and row-stripe Stage 1 servers.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "shell_encryption/testing/status_testing.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "shell_encryption/prng/chacha_prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using CoeffType = uint64_t;
using Server = PirServer<uint16_t, CoeffType, int32_t>;
using Prep = PreprocessingServer<uint16_t, CoeffType, int32_t>;
constexpr size_t kD = 2048;
const std::string kPackingSeed(32, 'K');

std::string Seed(uint64_t v, char tag) {
  std::string seed(32, tag);
  for (int i = 0; i < 8; ++i) seed[24 + i] = static_cast<char>(v >> (8 * i));
  return seed;
}

std::unique_ptr<::rlwe::SecurePrng> Prng(const std::string& seed) {
  return std::move(::rlwe::ChaChaPrng::Create(seed)).value();
}

class TallTableTest : public ::testing::Test {
 protected:
  // A batch request: one selector per color at `selector_bits` and the shared
  // packing key, from a fixed client secret.
  struct Request {
    std::unique_ptr<PirClient<CoeffType>> client;
    std::vector<std::vector<CoeffType>> queries;
    PirRequest<CoeffType> request;
    ChunkedFft y_vec_h_fft;
    size_t upload_bytes = 0;
  };

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(fft_ctx_, FftContext::Create(11));
  }

  PirParams<CoeffType> Params(int rows, size_t cols, bool centered) const {
    return PirParams<CoeffType>::Create(rlwe_, 65536, pack_gadget_,
                                        eval_gadget_, rows, 1, cols, centered)
        .value();
  }

  static std::vector<std::vector<uint16_t>> RandomTables(
      const std::vector<int>& rows, size_t cols, uint64_t seed) {
    auto prng = Prng(Seed(seed, 'D'));
    std::vector<std::vector<uint16_t>> raw(rows.size());
    for (size_t s = 0; s < rows.size(); ++s) {
      raw[s].resize(rows[s] * cols * kD);
      for (auto& v : raw[s]) v = static_cast<uint16_t>(prng->Rand64().value());
    }
    return raw;
  }

  Request MakeRequest(const std::vector<int>& rows,
                      const std::vector<int>& targets, size_t cols,
                      bool centered, int selector_bits) {
    Request r;
    r.client = PirClient<CoeffType>::Create(Params(kD, cols, centered),
                                            Prng(Seed(0, 'S')),
                                            Prng(Seed(7, 'C')))
                   .value();
    auto key_prng = Prng(kPackingSeed);
    r.request.packing_key = r.client->CreatePackingKey(key_prng.get()).value();
    r.upload_bytes = 4 * (kD * 52 / 8);
    const int shift = 52 - selector_bits;
    r.queries.resize(rows.size());
    const auto groups = BuildSelectorGroupPlans(rows, kD).value();
    for (const auto& grp : groups) {
      auto mask = Prng(Seed(grp.group_index, 'G'));
      std::vector<int> grp_targets;
      for (const auto& sl : grp.slices) grp_targets.push_back(targets[sl.slot]);
      auto sel = r.client->CreateGroupedSelector(grp.slices, grp_targets,
                                                 mask.get())
                     .value();
      for (const auto& sl : grp.slices) {
        auto& q = r.queries[sl.slot];
        q.assign(sel.begin() + sl.offset, sel.begin() + sl.offset + sl.rows);
        for (auto& y : q) {
          y = (((y + (1ULL << (shift - 1))) >> shift) &
               ((1ULL << selector_bits) - 1))
              << shift;
        }
        r.upload_bytes += (sl.rows * selector_bits + 7) / 8;
      }
    }
    r.request.color_queries = &r.queries;
    const std::vector<Polynomial<CoeffType>> y_vec_h(
        r.request.packing_key.begin() + pack_gadget_.num_digits,
        r.request.packing_key.end());
    r.y_vec_h_fft = YVecHFft(rlwe_, y_vec_h, *fft_ctx_).value();
    return r;
  }

  // Seeds and offsets of each color's selector group.
  static std::pair<std::vector<std::string>, std::vector<size_t>> GroupLayout(
      const std::vector<int>& rows) {
    std::vector<std::string> seeds(rows.size());
    std::vector<size_t> offsets(rows.size());
    const auto groups = BuildSelectorGroupPlans(rows, kD).value();
    for (const auto& grp : groups) {
      for (const auto& sl : grp.slices) {
        seeds[sl.slot] = Seed(grp.group_index, 'G');
        offsets[sl.slot] = sl.offset;
      }
    }
    return {seeds, offsets};
  }

  // A color's server, preprocessed as `num_stripes` row stripes: stripe 0
  // packs (returned first), the others only contract their rows.
  std::vector<std::unique_ptr<Server>> StripedServers(
      int rows, size_t cols, bool centered, const std::string& seed,
      size_t offset, const std::vector<uint16_t>& raw, size_t num_stripes) {
    auto params = Params(rows, cols, centered);
    auto prep = Prep::Create(&params, seed).value();
    std::vector<std::unique_ptr<Server>> servers;
    Prep::Stage1Stripe first;
    for (size_t s = 0; s < num_stripes; ++s) {
      const auto [r0, r1] = RowStripe(rows, kD, s, num_stripes);
      if (r0 == r1) continue;
      auto part = prep->PreprocessStripe(raw, r0, r1, offset).value();
      if (s == 0) {
        first = std::move(part);
        servers.push_back(nullptr);
        continue;
      }
      for (size_t k = 0; k < cols; ++k)
        for (size_t u = 0; u < kD; ++u)
          for (size_t j = 0; j < kD; ++j)
            first.hints[k][u][j] += part.hints[k][u][j];
      servers.push_back(
          Server::CreateStage1Only(params, std::move(part.db_matrices), r0)
              .value());
    }
    auto outputs = Prep::PreprocessPacking(rlwe_, pack_gadget_, kPackingSeed,
                                           first.hints)
                       .value();
    std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, int32_t>>>
        per_column(outputs.size());
    for (size_t k = 0; k < outputs.size(); ++k)
      per_column[k].push_back(std::move(outputs[k]));
    servers[0] = Server::Create(params, {std::move(first.db_matrices),
                                         std::move(per_column), {}})
                     .value();
    return servers;
  }

  // Answers `query` with stripe servers (stripe 0 first), summing Stage 1.
  PirResponse<CoeffType> Answer(
      const std::vector<std::unique_ptr<Server>>& servers, Request& r,
      const std::vector<CoeffType>& query) {
    PirRequest<CoeffType> req = r.request;
    req.first_dimension_query = query;
    std::vector<CoeffType> b = servers[0]->ComputeStage1(req).value();
    for (size_t s = 1; s < servers.size(); ++s) {
      const auto part = servers[s]->ComputeStage1(req).value();
      for (size_t j = 0; j < b.size(); ++j) b[j] += part[j];
    }
    return servers[0]
        ->PackFromStage1(req, std::move(b), &r.y_vec_h_fft)
        .value();
  }

  // Variance (in q1 units squared) of the decryption noise of `resp`.
  static double NoiseVariance(const PirClient<CoeffType>& client,
                              const PirResponse<CoeffType>& resp,
                              const std::vector<CoeffType>& expected) {
    const auto noise = client.DecryptionNoise(resp, expected).value();
    double sq = 0;
    for (int64_t e : noise) sq += double(e) * double(e);
    return sq / noise.size();
  }

  const RlweParams<CoeffType> rlwe_ =
      RlweParams<CoeffType>::Create(kD, 1ULL << 52, 1ULL << 25, 1ULL << 18,
                                    /*variance=*/40)
          .value();
  const GadgetParams pack_gadget_{19, 2}, eval_gadget_{19, 3};
  std::unique_ptr<FftContext> fft_ctx_;
};

namespace pm = ::private_membership::rlwe::v2;

// FNV-1a over every response coefficient of a fixed M <= 2048 batch request
// (standard multi-column path if !cross, else cross-color path).
uint64_t GoldenResponseHash(bool cross) {
  constexpr size_t kD = 2048;
  auto seed = [](uint64_t v, char tag) {
    std::string s(32, tag);
    for (int i = 0; i < 8; ++i) s[24 + i] = static_cast<char>(v >> (8 * i));
    return s;
  };
  auto prng = [](const std::string& s) {
    return std::move(::rlwe::ChaChaPrng::Create(s)).value();
  };
  const auto rlwe = pm::RlweParams<uint64_t>::Create(kD, 1ULL << 52, 1ULL << 25,
                                                     1ULL << 18, 40)
                        .value();
  const pm::GadgetParams pack{19, 2}, eval{19, 3};
  auto params = [&](int m, size_t cols) {
    return pm::PirParams<uint64_t>::Create(rlwe, 65536, pack, eval, m, 1, cols)
        .value();
  };
  const std::vector<int> rows = cross ? std::vector<int>{256, 512, 64}
                                      : std::vector<int>{1311, 1000, 33};
  const size_t cols = cross ? 1 : 2, d_row = 984;
  const std::string packing_seed(32, 'K');
  auto db = prng(seed(9, 'D'));
  std::vector<std::vector<uint16_t>> raw(rows.size());
  for (size_t s = 0; s < rows.size(); ++s) {
    raw[s].resize(rows[s] * cols * kD);
    for (auto& v : raw[s]) v = static_cast<uint16_t>(db->Rand64().value());
  }
  const auto groups = pm::BuildSelectorGroupPlans(rows, kD).value();
  std::vector<std::string> seeds(rows.size());
  std::vector<size_t> offsets(rows.size());
  for (const auto& g : groups)
    for (const auto& sl : g.slices) {
      seeds[sl.slot] = seed(g.group_index, 'G');
      offsets[sl.slot] = sl.offset;
    }
  auto client = pm::PirClient<uint64_t>::Create(params(kD, cols),
                                                prng(seed(0, 'S')),
                                                prng(seed(5, 'C')))
                    .value();
  auto key_prng = prng(packing_seed);
  pm::PirRequest<uint64_t> req;
  req.packing_key = client->CreatePackingKey(key_prng.get()).value();
  std::vector<std::vector<uint64_t>> queries(rows.size());
  for (const auto& g : groups) {
    auto mask = prng(seed(g.group_index, 'G'));
    std::vector<int> targets;
    for (const auto& sl : g.slices) targets.push_back(int(sl.rows) - 1 - int(sl.slot));
    auto sel = client->CreateGroupedSelector(g.slices, targets, mask.get()).value();
    for (const auto& sl : g.slices) {
      auto& q = queries[sl.slot];
      q.assign(sel.begin() + sl.offset, sel.begin() + sl.offset + sl.rows);
      for (auto& y : q) y = (((y + (1ULL << 11)) >> 12) & ((1ULL << 40) - 1)) << 12;
    }
  }
  req.color_queries = &queries;
  auto fft_ctx = pm::FftContext::Create(11).value();
  const std::vector<pm::Polynomial<uint64_t>> y_vec_h(req.packing_key.begin() + 2,
                                                      req.packing_key.end());
  const auto y_fft = pm::YVecHFft(rlwe, y_vec_h, *fft_ctx).value();
  std::vector<std::unique_ptr<Server>> servers;
  if (cross) {
    for (size_t b = 0; b * kD < rows.size() * d_row; ++b) {
      auto blk = Prep::PreprocessCrossColorBlock(rlwe, pack, rows, raw, seeds,
                                                 offsets, packing_seed, d_row, b)
                     .value();
      std::vector<pm::PreprocessMatrixPackOutput<uint64_t, int32_t>> outs;
      outs.push_back(std::move(blk.pack_output));
      servers.push_back(Server::CreateCrossColor(params(kD, 1),
                                                 std::move(blk.db_slices),
                                                 std::move(outs))
                            .value());
    }
  } else {
    for (size_t s = 0; s < rows.size(); ++s) {
      auto p = params(rows[s], cols);
      auto prep = Prep::Create(&p, seeds[s]).value();
      servers.push_back(
          Server::Create(p, prep->Preprocess(raw[s], packing_seed, offsets[s])
                                .value())
              .value());
    }
  }
  uint64_t h = 0xCBF29CE484222325ULL;
  for (size_t i = 0; i < servers.size(); ++i) {
    if (!cross) req.first_dimension_query = queries[i];
    const auto resp = servers[i]->ProcessResponse(req, &y_fft).value();
    for (const auto& ct : resp.ciphertexts)
      for (const auto* poly : {&ct.a, &ct.b})
        for (uint64_t c : poly->Coeffs())
          for (int k = 0; k < 8; ++k) h = (h ^ ((c >> (8 * k)) & 0xFF)) * 0x100000001B3ULL;
  }
  return h;
}

// Tables of at most d rows keep the pre-M > 2048 responses byte for byte
// (hashes recorded from m2048/baseline with this same function).
TEST(TallTableGoldenTest, ShortTablesMatchBaselineResponses) {
  EXPECT_EQ(GoldenResponseHash(/*cross=*/false), 1058159575200759081ULL);
  EXPECT_EQ(GoldenResponseHash(/*cross=*/true), 6585932620692559633ULL);
}

TEST(TallTablePlanTest, BuildSelectorGroupPlansGivesTallColorsOwnGroups) {
  ASSERT_OK_AND_ASSIGN(auto groups,
                       BuildSelectorGroupPlans({100, 5000, 300, 2048, 4097}));
  ASSERT_EQ(groups.size(), 5u);
  const std::vector<std::pair<size_t, size_t>> expected = {
      {100, 1}, {5000, 3}, {300, 1}, {2048, 1}, {4097, 3}};
  for (size_t g = 0; g < groups.size(); ++g) {
    EXPECT_EQ(groups[g].group_index, g);
    ASSERT_EQ(groups[g].slices.size(), 1u);
    EXPECT_EQ(groups[g].slices[0].slot, g);
    EXPECT_EQ(groups[g].slices[0].offset, 0u);
    EXPECT_EQ(groups[g].total_rows, expected[g].first);
    EXPECT_EQ(groups[g].num_samples, expected[g].second);
  }
}

TEST(TallTablePlanTest, RowStripesAreBlockAlignedAndCoverAllRows) {
  using Range = std::pair<size_t, size_t>;
  EXPECT_EQ(RowStripe(5000, kD, 0, 2), Range(0, 4096));
  EXPECT_EQ(RowStripe(5000, kD, 1, 2), Range(4096, 5000));
  const std::vector<Range> four = {
      {0, 2048}, {2048, 4096}, {4096, 5000}, {5000, 5000}};
  for (size_t s = 0; s < 4; ++s) EXPECT_EQ(RowStripe(5000, kD, s, 4), four[s]);
  EXPECT_EQ(RowStripe(700, kD, 0, 3), Range(0, 700));
  EXPECT_EQ(RowStripe(700, kD, 2, 3), Range(700, 700));
}

TEST_F(TallTableTest, MultiSampleSelectorRejectsOutOfRangeTargets) {
  auto r = MakeRequest({4100}, {4099}, 1, false, 40);
  auto mask = Prng(Seed(0, 'G'));
  EXPECT_THAT(r.client->CreateMultiSampleSelector(4100, 4100, mask.get()),
              ::rlwe::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(auto sel,
                       r.client->CreateMultiSampleSelector(4100, 4099,
                                                           mask.get()));
  EXPECT_EQ(sel.size(), 4100u);
}

// Mixed tall and grouped short colors on multi-column tables decrypt exactly
// at every target position, for unsigned and centered databases and 40- and
// 42-bit selectors.
TEST_F(TallTableTest, MultiColumnTallColorsDecryptExactly) {
  const std::vector<int> rows = {3000, 700, 300, 4101};
  constexpr size_t kCols = 2;
  const auto raw = RandomTables(rows, kCols, 1);
  const auto [seeds, offsets] = GroupLayout(rows);
  for (const auto& [centered, bits] :
       std::vector<std::pair<bool, int>>{{false, 40}, {true, 40}, {true, 42}}) {
    std::vector<std::vector<std::unique_ptr<Server>>> servers;
    for (size_t s = 0; s < rows.size(); ++s) {
      servers.push_back(StripedServers(rows[s], kCols, centered, seeds[s],
                                       offsets[s], raw[s], 1));
    }
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      auto r = MakeRequest(rows, targets, kCols, centered, bits);
      size_t selector_bytes = 0;
      for (int m : rows) selector_bytes += (size_t(m) * bits + 7) / 8;
      EXPECT_EQ(r.upload_bytes, 53248 + selector_bytes);
      for (size_t s = 0; s < rows.size(); ++s) {
        const auto resp = Answer(servers[s], r, r.queries[s]);
        ASSERT_EQ(resp.ciphertexts.size(), kCols);
        ASSERT_OK_AND_ASSIGN(auto got, r.client->ProcessResponse(resp));
        const auto row = raw[s].begin() + size_t(targets[s]) * kCols * kD;
        EXPECT_TRUE(std::equal(got.begin(), got.end(), row))
            << "color " << s << " pos " << pos << " centered " << centered
            << " bits " << bits;
      }
    }
  }
}

// Row-stripe servers summed before packing answer bit-identically to one
// server over the whole table (S = 1 vs S = 3).
TEST_F(TallTableTest, RowStripesMatchSingleServerBitExactly) {
  const std::vector<int> rows = {5000};
  const auto raw = RandomTables(rows, 1, 2);
  for (bool centered : {false, true}) {
    auto whole = StripedServers(rows[0], 1, centered, Seed(0, 'G'), 0, raw[0], 1);
    auto striped =
        StripedServers(rows[0], 1, centered, Seed(0, 'G'), 0, raw[0], 3);
    ASSERT_EQ(striped.size(), 3u);
    for (int target : {0, 2047, 2048, 4999}) {
      auto r = MakeRequest(rows, {target}, 1, centered, 42);
      const auto a = Answer(whole, r, r.queries[0]);
      const auto b = Answer(striped, r, r.queries[0]);
      EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), b.ciphertexts[0].b.Coeffs());
      EXPECT_EQ(whole[0]->GetPrepackedAModSwitched(0),
                striped[0]->GetPrepackedAModSwitched(0));
      ASSERT_OK_AND_ASSIGN(auto got, r.client->ProcessResponse(b));
      EXPECT_TRUE(std::equal(got.begin(), got.end(),
                             raw[0].begin() + size_t(target) * kD));
    }
  }
}

// Cross-color response packing over tall colors: two-stripe blocks answer
// bit-identically to PreprocessCrossColorBlock, and decrypt exactly.
TEST_F(TallTableTest, CrossColorTallColorsMatchAcrossStripes) {
  const std::vector<int> rows = {4100, 2048, 300};
  constexpr size_t kRowU16 = 700;
  const auto raw = RandomTables(rows, 1, 3);
  const auto [seeds, offsets] = GroupLayout(rows);
  const size_t n_resp = (rows.size() * kRowU16 + kD - 1) / kD;
  const std::vector<int> targets = {4099, 1000, 0};
  auto r = MakeRequest(rows, targets, 1, /*centered=*/true, 40);
  std::vector<CoeffType> recovered(rows.size() * kRowU16);
  for (size_t blk = 0; blk < n_resp; ++blk) {
    ASSERT_OK_AND_ASSIGN(auto whole,
                         Prep::PreprocessCrossColorBlock(
                             rlwe_, pack_gadget_, rows, raw, seeds, offsets,
                             kPackingSeed, kRowU16, blk, /*centered_db=*/true));
    ASSERT_OK_AND_ASSIGN(
        auto server_whole,
        Server::CreateCrossColor(Params(kD, 1, true), std::move(whole.db_slices),
                                 {std::move(whole.pack_output)}));
    std::vector<Prep::Stage1Stripe> parts;
    for (size_t s = 0; s < 2; ++s) {
      ASSERT_OK_AND_ASSIGN(
          auto part, Prep::PreprocessCrossColorStripe(rlwe_, rows, raw, seeds,
                                                      offsets, kRowU16, blk, s,
                                                      2, true));
      parts.push_back(std::move(part));
    }
    for (size_t u = 0; u < kD; ++u)
      for (size_t j = 0; j < kD; ++j)
        parts[0].hints[0][u][j] += parts[1].hints[0][u][j];
    ASSERT_OK_AND_ASSIGN(auto outputs,
                         Prep::PreprocessPacking(rlwe_, pack_gadget_,
                                                 kPackingSeed, parts[0].hints));
    ASSERT_OK_AND_ASSIGN(auto server0, Server::CreateCrossColor(
                                           Params(kD, 1, true),
                                           std::move(parts[0].db_slices),
                                           std::move(outputs)));
    ASSERT_OK_AND_ASSIGN(auto a, server_whole->ProcessResponse(
                                     r.request, &r.y_vec_h_fft));
    ASSERT_OK_AND_ASSIGN(auto b, server0->ComputeStage1(r.request));
    // Block 1 only holds rows of short colors, so its stripe 1 is empty.
    EXPECT_EQ(parts[1].db_slices.empty(), blk == 1);
    if (!parts[1].db_slices.empty()) {
      ASSERT_OK_AND_ASSIGN(auto server1, Server::CreateCrossColor(
                                             Params(kD, 1, true),
                                             std::move(parts[1].db_slices), {}));
      ASSERT_OK_AND_ASSIGN(auto b1, server1->ComputeStage1(r.request));
      for (size_t j = 0; j < b.size(); ++j) b[j] += b1[j];
    }
    ASSERT_OK_AND_ASSIGN(auto c, server0->PackFromStage1(r.request, b,
                                                         &r.y_vec_h_fft));
    EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), c.ciphertexts[0].b.Coeffs());
    EXPECT_EQ(server_whole->GetPrepackedAModSwitched(0),
              server0->GetPrepackedAModSwitched(0));
    ASSERT_OK_AND_ASSIGN(auto blk_plain, r.client->ProcessResponse(c));
    for (size_t i = blk * kD; i < std::min((blk + 1) * kD, recovered.size());
         ++i) {
      recovered[i] = blk_plain[i - blk * kD];
    }
  }
  for (size_t s = 0; s < rows.size(); ++s) {
    for (size_t u = 0; u < kRowU16; ++u) {
      ASSERT_EQ(recovered[s * kRowU16 + u], raw[s][targets[s] * kD + u])
          << "color " << s << " u " << u;
    }
  }
}

// On adversarial all-0xFFFF data at R = 8192, centering cuts the Stage 1
// noise variance about 4x and 42-bit selectors cut the rest about 16x
// (variances in q1^2: about 4600 unsigned, 2600 centered, 2000 at 42 bits).
TEST_F(TallTableTest, CenteredDatabaseAndWideSelectorsCutNoise) {
  const std::vector<int> rows = {8192};
  const std::vector<std::vector<uint16_t>> raw = {
      std::vector<uint16_t>(rows[0] * kD, 0xFFFF)};
  const std::vector<CoeffType> expected(kD, 0xFFFF);
  auto Variance = [&](bool centered, int bits) {
    auto servers =
        StripedServers(rows[0], 1, centered, Seed(0, 'G'), 0, raw[0], 1);
    auto r = MakeRequest(rows, {1234}, 1, centered, bits);
    const auto resp = Answer(servers, r, r.queries[0]);
    if (centered) {
      EXPECT_EQ(r.client->ProcessResponse(resp).value(), expected);
    }
    return NoiseVariance(*r.client, resp, expected);
  };
  const double unsigned40 = Variance(false, 40);
  const double centered40 = Variance(true, 40);
  const double centered42 = Variance(true, 42);
  EXPECT_GT(unsigned40 - centered40, 1500);
  EXPECT_GT(centered40 - centered42, 400);
  EXPECT_LT(centered42, 2300);
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
