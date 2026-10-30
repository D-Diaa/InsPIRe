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

// Palette-parameter (d=2048, q=2^52, q1=2^25, q2=2^18, p=2^16, t=1) tests of
// color tables taller than d: multi-sample selectors, signed (centered)
// entries, 42-bit selectors and row-stripe Stage 1 servers.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
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
  // A cross-color response block: stripe 0's packing server, then the other
  // stripes' Stage 1 servers.
  using Block = std::vector<std::unique_ptr<Server>>;

  void SetUp() override { ASSERT_OK_AND_ASSIGN(fft_ctx_, FftContext::Create(11)); }

  PirParams<CoeffType> Params(int rows, size_t cols) const {
    return PirParams<CoeffType>::Create(rlwe_, 65536, pack_gadget_,
                                        eval_gadget_, rows, 1, cols)
        .value();
  }

  // Tables of `row_words` uint16_t words per row: random, or the constant
  // `fill`.
  static std::vector<std::vector<uint16_t>> Tables(const std::vector<int>& rows,
                                                   size_t row_words,
                                                   uint64_t seed,
                                                   int fill = -1) {
    auto prng = Prng(Seed(seed, 'D'));
    std::vector<std::vector<uint16_t>> raw(rows.size());
    for (size_t s = 0; s < rows.size(); ++s) {
      raw[s].resize(rows[s] * row_words);
      for (auto& v : raw[s]) {
        v = fill >= 0 ? fill : static_cast<uint16_t>(prng->Rand64().value());
      }
    }
    return raw;
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

  Request MakeRequest(const std::vector<int>& rows,
                      const std::vector<int>& targets, int selector_bits,
                      size_t cols = 1, uint64_t client_seed = 7) {
    Request r;
    r.client = PirClient<CoeffType>::Create(Params(kD, cols),
                                            Prng(Seed(0, 'S')),
                                            Prng(Seed(client_seed, 'C')))
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

  // Every response block of the colors' rows of `d_row` words, preprocessed
  // as `num_stripes` row stripes whose hints are summed for stripe 0.
  std::vector<Block> Blocks(const std::vector<int>& rows,
                            const std::vector<std::vector<uint16_t>>& raw,
                            size_t d_row, bool centered, size_t num_stripes) {
    const auto [seeds, offsets] = GroupLayout(rows);
    const size_t n_blocks = (rows.size() * d_row + kD - 1) / kD;
    std::vector<std::future<Block>> jobs;
    jobs.reserve(n_blocks);
    for (size_t b = 0; b < n_blocks; ++b) {
      jobs.push_back(std::async(std::launch::async, [&, b] {
        Block block(1);
        Prep::CrossColorStripe first;
        for (size_t s = 0; s < num_stripes; ++s) {
          auto part = Prep::PreprocessCrossColorStripe(rlwe_, rows, raw, seeds,
                                                       offsets, d_row, b, s,
                                                       num_stripes, centered)
                          .value();
          if (s == 0) {
            first = std::move(part);
            continue;
          }
          for (size_t u = 0; u < kD; ++u)
            for (size_t j = 0; j < kD; ++j) first.hint[u][j] += part.hint[u][j];
          if (!part.db_slices.empty()) {
            block.push_back(Server::CreateCrossColor(
                                Params(kD, 1), std::move(part.db_slices), {})
                                .value());
          }
        }
        std::vector<PreprocessMatrixPackOutput<CoeffType, int32_t>> outputs;
        outputs.push_back(Prep::PreprocessPacking(rlwe_, pack_gadget_,
                                                  kPackingSeed, first.hint)
                              .value());
        block[0] = Server::CreateCrossColor(Params(kD, 1),
                                            std::move(first.db_slices),
                                            std::move(outputs))
                       .value();
        return block;
      }));
    }
    std::vector<Block> blocks;
    blocks.reserve(n_blocks);
    for (auto& job : jobs) blocks.push_back(job.get());
    return blocks;
  }

  // Answers `r` with `block`, summing the stripes' Stage 1 outputs.
  PirResponse<CoeffType> Answer(const Block& block, Request& r) {
    std::vector<CoeffType> b = block[0]->ComputeStage1(r.request).value();
    for (size_t s = 1; s < block.size(); ++s) {
      const auto part = block[s]->ComputeStage1(r.request).value();
      for (size_t j = 0; j < b.size(); ++j) b[j] += part[j];
    }
    return block[0]->PackFromStage1(r.request, std::move(b), &r.y_vec_h_fft)
        .value();
  }

  // Decrypts every block and (if `check`) checks each color's target row;
  // returns the decryption noise of the data coefficients.
  std::vector<int64_t> CheckDecryption(
      const std::vector<Block>& blocks, Request& r, const std::vector<int>& rows,
      const std::vector<std::vector<uint16_t>>& raw,
      const std::vector<int>& targets, size_t d_row, bool check = true) {
    std::vector<int64_t> noise;
    const size_t total = rows.size() * d_row;
    for (size_t b = 0; b < blocks.size(); ++b) {
      std::vector<int64_t> block_noise;
      const auto got =
          r.client->ProcessResponse(Answer(blocks[b], r), &block_noise).value();
      for (size_t i = 0; i < kD && b * kD + i < total; ++i) {
        const size_t s = (b * kD + i) / d_row, u = (b * kD + i) % d_row;
        const size_t row_words = raw[s].size() / rows[s];
        if (check) {
          EXPECT_EQ(got[i], raw[s][targets[s] * row_words + u])
              << "color " << s << " word " << u;
        }
        noise.push_back(block_noise[i]);
      }
    }
    return noise;
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

// With continuous cross-color packing across all row widths, PackedRowWords is
// always row_u16 and n_resp = ceil(P * bundle * row_u16 / 2048): at P = 64,
// M = 8192, 1024-word rows, bundle = 3 uses 96 blocks (1,983,936 B) vs 64
// blocks at bundle = 2 (2,068,480 B) and 32 blocks at bundle = 1 (3,026,944 B).
TEST(TallTablePlanTest, ChooseBundleFactorMinimizesExactWireBytes) {
  for (size_t w : {size_t{1}, size_t{1024}, size_t{1025}, size_t{1536},
                   size_t{1552}, size_t{2048}, size_t{2064}, size_t{3072}}) {
    EXPECT_EQ(PackedRowWords(w), w);
  }
  const std::vector<int> rows(64, 8192);
  size_t resp = 0;
  EXPECT_EQ(BundledWireBytes(rows, 1024, 1, 0, &resp), 3026944u);
  EXPECT_EQ(resp, 32u);
  EXPECT_EQ(BundledWireBytes(rows, 1024, 2, 0, &resp), 2068480u);
  EXPECT_EQ(resp, 64u);
  EXPECT_EQ(BundledWireBytes(rows, 1024, 3, 0, &resp), 1983936u);
  EXPECT_EQ(resp, 96u);
  EXPECT_EQ(BundledWireBytes(rows, 1024, 4, 0, &resp), 2117632u);
  EXPECT_EQ(resp, 128u);
  EXPECT_EQ(ChooseBundleFactor(rows, 1024), 3u);
  for (size_t w : {size_t{1024}, size_t{1552}, size_t{2064}}) {
    EXPECT_EQ(ChooseBundleFactor(std::vector<int>(64, 2048), w), 1u);
  }
  // Sub-full-tile (1024 < w < 2048) and multi-tile (w > 2048) continuous counts.
  EXPECT_EQ(BundledWireBytes(std::vector<int>(166, 2048), 1552, 1, 0, &resp),
            53248u + 166u * 10240u + 126u * 11008u);
  EXPECT_EQ(resp, 126u);
  EXPECT_EQ(BundledWireBytes(std::vector<int>(128, 2048), 2064, 1, 0, &resp),
            53248u + 128u * 10240u + 129u * 11008u);
  EXPECT_EQ(resp, 129u);
  EXPECT_EQ(BundledWireBytes(std::vector<int>(8, 8192), 2064, 2, 0, &resp),
            404224u);
  EXPECT_EQ(resp, 17u);
  EXPECT_EQ(ChooseBundleFactor(std::vector<int>(8, 8192), 2064), 2u);
  // A 42-bit override is priced at 42 bits per selector word.
  EXPECT_EQ(BundledWireBytes({8192}, 1024, 1, 42), 53248u + 43008u + 5504u * 2);
}

TEST(TallTablePlanTest, BundleRowsPlacesLogicalRowsAndZeroPadsTail) {
  // 5 logical rows of 3 words (stride 4) into bundles of 2 (stride 8).
  std::vector<uint16_t> logical(5 * 4);
  for (size_t i = 0; i < logical.size(); ++i) logical[i] = i + 1;
  const auto phys = BundleRows(logical, 5, 4, 3, 2, 8);
  ASSERT_EQ(phys.size(), 3u * 8);
  const std::vector<uint16_t> expected = {1,  2,  3,  5,  6,  7,  0, 0,
                                          9,  10, 11, 13, 14, 15, 0, 0,
                                          17, 18, 19, 0,  0,  0,  0, 0};
  EXPECT_EQ(phys, expected);
}

TEST_F(TallTableTest, MultiSampleSelectorRejectsOutOfRangeTargets) {
  auto r = MakeRequest({4100}, {4099}, 40);
  auto mask = Prng(Seed(0, 'G'));
  EXPECT_THAT(r.client->CreateMultiSampleSelector(4100, 4100, mask.get()),
              ::rlwe::testing::StatusIs(absl::StatusCode::kInvalidArgument));
  ASSERT_OK_AND_ASSIGN(auto sel,
                       r.client->CreateMultiSampleSelector(4100, 4099,
                                                           mask.get()));
  EXPECT_EQ(sel.size(), 4100u);
}

// Mixed tall and grouped short colors decrypt exactly at their first, last and
// middle rows, for sub-tile rows (cross-color blocks) and two-column rows, with
// unsigned or signed entries and 40- or 42-bit selectors.
TEST_F(TallTableTest, TallColorsDecryptExactly) {
  const std::vector<int> rows = {3000, 700, 300, 4101};
  struct Case {
    size_t d_row;
    bool centered;
    int bits;
  };
  for (const Case& c : {Case{700, false, 40}, Case{700, true, 40},
                        Case{700, true, 42}, Case{2 * kD, true, 42}}) {
    const size_t row_words = c.d_row <= kD ? kD : c.d_row;
    const auto raw = Tables(rows, row_words, 1);
    const auto blocks = Blocks(rows, raw, c.d_row, c.centered, 1);
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      auto r = MakeRequest(rows, targets, c.bits);
      size_t selector_bytes = 0;
      for (int m : rows) selector_bytes += (size_t(m) * c.bits + 7) / 8;
      EXPECT_EQ(r.upload_bytes, 53248 + selector_bytes);
      SCOPED_TRACE(::testing::Message() << "d_row " << c.d_row << " centered "
                                        << c.centered << " bits " << c.bits);
      CheckDecryption(blocks, r, rows, raw, targets, c.d_row);
    }
  }
}

// Row-stripe servers summed before packing answer bit-identically to one
// server per block, for sub-tile and full-column rows.
TEST_F(TallTableTest, RowStripesMatchSingleServerBitExactly) {
  struct Case {
    std::vector<int> rows;
    size_t d_row;
    std::vector<int> targets;
  };
  for (const Case& c : {Case{{4100, 2048, 300}, 700, {4099, 1000, 0}},
                        Case{{5000}, kD, {2048}}}) {
    const auto raw = Tables(c.rows, kD, 2);
    const auto whole = Blocks(c.rows, raw, c.d_row, true, 1);
    auto r = MakeRequest(c.rows, c.targets, 42);
    for (size_t num_stripes : {2, 3}) {
      const auto striped = Blocks(c.rows, raw, c.d_row, true, num_stripes);
      ASSERT_EQ(striped.size(), whole.size());
      for (size_t b = 0; b < whole.size(); ++b) {
        const auto a = Answer(whole[b], r), s = Answer(striped[b], r);
        EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), s.ciphertexts[0].b.Coeffs());
        EXPECT_EQ(whole[b][0]->GetPrepackedAModSwitched(0),
                  striped[b][0]->GetPrepackedAModSwitched(0));
      }
      CheckDecryption(striped, r, c.rows, raw, c.targets, c.d_row);
    }
  }
}

// Rows of C whole columns served as C cross-color blocks per color answer
// bit-identically to the per-color Preprocess + Create servers, tall colors
// included.
TEST_F(TallTableTest, WideRowBlocksMatchPerColorServers) {
  const std::vector<int> rows = {1311, 4100};
  constexpr size_t kCols = 2;
  const auto raw = Tables(rows, kCols * kD, 3);
  const auto [seeds, offsets] = GroupLayout(rows);
  const auto blocks = Blocks(rows, raw, kCols * kD, false, 1);
  ASSERT_EQ(blocks.size(), rows.size() * kCols);
  const std::vector<int> targets = {1310, 4099};
  auto r = MakeRequest(rows, targets, 40, kCols);
  for (size_t s = 0; s < rows.size(); ++s) {
    auto params = Params(rows[s], kCols);
    ASSERT_OK_AND_ASSIGN(auto prep, Prep::Create(&params, seeds[s]));
    ASSERT_OK_AND_ASSIGN(auto data,
                         prep->Preprocess(raw[s], kPackingSeed, offsets[s]));
    ASSERT_OK_AND_ASSIGN(auto server, Server::Create(params, std::move(data)));
    PirRequest<CoeffType> req = r.request;
    req.first_dimension_query = r.queries[s];
    ASSERT_OK_AND_ASSIGN(auto resp, server->ProcessResponse(req, &r.y_vec_h_fft));
    for (size_t k = 0; k < kCols; ++k) {
      const auto blk = Answer(blocks[s * kCols + k], r);
      EXPECT_EQ(resp.ciphertexts[k].a.Coeffs(), blk.ciphertexts[0].a.Coeffs());
      EXPECT_EQ(resp.ciphertexts[k].b.Coeffs(), blk.ciphertexts[0].b.Coeffs());
    }
    ASSERT_OK_AND_ASSIGN(auto got, r.client->ProcessResponse(resp));
    EXPECT_TRUE(std::equal(got.begin(), got.end(),
                           raw[s].begin() + targets[s] * kCols * kD));
  }
}

// At R = 16384 with uniform entries, signed entries cut the Stage 1 noise
// variance 4x and 42-bit selectors cut the remainder about 16x (expected
// variances in q1^2: about 3700 unsigned, 2400 signed, 1950 signed at 42
// bits). Most of the unsigned excess is the 2^15 * sum_r (selector error)
// offset shared by a response's coefficients, hence 64 fresh secrets. Unsigned
// 40-bit selectors are not decryption-safe here (about 2^-17 per coefficient),
// so only their noise is measured. Constant words decrypt exactly with signed
// entries and 42-bit selectors, within the noise bound.
TEST_F(TallTableTest, SignedEntriesAndWideSelectorsCutNoise) {
  const std::vector<int> rows = {16384};
  auto Variance = [&](bool centered, int bits, int fill, int queries) {
    const auto raw = Tables(rows, kD, 4, fill);
    const auto blocks = Blocks(rows, raw, kD, centered, 1);
    double sq = 0;
    size_t n = 0;
    for (int i = 0; i < queries; ++i) {
      const int target = i * 2477 % rows[0];
      auto r = MakeRequest(rows, {target}, bits, 1, /*client_seed=*/100 + i);
      for (int64_t e :
           CheckDecryption(blocks, r, rows, raw, {target}, kD, centered)) {
        sq += double(e) * double(e);
        ++n;
      }
    }
    return sq / n;
  };
  const double unsigned40 = Variance(false, 40, -1, 64);
  const double signed40 = Variance(true, 40, -1, 64);
  const double signed42 = Variance(true, 42, -1, 64);
  EXPECT_GT(unsigned40 - signed40, 600);
  EXPECT_GT(signed40 - signed42, 250);
  EXPECT_LT(signed42, 2150);
  // Constant words share their Stage 1 noise across a response; 0x8000 is the
  // signed worst case (about 85 q1^2 here), 0x0001 and 0xFFFF add almost none.
  for (int fill : {0x0000, 0x0001, 0x8000, 0xFFFF}) {
    EXPECT_LT(Variance(true, 42, fill, 4), 2800) << "fill " << fill;
  }
}

// Colors of up to 262,144 short logical rows decrypt exactly unbundled and
// with the wire-optimal bundle (including a color whose height the bundle does
// not divide); the client keeps the target's words of the physical row.
TEST_F(TallTableTest, BundledAndUnbundledTallColorsDecryptExactly) {
  const std::vector<int> rows = {262144, 70001};
  constexpr size_t kRowU16 = 64;
  const auto raw = Tables(rows, kRowU16, 5);
  const std::vector<int> targets = {262143, 70000};
  const size_t chosen = ChooseBundleFactor(rows, kRowU16);
  EXPECT_GT(chosen, 1u);
  for (size_t bundle : {size_t{1}, chosen}) {
    std::vector<int> phys_rows, phys_targets;
    std::vector<std::vector<uint16_t>> phys(rows.size());
    for (size_t s = 0; s < rows.size(); ++s) {
      phys[s] = BundleRows(raw[s], rows[s], kRowU16, kRowU16, bundle,
                           bundle * kRowU16);
      phys_rows.push_back((rows[s] + bundle - 1) / bundle);
      phys_targets.push_back(targets[s] / bundle);
    }
    const size_t d_row = PackedRowWords(bundle * kRowU16);
    const size_t max_phys = *std::max_element(phys_rows.begin(), phys_rows.end());
    const auto blocks = Blocks(phys_rows, phys, d_row, true, 1);
    auto r = MakeRequest(phys_rows, phys_targets, AutoSelectorBits(max_phys));
    for (size_t b = 0; b < blocks.size(); ++b) {
      const auto got = r.client->ProcessResponse(Answer(blocks[b], r)).value();
      for (size_t s = 0; s < rows.size(); ++s) {
        // The logical row's words within the decrypted physical row.
        const size_t first = s * d_row + targets[s] % bundle * kRowU16;
        for (size_t u = 0; u < kRowU16; ++u) {
          if (first + u < b * kD || first + u >= (b + 1) * kD) continue;
          EXPECT_EQ(got[first + u - b * kD], raw[s][targets[s] * kRowU16 + u])
              << "bundle " << bundle << " color " << s << " word " << u;
        }
      }
    }
  }
}

// Rows in (1024, 2048) words share 2048-word response blocks continuously
// across colors (n_resp = ceil(P * d_row / 2048) < P), straddling block
// boundaries for both M <= 2048 and tall M > 2048 with multi-stripe bit
// identity and unchanged RLWE decryption noise.
TEST_F(TallTableTest, ContinuousCrossColorSubFullTileRowsDecryptExactly) {
  // d_row = 1536 (3,072 B), P = 8 short/grouped colors -> 6 response blocks.
  {
    const std::vector<int> rows = {512, 1024, 2048, 300, 700, 1500, 64, 2048};
    constexpr size_t kDRow = 1536;
    const auto raw = Tables(rows, kD, /*seed=*/11);
    const auto blocks = Blocks(rows, raw, kDRow, /*centered=*/false, 1);
    ASSERT_EQ(blocks.size(), 6u);
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) {
        targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      }
      auto r = MakeRequest(rows, targets, 40);
      EXPECT_EQ(r.upload_bytes + blocks.size() * 11008u,
                BundledWireBytes(rows, kDRow, 1, 40));
      CheckDecryption(blocks, r, rows, raw, targets, kDRow);
    }
  }
  // d_row = 1552 (3,104 B), P = 5 mixed/tall colors -> 4 response blocks,
  // with non-zero trailing words in [1552, 2048) of raw[s], multi-stripe
  // bit-identity, and noise verification across 16 fresh client secrets.
  {
    const std::vector<int> rows = {4096, 1300, 700, 8192, 2048};
    constexpr size_t kDRow = 1552;
    const auto raw = Tables(rows, kD, /*seed=*/12);
    const auto whole = Blocks(rows, raw, kDRow, /*centered=*/true, 1);
    const auto striped = Blocks(rows, raw, kDRow, /*centered=*/true, 4);
    ASSERT_EQ(whole.size(), 4u);
    ASSERT_EQ(striped.size(), whole.size());
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) {
        targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      }
      auto r = MakeRequest(rows, targets, 40, 1, /*client_seed=*/20 + pos);
      if (pos == 0) {
        for (size_t b = 0; b < whole.size(); ++b) {
          const auto a = Answer(whole[b], r), s = Answer(striped[b], r);
          EXPECT_EQ(a.ciphertexts[0].a.Coeffs(), s.ciphertexts[0].a.Coeffs());
          EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), s.ciphertexts[0].b.Coeffs());
          EXPECT_EQ(whole[b][0]->GetPrepackedAModSwitched(0),
                    striped[b][0]->GetPrepackedAModSwitched(0));
        }
      }
      CheckDecryption(striped, r, rows, raw, targets, kDRow);
    }
    double sq = 0, max_abs = 0;
    size_t count = 0;
    for (int q = 0; q < 16; ++q) {
      std::vector<int> targets;
      for (size_t s = 0; s < rows.size(); ++s) {
        targets.push_back((q * 997 + int(s) * 131) % rows[s]);
      }
      auto r = MakeRequest(rows, targets, 40, 1, /*client_seed=*/200 + q);
      for (int64_t e : CheckDecryption(whole, r, rows, raw, targets, kDRow)) {
        sq += double(e) * double(e);
        max_abs = std::max(max_abs, std::abs(double(e)));
        ++count;
      }
    }
    EXPECT_LT(sq / count, 2150);
    EXPECT_LT(max_abs, 256);
  }
}

// Multi-tile rows (d_row > 2048) pack continuously across colors into
// n_resp = ceil(P * d_row / 2048) < P * c response blocks, splitting single
// colors across multiple blocks for both M <= 2048 and tall M > 2048.
TEST_F(TallTableTest, ContinuousCrossColorMultiTileRowsDecryptExactly) {
  // d_row = 2064 (4,128 B, c = 2), P = 5 -> 6 blocks (< 10 = P * c).
  for (bool tall : {false, true}) {
    const std::vector<int> rows = tall
                                      ? std::vector<int>{4096, 8192, 1500, 500, 4096}
                                      : std::vector<int>{2048, 512, 1024, 333, 2048};
    constexpr size_t kDRow = 2064;
    const auto raw = Tables(rows, 2 * kD, /*seed=*/tall ? 14 : 13);
    const auto whole = Blocks(rows, raw, kDRow, /*centered=*/tall, 1);
    ASSERT_EQ(whole.size(), 6u);
    ASSERT_LT(whole.size(), rows.size() * 2u);
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) {
        targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      }
      auto r = MakeRequest(rows, targets, 40, 1, /*client_seed=*/30 + pos);
      EXPECT_EQ(r.upload_bytes + whole.size() * 11008u,
                BundledWireBytes(rows, kDRow, 1, 40));
      CheckDecryption(whole, r, rows, raw, targets, kDRow);
    }
    if (tall) {
      const auto striped = Blocks(rows, raw, kDRow, /*centered=*/true, 3);
      ASSERT_EQ(striped.size(), whole.size());
      const std::vector<int> targets = {4095, 8191, 0, 250, 2048};
      auto r = MakeRequest(rows, targets, 40, 1, /*client_seed=*/40);
      for (size_t b = 0; b < whole.size(); ++b) {
        const auto a = Answer(whole[b], r), s = Answer(striped[b], r);
        EXPECT_EQ(a.ciphertexts[0].a.Coeffs(), s.ciphertexts[0].a.Coeffs());
        EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), s.ciphertexts[0].b.Coeffs());
      }
      CheckDecryption(striped, r, rows, raw, targets, kDRow);

      double sq = 0, max_abs = 0;
      size_t count = 0;
      for (int q = 0; q < 16; ++q) {
        std::vector<int> q_targets;
        for (size_t s = 0; s < rows.size(); ++s) {
          q_targets.push_back((q * 1237 + int(s) * 257) % rows[s]);
        }
        auto qr = MakeRequest(rows, q_targets, 40, 1, /*client_seed=*/300 + q);
        for (int64_t e :
             CheckDecryption(whole, qr, rows, raw, q_targets, kDRow)) {
          sq += double(e) * double(e);
          max_abs = std::max(max_abs, std::abs(double(e)));
          ++count;
        }
      }
      EXPECT_LT(sq / count, 2150);
      EXPECT_LT(max_abs, 256);
    }
  }
  // d_row = 3072 (6,144 B, c = 2), P = 4 -> 6 blocks (< 8 = P * c).
  {
    const std::vector<int> rows = {4096, 2048, 8192, 1000};
    constexpr size_t kDRow = 3072;
    const auto raw = Tables(rows, 2 * kD, /*seed=*/15);
    const auto whole = Blocks(rows, raw, kDRow, /*centered=*/true, 1);
    const auto striped = Blocks(rows, raw, kDRow, /*centered=*/true, 2);
    ASSERT_EQ(whole.size(), 6u);
    ASSERT_EQ(striped.size(), 6u);
    for (int pos = 0; pos < 3; ++pos) {
      std::vector<int> targets;
      for (int m : rows) {
        targets.push_back(pos == 0 ? 0 : pos == 1 ? m - 1 : m / 2);
      }
      auto r = MakeRequest(rows, targets, 40, 1, /*client_seed=*/50 + pos);
      if (pos == 0) {
        for (size_t b = 0; b < whole.size(); ++b) {
          const auto a = Answer(whole[b], r), s = Answer(striped[b], r);
          EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), s.ciphertexts[0].b.Coeffs());
        }
      }
      CheckDecryption(striped, r, rows, raw, targets, kDRow);
    }
  }
}

// Bundling 1 (compute-optimal default), 0 (wire-optimal ChooseBundleFactor),
// and explicit 2 works across M <= 2048 and tall M = 4096, 8192 for sub-tile
// (1024 -> bundle 3), sub-full-tile (1552), and multi-tile orig_c > 1 (2064,
// where bundle = 2 yields phys_c = 3 and d_row = 4128 -> 9 blocks < 12), with
// multi-stripe bit-identity.
TEST_F(TallTableTest, BundledContinuousCrossColorDecryptsAcrossModesAndStripes) {
  struct Config {
    std::vector<int> rows;
    size_t orig_row_u16;
    size_t bundle_flag;      // 1 = default, 0 = ChooseBundleFactor, 2 = explicit
    size_t expected_bundle;  // Resolved bundle factor
  };
  const std::vector<Config> configs = {
      {{2048, 1500, 1024, 2048}, 1552, 0, 1},
      {{2048, 1500, 1024, 2048}, 2064, 2, 2},
      {{8192, 8192, 8192, 8192}, 1024, 0, 3},
      {{8192, 4096, 8192, 4097}, 1552, 1, 1},
      {{8192, 4096, 8192, 4097}, 1552, 0, 2},
      {{8192, 4096, 8192, 4096}, 2064, 1, 1},
      {{8192, 4096, 8192, 4096}, 2064, 2, 2},
  };
  EXPECT_EQ(ChooseBundleFactor({2048, 1500, 1024, 2048}, 2064), 1u);
  EXPECT_EQ(ChooseBundleFactor({8192, 4096, 8192, 4096}, 2064), 2u);
  for (size_t cfg_idx = 0; cfg_idx < configs.size(); ++cfg_idx) {
    const auto& cfg = configs[cfg_idx];
    const size_t orig_c = (cfg.orig_row_u16 + kD - 1) / kD;
    const size_t src_stride = orig_c * kD;
    const auto logical = Tables(cfg.rows, src_stride, /*seed=*/60 + cfg_idx);
    const size_t bundle =
        cfg.bundle_flag == 0 ? ChooseBundleFactor(cfg.rows, cfg.orig_row_u16)
                             : cfg.bundle_flag;
    ASSERT_EQ(bundle, cfg.expected_bundle);
    const size_t phys_c = (bundle * cfg.orig_row_u16 + kD - 1) / kD;
    const size_t dst_stride = phys_c * kD;
    std::vector<int> phys_rows(cfg.rows.size());
    std::vector<std::vector<uint16_t>> phys(cfg.rows.size());
    for (size_t s = 0; s < cfg.rows.size(); ++s) {
      if (bundle > 1) {
        phys[s] = BundleRows(logical[s], cfg.rows[s], src_stride,
                             cfg.orig_row_u16, bundle, dst_stride);
        phys_rows[s] = (cfg.rows[s] + int(bundle) - 1) / int(bundle);
      } else {
        phys[s] = logical[s];
        phys_rows[s] = cfg.rows[s];
      }
    }
    const size_t d_row = PackedRowWords(bundle * cfg.orig_row_u16);
    const size_t max_phys =
        *std::max_element(phys_rows.begin(), phys_rows.end());
    const bool centered = max_phys > kD;
    const int bits = AutoSelectorBits(max_phys);
    const auto whole = Blocks(phys_rows, phys, d_row, centered, 1);
    size_t expected_resp = 0;
    const size_t expected_wire = BundledWireBytes(
        cfg.rows, cfg.orig_row_u16, bundle, bits, &expected_resp);
    ASSERT_EQ(whole.size(), expected_resp);

    const bool check_stripes = max_phys > kD && bundle == 2;
    const auto striped =
        check_stripes ? Blocks(phys_rows, phys, d_row, centered, 2)
                      : std::vector<Block>{};

    for (int pos = 0; pos < 2; ++pos) {
      std::vector<int> targets, phys_targets;
      for (size_t s = 0; s < cfg.rows.size(); ++s) {
        const int t = pos == 0 ? int(s % bundle)
                               : cfg.rows[s] - 1 - int(s % bundle);
        targets.push_back(t);
        phys_targets.push_back(t / int(bundle));
      }
      auto r = MakeRequest(phys_rows, phys_targets, bits, 1,
                           /*client_seed=*/80 + cfg_idx * 2 + pos);
      EXPECT_EQ(r.upload_bytes + whole.size() * 11008u, expected_wire);
      if (check_stripes && pos == 0) {
        for (size_t b = 0; b < whole.size(); ++b) {
          const auto a = Answer(whole[b], r), st = Answer(striped[b], r);
          EXPECT_EQ(a.ciphertexts[0].b.Coeffs(), st.ciphertexts[0].b.Coeffs());
        }
      }
      const auto& active = check_stripes ? striped : whole;
      for (size_t b = 0; b < active.size(); ++b) {
        const auto got =
            r.client->ProcessResponse(Answer(active[b], r)).value();
        for (size_t s = 0; s < cfg.rows.size(); ++s) {
          const size_t first =
              s * d_row + (targets[s] % bundle) * cfg.orig_row_u16;
          for (size_t u = 0; u < cfg.orig_row_u16; ++u) {
            if (first + u < b * kD || first + u >= (b + 1) * kD) continue;
            EXPECT_EQ(got[first + u - b * kD],
                      logical[s][targets[s] * src_stride + u])
                << "cfg " << cfg_idx << " bundle " << bundle << " color " << s
                << " word " << u;
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
