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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
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
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace pir {
namespace {

using CoeffType = uint64_t;
using MatCoeffType = int32_t;
using DbDataType = uint16_t;

struct PirTestParams {
  // Database parameters
  int num_entries = 4096;
  int entry_size_multiple = 1;
  int degree = 2048;
  // Database size = num_entries * degree * entry_size_multiple * 2 bytes

  // Crypto parameters
  int variance = 6;
  int plaintext_modulus = 65535;
  uint64_t modulus = 1ULL << 56;

  // Tunable parameters for performance
  int pack_log_digit = 19;
  int pack_num_digits = 2;
  int eval_log_digit = 19;
  int eval_num_digits = 3;
  int interpolation_degree = 2;

  uint64_t mod1_after_switch;
  uint64_t mod2_after_switch;

  PirTestParams() {
    int log_p = absl::bit_width(static_cast<uint32_t>(plaintext_modulus));
    int log_d = absl::bit_width(static_cast<uint32_t>(degree)) - 1;
    mod1_after_switch = 1ULL << (log_p + log_d + 2);
    mod2_after_switch = 1ULL << (log_p + 2);
  }
};

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

class PirEndToEndTest : public ::testing::Test {
 protected:
  absl::StatusOr<std::unique_ptr<PirParams<CoeffType>>> CreateParams(
      const PirTestParams& p) {
    ASSIGN_OR_RETURN(
        auto rlwe_params,
        RlweParams<CoeffType>::Create(p.degree, p.modulus, p.mod1_after_switch,
                                      p.mod2_after_switch, p.variance));

    GadgetParams pack_gadget{p.pack_log_digit, p.pack_num_digits};
    GadgetParams poly_eval_gadget{p.eval_log_digit, p.eval_num_digits};

    ASSIGN_OR_RETURN(
        auto params,
        PirParams<CoeffType>::Create(
            rlwe_params, p.plaintext_modulus, pack_gadget, poly_eval_gadget,
            p.num_entries, p.interpolation_degree, p.entry_size_multiple));
    return std::make_unique<PirParams<CoeffType>>(params);
  }
};

TEST_F(PirEndToEndTest, ExecutesSuccessfullyAndMatchesExpectedOutput) {
  PirTestParams p;
  ASSERT_OK_AND_ASSIGN(auto params, CreateParams(p));

  // 1. Generate a random database
  std::vector<DbDataType> db_raw(static_cast<size_t>(p.num_entries) * p.degree *
                                 p.entry_size_multiple);
  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    db_raw[i] = (*db_prng->Rand64()) % p.plaintext_modulus;
  }

  // 2. Create server and preprocess the database
  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  auto prng_seed = prep_server->GetPrngSeed();

  // Directly pass to online server (in memory)
  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *params, std::move(preprocessed_data))));

  // 3. Client setup
  ASSERT_OK_AND_ASSIGN(auto client_a_prng, CreatePrng(prng_seed));
  ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
  ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                        *params, std::move(client_a_prng),
                                        std::move(client_prng)));

  // Try out a random index
  int target_index = (*db_prng->Rand8()) % p.num_entries;

  // 4. Request / Response Phase
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
  ASSERT_OK_AND_ASSIGN(auto extracted_data, client->ProcessResponse(response));

  // 5. Verification Phase
  ASSERT_EQ(extracted_data.size(),
            static_cast<size_t>(p.degree) * p.entry_size_multiple);
  size_t offset = target_index * p.degree * p.entry_size_multiple;
  for (size_t i = 0; i < extracted_data.size(); ++i) {
    EXPECT_EQ(extracted_data[i], db_raw[offset + i])
        << "Mismatch at fetched index " << i << " (DB offset: " << offset + i
        << ")";
  }
}

TEST_F(PirEndToEndTest,
       ExecutesSuccessfullyWithNonRingDegreeMultipleNumEntries) {
  PirTestParams p;
  // 1000 is divisible by interpolation_degree = 2, but 500 is not divisible by
  // degree = 2048.
  p.num_entries = 1000;
  ASSERT_OK_AND_ASSIGN(auto params, CreateParams(p));

  // 1. Generate a random database
  std::vector<DbDataType> db_raw(static_cast<size_t>(p.num_entries) * p.degree *
                                 p.entry_size_multiple);
  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    db_raw[i] = (*db_prng->Rand64()) % p.plaintext_modulus;
  }

  // 2. Create server and preprocess the database
  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  auto prng_seed = prep_server->GetPrngSeed();

  // Directly pass to online server (in memory)
  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *params, std::move(preprocessed_data))));

  // Test boundaries and random index
  std::vector<int> test_indices = {0, 1, 499, p.num_entries - 1};
  for (int target_index : test_indices) {
    // 3. Client setup
    ASSERT_OK_AND_ASSIGN(auto client_a_prng, CreatePrng(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                          *params, std::move(client_a_prng),
                                          std::move(client_prng)));

    // 4. Request / Response Phase
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
    EXPECT_EQ(request.first_dimension_query.size(),
              p.num_entries / p.interpolation_degree);
    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
    ASSERT_OK_AND_ASSIGN(auto extracted_data,
                         client->ProcessResponse(response));

    // 5. Verification Phase
    ASSERT_EQ(extracted_data.size(),
              static_cast<size_t>(p.degree) * p.entry_size_multiple);
    size_t offset = target_index * p.degree * p.entry_size_multiple;
    for (size_t i = 0; i < extracted_data.size(); ++i) {
      EXPECT_EQ(extracted_data[i], db_raw[offset + i])
          << "Mismatch at fetched index " << i << " (DB offset: " << offset + i
          << ")";
    }
  }
}

std::string SelectorSeedForSlot(uint64_t slot) {
  std::string seed(32, 'S');
  for (int i = 0; i < 8; ++i) seed[24 + i] = static_cast<char>(slot >> (8 * i));
  return seed;
}

int NextPow2(int m) {
  int p = 1;
  while (p < m) p <<= 1;
  return p;
}

TEST_F(PirEndToEndTest,
       TiptoeNonPowerOfTwoDownloadIsBitIdenticalToPaddedAndThreaded) {
  // Exact Tiptoe adapter parameters (d=2048, q=2^52, q1=2^25, q2=2^18, t=1).
  ASSERT_OK_AND_ASSIGN(
      auto rlwe_params,
      RlweParams<CoeffType>::Create(2048, 1ULL << 52, 1ULL << 25, 1ULL << 18,
                                    /*variance=*/40));
  const GadgetParams pack_gadget{19, 2};
  const GadgetParams eval_gadget{19, 3};
  const std::string packing_seed(32, 'K');

  auto MakeParams = [&](int m, size_t cols) {
    return PirParams<CoeffType>::Create(rlwe_params, /*plaintext_modulus=*/65536,
                                        pack_gadget, eval_gadget, m,
                                        /*interpolation_degree=*/1, cols);
  };

  auto RunGeometry = [&](const std::vector<int>& real_rows, size_t c,
                         size_t threads) {
    const size_t p = real_rows.size();
    std::vector<int> pad_rows(p);
    for (size_t s = 0; s < p; ++s) pad_rows[s] = NextPow2(real_rows[s]);

    ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng(std::string(32, 'D')));
    std::vector<std::vector<DbDataType>> raw_real(p), raw_pad(p);
    for (size_t s = 0; s < p; ++s) {
      raw_real[s].resize(static_cast<size_t>(real_rows[s]) * c * 2048);
      for (auto& v : raw_real[s]) {
        ASSERT_OK_AND_ASSIGN(uint64_t r, db_prng->Rand64());
        v = static_cast<DbDataType>(r & 0xFFFFULL);
      }
      raw_pad[s].assign(static_cast<size_t>(pad_rows[s]) * c * 2048, 0);
      std::copy(raw_real[s].begin(), raw_real[s].end(), raw_pad[s].begin());
    }

    auto BuildServer = [&](size_t slot, int m, const std::vector<DbDataType>& db,
                           size_t k0, size_t k1)
        -> absl::StatusOr<
            std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>> {
      std::vector<DbDataType> slice;
      if (k1 - k0 < c) {
        slice.reserve(static_cast<size_t>(m) * (k1 - k0) * 2048);
        for (int row = 0; row < m; ++row) {
          slice.insert(slice.end(), db.begin() + (row * c + k0) * 2048,
                       db.begin() + (row * c + k1) * 2048);
        }
      }
      ASSIGN_OR_RETURN(auto params, MakeParams(m, k1 - k0));
      ASSIGN_OR_RETURN(
          auto prep,
          (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
              &params, SelectorSeedForSlot(slot))));
      ASSIGN_OR_RETURN(auto data, prep->Preprocess(k1 - k0 < c ? slice : db,
                                                   packing_seed));
      return PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(data));
    };

    std::vector<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
        servers_real(p), servers_pad(p);
    for (size_t s = 0; s < p; ++s) {
      ASSERT_OK_AND_ASSIGN(servers_real[s],
                           BuildServer(s, real_rows[s], raw_real[s], 0, c));
      ASSERT_OK_AND_ASSIGN(servers_pad[s],
                           BuildServer(s, pad_rows[s], raw_pad[s], 0, c));
    }

    // Threaded decomposition: G = min(P, T) color groups x H = min(C, T / G)
    // column groups.
    const size_t color_groups = std::min(p, threads);
    const size_t column_groups = std::min(c, threads / color_groups);
    struct Task {
      size_t p0, p1, k0, k1;
      std::vector<
          std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
          servers;
    };
    std::vector<Task> tasks;
    for (size_t g = 0; g < color_groups; ++g) {
      for (size_t h = 0; h < column_groups; ++h) {
        Task task{g * p / color_groups, (g + 1) * p / color_groups,
                  h * c / column_groups, (h + 1) * c / column_groups, {}};
        for (size_t s = task.p0; s < task.p1; ++s) {
          ASSERT_OK_AND_ASSIGN(
              auto srv,
              BuildServer(s, real_rows[s], raw_real[s], task.k0, task.k1));
          task.servers.push_back(std::move(srv));
        }
        tasks.push_back(std::move(task));
      }
    }

    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(11));
    ASSERT_OK_AND_ASSIGN(auto client_params, MakeParams(2048, c));

    // Test 3 queries: first row (0), last row (R_i - 1), and midpoint (R_i / 2).
    for (int q_idx = 0; q_idx < 3; ++q_idx) {
      std::vector<int> targets(p);
      for (size_t s = 0; s < p; ++s) {
        if (q_idx == 0) {
          targets[s] = 0;
        } else if (q_idx == 1) {
          targets[s] = real_rows[s] - 1;
        } else {
          targets[s] = real_rows[s] / 2;
        }
      }

      const std::string client_seed(32, static_cast<char>('A' + q_idx));
      auto EncodeRequest = [&](const std::vector<int>& rows_vec,
                               std::vector<std::string>* selector_wires)
          -> absl::StatusOr<std::string> {
        ASSIGN_OR_RETURN(auto a_prng, CreatePrng(SelectorSeedForSlot(0)));
        ASSIGN_OR_RETURN(auto s_prng, CreatePrng(client_seed));
        ASSIGN_OR_RETURN(auto client,
                         PirClient<CoeffType>::Create(
                             client_params, std::move(a_prng),
                             std::move(s_prng)));
        ASSIGN_OR_RETURN(auto key_prng, CreatePrng(packing_seed));
        ASSIGN_OR_RETURN(auto keys, client->CreatePackingKey(key_prng.get()));
        std::string upload;
        for (const auto& key : keys) {
          Polynomial<CoeffType>::PackCoefficientsAppend(key.Coeffs(), 52,
                                                        upload);
        }
        EXPECT_EQ(upload.size(), 53248u);
        for (size_t s = 0; s < p; ++s) {
          ASSIGN_OR_RETURN(auto mask, CreatePrng(SelectorSeedForSlot(s)));
          ASSIGN_OR_RETURN(auto sel,
                           client->CreateSelector(targets[s], mask.get()));
          sel.resize(rows_vec[s]);
          for (uint64_t& y : sel) {
            y = ((y + (1ULL << 11)) >> 12) & ((1ULL << 40) - 1);
          }
          std::string sel_wire =
              Polynomial<CoeffType>::PackCoefficients(sel, 40);
          EXPECT_EQ(sel_wire.size(), static_cast<size_t>(rows_vec[s]) * 5);
          if (selector_wires) selector_wires->push_back(sel_wire);
          upload.append(sel_wire);
        }
        return upload;
      };

      std::vector<std::string> sel_real, sel_pad;
      ASSERT_OK_AND_ASSIGN(std::string upload_real,
                           EncodeRequest(real_rows, &sel_real));
      ASSERT_OK_AND_ASSIGN(std::string upload_pad,
                           EncodeRequest(pad_rows, &sel_pad));

      size_t sum_real = 0, sum_pad = 0;
      for (size_t s = 0; s < p; ++s) {
        sum_real += real_rows[s];
        sum_pad += pad_rows[s];
        // Real-row selector wire is the exact 5*R_i byte prefix of the padded
        // selector wire.
        EXPECT_EQ(sel_real[s], sel_pad[s].substr(0, sel_real[s].size()));
      }
      EXPECT_EQ(upload_real.size(), 53248 + 5 * sum_real);
      EXPECT_EQ(upload_pad.size(), 53248 + 5 * sum_pad);
      // Key bundles are bit-identical.
      EXPECT_EQ(upload_real.substr(0, 53248), upload_pad.substr(0, 53248));

      auto DecodeAndAnswer =
          [&](const std::string& upload, const std::vector<int>& rows_vec,
              std::vector<std::unique_ptr<
                  PirServer<DbDataType, CoeffType, MatCoeffType>>>& servers)
          -> absl::StatusOr<std::string> {
        size_t off = 0;
        PirRequest<CoeffType> req;
        for (int k = 0; k < 4; ++k) {
          const size_t nbytes = (2048 * 52 + 7) / 8;
          ASSIGN_OR_RETURN(
              auto coeffs,
              Polynomial<CoeffType>::UnpackCoefficients(
                  absl::string_view(upload.data() + off, nbytes), 2048, 52));
          off += nbytes;
          ASSIGN_OR_RETURN(auto poly,
                           Polynomial<CoeffType>::Create(std::move(coeffs)));
          req.packing_key.push_back(std::move(poly));
        }
        std::vector<std::vector<CoeffType>> queries(p);
        for (size_t s = 0; s < p; ++s) {
          const size_t nbytes = (static_cast<size_t>(rows_vec[s]) * 40 + 7) / 8;
          ASSIGN_OR_RETURN(queries[s],
                           Polynomial<CoeffType>::UnpackCoefficients(
                               absl::string_view(upload.data() + off, nbytes),
                               rows_vec[s], 40));
          off += nbytes;
          for (uint64_t& y : queries[s]) y <<= 12;
        }
        EXPECT_EQ(off, upload.size());
        const std::vector<Polynomial<CoeffType>> y_vec_h(
            req.packing_key.begin() + pack_gadget.num_digits,
            req.packing_key.end());
        ASSIGN_OR_RETURN(ChunkedFft y_vec_h_fft,
                         YVecHFft(rlwe_params, y_vec_h, *fft_ctx));
        std::string download;
        for (size_t s = 0; s < p; ++s) {
          req.first_dimension_query = queries[s];
          ASSIGN_OR_RETURN(
              auto resp,
              servers[s]->ProcessResponse(req, &y_vec_h_fft,
                                          /*skip_a_t1=*/true));
          for (size_t k = 0; k < resp.ciphertexts.size(); ++k) {
            download.append(servers[s]->GetPrepackedAModSwitched(k));
            Polynomial<CoeffType>::PackCoefficientsAppend(
                resp.ciphertexts[k].b.Coeffs(), 18, download);
          }
        }
        return download;
      };

      ASSERT_OK_AND_ASSIGN(std::string download_real,
                           DecodeAndAnswer(upload_real, real_rows,
                                           servers_real));
      ASSERT_OK_AND_ASSIGN(std::string download_pad,
                           DecodeAndAnswer(upload_pad, pad_rows, servers_pad));

      // Also execute the threaded color-group x column-group path on real_rows.
      size_t off = 0;
      PirRequest<CoeffType> req;
      for (int k = 0; k < 4; ++k) {
        const size_t nbytes = (2048 * 52 + 7) / 8;
        ASSERT_OK_AND_ASSIGN(
            auto coeffs,
            Polynomial<CoeffType>::UnpackCoefficients(
                absl::string_view(upload_real.data() + off, nbytes), 2048, 52));
        off += nbytes;
        ASSERT_OK_AND_ASSIGN(auto poly,
                             Polynomial<CoeffType>::Create(std::move(coeffs)));
        req.packing_key.push_back(std::move(poly));
      }
      std::vector<std::vector<CoeffType>> queries(p);
      for (size_t s = 0; s < p; ++s) {
        const size_t nbytes = (static_cast<size_t>(real_rows[s]) * 40 + 7) / 8;
        ASSERT_OK_AND_ASSIGN(
            queries[s],
            Polynomial<CoeffType>::UnpackCoefficients(
                absl::string_view(upload_real.data() + off, nbytes),
                real_rows[s], 40));
        off += nbytes;
        for (uint64_t& y : queries[s]) y <<= 12;
      }
      const std::vector<Polynomial<CoeffType>> y_vec_h(
          req.packing_key.begin() + pack_gadget.num_digits,
          req.packing_key.end());
      ASSERT_OK_AND_ASSIGN(ChunkedFft y_vec_h_fft,
                           YVecHFft(rlwe_params, y_vec_h, *fft_ctx));
      std::vector<std::string> wires(p * column_groups);
      std::vector<std::thread> workers;
      for (size_t w = 0; w < tasks.size(); ++w) {
        workers.emplace_back([&, w] {
          auto task_req = req;
          for (size_t s = tasks[w].p0; s < tasks[w].p1; ++s) {
            task_req.first_dimension_query = queries[s];
            auto& srv = tasks[w].servers[s - tasks[w].p0];
            auto resp = srv->ProcessResponse(task_req, &y_vec_h_fft,
                                             /*skip_a_t1=*/true)
                            .value();
            std::string wire;
            for (size_t k = 0; k < resp.ciphertexts.size(); ++k) {
              wire.append(srv->GetPrepackedAModSwitched(k));
              Polynomial<CoeffType>::PackCoefficientsAppend(
                  resp.ciphertexts[k].b.Coeffs(), 18, wire);
            }
            wires[s * column_groups + w % column_groups] = std::move(wire);
          }
        });
      }
      for (auto& th : workers) th.join();
      std::string download_threaded;
      for (const auto& wire : wires) download_threaded.append(wire);

      EXPECT_EQ(download_real.size(), 11008 * p * c);
      EXPECT_EQ(download_real, download_pad);
      EXPECT_EQ(download_threaded, download_pad);

      // Verify decryption of download_real matches target rows in raw_real.
      ASSERT_OK_AND_ASSIGN(auto a_prng, CreatePrng(SelectorSeedForSlot(0)));
      ASSERT_OK_AND_ASSIGN(auto s_prng, CreatePrng(client_seed));
      ASSERT_OK_AND_ASSIGN(
          auto client,
          PirClient<CoeffType>::Create(client_params, std::move(a_prng),
                                       std::move(s_prng)));
      size_t d_off = 0;
      for (size_t s = 0; s < p; ++s) {
        PirResponse<CoeffType> resp;
        for (size_t col = 0; col < c; ++col) {
          ASSERT_OK_AND_ASSIGN(
              auto a_coeffs,
              Polynomial<CoeffType>::UnpackCoefficients(
                  absl::string_view(download_real.data() + d_off, 6400), 2048,
                  25));
          d_off += 6400;
          ASSERT_OK_AND_ASSIGN(
              auto b_coeffs,
              Polynomial<CoeffType>::UnpackCoefficients(
                  absl::string_view(download_real.data() + d_off, 4608), 2048,
                  18));
          d_off += 4608;
          ASSERT_OK_AND_ASSIGN(auto a_poly, Polynomial<CoeffType>::Create(
                                                std::move(a_coeffs)));
          ASSERT_OK_AND_ASSIGN(auto b_poly, Polynomial<CoeffType>::Create(
                                                std::move(b_coeffs)));
          resp.ciphertexts.push_back({std::move(a_poly), std::move(b_poly)});
        }
        ASSERT_OK_AND_ASSIGN(auto recovered, client->ProcessResponse(resp));
        ASSERT_EQ(recovered.size(), c * 2048);
        for (size_t i = 0; i < c * 2048; ++i) {
          EXPECT_EQ(recovered[i],
                    raw_real[s][static_cast<size_t>(targets[s]) * c * 2048 + i])
              << "Decryption mismatch at slot=" << s << " q_idx=" << q_idx;
        }
      }
      EXPECT_EQ(d_off, download_real.size());
    }
  };

  // Case 1: (1311, 1000; C=2) with 4 threads (2 color groups x 2 column groups)
  RunGeometry({1311, 1000}, /*c=*/2, /*threads=*/4);
  // Case 2: (874, 239, 33, 31, 1; C=3) with 8 threads (5 color groups x 1 col)
  RunGeometry({874, 239, 33, 31, 1}, /*c=*/3, /*threads=*/8);
}

}  // namespace
}  // namespace pir
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
