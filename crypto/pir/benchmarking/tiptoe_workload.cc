// Copyright 2026. Apache-2.0; uses the Google ReInsPIRe implementation.
// A real multi-color t=1, p=65536 request, with a fresh shared secret per
// request.
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "hwy/highway.h"
#include "hwy/targets.h"
#include "shell_encryption/prng/chacha_prng.h"

namespace pm = private_membership::rlwe::v2;
using Clock = std::chrono::steady_clock;
using Server = pm::PirServer<uint16_t, uint64_t, int32_t>;
using Prep = pm::PreprocessingServer<uint16_t, uint64_t, int32_t>;
using Poly = pm::Polynomial<uint64_t>;
template <class T>
T Take(absl::StatusOr<T> result) {
  if (!result.ok()) throw std::runtime_error(result.status().ToString());
  return std::move(result).value();
}
void Check(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
double Ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
uint64_t Read64(std::istream& input) {
  unsigned char b[8];
  input.read(reinterpret_cast<char*>(b), 8);
  Check(bool(input), "Truncated workload header");
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(b[i]) << (8 * i);
  return v;
}
std::string SelectorSeed(uint64_t slot) {
  std::string seed(32, 'S');
  for (int i = 0; i < 8; ++i) seed[24 + i] = char(slot >> (8 * i));
  return seed;
}
std::unique_ptr<::rlwe::SecurePrng> Prng(const std::string& seed) {
  return Take(::rlwe::ChaChaPrng::Create(seed));
}
void Encode(const Poly& poly, int bits, std::string& wire) {
  wire += Take(poly.ToProto(bits)).SerializeAsString();
}
// Each upstream polynomial protobuf is a bytes field: tag, varint, coefficient
// bytes. Its own length delimiter lets a complete request use concatenated
// components.
Poly Decode(const std::string& wire, size_t& off, int degree, int bits) {
  const size_t start = off;
  Check(off < wire.size() && uint8_t(wire[off++]) == 10,
        "Invalid polynomial tag");
  size_t len = 0;
  int shift = 0;
  while (true) {
    Check(off < wire.size() && shift < 63, "Invalid polynomial length");
    uint8_t byte = wire[off++];
    len |= size_t(byte & 127) << shift;
    if (!(byte & 128)) break;
    shift += 7;
  }
  Check(len == (size_t(degree) * bits + 7) / 8 && len <= wire.size() - off,
        "Invalid polynomial payload");
  off += len;
  pm::proto::Polynomial encoded;
  Check(encoded.ParseFromArray(wire.data() + start, off - start),
        "Polynomial parse failed");
  return Take(Poly::CreateFromProto(encoded, degree, bits));
}
int Run(int argc, char** argv) {
  Check(argc == 4, "usage: reinspire-workload WORKLOAD.bin WARMUPS REPEATS");
  Check(std::string(hwy::TargetName(HWY_STATIC_TARGET)) == "AVX2",
        "AVX2 backend required");
  const int warmups = std::stoi(argv[2]), repeats = std::stoi(argv[3]);
  Check(warmups >= 0 && repeats > 0, "Invalid repetitions");
  std::ifstream input(argv[1], std::ios::binary);
  char magic[8];
  input.read(magic, 8);
  Check(std::string(magic, 8) == "TPIRv001", "Invalid workload version");
  const size_t p = Read64(input), q = Read64(input), c = Read64(input);
  Check(p > 0 && q > 0 && c > 0 && p < 100000 && c < 100000,
        "Invalid workload dimensions");
  std::vector<int> rows(p);
  for (auto& m : rows) {
    m = Read64(input);
    Check(m > 0 && m <= 2048 && !(m & (m - 1)), "Unsupported M");
  }
  std::vector<size_t> targets(p * q);
  for (size_t i = 0; i < targets.size(); ++i) {
    targets[i] = Read64(input);
    Check(targets[i] < size_t(rows[i % p]), "Invalid target");
  }
  const auto rlwe_params = Take(pm::RlweParams<uint64_t>::Create(
      2048, 1ULL << 52, 1ULL << 29, 1ULL << 18, 40));
  const pm::GadgetParams pack_gadget{19, 2}, eval_gadget{19, 3};
  auto Params = [&](int m) {
    return Take(pm::PirParams<uint64_t>::Create(rlwe_params, 65536, pack_gadget,
                                                eval_gadget, m, 1, c));
  };
  const std::string packing_seed(32, 'K');
  std::vector<std::vector<uint16_t>> raw(p);
  std::vector<std::unique_ptr<Server>> servers(p);
  for (size_t slot = 0; slot < p; ++slot) {
    raw[slot].resize(size_t(rows[slot]) * c * 2048);
    // Both native harnesses consume the same little-endian uint16 payloads.
    std::vector<unsigned char> bytes(raw[slot].size() * 2);
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    Check(bool(input), "Truncated database");
    for (size_t i = 0; i < raw[slot].size(); ++i)
      raw[slot][i] = uint16_t(bytes[2 * i]) | (uint16_t(bytes[2 * i + 1]) << 8);
  }
  // Independent fixed-mask preprocessing can run concurrently. All online
  // requests below still execute sequentially on one server thread.
  const char* prep_env = std::getenv("PIR_PREPROCESS_THREADS");
  const int requested_threads = prep_env ? std::stoi(prep_env) : 8;
  Check(requested_threads > 0 && requested_threads <= 64,
        "PIR_PREPROCESS_THREADS must be in [1,64]");
  const size_t offline_threads = std::min(size_t(requested_threads), p);
  auto offline_begin = Clock::now();
  for (size_t first = 0; first < p; first += offline_threads) {
    std::vector<std::future<void>> jobs;
    for (size_t slot = first; slot < std::min(p, first + offline_threads);
         ++slot) {
      jobs.push_back(std::async(std::launch::async, [&, slot] {
        auto params = Params(rows[slot]);
        auto prep = Take(Prep::Create(&params, SelectorSeed(slot)));
        auto data = Take(prep->Preprocess(raw[slot], packing_seed));
        servers[slot] = Take(Server::Create(params, std::move(data)));
      }));
    }
    for (auto& job : jobs) job.get();
  }
  const double offline_ms = Ms(offline_begin);
  Check(input.peek() == std::char_traits<char>::eof(),
        "Trailing workload data");
  const char* cpu_env = std::getenv("PIR_ONLINE_CPU");
  const int online_cpu = cpu_env ? std::stoi(cpu_env) : -1;
  if (cpu_env) {
    Check(online_cpu >= 0 && online_cpu < CPU_SETSIZE,
          "Invalid PIR_ONLINE_CPU");
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(online_cpu, &mask);
    Check(sched_setaffinity(0, sizeof(mask), &mask) == 0,
          "Failed to bind online execution to PIR_ONLINE_CPU");
  }
  std::cout << "{\"type\":\"setup\",\"protocol\":\"reinspire\",\"backend\":"
               "\"avx2\",\"online_threads\":1,\"plaintext_modulus\":65536,"
               "\"interpolation_degree\":1,\"offline_ms\":"
            << offline_ms << ",\"online_cpu\":" << online_cpu
            << ",\"offline_threads\":" << offline_threads << ",\"P\":" << p
            << ",\"Q\":" << q << ",\"C\":" << c << ",\"rows\":[";
  for (size_t i = 0; i < p; ++i) std::cout << (i ? "," : "") << rows[i];
  std::cout << "]}" << std::endl;
  for (size_t trial = 0; trial < size_t(warmups) + q * repeats; ++trial) {
    const bool warmup = trial < size_t(warmups);
    const size_t qi = warmup ? trial % q : (trial - warmups) / repeats;
    const int rep = warmup ? -1 : (trial - warmups) % repeats;
    const auto client_begin = Clock::now();
    auto client = Take(pm::PirClient<uint64_t>::Create(
        Params(2048), Prng(SelectorSeed(0)),
        Prng(Take(::rlwe::ChaChaPrng::GenerateSeed()))));
    auto key_prng = Prng(packing_seed);
    auto keys = Take(client->CreatePackingKey(key_prng.get()));
    Check(keys.size() == 4, "Unexpected packing key count");
    const double client_key_generation_ms = Ms(client_begin);
    auto t_pack = Clock::now();
    std::string upload;
    for (const auto& key : keys) Encode(key, 52, upload);
    const size_t key_bytes = upload.size();
    double client_request_pack_ms = Ms(t_pack), client_selector_ms = 0;
    for (size_t slot = 0; slot < p; ++slot) {
      auto t = Clock::now();
      auto mask = Prng(SelectorSeed(slot));
      auto selector =
          Take(client->CreateSelector(targets[qi * p + slot], mask.get()));
      selector.resize(rows[slot]);
      client_selector_ms += Ms(t);
      t = Clock::now();
      Encode(Take(Poly::Create(std::move(selector))), 52, upload);
      client_request_pack_ms += Ms(t);
    }
    const double client_query_ms = Ms(client_begin);
    const auto server_begin = Clock::now();
    size_t off = 0;
    // One serialized key bundle decoded once, shared by all color servers.
    pm::PirRequest<uint64_t> request;
    for (int k = 0; k < 4; ++k)
      request.packing_key.push_back(Decode(upload, off, 2048, 52));
    double ingest_ms = Ms(server_begin), db_ms = 0, packing_ms = 0,
           encode_ms = 0;
    double matrix_ms = 0, finalize_ms = 0, modswitch_ms = 0, poly_eval_ms = 0;
    std::string download;
    for (size_t slot = 0; slot < p; ++slot) {
      auto t = Clock::now();
      request.first_dimension_query =
          Decode(upload, off, rows[slot], 52).Coeffs();
      ingest_ms += Ms(t);
      auto& server = servers[slot];
      auto db_before = server->GetDbMultiplyTime(),
           pack_before = server->GetPackingTime();
      const auto matrix_before = server->GetPackMatrixTime(),
                 finalize_before = server->GetFinalizeTime();
      const auto modswitch_before = server->GetModswitchTime(),
                 eval_before = server->GetPolyEvalTime();
      auto response = Take(server->ProcessResponse(request));
      matrix_ms += absl::ToDoubleMilliseconds(server->GetPackMatrixTime() -
                                              matrix_before);
      finalize_ms += absl::ToDoubleMilliseconds(server->GetFinalizeTime() -
                                                finalize_before);
      modswitch_ms += absl::ToDoubleMilliseconds(server->GetModswitchTime() -
                                                 modswitch_before);
      poly_eval_ms +=
          absl::ToDoubleMilliseconds(server->GetPolyEvalTime() - eval_before);
      db_ms +=
          absl::ToDoubleMilliseconds(server->GetDbMultiplyTime() - db_before);
      packing_ms +=
          absl::ToDoubleMilliseconds(server->GetPackingTime() - pack_before);
      t = Clock::now();
      for (const auto& ct : response.ciphertexts) {
        Encode(ct.a, 29, download);
        Encode(ct.b, 18, download);
      }
      encode_ms += Ms(t);
    }
    Check(off == upload.size(), "Unconsumed request bytes");
    const double server_ms = Ms(server_begin);
    const auto decode_begin = Clock::now();
    off = 0;
    std::vector<std::vector<uint64_t>> recovered;
    double client_response_unpack_ms = 0, client_decrypt_ms = 0;
    for (size_t slot = 0; slot < p; ++slot) {
      auto t = Clock::now();
      pm::PirResponse<uint64_t> response;
      for (size_t col = 0; col < c; ++col) {
        auto a = Decode(download, off, 2048, 29);
        auto b = Decode(download, off, 2048, 18);
        response.ciphertexts.push_back({std::move(a), std::move(b)});
      }
      client_response_unpack_ms += Ms(t);
      t = Clock::now();
      recovered.push_back(Take(client->ProcessResponse(response)));
      client_decrypt_ms += Ms(t);
    }
    Check(off == download.size(), "Unconsumed response bytes");
    const double client_decode_ms = Ms(decode_begin);
    const double protocol_roundtrip_ms = Ms(client_begin);
    const auto verify_begin = Clock::now();
    for (size_t slot = 0; slot < p; ++slot) {
      Check(recovered[slot].size() == c * 2048, "Invalid response length");
      for (size_t i = 0; i < c * 2048; ++i)
        Check(recovered[slot][i] ==
                  raw[slot][targets[qi * p + slot] * c * 2048 + i],
              "Decryption mismatch");
    }
    const double client_verify_ms = Ms(verify_begin);
    std::cout << "{\"type\":\"query\",\"protocol\":\"reinspire\",\"correct\":"
                 "true,\"query_index\":"
              << qi << ",\"repeat\":" << rep
              << ",\"warmup\":" << (warmup ? "true" : "false")
              << ",\"server_ms\":" << server_ms
              << ",\"client_query_ms\":" << client_query_ms
              << ",\"client_decode_ms\":" << client_decode_ms
              << ",\"server_ingest_ms\":" << ingest_ms
              << ",\"server_inner_product_ms\":" << db_ms
              << ",\"server_packing_ms\":" << packing_ms
              << ",\"server_encode_ms\":" << encode_ms
              << ",\"client_key_generation_ms\":" << client_key_generation_ms
              << ",\"client_selector_ms\":" << client_selector_ms
              << ",\"client_request_pack_ms\":" << client_request_pack_ms
              << ",\"client_response_unpack_ms\":" << client_response_unpack_ms
              << ",\"client_decrypt_ms\":" << client_decrypt_ms
              << ",\"client_verify_ms\":" << client_verify_ms
              << ",\"protocol_roundtrip_ms\":" << protocol_roundtrip_ms
              << ",\"server_request_unpack_ms\":" << ingest_ms
              << ",\"server_pack_matrix_ms\":" << matrix_ms
              << ",\"server_pack_finalize_ms\":" << finalize_ms
              << ",\"server_modswitch_ms\":" << modswitch_ms
              << ",\"server_poly_eval_ms\":" << poly_eval_ms
              << ",\"server_other_ms\":"
              << server_ms - ingest_ms - db_ms - packing_ms - modswitch_ms -
                     poly_eval_ms - encode_ms
              << ",\"key_bundle_upload_bytes\":" << key_bytes
              << ",\"upload_bytes\":" << upload.size()
              << ",\"download_bytes\":" << download.size()
              << ",\"total_communication_bytes\":"
              << upload.size() + download.size() << "}" << std::endl;
  }
  return 0;
}
int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
