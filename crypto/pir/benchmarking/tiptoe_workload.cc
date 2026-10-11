// Copyright 2026. Apache-2.0; uses the Google ReInsPIRe implementation.
// A real multi-color t=1, p=65536 request, with a fresh shared secret per
// request. With PIR_THREAD_CPUS, pinned worker threads answer it as color
// group x column group tasks (see improvements.md).
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial_fft.h"
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
// Lock-free single-threaded ChaCha20 PRNG producing the exact same byte stream
// as ::rlwe::ChaChaPrng without per-call mutex locking or byte-by-byte loops.
class FastChaChaPrng final : public ::rlwe::SecurePrng {
 public:
  explicit FastChaChaPrng(std::string key) : key_(std::move(key)) {}

  absl::StatusOr<::rlwe::Uint8> Rand8() override {
    if (position_in_buffer_ >= static_cast<int>(buffer_.size())) {
      RETURN_IF_ERROR(::rlwe::internal::ChaChaPrngResalt(
          key_, ::rlwe::internal::kChaChaOutputBytes, &salt_counter_,
          &position_in_buffer_, &buffer_));
    }
    return buffer_[position_in_buffer_++];
  }

  absl::StatusOr<::rlwe::Uint64> Rand64() override {
    if (position_in_buffer_ + 8 > static_cast<int>(buffer_.size())) {
      if (position_in_buffer_ == static_cast<int>(buffer_.size())) {
        RETURN_IF_ERROR(::rlwe::internal::ChaChaPrngResalt(
            key_, ::rlwe::internal::kChaChaOutputBytes, &salt_counter_,
            &position_in_buffer_, &buffer_));
      } else {
        return ::rlwe::internal::ChaChaPrngRand64(
            key_, &position_in_buffer_, &salt_counter_, &buffer_);
      }
    }
    ::rlwe::Uint64 rand64;
    std::memcpy(&rand64, buffer_.data() + position_in_buffer_, sizeof(rand64));
    position_in_buffer_ += 8;
    return rand64;
  }

 private:
  const std::string key_;
  int position_in_buffer_ = 0;
  int salt_counter_ = 0;
  std::vector<::rlwe::Uint8> buffer_;
};
std::unique_ptr<::rlwe::SecurePrng> Prng(const std::string& seed) {
  Check(static_cast<int>(seed.size()) == ::rlwe::internal::kChaChaKeyBytesSize,
        "Invalid ChaCha PRNG seed size");
  return std::make_unique<FastChaChaPrng>(seed);
}
void PackCoeffsAppend(const std::vector<uint64_t>& coeffs, int bits,
                      std::string& wire) {
  Poly::PackCoefficientsAppend(coeffs, bits, wire);
}

std::vector<uint64_t> UnpackCoeffs(const std::string& wire, size_t& off,
                                   int degree, int bits) {
  const size_t num_bytes = (static_cast<size_t>(degree) * bits + 7) / 8;
  Check(num_bytes <= wire.size() - off, "Invalid polynomial payload");
  absl::string_view slice(wire.data() + off, num_bytes);
  off += num_bytes;
  return Take(Poly::UnpackCoefficients(slice, degree, bits));
}
void Pin(int cpu, const char* error) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  Check(sched_setaffinity(0, sizeof(mask), &mask) == 0, error);
}
// PIR_THREAD_CPUS: comma-separated logical CPU ids, unique and inside the
// process affinity mask. Worker w runs on the w-th.
std::vector<int> ThreadCpus() {
  const char* env = std::getenv("PIR_THREAD_CPUS");
  if (!env || !*env) return {};
  Check(!std::getenv("PIR_ONLINE_CPU"),
        "PIR_THREAD_CPUS and PIR_ONLINE_CPU are mutually exclusive");
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  Check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0,
        "Failed to read CPU affinity mask");
  std::vector<int> cpus;
  for (const char* p = env;;) {
    char* end = nullptr;
    const long v = std::strtol(p, &end, 10);
    Check(end != p && (*end == ',' || *end == '\0') && v >= 0 &&
              v < CPU_SETSIZE,
          "Invalid CPU id in PIR_THREAD_CPUS");
    Check(CPU_ISSET(int(v), &allowed),
          "PIR_THREAD_CPUS entry is outside the process affinity mask");
    Check(std::find(cpus.begin(), cpus.end(), int(v)) == cpus.end(),
          "Duplicate CPU in PIR_THREAD_CPUS");
    cpus.push_back(int(v));
    if (*end == '\0') break;
    p = end + 1;
  }
  return cpus;
}
// Persistent workers; the calling thread is worker 0.
class Workers {
 public:
  explicit Workers(size_t n) : errors_(n) {
    for (size_t w = 1; w < n; ++w)
      threads_.emplace_back([this, w] { Loop(w); });
  }
  ~Workers() {
    Start(nullptr);
    for (auto& t : threads_) t.join();
  }
  void Run(const std::function<void(size_t)>& job) {
    Start(&job);
    Call(0);
    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [this] { return busy_ == 0; });
    for (const auto& error : errors_)
      if (error) std::rethrow_exception(error);
  }

 private:
  void Start(const std::function<void(size_t)>* job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = job;
      busy_ = threads_.size();
      ++round_;
    }
    start_.notify_all();
  }
  void Call(size_t w) {
    try {
      (*job_)(w);
    } catch (...) {
      errors_[w] = std::current_exception();
    }
  }
  void Loop(size_t w) {
    for (size_t round = 1;; ++round) {
      std::unique_lock<std::mutex> lock(mu_);
      start_.wait(lock, [&] { return round_ >= round; });
      if (!job_) return;
      lock.unlock();
      Call(w);
      lock.lock();
      if (--busy_ == 0) done_.notify_one();
    }
  }
  std::mutex mu_;
  std::condition_variable start_, done_;
  const std::function<void(size_t)>* job_ = nullptr;
  size_t round_ = 0, busy_ = 0;
  std::vector<std::exception_ptr> errors_;
  std::vector<std::thread> threads_;
};
// Colors [p0, p1) x columns [k0, k1) of every request, with one server per
// color holding just those columns.
struct Task {
  size_t p0, p1, k0, k1;
  std::vector<std::unique_ptr<Server>> servers;
};
// One task's server time for one request, in ms.
struct Phases {
  double db = 0, packing = 0, matrix = 0, finalize = 0, modswitch = 0,
         poly_eval = 0, encode = 0, busy = 0;
};
int Run(int argc, char** argv) {
  Check(argc == 4, "usage: reinspire-workload WORKLOAD.bin WARMUPS REPEATS");
  Check(std::string(hwy::TargetName(HWY_STATIC_TARGET)) == "AVX2",
        "AVX2 backend required");
  const std::vector<int> cpus = ThreadCpus();
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
    Check(m > 0 && m <= 2048, "Unsupported M");
  }
  std::vector<size_t> targets(p * q);
  for (size_t i = 0; i < targets.size(); ++i) {
    targets[i] = Read64(input);
    Check(targets[i] < size_t(rows[i % p]), "Invalid target");
  }
  const auto rlwe_params = Take(pm::RlweParams<uint64_t>::Create(
      2048, 1ULL << 52, 1ULL << 25, 1ULL << 18, 40));
  const pm::GadgetParams pack_gadget{19, 2}, eval_gadget{19, 3};
  auto Params = [&](int m, size_t cols) {
    return Take(pm::PirParams<uint64_t>::Create(rlwe_params, 65536, pack_gadget,
                                                eval_gadget, m, 1, cols));
  };
  const std::string packing_seed(32, 'K');
  std::vector<std::vector<uint16_t>> raw(p);
  for (size_t slot = 0; slot < p; ++slot) {
    raw[slot].resize(size_t(rows[slot]) * c * 2048);
    // Both native harnesses consume the same little-endian uint16 payloads.
    std::vector<unsigned char> bytes(raw[slot].size() * 2);
    input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    Check(bool(input), "Truncated database");
    for (size_t i = 0; i < raw[slot].size(); ++i)
      raw[slot][i] = uint16_t(bytes[2 * i]) | (uint16_t(bytes[2 * i + 1]) << 8);
  }
  // The server for columns [k0, k1) of table `slot`. A column is one
  // 2048-coefficient plaintext of every row; per-column preprocessing does not
  // depend on the other columns, so its responses match the full table's.
  auto Build = [&](size_t slot, size_t k0, size_t k1) {
    std::vector<uint16_t> slice;
    if (k1 - k0 < c) {
      for (size_t row = 0; row < size_t(rows[slot]); ++row)
        slice.insert(slice.end(), raw[slot].begin() + (row * c + k0) * 2048,
                     raw[slot].begin() + (row * c + k1) * 2048);
    }
    auto params = Params(rows[slot], k1 - k0);
    auto prep = Take(Prep::Create(&params, SelectorSeed(slot)));
    auto data =
        Take(prep->Preprocess(k1 - k0 < c ? slice : raw[slot], packing_seed));
    return Take(Server::Create(params, std::move(data)));
  };
  // Task (g, h) covers color group g x column group h, both balanced
  // contiguous splits, and runs on worker w = g * H + h.
  const size_t threads = std::max<size_t>(cpus.size(), 1);
  const size_t color_groups = std::min(p, threads);
  const size_t column_groups = std::min(c, threads / color_groups);
  std::vector<Task> tasks;
  for (size_t g = 0; g < color_groups; ++g)
    for (size_t h = 0; h < column_groups; ++h)
      tasks.push_back({g * p / color_groups, (g + 1) * p / color_groups,
                       h * c / column_groups, (h + 1) * c / column_groups});
  Workers workers(tasks.size());
  size_t offline_threads = tasks.size();
  const auto offline_begin = Clock::now();
  if (!cpus.empty()) {
    // Owner computes: each worker pins itself and builds its own task's
    // servers, so their memory is local to the CPU that serves them.
    workers.Run([&](size_t w) {
      Pin(cpus[w], "Failed to pin a worker to its PIR_THREAD_CPUS entry");
      for (size_t slot = tasks[w].p0; slot < tasks[w].p1; ++slot)
        tasks[w].servers.push_back(Build(slot, tasks[w].k0, tasks[w].k1));
    });
  } else {
    // Independent fixed-mask preprocessing can run concurrently.
    const char* prep_env = std::getenv("PIR_PREPROCESS_THREADS");
    const int requested_threads = prep_env ? std::stoi(prep_env) : 8;
    Check(requested_threads > 0 && requested_threads <= 64,
          "PIR_PREPROCESS_THREADS must be in [1,64]");
    offline_threads = std::min(size_t(requested_threads), p);
    tasks[0].servers.resize(p);
    for (size_t first = 0; first < p; first += offline_threads) {
      std::vector<std::future<void>> jobs;
      for (size_t slot = first; slot < std::min(p, first + offline_threads);
           ++slot) {
        jobs.push_back(std::async(std::launch::async, [&, slot] {
          tasks[0].servers[slot] = Build(slot, 0, c);
        }));
      }
      for (auto& job : jobs) job.get();
    }
  }
  const double offline_ms = Ms(offline_begin);
  Check(input.peek() == std::char_traits<char>::eof(),
        "Trailing workload data");
  const char* cpu_env = std::getenv("PIR_ONLINE_CPU");
  const int online_cpu = cpu_env ? std::stoi(cpu_env) : -1;
  if (cpu_env) {
    Check(online_cpu >= 0 && online_cpu < CPU_SETSIZE,
          "Invalid PIR_ONLINE_CPU");
    Pin(online_cpu, "Failed to bind online execution to PIR_ONLINE_CPU");
  }
  const std::vector<int> online_cpus =
      cpus.empty() && cpu_env ? std::vector<int>{online_cpu} : cpus;
  std::cout << "{\"type\":\"setup\",\"protocol\":\"reinspire\",\"backend\":"
               "\"avx2\",\"online_threads\":"
            << threads
            << ",\"plaintext_modulus\":65536,\"interpolation_degree\":1,"
               "\"offline_ms\":"
            << offline_ms << ",\"online_cpu\":" << online_cpu
            << ",\"offline_threads\":" << offline_threads << ",\"P\":" << p
            << ",\"Q\":" << q << ",\"C\":" << c << ",\"rows\":[";
  for (size_t i = 0; i < p; ++i) std::cout << (i ? "," : "") << rows[i];
  std::cout << "],\"online_cpus\":[";
  for (size_t i = 0; i < online_cpus.size(); ++i)
    std::cout << (i ? "," : "") << online_cpus[i];
  std::cout << "],\"color_groups\":" << color_groups
            << ",\"column_groups\":" << column_groups << ",\"tasks\":[";
  for (size_t w = 0; w < tasks.size(); ++w) {
    std::cout << (w ? "," : "") << "{\"worker\":" << w << ",\"cpu\":"
              << (w < online_cpus.size() ? std::to_string(online_cpus[w])
                                         : "null")
              << ",\"colors\":[" << tasks[w].p0 << "," << tasks[w].p1
              << "],\"columns\":[" << tasks[w].k0 << "," << tasks[w].k1 << "]}";
  }
  std::cout << "]}" << std::endl;
  auto fft_ctx = Take(pm::FftContext::Create(11));  // log2(2048)
  for (size_t trial = 0; trial < size_t(warmups) + q * repeats; ++trial) {
    const bool warmup = trial < size_t(warmups);
    const size_t qi = warmup ? trial % q : (trial - warmups) / repeats;
    const int rep = warmup ? -1 : (trial - warmups) % repeats;
    const auto client_begin = Clock::now();
    auto client = Take(pm::PirClient<uint64_t>::Create(
        Params(2048, c), Prng(SelectorSeed(0)),
        Prng(Take(::rlwe::ChaChaPrng::GenerateSeed()))));
    auto key_prng = Prng(packing_seed);
    auto keys = Take(client->CreatePackingKey(key_prng.get()));
    Check(keys.size() == 4, "Unexpected packing key count");
    const double client_key_generation_ms = Ms(client_begin);
    auto t_pack = Clock::now();
    std::string upload;
    for (const auto& key : keys) PackCoeffsAppend(key.Coeffs(), 52, upload);
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
      for (uint64_t& y : selector) {
        y = ((y + (1ULL << 11)) >> 12) & ((1ULL << 40) - 1);
      }
      PackCoeffsAppend(selector, 40, upload);
      client_request_pack_ms += Ms(t);
    }
    const double client_query_ms = Ms(client_begin);
    const auto server_begin = Clock::now();
    size_t off = 0;
    // The serialized key bundle and every selector are decoded once and only
    // read by the tasks.
    pm::PirRequest<uint64_t> request;
    for (int k = 0; k < 4; ++k)
      request.packing_key.push_back(
          Take(Poly::Create(UnpackCoeffs(upload, off, 2048, 52))));
    std::vector<std::vector<uint64_t>> queries(p);
    for (size_t slot = 0; slot < p; ++slot) {
      queries[slot] = UnpackCoeffs(upload, off, rows[slot], 40);
      for (uint64_t& y : queries[slot]) y <<= 12;
    }
    Check(off == upload.size(), "Unconsumed request bytes");
    const double ingest_ms = Ms(server_begin);
    // The key's y_vec_h is transformed once, not once per color; the transform
    // counts towards packing finalize.
    const std::vector<Poly> y_vec_h(
        request.packing_key.begin() + pack_gadget.num_digits,
        request.packing_key.end());
    const auto fft_begin = Clock::now();
    const auto y_vec_h_fft = Take(pm::YVecHFft(rlwe_params, y_vec_h, *fft_ctx));
    const double fft_ms = Ms(fft_begin);
    // Task w encodes color slot's responses for column group w % H into
    // wires[slot * H + w % H].
    std::vector<std::string> wires(p * column_groups);
    std::vector<Phases> phases(tasks.size());
    const auto tasks_begin = Clock::now();
    workers.Run([&](size_t w) {
      const auto task_begin = Clock::now();
      Phases ms;
      auto task_request = request;
      for (size_t slot = tasks[w].p0; slot < tasks[w].p1; ++slot) {
        task_request.first_dimension_query = queries[slot];
        auto& server = tasks[w].servers[slot - tasks[w].p0];
        auto db_before = server->GetDbMultiplyTime(),
             pack_before = server->GetPackingTime();
        const auto matrix_before = server->GetPackMatrixTime(),
                   finalize_before = server->GetFinalizeTime();
        const auto modswitch_before = server->GetModswitchTime(),
                   eval_before = server->GetPolyEvalTime();
        auto response = Take(server->ProcessResponse(
            task_request, &y_vec_h_fft, /*skip_a_t1=*/true));
        ms.matrix += absl::ToDoubleMilliseconds(server->GetPackMatrixTime() -
                                                matrix_before);
        ms.finalize += absl::ToDoubleMilliseconds(server->GetFinalizeTime() -
                                                  finalize_before);
        ms.modswitch += absl::ToDoubleMilliseconds(server->GetModswitchTime() -
                                                   modswitch_before);
        ms.poly_eval +=
            absl::ToDoubleMilliseconds(server->GetPolyEvalTime() - eval_before);
        ms.db +=
            absl::ToDoubleMilliseconds(server->GetDbMultiplyTime() - db_before);
        ms.packing +=
            absl::ToDoubleMilliseconds(server->GetPackingTime() - pack_before);
        const auto encode_begin = Clock::now();
        std::string wire;
        wire.reserve(response.ciphertexts.size() * (6400 + 4608));
        for (size_t k = 0; k < response.ciphertexts.size(); ++k) {
          wire.append(server->GetPrepackedAModSwitched(k));
          PackCoeffsAppend(response.ciphertexts[k].b.Coeffs(), 18, wire);
        }
        wires[slot * column_groups + w % column_groups] = std::move(wire);
        ms.encode += Ms(encode_begin);
      }
      ms.busy = Ms(task_begin);
      phases[w] = ms;
    });
    const double tasks_ms = Ms(tasks_begin);
    // Responses in color order, and each color's columns in order.
    const auto assemble_begin = Clock::now();
    size_t download_size = 0;
    for (const auto& wire : wires) download_size += wire.size();
    std::string download;
    download.reserve(download_size);
    for (const auto& wire : wires) download += wire;
    const double assemble_ms = Ms(assemble_begin);
    const double server_ms = Ms(server_begin);
    // Phase fields follow the critical path: the serial phases plus those of
    // the slowest task. server_cpu_ms adds up the busy time of all workers.
    const Phases& slow = *std::max_element(
        phases.begin(), phases.end(),
        [](const Phases& a, const Phases& b) { return a.busy < b.busy; });
    const double db_ms = slow.db, packing_ms = fft_ms + slow.packing,
                 matrix_ms = slow.matrix, finalize_ms = fft_ms + slow.finalize,
                 modswitch_ms = slow.modswitch, poly_eval_ms = slow.poly_eval,
                 encode_ms = slow.encode + assemble_ms;
    double server_cpu_ms = server_ms - tasks_ms;
    for (const auto& task_ms : phases) server_cpu_ms += task_ms.busy;
    const auto decode_begin = Clock::now();
    off = 0;
    std::vector<std::vector<uint64_t>> recovered;
    double client_response_unpack_ms = 0, client_decrypt_ms = 0;
    for (size_t slot = 0; slot < p; ++slot) {
      auto t = Clock::now();
      pm::PirResponse<uint64_t> response;
      for (size_t col = 0; col < c; ++col) {
        auto a = Take(Poly::Create(UnpackCoeffs(download, off, 2048, 25)));
        auto b = Take(Poly::Create(UnpackCoeffs(download, off, 2048, 18)));
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
              << upload.size() + download.size();
    if (!cpus.empty()) std::cout << ",\"server_cpu_ms\":" << server_cpu_ms;
    std::cout << "}" << std::endl;
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
