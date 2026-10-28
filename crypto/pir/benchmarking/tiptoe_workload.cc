// Copyright 2026. Apache-2.0; uses the Google ReInsPIRe implementation.
// A real multi-color t=1, p=65536 request, with a fresh shared secret per
// request. Colors may be taller than the ring degree (multi-sample selectors).
// With PIR_THREAD_CPUS, pinned worker threads answer it as response block
// group x row stripe tasks (see improvements.md).
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <random>
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
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
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
std::string SelectorSeed(uint64_t slot, char tag = 'S') {
  std::string seed(32, tag);
  for (int i = 0; i < 8; ++i) seed[24 + i] = char(slot >> (8 * i));
  return seed;
}
std::string GroupSelectorSeed(uint64_t group) {
  return SelectorSeed(group, 'G');
}
size_t ResolveRowU16(const std::string& workload_path, size_t c,
                     const std::vector<int>& rows,
                     const std::vector<std::vector<uint16_t>>& raw) {
  if (c != 1) return c * 2048;
  const char* env = std::getenv("PIR_ROW_U16");
  if (env && *env) {
    const size_t v = std::stoul(env);
    Check(v > 0 && v <= 2048, "Invalid PIR_ROW_U16");
    return v;
  }
  const size_t slash = workload_path.find_last_of('/');
  const std::string dir =
      slash == std::string::npos ? "." : workload_path.substr(0, slash);
  std::ifstream cf(dir + "/client.pal", std::ios::binary);
  if (cf) {
    char cmagic[8];
    cf.read(cmagic, 8);
    if (cf && std::memcmp(cmagic, "PALETTE\0", 8) == 0) {
      uint64_t hdr[10];
      for (int i = 0; i < 10; ++i) hdr[i] = Read64(cf);
      const uint64_t msub = hdr[1], nbits = hdr[2], h = hdr[9];
      const size_t row_bytes = static_cast<size_t>((h * msub * nbits + 7) / 8);
      const size_t row_u16 = (row_bytes + 1) / 2;
      if (row_u16 > 0 && row_u16 <= 2048) return row_u16;
    }
  }
  size_t max_u16 = 0;
  for (size_t s = 0; s < raw.size(); ++s) {
    for (int r = 0; r < rows[s]; ++r) {
      const uint16_t* row_ptr = raw[s].data() + static_cast<size_t>(r) * 2048;
      for (size_t u = 2048; u > max_u16; --u) {
        if (row_ptr[u - 1] != 0) {
          max_u16 = u;
          break;
        }
      }
    }
  }
  return max_u16 > 0 ? max_u16 : 2048;
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
ABSL_FLAG(size_t, synthetic_p, 0,
          "Colors of a synthetic workload (replaces WORKLOAD.bin when > 0).");
ABSL_FLAG(size_t, synthetic_m, 0, "Rows per synthetic color.");
ABSL_FLAG(size_t, synthetic_row_bytes, 4096, "Payload bytes per synthetic row.");
ABSL_FLAG(size_t, synthetic_q, 4,
          "Synthetic queries (first row, last row, then random rows).");
ABSL_FLAG(std::string, synthetic_fill, "random",
          "Synthetic payload: random, or a constant 16-bit word such as 0xffff.");
ABSL_FLAG(uint64_t, seed, 1, "Synthetic payload and target seed.");
ABSL_FLAG(int, selector_bits, 0,
          "Selector bits per row: 40, 42, or 0 for 42 iff max R > 8192.");
ABSL_FLAG(std::string, centered_db, "auto",
          "Signed int16 Stage 1 entries: 0, 1, or auto (iff max R > 2048).");
ABSL_FLAG(bool, noise_stats, false,
          "Report decryption noise (q1 units, failure at 256) of timed queries.");
ABSL_FLAG(uint64_t, client_seed, 0,
          "Fixed client secret seed (0: fresh), for comparing download_fnv64.");
uint64_t Fnv64(const std::string& bytes) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char b : bytes) h = (h ^ b) * 0x100000001B3ULL;
  return h;
}
// Cross-color response blocks [p0, p1) x row stripe s of every request, with
// one server per block holding just the stripe's rows (null when the stripe
// has none). Stripe 0 servers also hold the packing material.
struct Task {
  size_t p0, p1, s;
  std::vector<std::unique_ptr<Server>> servers;
};
// One task's server time for one request, in ms.
struct Phases {
  double db = 0, reduce = 0, packing = 0, matrix = 0, finalize = 0,
         modswitch = 0, poly_eval = 0, encode = 0, busy = 0;
};
int Run(int argc, char** argv) {
  const std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  const size_t synthetic_p = absl::GetFlag(FLAGS_synthetic_p);
  const bool synthetic = synthetic_p > 0;
  Check(args.size() == (synthetic ? 3u : 4u),
        "usage: reinspire-workload [flags] WORKLOAD.bin WARMUPS REPEATS, or "
        "--synthetic_p=P --synthetic_m=M [flags] WARMUPS REPEATS");
  Check(std::string(hwy::TargetName(HWY_STATIC_TARGET)) == "AVX2",
        "AVX2 backend required");
  const std::vector<int> cpus = ThreadCpus();
  const int warmups = std::stoi(args[args.size() - 2]),
            repeats = std::stoi(args.back());
  Check(warmups >= 0 && repeats > 0, "Invalid repetitions");
  constexpr size_t kMaxRows = 1 << 22;
  size_t p, q, c, row_u16 = 0;
  std::vector<int> rows;
  std::vector<size_t> targets;
  std::vector<std::vector<uint16_t>> raw;
  std::string workload_path;
  std::ifstream input;
  if (synthetic) {
    const size_t m = absl::GetFlag(FLAGS_synthetic_m),
                 row_bytes = absl::GetFlag(FLAGS_synthetic_row_bytes);
    p = synthetic_p, q = absl::GetFlag(FLAGS_synthetic_q);
    c = (row_bytes + 4095) / 4096;
    Check(m > 0 && m <= kMaxRows && row_bytes > 0 && q > 0 && p < 100000,
          "Invalid synthetic geometry");
    const std::string fill = absl::GetFlag(FLAGS_synthetic_fill);
    const bool random_fill = fill == "random";
    const uint16_t word = random_fill ? 0 : uint16_t(std::stoul(fill, nullptr, 0));
    std::mt19937_64 rng(absl::GetFlag(FLAGS_seed));
    rows.assign(p, static_cast<int>(m));
    targets.resize(p * q);
    for (size_t i = 0; i < targets.size(); ++i) {
      targets[i] = i < p ? 0 : i < 2 * p ? m - 1 : rng() % m;
    }
    row_u16 = (row_bytes + 1) / 2;
    raw.resize(p);
    for (auto& table : raw) {
      table.assign(m * c * 2048, 0);
      for (size_t r = 0; r < m; ++r) {
        uint16_t* row = table.data() + r * c * 2048;
        for (size_t u = 0; u < row_u16; ++u) {
          row[u] = random_fill ? uint16_t(rng()) : word;
        }
        if (row_bytes % 2) row[row_u16 - 1] &= 0xFF;
      }
    }
  } else {
    workload_path = args[1];
    input.open(workload_path, std::ios::binary);
    char magic[8];
    input.read(magic, 8);
    Check(std::string(magic, 8) == "TPIRv001", "Invalid workload version");
    p = Read64(input), q = Read64(input), c = Read64(input);
    Check(p > 0 && q > 0 && c > 0 && p < 100000 && c < 100000,
          "Invalid workload dimensions");
    rows.resize(p);
    for (auto& m : rows) {
      const uint64_t v = Read64(input);
      Check(v > 0 && v <= kMaxRows, "Unsupported M");
      m = static_cast<int>(v);
    }
    targets.resize(p * q);
    for (size_t i = 0; i < targets.size(); ++i) {
      targets[i] = Read64(input);
      Check(targets[i] < size_t(rows[i % p]), "Invalid target");
    }
    raw.resize(p);
    for (size_t slot = 0; slot < p; ++slot) {
      raw[slot].resize(size_t(rows[slot]) * c * 2048);
      // Both native harnesses consume the same little-endian uint16 payloads.
      std::vector<unsigned char> bytes(raw[slot].size() * 2);
      input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
      Check(bool(input), "Truncated database");
      for (size_t i = 0; i < raw[slot].size(); ++i)
        raw[slot][i] = uint16_t(bytes[2 * i]) | (uint16_t(bytes[2 * i + 1]) << 8);
    }
    Check(input.peek() == std::char_traits<char>::eof(),
          "Trailing workload data");
  }
  const size_t max_rows = *std::max_element(rows.begin(), rows.end());
  // 40-bit selectors stay noise-safe to about R = 10k with signed entries;
  // 42-bit ones (same 3-digit kernel) to about R = 168k.
  int selector_bits = absl::GetFlag(FLAGS_selector_bits);
  Check(selector_bits == 0 || selector_bits == 40 || selector_bits == 42,
        "--selector_bits must be 0, 40 or 42");
  if (selector_bits == 0) selector_bits = max_rows > 8192 ? 42 : 40;
  const int selector_shift = 52 - selector_bits;
  const std::string centered_flag = absl::GetFlag(FLAGS_centered_db);
  Check(centered_flag == "auto" || centered_flag == "0" || centered_flag == "1",
        "--centered_db must be auto, 0 or 1");
  const bool centered =
      centered_flag == "auto" ? max_rows > 2048 : centered_flag == "1";
  const auto rlwe_params = Take(pm::RlweParams<uint64_t>::Create(
      2048, 1ULL << 52, 1ULL << 25, 1ULL << 18, 40));
  const pm::GadgetParams pack_gadget{19, 2}, eval_gadget{19, 3};
  const auto params = Take(pm::PirParams<uint64_t>::Create(
      rlwe_params, 65536, pack_gadget, eval_gadget, 2048, 1, 1));
  const std::string packing_seed(32, 'K');
  const auto selector_groups = Take(pm::BuildSelectorGroupPlans(rows, 2048));
  std::vector<size_t> slot_offset(p);
  std::vector<std::string> slot_seeds(p);
  for (const auto& grp : selector_groups) {
    for (const auto& sl : grp.slices) {
      slot_offset[sl.slot] = sl.offset;
      slot_seeds[sl.slot] = GroupSelectorSeed(grp.group_index);
    }
  }
  // Every response is one RLWE ciphertext packing 2048 consecutive LWE outputs
  // of the colors' concatenated rows: sub-tile rows of d_row <= 1024 words
  // share blocks across colors, wider rows take C whole blocks per color.
  if (!synthetic) row_u16 = ResolveRowU16(workload_path, c, rows, raw);
  const size_t d_row = (c == 1 && row_u16 <= 1024) ? row_u16 : c * 2048;
  const size_t n_resp = (p * d_row + 2047) / 2048;
  // Task (g, s) covers block group g x row stripe s, both balanced contiguous
  // splits, and runs on worker w = g * S + s. Stripes split every color's
  // selector rows into 2048-row blocks; their Stage 1 outputs add up mod 2^64
  // before stripe 0 packs (S = 1 for R <= 2048).
  const size_t threads = std::max<size_t>(cpus.size(), 1);
  const size_t block_groups = std::min(n_resp, threads);
  const size_t stripes =
      std::min((max_rows + 2047) / 2048, threads / block_groups);
  std::vector<Task> tasks;
  for (size_t g = 0; g < block_groups; ++g)
    for (size_t s = 0; s < stripes; ++s)
      tasks.push_back({g * n_resp / block_groups,
                       (g + 1) * n_resp / block_groups, s, {}});
  auto BuildStripe = [&](size_t block, size_t s) {
    return Take(Prep::PreprocessCrossColorStripe(rlwe_params, rows, raw,
                                                 slot_seeds, slot_offset, d_row,
                                                 block, s, stripes, centered));
  };
  // The stripe 0 server, packing the hint summed over all stripes.
  auto PackingServer = [&](Prep::CrossColorStripe part) {
    std::vector<pm::PreprocessMatrixPackOutput<uint64_t, int32_t>> outputs;
    outputs.push_back(Take(Prep::PreprocessPacking(rlwe_params, pack_gadget,
                                                   packing_seed, part.hint)));
    return Take(Server::CreateCrossColor(params, std::move(part.db_slices),
                                         std::move(outputs)));
  };
  Workers workers(tasks.size());
  size_t offline_threads = tasks.size();
  const auto offline_begin = Clock::now();
  if (!cpus.empty()) {
    // Owner computes: each worker pins itself and builds its own task's
    // servers, so their memory is local to the CPU that serves them.
    std::vector<std::vector<Prep::CrossColorStripe>> parts(tasks.size());
    workers.Run([&](size_t w) {
      Pin(cpus[w], "Failed to pin a worker to its PIR_THREAD_CPUS entry");
      auto& t = tasks[w];
      for (size_t block = t.p0; block < t.p1; ++block) {
        parts[w].push_back(BuildStripe(block, t.s));
        auto& slices = parts[w].back().db_slices;
        t.servers.push_back(t.s == 0 || slices.empty()
                                ? nullptr
                                : Take(Server::CreateCrossColor(
                                      params, std::move(slices), {})));
      }
    });
    workers.Run([&](size_t w) {
      auto& t = tasks[w];
      if (t.s) return;
      for (size_t i = 0; i < t.servers.size(); ++i) {
        auto& hint = parts[w][i].hint;
        for (size_t s = 1; s < stripes; ++s) {
          auto& other = parts[w + s][i].hint;
          for (size_t r = 0; r < other.size(); ++r)
            for (size_t j = 0; j < other[r].size(); ++j)
              hint[r][j] += other[r][j];
          other = {};
        }
        t.servers[i] = PackingServer(std::move(parts[w][i]));
      }
    });
  } else {
    // Independent fixed-mask preprocessing can run concurrently.
    const char* prep_env = std::getenv("PIR_PREPROCESS_THREADS");
    const int requested_threads = prep_env ? std::stoi(prep_env) : 8;
    Check(requested_threads > 0 && requested_threads <= 64,
          "PIR_PREPROCESS_THREADS must be in [1,64]");
    offline_threads = std::min(size_t(requested_threads), n_resp);
    tasks[0].servers.resize(n_resp);
    for (size_t first = 0; first < n_resp; first += offline_threads) {
      std::vector<std::future<void>> jobs;
      for (size_t block = first;
           block < std::min(n_resp, first + offline_threads); ++block) {
        jobs.push_back(std::async(std::launch::async, [&, block] {
          tasks[0].servers[block] = PackingServer(BuildStripe(block, 0));
        }));
      }
      for (auto& job : jobs) job.get();
    }
  }
  const double offline_ms = Ms(offline_begin);
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
            << ",\"Q\":" << q << ",\"C\":" << c << ",\"d_row\":" << d_row
            << ",\"responses\":" << n_resp
            << ",\"selector_bits\":" << selector_bits
            << ",\"centered_db\":" << (centered ? "true" : "false")
            << ",\"synthetic\":" << (synthetic ? "true" : "false")
            << ",\"rows\":[";
  for (size_t i = 0; i < p; ++i) std::cout << (i ? "," : "") << rows[i];
  std::cout << "],\"online_cpus\":[";
  for (size_t i = 0; i < online_cpus.size(); ++i)
    std::cout << (i ? "," : "") << online_cpus[i];
  std::cout << "],\"block_groups\":" << block_groups
            << ",\"stripes\":" << stripes << ",\"tasks\":[";
  for (size_t w = 0; w < tasks.size(); ++w) {
    std::cout << (w ? "," : "") << "{\"worker\":" << w << ",\"cpu\":"
              << (w < online_cpus.size() ? std::to_string(online_cpus[w])
                                         : "null")
              << ",\"blocks\":[" << tasks[w].p0 << "," << tasks[w].p1
              << "],\"stripe\":" << tasks[w].s << "}";
  }
  std::cout << "]}" << std::endl;
  auto fft_ctx = Take(pm::FftContext::Create(11));  // log2(2048)
  const bool noise_stats = absl::GetFlag(FLAGS_noise_stats);
  const uint64_t client_seed = absl::GetFlag(FLAGS_client_seed);
  // Decryption noise of the data coefficients of every timed query.
  double noise_sq = 0, noise_max = 0;
  size_t noise_count = 0;
  for (size_t trial = 0; trial < size_t(warmups) + q * repeats; ++trial) {
    const bool warmup = trial < size_t(warmups);
    const size_t qi = warmup ? trial % q : (trial - warmups) / repeats;
    const int rep = warmup ? -1 : (trial - warmups) % repeats;
    const auto client_begin = Clock::now();
    auto client = Take(pm::PirClient<uint64_t>::Create(
        params, Prng(SelectorSeed(0)),
        Prng(client_seed ? SelectorSeed(client_seed + qi, 'C')
                         : Take(::rlwe::ChaChaPrng::GenerateSeed()))));
    auto key_prng = Prng(packing_seed);
    auto keys = Take(client->CreatePackingKey(key_prng.get()));
    Check(keys.size() == 4, "Unexpected packing key count");
    const double client_key_generation_ms = Ms(client_begin);
    auto t_pack = Clock::now();
    std::string upload;
    for (const auto& key : keys) PackCoeffsAppend(key.Coeffs(), 52, upload);
    const size_t key_bytes = upload.size();
    double client_request_pack_ms = Ms(t_pack), client_selector_ms = 0;
    for (const auto& grp : selector_groups) {
      auto t = Clock::now();
      auto mask = Prng(slot_seeds[grp.slices[0].slot]);
      std::vector<int> group_targets;
      group_targets.reserve(grp.slices.size());
      for (const auto& sl : grp.slices) {
        group_targets.push_back(static_cast<int>(targets[qi * p + sl.slot]));
      }
      auto grouped_selector = Take(
          client->CreateGroupedSelector(grp.slices, group_targets, mask.get()));
      client_selector_ms += Ms(t);
      t = Clock::now();
      for (const auto& sl : grp.slices) {
        std::vector<uint64_t> selector(
            grouped_selector.begin() + sl.offset,
            grouped_selector.begin() + sl.offset + sl.rows);
        for (uint64_t& y : selector) {
          y = ((y + (1ULL << (selector_shift - 1))) >> selector_shift) &
              ((1ULL << selector_bits) - 1);
        }
        PackCoeffsAppend(selector, selector_bits, upload);
      }
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
      queries[slot] = UnpackCoeffs(upload, off, rows[slot], selector_bits);
      for (uint64_t& y : queries[slot]) y <<= selector_shift;
    }
    request.color_queries = &queries;
    Check(off == upload.size(), "Unconsumed request bytes");
    const double ingest_ms = Ms(server_begin);
    // The key's y_vec_h is transformed once, not once per block; the transform
    // counts towards packing finalize.
    const std::vector<Poly> y_vec_h(
        request.packing_key.begin() + pack_gadget.num_digits,
        request.packing_key.end());
    const auto fft_begin = Clock::now();
    const auto y_vec_h_fft = Take(pm::YVecHFft(rlwe_params, y_vec_h, *fft_ctx));
    const double fft_ms = Ms(fft_begin);
    // Stage 1 outputs of task w's i-th block; stripe 0 tasks sum their
    // stripes' and encode the block's response into wires[block].
    std::vector<std::vector<std::vector<uint64_t>>> partial(tasks.size());
    std::vector<std::string> wires(n_resp);
    std::vector<Phases> phases_a(tasks.size()), phases_b(tasks.size());
    auto Pack = [&](size_t w, Phases& ms) {
      const auto& t = tasks[w];
      for (size_t i = 0; i < t.servers.size(); ++i) {
        auto reduce_begin = Clock::now();
        std::vector<uint64_t> b_prime = std::move(partial[w][i]);
        for (size_t s = 1; s < stripes; ++s) {
          const auto& other = partial[w + s][i];
          for (size_t j = 0; j < other.size(); ++j) b_prime[j] += other[j];
        }
        ms.reduce += Ms(reduce_begin);
        auto& server = t.servers[i];
        const auto pack_before = server->GetPackingTime(),
                   matrix_before = server->GetPackMatrixTime(),
                   finalize_before = server->GetFinalizeTime(),
                   modswitch_before = server->GetModswitchTime(),
                   eval_before = server->GetPolyEvalTime();
        auto response = Take(server->PackFromStage1(
            request, std::move(b_prime), &y_vec_h_fft, /*skip_a_t1=*/true));
        auto Delta = [](absl::Duration after, absl::Duration before) {
          return absl::ToDoubleMilliseconds(after - before);
        };
        ms.matrix += Delta(server->GetPackMatrixTime(), matrix_before);
        ms.finalize += Delta(server->GetFinalizeTime(), finalize_before);
        ms.modswitch += Delta(server->GetModswitchTime(), modswitch_before);
        ms.poly_eval += Delta(server->GetPolyEvalTime(), eval_before);
        ms.packing += Delta(server->GetPackingTime(), pack_before);
        const auto encode_begin = Clock::now();
        std::string wire;
        wire.reserve(6400 + 4608);
        wire.append(server->GetPrepackedAModSwitched(0));
        PackCoeffsAppend(response.ciphertexts[0].b.Coeffs(), 18, wire);
        wires[t.p0 + i] = std::move(wire);
        ms.encode += Ms(encode_begin);
      }
    };
    const auto tasks_begin = Clock::now();
    workers.Run([&](size_t w) {
      const auto task_begin = Clock::now();
      Phases& ms = phases_a[w];
      for (const auto& server : tasks[w].servers) {
        partial[w].emplace_back();
        if (!server) continue;
        const auto db_before = server->GetDbMultiplyTime();
        partial[w].back() = Take(server->ComputeStage1(request));
        ms.db +=
            absl::ToDoubleMilliseconds(server->GetDbMultiplyTime() - db_before);
      }
      if (stripes == 1) Pack(w, ms);
      ms.busy = Ms(task_begin);
    });
    if (stripes > 1) {
      workers.Run([&](size_t w) {
        if (tasks[w].s) return;
        const auto task_begin = Clock::now();
        Pack(w, phases_b[w]);
        phases_b[w].busy = Ms(task_begin);
      });
    }
    const double tasks_ms = Ms(tasks_begin);
    // Responses in block order.
    const auto assemble_begin = Clock::now();
    size_t download_size = 0;
    for (const auto& wire : wires) download_size += wire.size();
    std::string download;
    download.reserve(download_size);
    for (const auto& wire : wires) download += wire;
    const double assemble_ms = Ms(assemble_begin);
    const double server_ms = Ms(server_begin);
    // Phase fields follow the critical path: the serial phases plus those of
    // the slowest task of each parallel phase. server_cpu_ms adds up the busy
    // time of all workers.
    auto Slowest = [](const std::vector<Phases>& ph) {
      return *std::max_element(
          ph.begin(), ph.end(),
          [](const Phases& a, const Phases& b) { return a.busy < b.busy; });
    };
    const Phases sa = Slowest(phases_a), sb = Slowest(phases_b);
    const double db_ms = sa.db, reduce_ms = sa.reduce + sb.reduce,
                 packing_ms = fft_ms + sa.packing + sb.packing,
                 matrix_ms = sa.matrix + sb.matrix,
                 finalize_ms = fft_ms + sa.finalize + sb.finalize,
                 modswitch_ms = sa.modswitch + sb.modswitch,
                 poly_eval_ms = sa.poly_eval + sb.poly_eval,
                 encode_ms = sa.encode + sb.encode + assemble_ms;
    double server_cpu_ms = server_ms - tasks_ms;
    for (const auto* ph : {&phases_a, &phases_b})
      for (const auto& task_ms : *ph) server_cpu_ms += task_ms.busy;
    const auto decode_begin = Clock::now();
    off = 0;
    std::vector<std::vector<uint64_t>> recovered;
    recovered.reserve(p);
    std::vector<pm::PirResponse<uint64_t>> responses;
    double client_response_unpack_ms = 0, client_decrypt_ms = 0;
    for (size_t b_idx = 0; b_idx < n_resp; ++b_idx) {
      auto t = Clock::now();
      pm::PirResponse<uint64_t> response;
      auto a = Take(Poly::Create(UnpackCoeffs(download, off, 2048, 25)));
      auto b = Take(Poly::Create(UnpackCoeffs(download, off, 2048, 18)));
      response.ciphertexts.push_back({std::move(a), std::move(b)});
      client_response_unpack_ms += Ms(t);
      t = Clock::now();
      auto blk = Take(client->ProcessResponse(response));
      const size_t base = b_idx * 2048;
      const size_t end = std::min(base + 2048, p * d_row);
      for (size_t idx = base; idx < end;) {
        const size_t slot = idx / d_row;
        const size_t off_in_row = idx % d_row;
        const size_t take = std::min(d_row - off_in_row, end - idx);
        if (recovered.size() <= slot) recovered.emplace_back(c * 2048, 0);
        std::copy_n(blk.begin() + (idx - base), take,
                    recovered[slot].begin() + off_in_row);
        idx += take;
      }
      client_decrypt_ms += Ms(t);
      if (noise_stats && !warmup) responses.push_back(std::move(response));
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
    // Untimed: noise of the coefficients that carry data (the rest of the last
    // block is empty).
    for (size_t b_idx = 0; b_idx < responses.size(); ++b_idx) {
      std::vector<int64_t> noise;
      Take(client->ProcessResponse(responses[b_idx], &noise));
      const size_t used = std::min<size_t>(2048, p * d_row - b_idx * 2048);
      for (size_t i = 0; i < used; ++i) {
        const double e = double(noise[i]);
        noise_sq += e * e;
        noise_max = std::max(noise_max, std::abs(e));
      }
      noise_count += used;
    }
    std::cout << "{\"type\":\"query\",\"protocol\":\"reinspire\",\"correct\":"
                 "true,\"query_index\":"
              << qi << ",\"repeat\":" << rep
              << ",\"warmup\":" << (warmup ? "true" : "false")
              << ",\"server_ms\":" << server_ms
              << ",\"client_query_ms\":" << client_query_ms
              << ",\"client_decode_ms\":" << client_decode_ms
              << ",\"server_ingest_ms\":" << ingest_ms
              << ",\"server_inner_product_ms\":" << db_ms
              << ",\"server_stripe_reduce_ms\":" << reduce_ms
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
              << server_ms - ingest_ms - db_ms - reduce_ms - packing_ms -
                     modswitch_ms - poly_eval_ms - encode_ms
              << ",\"key_bundle_upload_bytes\":" << key_bytes
              << ",\"upload_bytes\":" << upload.size()
              << ",\"download_bytes\":" << download.size()
              << ",\"total_communication_bytes\":"
              << upload.size() + download.size()
              << ",\"download_fnv64\":" << Fnv64(download);
    if (!cpus.empty()) std::cout << ",\"server_cpu_ms\":" << server_cpu_ms;
    std::cout << "}" << std::endl;
  }
  if (noise_stats) {
    std::cout << "{\"type\":\"noise\",\"coefficients\":" << noise_count
              << ",\"std_q1\":" << std::sqrt(noise_sq / noise_count)
              << ",\"max_abs_q1\":" << noise_max << "}" << std::endl;
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
