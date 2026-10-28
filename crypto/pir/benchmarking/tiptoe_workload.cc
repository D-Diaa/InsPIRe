// Copyright 2026. Apache-2.0; uses the Google ReInsPIRe implementation.
// A real multi-color t=1, p=65536 request, with a fresh shared secret per
// request. Colors may be taller than the ring degree (multi-sample selectors).
// With PIR_THREAD_CPUS, pinned worker threads answer it as color group x column
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
// Options: positional WORKLOAD.bin WARMUPS REPEATS, or WARMUPS REPEATS with
// --synthetic_p/--synthetic_m (random tables, no workload file).
struct Options {
  std::string workload;
  int warmups = 0, repeats = 0;
  size_t synthetic_p = 0, synthetic_m = 0, synthetic_row_bytes = 4096;
  size_t synthetic_q = 4;
  std::string synthetic_fill = "random";  // random | zero | ffff
  uint64_t seed = 1;
  uint64_t client_seed = 0;  // 0: a fresh random secret per request
  int selector_bits = 0;  // 0: 40 when max R <= 8192, else 42
  int centered_db = -1;   // -1: centered iff max R > 2048
  bool noise_stats = false;
};
Options ParseOptions(int argc, char** argv) {
  Options o;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      positional.push_back(arg);
      continue;
    }
    const size_t eq = arg.find('=');
    Check(eq != std::string::npos, "Flags take the form --name=value");
    const std::string key = arg.substr(2, eq - 2), value = arg.substr(eq + 1);
    if (key == "synthetic_p") {
      o.synthetic_p = std::stoul(value);
    } else if (key == "synthetic_m") {
      o.synthetic_m = std::stoul(value);
    } else if (key == "synthetic_row_bytes") {
      o.synthetic_row_bytes = std::stoul(value);
    } else if (key == "synthetic_q") {
      o.synthetic_q = std::stoul(value);
    } else if (key == "synthetic_fill") {
      o.synthetic_fill = value;
    } else if (key == "seed") {
      o.seed = std::stoull(value);
    } else if (key == "client_seed") {
      o.client_seed = std::stoull(value);
    } else if (key == "selector_bits") {
      o.selector_bits = std::stoi(value);
    } else if (key == "centered_db") {
      o.centered_db = value == "auto" ? -1 : std::stoi(value);
    } else if (key == "noise_stats") {
      o.noise_stats = std::stoi(value) != 0;
    } else {
      throw std::runtime_error("Unknown flag --" + key);
    }
  }
  const bool synthetic = o.synthetic_p > 0;
  Check(positional.size() == (synthetic ? 2u : 3u),
        "usage: reinspire-workload WORKLOAD.bin WARMUPS REPEATS [flags] | "
        "--synthetic_p=P --synthetic_m=M [flags] WARMUPS REPEATS");
  if (!synthetic) o.workload = positional[0];
  o.warmups = std::stoi(positional[positional.size() - 2]);
  o.repeats = std::stoi(positional.back());
  Check(o.warmups >= 0 && o.repeats > 0, "Invalid repetitions");
  Check(!synthetic || (o.synthetic_m > 0 && o.synthetic_row_bytes > 0 &&
                       o.synthetic_q > 0),
        "Invalid synthetic geometry");
  Check(o.synthetic_fill == "random" || o.synthetic_fill == "zero" ||
            o.synthetic_fill == "ffff",
        "--synthetic_fill must be random, zero or ffff");
  Check(o.selector_bits == 0 || o.selector_bits == 40 || o.selector_bits == 42,
        "--selector_bits must be 0, 40 or 42");
  Check(o.centered_db >= -1 && o.centered_db <= 1,
        "--centered_db must be auto, 0 or 1");
  return o;
}
uint64_t SplitMix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
uint64_t Fnv64(const std::string& bytes) {
  uint64_t h = 0xCBF29CE484222325ULL;
  for (unsigned char b : bytes) h = (h ^ b) * 0x100000001B3ULL;
  return h;
}
// Colors (or cross-color response blocks) [p0, p1) x columns [k0, k1) x row
// stripe s of every request, with one server per unit holding just those
// columns and rows (null when the stripe has no rows of that unit). Stripe 0
// servers also hold the packing material.
struct Task {
  size_t p0, p1, k0, k1, s;
  std::vector<std::unique_ptr<Server>> servers;
};
// One task's server time for one request, in ms.
struct Phases {
  double db = 0, reduce = 0, packing = 0, matrix = 0, finalize = 0,
         modswitch = 0, poly_eval = 0, encode = 0, busy = 0;
};
int Run(int argc, char** argv) {
  const Options opt = ParseOptions(argc, argv);
  Check(std::string(hwy::TargetName(HWY_STATIC_TARGET)) == "AVX2",
        "AVX2 backend required");
  const std::vector<int> cpus = ThreadCpus();
  const int warmups = opt.warmups, repeats = opt.repeats;
  const bool synthetic = opt.synthetic_p > 0;
  constexpr int kMaxRows = 1 << 22;
  size_t p, q, c;
  std::vector<int> rows;
  std::vector<size_t> targets;
  std::vector<std::vector<uint16_t>> raw;
  std::ifstream input;
  if (synthetic) {
    p = opt.synthetic_p, q = opt.synthetic_q;
    c = (opt.synthetic_row_bytes + 4095) / 4096;
    Check(p < 100000 && c < 100000 && opt.synthetic_m <= kMaxRows,
          "Invalid synthetic dimensions");
    rows.assign(p, static_cast<int>(opt.synthetic_m));
    uint64_t state = opt.seed;
    targets.resize(p * q);
    for (size_t i = 0; i < targets.size(); ++i) {
      const size_t qi = i / p, m = rows[i % p];
      targets[i] = qi == 0 ? 0 : qi == 1 ? m - 1 : SplitMix64(state) % m;
    }
    raw.resize(p);
    const size_t row_u16 = (opt.synthetic_row_bytes + 1) / 2;
    for (size_t slot = 0; slot < p; ++slot) {
      raw[slot].assign(size_t(rows[slot]) * c * 2048, 0);
      for (size_t r = 0; r < size_t(rows[slot]); ++r) {
        uint16_t* row = raw[slot].data() + r * c * 2048;
        for (size_t u = 0; u < row_u16; ++u) {
          row[u] = opt.synthetic_fill == "zero"   ? 0
                   : opt.synthetic_fill == "ffff" ? 0xFFFF
                                                  : uint16_t(SplitMix64(state));
        }
        if (opt.synthetic_row_bytes % 2) row[row_u16 - 1] &= 0xFF;
      }
    }
  } else {
    input.open(opt.workload, std::ios::binary);
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
  }
  const size_t max_rows = *std::max_element(rows.begin(), rows.end());
  // 40-bit selectors stay noise-safe to about R = 10k with a centered database;
  // 42-bit ones (same 3-digit kernel) to about R = 168k.
  const int selector_bits =
      opt.selector_bits ? opt.selector_bits : (max_rows > 8192 ? 42 : 40);
  const int selector_shift = 52 - selector_bits;
  const bool centered =
      opt.centered_db < 0 ? max_rows > 2048 : opt.centered_db == 1;
  const auto rlwe_params = Take(pm::RlweParams<uint64_t>::Create(
      2048, 1ULL << 52, 1ULL << 25, 1ULL << 18, 40));
  const pm::GadgetParams pack_gadget{19, 2}, eval_gadget{19, 3};
  auto Params = [&](int m, size_t cols) {
    return Take(pm::PirParams<uint64_t>::Create(rlwe_params, 65536, pack_gadget,
                                                eval_gadget, m, 1, cols,
                                                centered));
  };
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
  const size_t d_row =
      synthetic ? (c == 1 ? (opt.synthetic_row_bytes + 1) / 2 : c * 2048)
                : ResolveRowU16(opt.workload, c, rows, raw);
  const bool cross_color_resp = (c == 1) && ((2048 / d_row) >= 2);
  const size_t n_resp =
      cross_color_resp ? ((p * d_row + 2047) / 2048) : (p * c);
  const size_t eff_c = cross_color_resp ? 1 : c;
  const size_t eff_p = cross_color_resp ? n_resp : p;
  // Task (g, h, s) covers unit group g x column group h x row stripe s, all
  // balanced contiguous splits, and runs on worker w = (g * H + h) * S + s.
  // Row stripes split every unit's selector rows into 2048-row blocks; their
  // Stage 1 outputs add up mod 2^64 before stripe 0 packs (S = 1 for R <= 2048).
  const size_t threads = std::max<size_t>(cpus.size(), 1);
  const size_t color_groups = std::min(eff_p, threads);
  const size_t column_groups = std::min(eff_c, threads / color_groups);
  const size_t stripes = std::min((max_rows + 2047) / 2048,
                                  threads / (color_groups * column_groups));
  std::vector<Task> tasks;
  for (size_t g = 0; g < color_groups; ++g)
    for (size_t h = 0; h < column_groups; ++h)
      for (size_t s = 0; s < stripes; ++s)
        tasks.push_back({g * eff_p / color_groups,
                         (g + 1) * eff_p / color_groups,
                         h * eff_c / column_groups,
                         (h + 1) * eff_c / column_groups, s, {}});
  // Stage 1 material of unit u (a color, or a cross-color block), columns
  // [k0, k1), row stripe s.
  auto BuildStage1 = [&](size_t u, size_t k0, size_t k1, size_t s) {
    if (cross_color_resp) {
      return Take(Prep::PreprocessCrossColorStripe(
          rlwe_params, rows, raw, slot_seeds, slot_offset, d_row, u, s,
          stripes, centered));
    }
    const auto [r0, r1] = pm::RowStripe(rows[u], 2048, s, stripes);
    if (r0 == r1) return Prep::Stage1Stripe{};
    std::vector<uint16_t> slice;
    if (k1 - k0 < c) {
      for (size_t row = 0; row < size_t(rows[u]); ++row)
        slice.insert(slice.end(), raw[u].begin() + (row * c + k0) * 2048,
                     raw[u].begin() + (row * c + k1) * 2048);
    }
    auto params = Params(rows[u], k1 - k0);
    auto prep = Take(Prep::Create(&params, slot_seeds[u]));
    return Take(prep->PreprocessStripe(k1 - k0 < c ? slice : raw[u], r0, r1,
                                       slot_offset[u]));
  };
  // A Stage 1 only server for stripe s > 0 (null if the stripe is empty).
  auto Stage1Server = [&](size_t u, size_t k1_minus_k0, size_t s,
                          Prep::Stage1Stripe& part) -> std::unique_ptr<Server> {
    if (cross_color_resp) {
      if (part.db_slices.empty()) return nullptr;
      return Take(Server::CreateCrossColor(Params(2048, 1),
                                           std::move(part.db_slices), {}));
    }
    if (part.db_matrices.empty()) return nullptr;
    return Take(Server::CreateStage1Only(
        Params(rows[u], k1_minus_k0), std::move(part.db_matrices),
        pm::RowStripe(rows[u], 2048, s, stripes).first));
  };
  // The stripe 0 server, packing the hints summed over all stripes.
  auto PackingServer = [&](size_t u, size_t k1_minus_k0,
                           Prep::Stage1Stripe part) {
    auto outputs = Take(Prep::PreprocessPacking(rlwe_params, pack_gadget,
                                                packing_seed, part.hints));
    if (cross_color_resp) {
      return Take(Server::CreateCrossColor(
          Params(2048, 1), std::move(part.db_slices), std::move(outputs)));
    }
    std::vector<std::vector<pm::PreprocessMatrixPackOutput<uint64_t, int32_t>>>
        per_column;
    for (auto& output : outputs) {
      per_column.emplace_back();
      per_column.back().push_back(std::move(output));
    }
    return Take(Server::Create(
        Params(rows[u], k1_minus_k0),
        {std::move(part.db_matrices), std::move(per_column), {}}));
  };
  Workers workers(tasks.size());
  size_t offline_threads = tasks.size();
  const auto offline_begin = Clock::now();
  if (!cpus.empty()) {
    // Owner computes: each worker pins itself and builds its own task's
    // servers, so their memory is local to the CPU that serves them.
    std::vector<std::vector<Prep::Stage1Stripe>> parts(tasks.size());
    workers.Run([&](size_t w) {
      Pin(cpus[w], "Failed to pin a worker to its PIR_THREAD_CPUS entry");
      auto& t = tasks[w];
      for (size_t u = t.p0; u < t.p1; ++u) {
        parts[w].push_back(BuildStage1(u, t.k0, t.k1, t.s));
        t.servers.push_back(
            t.s ? Stage1Server(u, t.k1 - t.k0, t.s, parts[w].back()) : nullptr);
      }
    });
    workers.Run([&](size_t w) {
      auto& t = tasks[w];
      if (t.s) return;
      for (size_t i = 0; i < t.servers.size(); ++i) {
        auto& hints = parts[w][i].hints;
        for (size_t s = 1; s < stripes; ++s) {
          auto& other = parts[w + s][i].hints;
          for (size_t k = 0; k < other.size(); ++k)
            for (size_t r = 0; r < other[k].size(); ++r)
              for (size_t j = 0; j < other[k][r].size(); ++j)
                hints[k][r][j] += other[k][r][j];
          other = {};
        }
        t.servers[i] =
            PackingServer(t.p0 + i, t.k1 - t.k0, std::move(parts[w][i]));
      }
    });
  } else {
    // Independent fixed-mask preprocessing can run concurrently.
    const char* prep_env = std::getenv("PIR_PREPROCESS_THREADS");
    const int requested_threads = prep_env ? std::stoi(prep_env) : 8;
    Check(requested_threads > 0 && requested_threads <= 64,
          "PIR_PREPROCESS_THREADS must be in [1,64]");
    offline_threads = std::min(size_t(requested_threads), eff_p);
    tasks[0].servers.resize(eff_p);
    for (size_t first = 0; first < eff_p; first += offline_threads) {
      std::vector<std::future<void>> jobs;
      for (size_t u = first; u < std::min(eff_p, first + offline_threads);
           ++u) {
        jobs.push_back(std::async(std::launch::async, [&, u] {
          tasks[0].servers[u] =
              PackingServer(u, eff_c, BuildStage1(u, 0, eff_c, 0));
        }));
      }
      for (auto& job : jobs) job.get();
    }
  }
  const double offline_ms = Ms(offline_begin);
  if (!synthetic) {
    Check(input.peek() == std::char_traits<char>::eof(),
          "Trailing workload data");
  }
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
            << ",\"cross_color\":" << (cross_color_resp ? "true" : "false")
            << ",\"selector_bits\":" << selector_bits
            << ",\"centered_db\":" << (centered ? "true" : "false")
            << ",\"synthetic\":" << (synthetic ? "true" : "false")
            << ",\"rows\":[";
  for (size_t i = 0; i < p; ++i) std::cout << (i ? "," : "") << rows[i];
  std::cout << "],\"online_cpus\":[";
  for (size_t i = 0; i < online_cpus.size(); ++i)
    std::cout << (i ? "," : "") << online_cpus[i];
  std::cout << "],\"color_groups\":" << color_groups
            << ",\"column_groups\":" << column_groups
            << ",\"stripes\":" << stripes << ",\"tasks\":[";
  for (size_t w = 0; w < tasks.size(); ++w) {
    std::cout << (w ? "," : "") << "{\"worker\":" << w << ",\"cpu\":"
              << (w < online_cpus.size() ? std::to_string(online_cpus[w])
                                         : "null")
              << ",\"colors\":[" << tasks[w].p0 << "," << tasks[w].p1
              << "],\"columns\":[" << tasks[w].k0 << "," << tasks[w].k1
              << "],\"stripe\":" << tasks[w].s << "}";
  }
  std::cout << "]}" << std::endl;
  auto fft_ctx = Take(pm::FftContext::Create(11));  // log2(2048)
  // Decryption noise of every timed query, in units of q1 (failure at 256).
  double noise_sq = 0, noise_max = 0;
  size_t noise_count = 0;
  for (size_t trial = 0; trial < size_t(warmups) + q * repeats; ++trial) {
    const bool warmup = trial < size_t(warmups);
    const size_t qi = warmup ? trial % q : (trial - warmups) / repeats;
    const int rep = warmup ? -1 : (trial - warmups) % repeats;
    const auto client_begin = Clock::now();
    // A fixed --client_seed makes downloads reproducible across thread
    // layouts (compare download_fnv64).
    auto client = Take(pm::PirClient<uint64_t>::Create(
        Params(2048, eff_c), Prng(SelectorSeed(0)),
        Prng(opt.client_seed ? SelectorSeed(opt.client_seed + qi, 'C')
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
    // The key's y_vec_h is transformed once, not once per color; the transform
    // counts towards packing finalize.
    const std::vector<Poly> y_vec_h(
        request.packing_key.begin() + pack_gadget.num_digits,
        request.packing_key.end());
    const auto fft_begin = Clock::now();
    const auto y_vec_h_fft = Take(pm::YVecHFft(rlwe_params, y_vec_h, *fft_ctx));
    const double fft_ms = Ms(fft_begin);
    // Stage 1 outputs of task w's i-th unit; stripe 0 tasks sum their stripes'
    // and encode the unit's responses for column group h into
    // wires[unit * H + h].
    std::vector<std::vector<std::vector<uint64_t>>> partial(tasks.size());
    std::vector<std::string> wires(eff_p * column_groups);
    std::vector<Phases> phases_a(tasks.size()), phases_b(tasks.size());
    auto Pack = [&](size_t w, Phases& ms) {
      const auto& t = tasks[w];
      for (size_t i = 0; i < t.servers.size(); ++i) {
        const size_t u = t.p0 + i;
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
        wire.reserve(response.ciphertexts.size() * (6400 + 4608));
        for (size_t k = 0; k < response.ciphertexts.size(); ++k) {
          wire.append(server->GetPrepackedAModSwitched(k));
          PackCoeffsAppend(response.ciphertexts[k].b.Coeffs(), 18, wire);
        }
        wires[u * column_groups + (w / stripes) % column_groups] =
            std::move(wire);
        ms.encode += Ms(encode_begin);
      }
    };
    const auto tasks_begin = Clock::now();
    workers.Run([&](size_t w) {
      const auto task_begin = Clock::now();
      Phases& ms = phases_a[w];
      const auto& t = tasks[w];
      auto task_request = request;
      for (size_t i = 0; i < t.servers.size(); ++i) {
        partial[w].emplace_back();
        if (!t.servers[i]) continue;
        if (!cross_color_resp)
          task_request.first_dimension_query = queries[t.p0 + i];
        const auto db_before = t.servers[i]->GetDbMultiplyTime();
        partial[w].back() = Take(t.servers[i]->ComputeStage1(task_request));
        ms.db += absl::ToDoubleMilliseconds(t.servers[i]->GetDbMultiplyTime() -
                                            db_before);
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
    for (const auto& ph : {&phases_a, &phases_b})
      for (const auto& task_ms : *ph) server_cpu_ms += task_ms.busy;
    const auto decode_begin = Clock::now();
    off = 0;
    std::vector<std::vector<uint64_t>> recovered;
    double client_response_unpack_ms = 0, client_decrypt_ms = 0;
    // Expected plaintext of one response and which coefficients carry data.
    std::vector<uint64_t> expected;
    std::vector<bool> used;
    auto AddNoise = [&](const pm::PirResponse<uint64_t>& response) {
      if (warmup || !opt.noise_stats) return;
      const auto noise = Take(client->DecryptionNoise(response, expected));
      for (size_t i = 0; i < noise.size(); ++i) {
        if (!used[i]) continue;
        const double e = double(noise[i]);
        noise_sq += e * e;
        noise_max = std::max(noise_max, std::abs(e));
        ++noise_count;
      }
    };
    if (cross_color_resp) {
      recovered.reserve(p);
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
        expected.assign(2048, 0);
        used.assign(2048, false);
        for (size_t idx = base; idx < end;) {
          const size_t slot = idx / d_row;
          const size_t off_in_row = idx % d_row;
          const size_t take = std::min(d_row - off_in_row, end - idx);
          if (recovered.size() <= slot)
            recovered.emplace_back(c * 2048, 0);
          std::copy_n(blk.begin() + (idx - base), take,
                      recovered[slot].begin() + off_in_row);
          for (size_t i = 0; i < take; ++i) {
            expected[idx - base + i] =
                raw[slot][targets[qi * p + slot] * c * 2048 + off_in_row + i];
            used[idx - base + i] = true;
          }
          idx += take;
        }
        client_decrypt_ms += Ms(t);
        AddNoise(response);
      }
    } else {
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
        const auto row = raw[slot].begin() + targets[qi * p + slot] * c * 2048;
        expected.assign(row, row + c * 2048);
        used.assign(c * 2048, true);
        AddNoise(response);
      }
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
  if (opt.noise_stats) {
    // Decryption fails once |noise| reaches q1 / (2p) = 2^(25 - 16 - 1). The
    // noise is a Gaussian part plus the uniform +-2^(25 - 18 - 1) rounding of b
    // to q2, so the failure estimate convolves the two.
    const double threshold = 256, half = 64;
    const double sigma = std::sqrt(noise_sq / noise_count);
    const double sigma_g =
        std::sqrt(std::max(sigma * sigma - half * half / 3, 1.0));
    double fail = 0;
    constexpr int kSteps = 512;
    for (int i = 0; i < kSteps; ++i) {
      const double u = -half + 2 * half * (i + 0.5) / kSteps;
      fail += 0.5 * (std::erfc((threshold - u) / sigma_g / std::sqrt(2.0)) +
                     std::erfc((threshold + u) / sigma_g / std::sqrt(2.0)));
    }
    std::cout << "{\"type\":\"noise\",\"coefficients\":" << noise_count
              << ",\"max_abs_q1\":" << noise_max << ",\"std_q1\":" << sigma
              << ",\"gaussian_std_q1\":" << sigma_g
              << ",\"threshold_q1\":" << threshold
              << ",\"margin_sigma\":" << threshold / sigma
              << ",\"log2_failure_per_coefficient\":"
              << std::log2(fail / kSteps) << "}" << std::endl;
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
