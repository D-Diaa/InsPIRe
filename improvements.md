# ReInsPIRe Improvements (Gen 1, Gen 2 & Gen 3)

All online, offline, and wire optimizations across **Gen 1** (`I5, I3a, I3b, I4`), **Gen 2** (`I1, I2, I3, I4`), and **Gen 3** (`S3, S4, S5a`), classified into three tiers: **(1) Algorithmic & Transform-Domain**, **(2) Arithmetic Radix & Cache/SIMD Kernels**, and **(3) Wire Serialization, Modulus Switching & Sampling**. Detailed benchmarks and 3-way comparisons are in [`results.md`](./results.md); the correctness, noise, and security audit is in [`audit.md`](./audit.md).

## Summary by Class

| Class | ID | Target Phase | Change | Impact |
|---|---|---|---|---|
| **1. Algorithmic & Transform-Domain** | **Gen 1 `I3a`** | Server `pack_finalize` | Hoist query-independent `FFT(t_vec_h)` to offline setup and `FFT(y_vec_h)` to once per query ($8 \to 0$ forward FFTs per call) | `pack_finalize`: `109.0 → 54.1 ms` (**2.02×**) |
| **1. Algorithmic & Transform-Domain** | **Gen 2 `I2a/b`** | Server `pack_finalize` & offline | Conjugate-symmetric `TwoDigitsFft` ($d_0 + i d_1$) + paired real IFFT ($11 \to 2$ transforms per packed RLWE) | `pack_finalize`: `54.1 → 23.5 ms` (**4.63×** vs Gen 0) |
| **1. Algorithmic & Transform-Domain** | **Gen 3 `S4c`** | Server `pack_finalize` & offline `t_vec_h_ffts_` | Replace length-$2d=4096$ zero-padded real FFTs + cyclic wrap subtraction with length-$d=2048$ twisted negacyclic FFTs (`ctx.negacyclic_plan`) | `pack_finalize`: **-52% to -65%** (`36.0 → 16.2 ms` at $P=297$); `2×` smaller `t_vec_h_ffts_` |
| **1. Algorithmic & Transform-Domain** | **Gen 1 `I4` + Gen 2 `I1b`** | Client `keygen/selector/decrypt`, Server `modswitch` | Cache secret key in twisted `f64` negacyclic FFT domain (`MultFft`) + pre-modswitch CRS mask `a_resp` offline | Client crypto: `370.9 → 22.5 ms` (**16.5×**); `decrypt`: `261.6 → 6.2 ms` (**42.3×**) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 1 `I5` + Gen 2 `I4a/b`** | Server `pack_matrix` (`CondensedMultiply<int32_t>`) | $\mathbb{Z}_{2^{64}}$ $2^{32}$-radix split ($y = y_{\text{lo}} + 2^{32}y_{\text{hi}}$) + interleaved `y_packed` + 6-row register blocking (12 YMM accumulators) | `pack_matrix`: `445.9 → 224.3 → 176.7 ms` (**2.52×**, ~25 GB/s DRAM ceiling) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 1 `I3b` + Gen 2 `I3` + Gen 3 `S3b`** | Server `inner_product` (`CondensedMultiply<uint16_t>`) | Sign-flipped `int16_t` DB ($D' = D - 32768$) + 16-lane AVX2 `vpmaddwd` (`_mm256_madd_epi16`) 3-digit ($14$-bit) kernel with hoisted column decomposition | `inner_product`: **-11% to -25%** on top of Gen 2 (`25.4 → 19.1 ms` on `1MiB_M2048_P7_C6`) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 2 `I2c`** | Server/Client FFT (`crypto/fft.h`) | AVX2+FMA stage-twiddle butterfly kernel (2 complex butterflies per `__m256d` FMA, radix-2/4 first stages) | Contributes to `pack_finalize` and `2.3×` offline speedup |
| **3. Wire, Modulus Switching & Sampling** | **Gen 2 `I1a/c`, `I2d`, `I4c`** | Bit-packing, `Rescale`, `FftContext`, offline `Transpose` | 128-bit word-stream bit-packer/unpacker; divisionless shift `Rescale`; persistent `fft_ctx_`; direct `CreateCondensed<int32_t>` | Wire pack/unpack: `43.0 → 3.6 ms` (**12.1×**); Offline setup: `117.7 → 51.8 s` (**2.27×**) |
| **3. Wire, Modulus Switching & Sampling** | **Gen 3 `S3a` + `S4a/b`** | Selector upload & response download wire | 40-bit selector modulus switching (`5 B/coeff` vs `6.5 B/coeff`, **-23.1%**) + single-rounding $q_1=2^{25}$ decrypt (**-1,024 B/ct** download, **-21%** noise std) + headerless wire & offline pre-packed `a_mod_switched_t1_packed_` | Total comm: **-8.5% to -15.3%** (`-65 KiB` to `-421 KiB`); `server_encode_ms`: **-45% to -56%** |
| **3. Wire, Modulus Switching & Sampling** | **Gen 3 `S5a`** | Client `SampleSecretKey` & `GenerateRlweSamples` | Bit-sliced `absl::popcount` centered binomial ($\sigma^2=40$, `2.5` `Rand64` words/coeff) & ternary samplers + lock-free `FastChaChaPrng` | `client_query_ms`: **3.2×–3.5× faster** (`110.5 → 31.6 ms` at $P=297$); `T=32` roundtrip **1.65×–1.73× faster** |
| **1. Algorithmic & Transform-Domain** | **Grouped Selectors** | Client `CreateGroupedSelector`, Offline `RotateGroupMaskForInterval` | Greedy whole-colour bin-packing ($\sum_{i \in g} R_i \le 2048$) into $G_{\text{sel}}$ RLWE encryptions + offline signed negacyclic mask rotation $X^{-\text{offset}_i} a_g(X) \bmod (X^d + 1)$ | Selector encryptions: $P \to G_{\text{sel}}$ (**up to 16× fewer** at $R_i \le 128$); zero wire or online server change |
| **1. Algorithmic & Transform-Domain** | **Cross-Colour Response Packing** | Server `PreprocessCrossColorBlock`, `CreateCrossColor` & Client unpack | When $C=1$ and $d_{\text{row}} \le 1024$, contract only $d_{\text{row}}$ active Stage 1 rows per colour and pack $P \cdot d_{\text{row}}$ LWE outputs across colours into $N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$ RLWE blocks | Download, online `pack_matrix`, `pack_finalize`, and client decrypt scale by $N_{\text{resp}} / P$ (**-50% to -75%**) |

---

## Class 1: Algorithmic & Transform-Domain Improvements

1. **Hoisted LWE-to-RLWE Packing Transforms (`Gen 1 I3a` — `crypto/lwes_to_rlwe.cc`, `crypto/pir/server/pir_server.cc`)**:
   - `FinalizeMatrixPack` computes $\langle \mathbf{t}_{\text{vec}, h}, \mathbf{y}_{\text{vec}, h} \rangle$ via a chunked negacyclic FFT inner product. Because $\mathbf{t}_{\text{vec}, h}$ depends only on the database/CRS and $\mathbf{y}_{\text{vec}, h}$ is shared across all partitions and columns of a request, precomputing `FFT(t_vec_h)` offline (`PrecomputeTVecHFfts`) and `FFT(y_vec_h)` once per request eliminates all 8 per-call forward FFTs.
   - A multi-color request (Tiptoe adapter, `crypto/pir/benchmarking/tiptoe_workload.cc`) calls `ProcessResponse` once per color, and every color shares the packing key. The adapter therefore computes `FFT(y_vec_h)` once per request and passes it to every call (optional `y_vec_h_fft` argument), which removes $P - 1$ identical transforms. Its time still counts in `server_pack_finalize_ms` and `server_packing_ms`.
2. **Hermitian-Folded `TwoDigitsFft` & Paired Real IFFT (`Gen 2 I2a/b` — `crypto/polynomial_fft.cc`)**:
   - Packs pairs of real digit chunks into one complex vector $Z[k] = d_0[k] + i d_1[k]$, runs a single twisted forward FFT, and separates $\text{FFT}(d_0), \text{FFT}(d_1)$ in $\mathcal{O}(d)$ via conjugate symmetry $\overline{Z[d - 1 - k]}$, halving forward FFTs in `ToChunkedFft` and offline preprocessing.
   - Dually, packs pairs of real output accumulators as $A[k] + i B[k]$ to recover both polynomials from a single inverse FFT (`2` IFFTs instead of `3` per packed ciphertext). Combined with `I3a`, reduces per-call transforms in `FinalizeMatrixPack` from **11 FFTs to 2 IFFTs**.
3. **Length-$d$ Twisted Negacyclic FFTs in `FinalizeMatrixPack` (`Gen 3 S4c` — `crypto/polynomial_fft.cc`, `crypto/lwes_to_rlwe.cc`)**:
   - Replaces the length-$2d = 4096$ zero-padded real FFTs (`ctx.real_plan`) and post-IFFT cyclic wrap-around subtraction (`out[j] - out[j + d]`) in `TwoDigitsFft`, `ToChunkedFft`, `InnerProductFft`, and `RecombineDigitProducts` with **length-$d = 2048$ twisted negacyclic FFTs** (`ctx.negacyclic_plan` with root $\psi = e^{i\pi/d}$).
   - Cuts the offline precomputed `t_vec_h_ffts_` footprint in half (`256 KiB` vs `512 KiB` per column), halves the frequency-domain complex FMA loop (`2048` vs `4096` complex points), and reduces `server_pack_finalize_ms` by **52%–65%** while producing **bit-for-bit identical** polynomial coefficients.
4. **Twisted-FFT Secret Key Cache & Pre-Modswitched CRS Mask (`Gen 1 I4` + `Gen 2 I1b` — `crypto/polynomial_fft.cc`, `crypto/encryption.cc`)**:
   - Caches the ternary secret key $\mathbf{s} \in \{-1, 0, 1\}^d$ once in a twisted $d$-point `f64` negacyclic FFT domain (`FftPolynomial`). Because $u + v + \log_2 d \le 44$ bits (`kFftExactProductBits`) guarantees exact integer recovery after `nearbyint`, the modulus-switched response decrypts in a **single unchunked `f64` transform** (replacing 52-bit NTT multiplications).
   - Pre-rescaling the deterministic CRS mask $\mathbf{a}_{\text{resp}}$ from $q=2^{52}$ to $q_1$ once at setup (`a_mod_switched_t1_`) eliminates online mask rescaling during both server `ModSwitch` and client `Decrypt`.

---

## Class 2: Arithmetic Radix & Cache/SIMD Kernel Optimizations

1. **$\mathbb{Z}_{2^{64}}$ $2^{32}$-Radix Split & 6-Row Blocked Packing Matrix (`Gen 1 I5` + `Gen 2 I4a/b` — `crypto/matrix.cc`)**:
   - Splits the 64-bit vector once per call as $y = y_{\text{lo}} + 2^{32} y_{\text{hi}} \pmod{2^{64}}$ with $y_{\text{lo}} = \text{int32}(y \bmod 2^{32})$ and $y_{\text{hi}} = (y \gg 32) + \text{bit}_{31}(y)$ (canceling sign extension), replacing emulated 64-bit SIMD multiplies with native 32-bit `MulEven` ($32\times 32\to 64$) + `Mul` ($32\times 32\to 32$).
   - Gen 2 `I4` interleaves `(y_lo, y_hi)` into a single `y_packed` stream and unrolls **6 matrix rows** per loop (12 YMM accumulators in 16 YMM registers), cutting vector L1 load traffic by `33%` and streaming the 128 MiB packing matrix at the single-core DRAM ceiling (~25 GB/s).
2. **Sign-Flipped `int16_t` DB & 16-Lane AVX2 `vpmaddwd` Contraction Kernel (`Gen 1 I3b` + `Gen 2 I3` + `Gen 3 S3b` — `crypto/matrix.cc`)**:
   - Gen 1 `I3b` and Gen 2 `I3` introduced the 6-row blocked database kernel with next-row-block software prefetching (`hwy::Prefetch`).
   - Gen 3 `S3b` sign-flips the condensed `uint16_t` database entries once offline during `Matrix<uint16_t>::Condense`:
     $$D'_{r,j} = D_{r,j} - 2^{15} \in [-32768, 32767]$$
     and decomposes each centered query coefficient $v_r \in [-2^{39}, 2^{39}] \cdot 2^{12}$ (or general 52-bit $v_r$ decomposed into 3 signed balanced 14-bit digits $d_0, d_1, d_2 \in [-8192, 8192]$ with $v_r \equiv d_0 + d_1 2^{14} + d_2 2^{28}$).
   - Pairs of rows $(2r, 2r+1)$ are multiplied and accumulated into 32-bit signed integers in a single **16-lane AVX2 `_mm256_madd_epi16` (`vpmaddwd`)** instruction (4× the lane density of 64-bit `pmuludq`), with exact 32-bit overflow safety ($\lvert -32768 \cdot 8192 \rvert \times 2 \times 1024 = 2^{29} \cdot 2 = 2^{30} < 2^{31}-1$) and a single hoisted scalar bias restoration $32768 \sum_r v_r \pmod{2^{52}}$ per output row.
3. **AVX2+FMA Stage-Twiddle Butterfly Kernel (`Gen 2 I2c` — `crypto/fft.h`)**:
   - Precomputes contiguous per-stage twiddle tables and executes 2 complex butterflies per 256-bit `__m256d` FMA with dedicated twiddle-free radix-2/radix-4 initial stages.

---

## Class 3: Wire Serialization, Modulus Switching & Sampling

1. **128-Bit Word-Stream Bit-Packing & Headerless Wire (`Gen 2 I1a` + `Gen 3 S4b` — `crypto/polynomial.cc`, `crypto/pir/benchmarking/tiptoe_workload.cc`, `crypto/pir/server/pir_server.cc`)**:
   - Gen 2 `I1a` replaced bit-by-bit loops in `ToProto`/`CreateFromProto` with a `uint128_t` word accumulator flushing/reading 64-bit words.
   - Gen 3 `S4b` removes redundant 32-bit length prefixes from fixed-schema wire messages (`PackCoeffsAppend`/`UnpackCoeffs`) and pre-packs the static modulus-switched response mask `a_mod_switched_t1_packed_` (`6,400 B` per column) once at server setup so online `ProcessResponse(..., /*skip_a_t1=*/true)` skips `ct.a` bit-packing entirely (`-45% to -56%` `server_encode_ms`).
2. **Divisionless Power-of-Two `Rescale` & Single-Rounding $q_1 = 2^{25}$ Decryption (`Gen 2 I1c` + `Gen 3 S4a` — `crypto/polynomial.cc`, `crypto/encryption.cc`)**:
   - Gen 2 `I1c` replaced 128-bit software divisions (`__udivti3`) by powers of two with `(x + round_offset) >> shift`.
   - Gen 3 `S4a` eliminates the intermediate rounding of $a_{\text{sw}} \cdot s$ from $q_1$ to $q_2 = 2^{18}$ in `DecryptAfterModulusSwitchImpl`: instead of rounding twice ($q \to q_1 \to q_2 \to p$), it left-shifts $b_{\text{sw}}$ by $\log_2(q_1/q_2) = 7$ bits into $\mathbb{Z}_{q_1}$, computes $(a_{\text{sw}} \cdot s + (b_{\text{sw}} \ll 7)) \bmod q_1$, and rounds **once** from $q_1 = 2^{25}$ to $p = 2^{16}$. This eliminates the $\sigma_{\text{mid}}^2 = \frac{d}{6} = 341.33\text{ LSB}_{q_2}^2$ intermediate rounding variance, allowing $\log_2 q_1$ to drop from `29` to `25` (**`-1,024 B` per response ciphertext** AND **`-20% to -22%` lower total decryption noise**).
3. **40-Bit First-Dimension Selector Modulus Switching (`Gen 3 S3a` — `crypto/pir/benchmarking/tiptoe_workload.cc`)**:
   - Modulus-switches each first-dimension selector coefficient from $q = 2^{52}$ to $q_{\text{sel}} = 2^{40}$ (`y_sw = (y + 2^11) >> 12`, `5 B/coeff` vs `6.5 B/coeff`, **`-23.1%` selector upload bytes**) and rescales back by `<< 12` on server unpack. Because $q/q_{\text{sel}} = 2^{12}$ aligns with the `vpmaddwd` kernel's 40-bit centered query input, the added selector rounding variance at $q_2 = 2^{18}$ is only $\le 0.16\text{ LSB}_{q_2}^2$ at $M=2048$ (over $2000\times$ smaller than the $341.33\text{ LSB}_{q_2}^2$ saved by `S4a`).
4. **Bit-Sliced `popcount` Centered Binomial & Ternary Samplers + Lock-Free `FastChaChaPrng` (`Gen 3 S5a` — `crypto/encryption.cc`, `crypto/pir/benchmarking/tiptoe_workload.cc`)**:
   - Replaces `shell_encryption`'s `SampleFromErrorDistribution<MontgomeryInt<Uint32>>` (which allocated a `MontgomeryInt` vector and called `Rand8()` 20 times per coefficient) with `SampleCenteredBinomialAndSubtract`, which draws `5` `Rand64()` words per `2` coefficients (`160` bits/coeff for $\sigma^2 = 40$) and computes $\text{popcount}(x) - \text{popcount}(y) - (a \cdot s)_i \bmod q$ directly in registers.
   - Replaces `SampleFromUniformTernary<MontgomeryInt<Uint32>>` in `SampleSecretKey` with direct bitsliced rejection sampling into `CoeffType`, and replaces mutex-locked `::rlwe::ChaChaPrng` with single-threaded `FastChaChaPrng`, speeding up `client_query_ms` by **`3.2×–3.5×`**.
5. **Persistent Server `FftContext` & Direct `CreateCondensed<int32_t>` (`Gen 2 I2d`, `I4c` — `crypto/pir/server/pir_server.cc`)**:
   - Reuses a per-server `fft_ctx_` workspace instead of allocating `320 KiB` per `FinalizeMatrixPack` call, and constructs `combined_pack_matrix_` directly from column-major `pack_matrices` without an intermediate `Matrix::Transpose()` allocation.

---

## Multicore Online Server (`PIR_THREAD_CPUS`, Tiptoe adapter)

`crypto/pir/benchmarking/tiptoe_workload.cc` can answer each request on several pinned threads. With `PIR_THREAD_CPUS` unset it behaves as before: one online thread, pinned by `PIR_ONLINE_CPU` if set, and `PIR_PREPROCESS_THREADS` offline jobs; the new setup fields then describe a single task (T = 1), and query records are unchanged.

- **`PIR_THREAD_CPUS=c0,c1,...`**: one worker per listed logical CPU, so T = number of entries. The ids must be unique and inside the process affinity mask. The variable cannot be combined with `PIR_ONLINE_CPU`, and `PIR_PREPROCESS_THREADS` is ignored. Invalid settings exit nonzero with a message.
- **Decomposition**: for P colors and C columns, $G = \min(P, T)$ color groups and $H = \min(C, \lfloor T/G \rfloor)$ column groups, each a balanced contiguous split with boundaries $\lfloor i n / k \rfloor$. Task $(g, h)$ runs on worker $w = gH + h$, pinned to `c_w`. Worker 0 is the main thread, which also runs the client. Workers persist across requests, and CPUs beyond the $G H$ tasks stay idle.
- **Owner computes**: each worker pins itself, then builds one `PirServer` per color of its task holding only the task's columns (a column is one 4096-byte plaintext of each row). Per-column preprocessing does not depend on the other columns, so these servers return exactly the full table's ciphertexts for those columns, and their memory is local to the CPU that streams it. Every server, with its `FftContext` and timers, is used by a single thread.
- **Per request**: the main thread decodes the key bundle and the selectors and computes `FFT(y_vec_h)` once (see Class 1). The workers only read them and encode their ciphertexts into per-(color, column group) buffers, which the main thread joins in color order, columns in order. The wire format and byte counts are those of the single-thread run.
- **Records**: the setup record adds `online_threads` (T), `online_cpus`, `color_groups`, `column_groups` and `tasks` (`worker`, `cpu`, half-open `colors` and `columns` ranges). In threaded mode `offline_threads` is the number of tasks. In query records, `server_ms` is the main thread's wall time from request decode to serialized response. The `server_*_ms` phase fields follow the critical path: the serial phases (ingest; the shared FFT within finalize and packing; joining the buffers within encode) plus the phases of the slowest task. The threaded mode adds `server_cpu_ms`, the busy server time summed over all workers.
- **Placement**: one thread per physical core (SMT siblings slow requests down), spread over both NUMA nodes (`pack_matrix` streams 32 MiB per (color, column), so DRAM bandwidth per node is the limit), and preferably not CPU 0.

### Non-Power-of-Two Colour Tables (Tiptoe adapter)

- **Change (`crypto/pir/benchmarking/tiptoe_workload.cc:237`)**: Relaxed the workload header check from `Check(m > 0 && m <= 2048 && !(m & (m - 1)), "Unsupported M")` to `Check(m > 0 && m <= 2048, "Unsupported M")`. The core ReInsPIRe library already accepts any $1 \le R_i \le 2048$ when $t = 1$ (`pir_params.cc:35-55` only requires $t \mid \text{num\_entries}$).
- **Wire & work formulas ($R_i$ = real rows of colour $i$, vs padded $M_i = 2^{\lceil \log_2 R_i \rceil}$)**:
  | Quantity | Formula ($R_i$ = real rows) |
  |---|---|
  | **Upload** | $53{,}248 + 5 \sum_{i=0}^{P-1} R_i$ bytes (was $53{,}248 + 5 \sum M_i$; wire granularity is **1 row = 5 B**) |
  | **Download** | $11{,}008 \cdot P \cdot C$ bytes (**unchanged**) |
  | **Online DB MACs** | $2048 \cdot C \cdot \sum_{i=0}^{P-1} \text{round\_up}(R_i, 32)$ (**32-row server granularity**) |
  | **Packing & offline hint** | $2 \cdot 2048^2 \cdot P \cdot C$ online packing MACs; offline hint pads to $2048$ (**unchanged**) |
- **32-row server granularity**: `PreprocessingServer` zero-pads database matrix columns to a multiple of 32 for the 16-lane AVX2 `uint16_t`/`int16_t` contraction kernel (`preprocessing_server.h:135-154`), and `PirServer::ProcessResponse` zero-pads the unpacked $R_i$-coefficient query vector to match (`pir_server.cc:266-277`), while the offline hint always zero-pads columns to $d = 2048$ (`preprocessing_server.h:232-246`).
- **Measured effect**:
  | Geometry | `T=1` Server | `T=32` Server | Upload Bytes |
  |---|---|---|---|
  | `2×1311, C=29` | **−15%** | noise | `73,728 → 66,358 B` (`-10.0%`, DB MACs `-35.9%`) |
  | `38×1300, C=1` | **−14%** | **−11%** | `442,368 → 300,248 B` (`-32.1%`) |
  | `11×239` and `38×2000` | noise | noise | small drop (`5 B` per unpadded row) |
  Because LWE-to-RLWE packing dominates online server time and is independent of $R_i$, the server latency speedup is modest while upload decreases by $5(M_i - R_i)$ bytes per colour.
- **Unchanged correctness, noise & security**:
  - **Correctness**: Verified across $R_i \in \{1, 2, 17, 31, 32, 33, 63, 64, 65, 96, 97, 239, 874, 1000, 1311, 2016, 2017, 2047, 2048\}$, mixed heights, $C \in \{1, 2, 3, 5, 29\}$, $T \in \{1, 4, 8, 16, 32\}$, boundary targets $0$ and $R_i - 1$, a 100-query stress run, and clean rejection of $M = 0$ and $M = 2049$.
  - **Noise (bit-identical download)**: With the client seed fixed, the serialized download for real-row tables ($R_i$) is **100% bit-for-bit identical** to the power-of-two padded tables ($M_i$) across both single-threaded and multi-threaded execution (padding rows are zero and contribute $0 \bmod 2^{64}$, and the hint is identical). Covered by `PirEndToEndTest::TiptoeNonPowerOfTwoDownloadIsBitIdenticalToPaddedAndThreaded` in `crypto/pir/pir_end_to_end_real_params_test.cc`.
  - **Security**: The uploaded selector is the first $R_i$ coefficients of the same 2048-coefficient RLWE ciphertext under the unchanged `SelectorSeed(slot)` CRS, packing key bundle, packing seed, and hint—a public projection of the padded upload. Revealing $R_i$ via upload length (`5·R_i`) leaks nothing new because table populations are already public layout metadata.

### Whole-Colour Grouped Selector Encryption (`CreateGroupedSelector`, `BuildSelectorGroupPlans`, `RotateGroupMaskForInterval`)

When colour table heights satisfy $R_i < d = 2048$, encrypting a separate degree-2048 RLWE selector per colour wastes $d - R_i$ coefficients of each RLWE sample. Whole-colour grouped selector encryption packs multiple consecutive colours into a single degree-$d$ RLWE encryption on the client while leaving the wire format and online server completely unchanged:

- **Greedy whole-colour bin-packing (`BuildSelectorGroupPlans` in `crypto/pir/pir_params.h`)**:
  - Consecutive colours $i \in g$ with $\sum_{i \in g} R_i \le d = 2048$ are assigned disjoint contiguous coefficient intervals $[\text{offset}_i, \text{offset}_i + R_i) \subset [0, d)$, reducing client selector RLWE encryptions from $P$ to $G_{\text{sel}} \ge \lceil \sum_i R_i / 2048 \rceil$.
  - For each group $g$, `PirClient::CreateGroupedSelector` (`crypto/pir/client/pir_client.cc`) constructs a single multi-hot plaintext $m_g(X) = \sum_{i \in g} X^{\text{offset}_i + r_i^*}$ ($r_i^* \in [0, R_i)$), samples one group CRS mask $a_g(X)$ from `GroupSelectorSeed(g)`, and encrypts $m_g(X)$ under one RLWE sample $(a_g(X), b_g(X))$.
  - The client slices each colour's interval $b_g[\text{offset}_i : \text{offset}_i + R_i)$, applies the 40-bit modulus switch, and appends the `5·R_i`-byte slice to the upload stream in colour order.
- **Offline signed negacyclic mask rotation (`RotateGroupMaskForInterval` in `crypto/pir/pir_params.h`, `PreprocessingServer::Preprocess` in `crypto/pir/server/preprocessing_server.h`)**:
  - In `ComputeHintMatrix`, column $r \in [0, R_i)$ of colour $i$'s database multiplies row $\text{offset}_i + r$ of the negacyclic matrix $\text{Negacyclic}(a_g)$.
  - For any $\text{offset}_i + r < d$, row $\text{offset}_i + r$ of $\text{Negacyclic}(a_g)$ is identically row $r$ of $\text{Negacyclic}(\tilde{a}_i)$, where
    $$\tilde{a}_i(X) = X^{-\text{offset}_i} a_g(X) \bmod (X^d + 1), \qquad \tilde{a}_i[k] = \begin{cases} a_g[k + \text{offset}_i] & 0 \le k < d - \text{offset}_i, \\ -a_g[k + \text{offset}_i - d] \bmod q & d - \text{offset}_i \le k < d. \end{cases}$$
  - Rotating $a_g(X)$ by $X^{-\text{offset}_i} \bmod (X^d + 1)$ once during offline `Preprocess` allows every colour server to keep its standard $[0, R_i)$ database layout and online contraction with zero online server changes.
- **Zero cross-colour contamination & RLWE security**:
  - Because each coefficient of $b_g(X) = \Delta m_g(X) - a_g(X)s(X) + e_g(X) \bmod q$ is rounded to 40 bits independently and colour $i$'s server contracts only $b_g[\text{offset}_i : \text{offset}_i + R_i)$, entries of $m_g$ outside $[\text{offset}_i, \text{offset}_i + R_i)$ have **zero** effect on colour $i$'s query slice or response noise.

### Sub-Tile Cross-Colour LWE Response Packing (`PreprocessCrossColorBlock`, `PirServer::CreateCrossColor`)

When $C = 1$ and each colour row occupies only $d_{\text{row}} \le 1024$ active `uint16_t` coefficients ($g = \lfloor 2048 / d_{\text{row}} \rfloor \ge 2$, e.g. $H = 41, M_{\text{sub}} = 24, b = 8 \implies d_{\text{row}} = 984$), the trailing $2048 - d_{\text{row}}$ coefficients of each 2048-coefficient table row are zero padding. Sub-tile cross-colour response packing contracts only the $d_{\text{row}}$ active rows per colour and packs the resulting $P \cdot d_{\text{row}}$ LWE outputs across colours into $N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$ RLWE response ciphertexts:

- **Offline cross-colour block preprocessing (`PreprocessingServer::PreprocessCrossColorBlock` in `crypto/pir/server/preprocessing_server.h`)**:
  - Response block $b \in [0, N_{\text{resp}})$ covers global LWE indices $[b \cdot 2048, \min((b + 1) \cdot 2048, P \cdot d_{\text{row}}))$, decomposing that range into contiguous colour slices `(color = s, u0, take)` (including slices that straddle a 2048-coefficient block boundary when $d_{\text{row}} \nmid 2048$).
  - For each slice, `PreprocessCrossColorBlock` extracts only rows `[u0, u0 + take)` of colour $s$, computes their `take` LWE hint polynomials under colour $s$'s rotated selector mask $\tilde{a}_s(X)$, and condenses the `take × round_up(R_s, 32)` `int16_t` Stage 1 submatrix `CrossColorDbSlice`.
  - The assembled 2048-row hint block is passed once to `PreprocessMatrixPack` under the shared `packing_seed`, producing a single `PreprocessMatrixPackOutput` for response block $b$.
- **Online cross-colour server & client unpack (`PirServer::CreateCrossColor` in `crypto/pir/server/pir_server.cc`, `tiptoe_workload.cc`)**:
  - Online, each cross-colour response server multiplies its `db_slices_` against the corresponding colour query vectors `(*request.color_queries)[sl.color]` to populate `b_prime_all` ($2048$ coefficients total across slices), then runs a single $2048 \times 4096$ `pack_matrix` multiply, a single `FinalizeMatrixPack`, and a single modulus switch.
  - The client decrypts the $N_{\text{resp}}$ RLWE response ciphertexts and scatters each slice `(s, off_in_row, take)` back into `recovered[s][off_in_row : off_in_row + take)`.
- **Wire & work formulas ($C = 1, d_{\text{row}} \le 1024, N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$)**:
  | Quantity | Per-Colour Response ($P$ blocks) | Cross-Colour Response ($N_{\text{resp}}$ blocks) |
  |---|---|---|
  | **Upload** | $53{,}248 + 5 \sum_{i=0}^{P-1} R_i$ B | $53{,}248 + 5 \sum_{i=0}^{P-1} R_i$ B (**unchanged**) |
  | **Download** | $11{,}008 \cdot P$ B | $11{,}008 \cdot N_{\text{resp}}$ B (**reduced by $N_{\text{resp}} / P$**) |
  | **Online Stage 1 DB MACs** | $2048 \sum_{i=0}^{P-1} \text{round\_up}(R_i, 32)$ | $d_{\text{row}} \sum_{i=0}^{P-1} \text{round\_up}(R_i, 32)$ (**reduced by $d_{\text{row}} / 2048$**) |
  | **Online Stage 2 Packing MACs** | $2 \cdot 2048^2 \cdot P$ | $2 \cdot 2048^2 \cdot N_{\text{resp}}$ (**reduced by $N_{\text{resp}} / P$**) |
  | **Client Decryptions** | $P$ RLWE decryptions | $N_{\text{resp}}$ RLWE decryptions (**reduced by $N_{\text{resp}} / P$**) |

---

## Assumptions, Caveats & Tradeoffs

- **Arbitrary table row counts ($1 \le R_i \le 2048$)**: Colour table row counts $R_i$ do **not** need to be powers of two or multiples of the ring degree ($t = 1$ requires only $1 \le R_i \le 2048$, and general `PirParams` requires only $t \mid \text{num\_entries}$). Upload scales at 1-row granularity (`5 B/row`), while online DB contraction scales at 32-row granularity (`round_up(R_i, 32)`).

- **Precomputed FFT & mask memory (`Gen 1 I3a`, `Gen 2 I1b`, `Gen 3 S4b/c`)**: With length-$d$ negacyclic FFTs (`S4c`), `PrecomputeTVecHFfts` uses `256 KiB` per column (`50%` less than Gen 2), `a_mod_switched_t1_` uses `16 KiB` per column, and `a_mod_switched_t1_packed_` uses `6.25 KiB` per column. Note: Under glibc's default arena allocator, long-lived allocations after multi-threaded preprocessing can retain freed thread arenas in RSS; `tcmalloc`/`jemalloc` avoids this.
- **Exactness budget (`Gen 1 I4`, `Gen 2 I2b`, `Gen 3 S4c`)**: `kFftExactProductBits = 44` is a conservative analytic bound ($53 - 8 - 1$) that holds with zero extra chunking at $d=2048$; `allow_paired_ifft` is guarded by $\log_2(N_{\text{polys}}) + \log_2 d + u_{\text{mag}} + v_{\text{mag}} \le 50.0$ (RMS error $\sim 0.0055 \ll 0.5$).
- **Single-threaded execution per `PirServer` / `PirClient` instance (`Gen 1 I4`, `Gen 2 I2d`)**: Scratch buffers in `fft_ctx_` are reused across calls on the same instance; concurrent queries sharing a single `PirServer` shard should mark `fft_ctx_` `thread_local`. The threaded Tiptoe adapter avoids this by giving every worker its own servers.
- **Hardware generalization**: Matrix kernels (`I5, I3b, I3, I4`) use Google Highway SIMD, while `S3b` (`MultiplyCondensedMatrixVector<uint16_t>`) and `I2c` (`crypto/fft.h`) include `#if defined(__AVX2__)` fast paths with clean portable Highway/scalar `#else` fallbacks.
