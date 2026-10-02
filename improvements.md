# ReInsPIRe Improvements (Gen 1 & Gen 2)

All online and offline optimizations across **Gen 1** (`I5, I3a, I3b, I4`) and **Gen 2** (`I1, I2, I3, I4`), classified into three tiers: **(1) Algorithmic & Transform-Domain**, **(2) Arithmetic Radix & Cache/SIMD Kernels**, and **(3) Wire Serialization & Plumbing**. None alters cryptographic parameters, noise margins, key material, or wire byte counts. Detailed benchmarks and 3-way comparisons are in [`results.md`](./results.md); the correctness, noise, and security audit is in [`audit.md`](./audit.md).

## Summary by Class

| Class | ID | Target Phase | Change | Impact (`E072_M2048_P41_C4`) |
|---|---|---|---|---|
| **1. Algorithmic & Transform-Domain** | **Gen 1 `I3a`** | Server `pack_finalize` | Hoist query-independent `FFT(t_vec_h)` to offline setup and `FFT(y_vec_h)` to once per query ($8 \to 0$ forward FFTs per call) | `pack_finalize`: `109.0 → 54.1 ms` (**2.02×**) |
| **1. Algorithmic & Transform-Domain** | **Gen 2 `I2a/b`** | Server `pack_finalize` & offline | Conjugate-symmetric `TwoDigitsFft` ($d_0 + i d_1$) + paired real IFFT ($11 \to 2$ transforms per packed RLWE) | `pack_finalize`: `54.1 → 23.5 ms` (**4.63×** vs Gen 0) |
| **1. Algorithmic & Transform-Domain** | **Gen 1 `I4` + Gen 2 `I1b`** | Client `keygen/selector/decrypt`, Server `modswitch` | Cache secret key in twisted `f64` negacyclic FFT domain (`MultFft`) + pre-modswitch CRS mask `a_resp` to $q_1=2^{29}$ offline | Client crypto: `370.9 → 22.5 ms` (**16.5×**); `decrypt`: `261.6 → 6.2 ms` (**42.3×**) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 1 `I5` + Gen 2 `I4a/b`** | Server `pack_matrix` (`CondensedMultiply<int32_t>`) | $\mathbb{Z}_{2^{64}}$ $2^{32}$-radix split ($y = y_{\text{lo}} + 2^{32}y_{\text{hi}}$) + interleaved `y_packed` + 6-row register blocking (12 YMM accumulators) | `pack_matrix`: `445.9 → 224.3 → 176.7 ms` (**2.52×**, ~25 GB/s DRAM ceiling) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 1 `I3b` + Gen 2 `I3`** | Server `inner_product` (`CondensedMultiply<uint16_t>`) | Unsigned $2^{32}$-radix split + 4-plane `+64B` cache-set-skewed `y_split` + 6-row blocking + next-block prefetch | `inner_product`: `173.7 → 92.7 → 85.9 ms` (**2.02×**) |
| **2. Arithmetic Radix & Cache/SIMD Kernels** | **Gen 2 `I2c`** | Server/Client FFT (`crypto/fft.h`) | AVX2+FMA stage-twiddle butterfly kernel (2 complex butterflies per `__m256d` FMA, radix-2/4 first stages) | Contributes to `pack_finalize` (**4.63×**) and `2.3×` offline speedup |
| **3. Wire Serialization & Plumbing** | **Gen 2 `I1a/c`, `I2d`, `I4c`** | Bit-packing, `Rescale`, `FftContext`, offline `Transpose` | 128-bit word-stream bit-packer/unpacker; divisionless shift `Rescale`; persistent `fft_ctx_`; direct `CreateCondensed<int32_t>` | Wire pack/unpack: `43.0 → 3.6 ms` (**12.1×**); Offline setup: `117.7 → 51.8 s` (**2.27×**) |

---

## Class 1: Algorithmic & Transform-Domain Improvements

1. **Hoisted LWE-to-RLWE Packing Transforms (`Gen 1 I3a` — `crypto/lwes_to_rlwe.cc`, `crypto/pir/server/pir_server.cc`)**:
   - `FinalizeMatrixPack` computes $\langle \mathbf{t}_{\text{vec}, h}, \mathbf{y}_{\text{vec}, h} \rangle$ via a chunked negacyclic FFT inner product. Because $\mathbf{t}_{\text{vec}, h}$ depends only on the database/CRS and $\mathbf{y}_{\text{vec}, h}$ is shared across all partitions and columns of a request, precomputing `FFT(t_vec_h)` offline (`PrecomputeTVecHFfts`) and `FFT(y_vec_h)` once per request eliminates all 8 per-call forward FFTs.
2. **Hermitian-Folded `TwoDigitsFft` & Paired Real IFFT (`Gen 2 I2a/b` — `crypto/polynomial_fft.cc`)**:
   - Packs pairs of real digit chunks into one complex vector $Z[k] = d_0[k] + i d_1[k]$, runs a single twisted forward FFT, and separates $\text{FFT}(d_0), \text{FFT}(d_1)$ in $\mathcal{O}(d)$ via conjugate symmetry $\overline{Z[d - 1 - k]}$, halving forward FFTs in `ToChunkedFft` and offline preprocessing.
   - Dually, packs pairs of real output accumulators as $A[k] + i B[k]$ to recover both polynomials from a single inverse FFT (`2` IFFTs instead of `3` per packed ciphertext). Combined with `I3a`, reduces per-call transforms in `FinalizeMatrixPack` from **11 FFTs to 2 IFFTs**.
3. **Twisted-FFT Secret Key Cache & Pre-Modswitched CRS Mask (`Gen 1 I4` + `Gen 2 I1b` — `crypto/polynomial_fft.cc`, `crypto/encryption.cc`)**:
   - Caches the ternary secret key $\mathbf{s} \in \{-1, 0, 1\}^d$ once in a twisted $d$-point `f64` negacyclic FFT domain (`FftPolynomial`). Because $u + v + \log_2 d \le 44$ bits (`kFftExactProductBits`) guarantees exact integer recovery after `nearbyint`, the 29-bit modulus-switched response decrypts in a **single unchunked `f64` transform** (replacing 52-bit NTT multiplications).
   - Pre-rescaling the deterministic CRS mask $\mathbf{a}_{\text{resp}}$ from $q=2^{52}$ to $q_1=2^{29}$ once at setup (`a_mod_switched_t1_`) eliminates online mask rescaling during both server `ModSwitch` and client `Decrypt`.

---

## Class 2: Arithmetic Radix & Cache/SIMD Kernel Optimizations

1. **$\mathbb{Z}_{2^{64}}$ $2^{32}$-Radix Split & 6-Row Blocked Packing Matrix (`Gen 1 I5` + `Gen 2 I4a/b` — `crypto/matrix.cc`)**:
   - Splits the 64-bit vector once per call as $y = y_{\text{lo}} + 2^{32} y_{\text{hi}} \pmod{2^{64}}$ with $y_{\text{lo}} = \text{int32}(y \bmod 2^{32})$ and $y_{\text{hi}} = (y \gg 32) + \text{bit}_{31}(y)$ (canceling sign extension), replacing emulated 64-bit SIMD multiplies with native 32-bit `MulEven` ($32\times 32\to 64$) + `Mul` ($32\times 32\to 32$).
   - Gen 2 `I4` interleaves `(y_lo, y_hi)` into a single `y_packed` stream and unrolls **6 matrix rows** per loop (12 YMM accumulators in 16 YMM registers), cutting vector L1 load traffic by `33%` and streaming the 128 MiB packing matrix at the single-core DRAM ceiling (~25 GB/s).
2. **4-Plane Set-Skewed `y_split` & 6-Row Blocked DB Kernel (`Gen 1 I3b` + `Gen 2 I3` — `crypto/matrix.cc`)**:
   - Applies the unsigned $2^{32}$-radix split to 16-bit database entries (`4` `uint16_t`s per 64-bit lane), using next-row-block software prefetching (`hwy::Prefetch`) to hide row-boundary hardware prefetcher resets.
   - Gen 2 `I3` compresses the 6-plane `y_split` into 4 planes padded by `+64 B` (`kPlaneSkewU32 = 16`) between planes to eliminate 8-way L1D cache set-associative conflicts at $16\text{ KiB}$ ($M=2048$) strides, and unrolls 6 rows at a time.
3. **AVX2+FMA Stage-Twiddle Butterfly Kernel (`Gen 2 I2c` — `crypto/fft.h`)**:
   - Precomputes contiguous per-stage twiddle tables and executes 2 complex butterflies per 256-bit `__m256d` FMA with dedicated twiddle-free radix-2/radix-4 initial stages.

---

## Class 3: Wire Serialization & Prototype Plumbing

1. **128-Bit Word-Stream Bit-Packing (`Gen 2 I1a` — `crypto/polynomial.cc`)**: Replaces bit-by-bit loops in `ToProto`/`CreateFromProto` with a `uint128_t` word accumulator flushing/reading 64-bit words (`12–20×` faster pack/unpack).
2. **Divisionless Power-of-Two `Rescale` & `Decrypt` (`Gen 2 I1c` — `crypto/polynomial.cc`, `crypto/encryption.cc`)**: Replaces 128-bit software divisions (`__udivti3`) by powers of two ($q=2^{52}, q_1=2^{29}, q_2=2^{18}, p=2^{16}$) with `(x + round_offset) >> shift`.
3. **Persistent Server `FftContext` & In-Place Offline NTT (`Gen 2 I2d` — `crypto/pir/server/pir_server.cc`)**: Reuses a per-server `fft_ctx_` workspace instead of allocating `320 KiB` per `FinalizeMatrixPack` call, and reuses a 4-polynomial buffer in offline InspiRING NTT evaluation.
4. **Direct `CreateCondensed<int32_t>` (`Gen 2 I4c` — `crypto/pir/server/pir_server.cc`)**: Constructs `combined_pack_matrix_` directly from column-major `pack_matrices` without an intermediate `Matrix::Transpose()` allocation.

---

## Quantitative Contribution by Class (`E072_M2048_P41_C4`: `1,149.8 ms → 314.0 ms`, `3.66×`)

| Class | Latency Saved (`E072`) | Share of Total Latency Saved | Primary Effect |
|---|---:|---:|---|
| **Class 1: Algorithmic & Transform-Domain** | `-433.9 ms` | **51.9%** | `16.5×` client crypto speedup; `4.6×` server `pack_finalize` speedup |
| **Class 2: Arithmetic Radix & Cache/SIMD Kernels** | `-357.0 ms` | **42.7%** | `2.52×` `pack_matrix` speedup; `2.02×` `inner_product` speedup |
| **Class 3: Wire Serialization & Plumbing** | `-44.9 ms` | **5.4%** | `12.1×` wire pack/unpack speedup; `2.27×` offline setup speedup |

---

## Assumptions, Caveats & Tradeoffs

- **Precomputed FFT & mask memory (`Gen 1 I3a`, `Gen 2 I1b`)**: `PrecomputeTVecHFfts` adds `512 KiB` per partition (`21 MiB` at `P=41`) and `a_mod_switched_t1_` adds `64 KiB` per partition (`2.6 MiB` at `P=41`). Note: Under glibc's default arena allocator, long-lived allocations after multi-threaded preprocessing can retain freed thread arenas in RSS; `tcmalloc`/`jemalloc` avoids this.
- **Exactness budget (`Gen 1 I4`, `Gen 2 I2b`)**: `kFftExactProductBits = 44` is a conservative analytic bound ($53 - 8 - 1$) that holds with zero extra chunking at $d=2048$; `allow_paired_ifft` is guarded by $\log_2(N_{\text{polys}}) + \log_2 d + u_{\text{mag}} + v_{\text{mag}} \le 50.0$ (RMS error $\sim 0.0055 \ll 0.5$).
- **Single-threaded execution per `PirServer` / `PirClient` instance (`Gen 1 I4`, `Gen 2 I2d`)**: Scratch buffers in `fft_ctx_` are reused across calls on the same instance; concurrent queries sharing a single `PirServer` shard should mark `fft_ctx_` `thread_local`.
- **Hardware generalization**: All matrix kernels (`I5, I3b, I3, I4`) use portable Google Highway SIMD (compiling natively to x86 AVX2/AVX-512 and ARM64 NEON/SVE). `I2c` uses `#if defined(__AVX2__) && defined(__FMA__)` with a clean portable scalar `#else` fallback.
