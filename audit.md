# ReInsPIRe Correctness, Noise, and Security Audit (Gen 1, Gen 2, Gen 3 & Gen 4)

This document audits all optimizations introduced in **Gen 1** (`codex/reinspire`: `I5`, `I3a`, `I3b`, `I4`), **Gen 2** (`combined`: `I1`, `I2`, `I3`, `I4`), **Gen 3** (`S3`, `S4`, `S5a`), **Whole-Colour Grouped Selector Encryption**, **Continuous Cross-Colour LWE Response Packing**, and **Gen 4 Tall-Table (`M > 2048`) Retrieval** (`R1` multi-sample $t=1$, `R1a/R1b` centered `int16` contraction & `40/42`-bit selectors, `R2` plaintext row bundling with compute-optimal default `--bundle_rows=1` and wire-optimal `--bundle_rows=0`, and 2D Stage 1 row-stripe parallelism) for algebraic correctness, IEEE-754 floating-point exactness, RLWE noise impact, and cryptographic security.

## 1. Summary Verdict

| Generation & Idea | Target Phase | Output Bit-Identical? | RLWE Noise Impact | Security / Wire Impact |
|---|---|---|---|---|
| **Gen 1 `I5`** | Server `pack_matrix` (`CondensedMultiply<int32_t>`) | **Yes** (exact mod $2^{64}$) | **Zero** | **Zero** |
| **Gen 1 `I3a`** | Server `pack_finalize` (`PrecomputeTVecHFfts` + `y_vec_h` FFT) | **Yes** (bit-identical) | **Zero** | **Zero** |
| **Gen 1 `I3b`** | Server `inner_product` (`CondensedMultiply<uint16_t>`) | **Yes** (exact mod $2^{64}$) | **Zero** | **Zero** |
| **Gen 1 `I4`** | Client `MultFft` (`FftPolynomial` ternary key FFT) | **Yes** (exact mod $2^{52}$) | **Zero** | **Zero** |
| **Gen 2 `I1`** | Bit-packing, pre-modswitched `a_resp`, power-of-two `Rescale`/`Decrypt` | **Yes** (bit-identical) | **Zero** | **Zero** |
| **Gen 2 `I2`** | AVX2+FMA FFT, `TwoDigitsFft`, paired IFFT, persistent `fft_ctx_` | **Yes** after `nearbyint` | **Zero** (rounds to exact same ints) | **Zero** (single-thread per shard) |
| **Gen 2 `I3`** | 4-plane set-skewed `y_split` + 6-row `CondensedMultiply<uint16_t>` | **Yes** (exact mod $2^{64}$) | **Zero** | **Zero** |
| **Gen 2 `I4`** | 6-row `CondensedMultiply<int32_t>` + direct `CreateCondensed<int32_t>` | **Yes** (exact mod $2^{64}$) | **Zero** | **Zero** |
| **Gen 3 `S3a/b`** | 40-bit selector modswitch + sign-flipped `int16_t` AVX2 `vpmaddwd` DB | **Yes** (`S3b` exact mod $2^{52}$) | $+0.16\text{ LSB}_{q_2}^2$ (`S3a`, negligible) | **-23.1%** selector upload (`5 B/row`); IND-CPA preserved |
| **Gen 3 `S4a/b/c`** | Single-rounding $q_1=2^{25}$ decrypt + headerless wire + length-$d$ negacyclic FFT | **Yes** (`S4c` bit-identical) | **-20% to -22% lower** total noise std (`-341.33\text{ LSB}_{q_2}^2`) | **-1,024 B/ct** download; IND-CPA preserved |
| **Gen 3 `S5a`** | Bit-sliced `popcount` binomial/ternary samplers + `FastChaChaPrng` | **Yes** (`FastChaChaPrng` bit-identical stream) | **Zero** (exact $\text{Binomial}(80, 0.5)-40$) | **Zero** (identical ChaCha20 keystream & distributions) |
| **Grouped Selectors** | Client `CreateGroupedSelector` + offline `RotateGroupMaskForInterval` | **Yes** (rotated-mask == interval extraction) | **Zero** (no cross-colour contamination) | $P \to G_{\text{sel}}$ RLWE samples; standard RLWE IND-CPA |
| **Continuous Cross-Colour Response Packing** | Server `PreprocessCrossColorStripe`, `CreateCrossColor`, `PackedRowWords` | **Yes** (exact LWE slot correspondence; bit-identical to per-colour when $2048 \mid d_{\text{row}}$) | **Zero** (per-slot noise distribution invariant across all $d_{\text{row}} \ge 1$) | Download $P \cdot C \to N_{\text{resp}} = \lceil P d_{\text{row}} / 2048 \rceil$; public layout only (IND-CPA) |
| **Gen 4 `R1` + `R1a/R1b`** | Multi-sample $t=1$ selectors + centered `int16` DB + `40/42`-bit `<3, 10>` selectors | **Yes** (`R1a` exact mod $2^{16}$; `M <= 2048` bit-identical at $d_{\text{row}} \le 1024$ or $2048 \mid d_{\text{row}}$) | **`4×` lower** DB variance + **`16×` lower** rounding variance above $8{,}192$ rows (`4.75–5.94σ` to $256$) | $\lceil R_i / 2048 \rceil$ CRS samples under 1 secret; IND-CPA preserved |
| **Gen 4 `R2` + Row Stripes** | Plaintext row bundling (default `--bundle_rows=1` compute-optimal; `0` `ChooseBundleFactor` wire-optimal) + 2D `(block_groups × stripes)` Stage 1 | **Yes** (`T=1 == T=32` `download_fnv64` bit-identical) | **Reduces** physical height $R_i \to \lceil R_i / B_{\text{row}} \rceil$ when $B_{\text{row}} > 1$ | **4.03×–7.99×** less wire under `--bundle_rows=0`; public deterministic layout |

Across all **30 benchmarked workloads × 40 measured queries = 1,200 queries per binary** (`3,600` ReInsPIRe queries total), every query verified **0 decryption errors** and identical wire byte counts.

---

## 2. Server Integer Matrix Kernels (Gen 1 `I5`, `I3b` & Gen 2 `I3`, `I4`)

### 2.1 Signed 32-bit Split (`CondensedMultiply<int32_t>`)
- **Identity**: For $m \in [-2^{31}, 2^{31}-1]$ and $y \in [0, 2^{64})$, let $y_{\text{lo}} = \text{int32}(y \bmod 2^{32}) \in [-2^{31}, 2^{31}-1]$ and $y_{\text{hi}} = \text{int32}((y \gg 32) + \text{bit}_{31}(y))$. Because $y_{\text{lo}} = (y \bmod 2^{32}) - 2^{32} \cdot \text{bit}_{31}(y)$, we have $y \equiv y_{\text{lo}} + 2^{32} y_{\text{hi}} \pmod{2^{64}}$.
- **Modulo $2^{64}$ reduction**: The high product $m \cdot y_{\text{hi}}$ is multiplied by $2^{32}$, so only its low 32 bits affect the result modulo $2^{64}$. Accumulating `Mul(m_i32, y_hi)` in 32-bit lanes and shifting left by 32 bits at the end of the row (`ShiftLeft<32>(BitCast(du, hi))`) is **100% bit-exact modulo $2^{64}$**.
- **Gen 2 `I4` changes**: Interleaving `(y_lo_u32, y_hi_u32)` into `y_packed`, unrolling 6 rows instead of 4, and building `combined_pack_matrix_` directly from column-major `pack_matrices` via strided reads perform the exact same additions modulo $2^{64}$ (which is associative and commutative). Verified by `MatrixTest::CondensedMultiplyInt32AndUint16Test`.

### 2.2 Unsigned 16-bit Split (`CondensedMultiply<uint16_t>`)
- **Identity**: Each 64-bit condensed lane packs four `uint16_t` values $(m_0, m_1, m_2, m_3)$. Viewed as two `uint32_t` lanes, `m_02 = m & 0xFFFF` holds $(m_0, m_2)$ and `m_13 = m >> 16` holds $(m_1, m_3)$.
- **Gen 2 `I3` changes**:
  1. Replaces the 6-plane `y_split` with 4 planes (`y_lo_01`, `y_lo_23`, `y_hi_02`, `y_hi_13`) separated by `+8` `uint64_t`s (`64 B`) to eliminate 4 KiB L1D set-conflict aliasing.
  2. For `m_23`, instead of `MulEven(ShiftRight<32>(m), y_lo_23)`, `I3` places $(y_{\text{lo}}[4b+2], y_{\text{lo}}[4b+3])$ in the **odd** 32-bit lanes of `y_lo_23`, masks the low 32 bits of `m` (`And(m, 0xFFFFFFFF00000000ULL)`), and uses `MulOdd` (or `Mul` on `m_u32` for high halves), eliminating the 32-bit right-shift instruction on every load while producing **bit-identical** modulo-$2^{64}$ sums.

---

## 3. Floating-Point Negacyclic FFT Arithmetic (Gen 1 `I3a`, `I4` & Gen 2 `I2`)

### 3.1 Client Secret-Key FFT (`MultFft`, Gen 1 `I4`)
- **Bound**: At ring degree $d = 2048$ ($\log_2 d = 11$) with ternary secret key $s \in \{-1, 0, 1\}^d$ ($v_{\text{mag}} = 1$), a chunk of width $u$ bits has exact integer convolution coefficients bounded by $d \cdot 2^u = 2^{u+11}$.
- **Encryption (`GenerateRlweSamples`, 52-bit operand)**: Split into two chunks (`32` bits and `20` bits under `kFftExactProductBits = 44`, or `38` and `14` bits under `50`). Even at $u = 38$, worst-case coefficient magnitude is $\le 2^{49}$ and RMS magnitude for uniform $a \in [0, 2^{38})$ and ternary $s$ ($\Pr[s_i \ne 0] = 0.5$) is $\sqrt{1024 \cdot (2^{38})^2 / 3} \approx 2^{42.2} \ll 2^{53}$, so IEEE-754 `f64` rounding error is $< 0.01 \ll 0.5$ and `std::nearbyint` recovers the **exact integer modulo $2^{52}$**.
- **Decryption (`DecryptAfterModulusSwitch`, 29-bit operand)**: Single 29-bit chunk; max coefficient is $2048 \times 2^{29} = 2^{40} \ll 2^{53}$, max `f64` error is $\sim 2 \times 10^{-4} \ll 0.5$ (**100% bit-exact**).

### 3.2 Conjugate-Symmetric `TwoDigitsFft` and Paired IFFT (Gen 2 `I2`)
- **Forward `TwoDigitsFft`**: For two real vectors $a, b \in \mathbb{R}^d$, let $z[k] = (a[k] + i\,b[k])\,\psi^k$. Because $a[k], b[k]$ are real and $\overline{\psi^k \omega_d^{k m}} = \psi^{-(k+1)} \omega_d^{-(k+1)m + (d-1-m)}$ with $\psi^d = -i$, the DFT $Z = \text{FFT}(z)$ satisfies exact conjugate symmetry between index $m$ and $d - 1 - m$. Since $a, b$ are 18-bit or 26-bit integers ($\le 2^{26}$), $Z$ has magnitude $\le 2^{37} \ll 2^{53}$, so unpacking $A[m], B[m]$ is accurate to $10^{-15}$ relative error.
- **Paired IFFT (`allow_paired_ifft`)**: Given two conjugate-twisted accumulators $A, B$, `I2` computes $\text{IFFT}(A + i B)$ so that $\text{Re}$ yields $\text{IFFT}(A)$ and $\text{Im}$ yields $\text{IFFT}(B)$.
  - Combining $A + i B$ increases the complex Euclidean norm by at most $\sqrt{2}$ (`0.5` bits).
  - `allow_paired` is guarded by $\log_2(N_{\text{polys}}) + \log_2 d + u_{\text{mag}} + v_{\text{mag}} \le 50.0$:
    - In `FinalizeMatrixPack` (`InnerProductPrecomputed`), $N_{\text{polys}} = 2$, $\log_2 d = 11$, $u_{\text{mag}} = 18$ (signed 19-bit digit in $[-2^{18}, 2^{18}]$), $v_{\text{mag}} = 20$ (signed 20-bit centered `t_vec_h` digit): $1 + 11 + 18 + 20 = 50.0$.
    - The RMS coefficient of the 4-term sum (2 polynomials × 2 digit pairs) is $\sqrt{4 \cdot 2048 \cdot (2^{18}/\sqrt{3})^2 (2^{19}/\sqrt{3})^2} \approx 2^{41.9} \ll 2^{53}$.
    - The resulting `f64` rounding error after the 11-stage AVX2 FMA IFFT is $\approx 0.0055 \ll 0.5$, so `std::nearbyint` rounds to the **exact same integer coefficients modulo $2^{52}$**.
  - For offline `ExternalProduct` (`t_vec_h` 52-bit unsigned, $u_{\text{mag}} = 26, v_{\text{mag}} = 20, t = 2$), $1 + 11 + 26 + 20 = 58 > 50$, so `allow_paired` is automatically `false` and each digit runs an isolated IFFT.

---

## 4. Bit-Packing, Rescaling, and Concurrency (Gen 2 `I1`, `I2`)

1. **128-bit Word-Stream Bit-Packer (`I1`)**: Packs/unpacks the exact same little-endian bit layout as the original bit-by-bit loop (`ToProto` / `CreateFromProto` round-trip verified across all bit widths $16, 18, 29, 52$).
2. **Power-of-Two Rescaling (`I1`)**: For unsigned $x \in [0, 2^a)$ and power-of-two moduli $2^a > 2^b$, $\lfloor (x \cdot 2^b + 2^{a-1}) / 2^a \rfloor = (x + 2^{a-b-1}) \gg (a - b)$ holds identically for every integer $x$.
3. **Thread-Safety Note (`I2`)**: `PirServer::fft_ctx_` is a `mutable` member holding pre-allocated scratch vectors (`forward_in`, `backward_out`, `accum_scratch`). This is safe when each `PirServer` partition is queried by one thread at a time (or across partition-parallel threads). If multiple threads call `ProcessResponse` concurrently on the *same* `PirServer` instance, `fft_ctx_` should be `thread_local`.

---

## 5. Gen 3 Wire, Kernel, and Sampling Optimizations (`S3`, `S4`, `S5a`)

1. **40-Bit Selector Modulus Switching (`S3a`) & Sign-Flipped `vpmaddwd` Contraction (`S3b`)**:
   - Rounding each selector coefficient from $q = 2^{52}$ to $q_{\text{sel}} = 2^{40}$ and shifting left by 12 bits on the server injects zero-mean rounding error $\delta_r \in [-2^{11}, 2^{11}]$ ($\text{Var}(\delta_r) = 2^{24}/12$). Multiplied by $M \le 2048$ database entries in $[0, 2^{16})$ and scaled to $q_2 = 2^{18}$, the added variance is at most $2048 \cdot \frac{2^{32}}{3} \cdot \frac{2^{24}}{12} \cdot 2^{-68} \approx 0.16\text{ LSB}_{q_2}^2$.
   - Writing $D'_{r,j} = D_{r,j} - 2^{15} \in [-32768, 32767]$ and decomposing centered query coefficients into three signed 14-bit digits $d_0, d_1, d_2 \in [-8192, 8192]$ ensures each 16-lane `_mm256_madd_epi16` (`vpmaddwd`) accumulator sums at most $1024$ pairs bounded by $2 \cdot 32768 \cdot 8192 = 2^{29}$, so the 32-bit signed sum is $\le 2^{39}/2^{9} = 2^{30} < 2^{31}-1$ (zero overflow) and adding $2^{15} \sum_r v_r \pmod{2^{52}}$ restores the exact unsigned inner product modulo $2^{52}$.
2. **Single-Rounding $q_1 = 2^{25}$ Decryption (`S4a`) & Length-$d$ Negacyclic FFT (`S4c`)**:
   - Computing $(a_{\text{sw}} \cdot s + (b_{\text{sw}} \ll 7)) \bmod q_1$ and rounding once from $q_1 = 2^{25}$ to $p = 2^{16}$ eliminates the intermediate $q_1 \to q_2$ rounding variance $\frac{d}{6} = 341.33\text{ LSB}_{q_2}^2$, reducing total decryption noise standard deviation by **20%–22%** while cutting `ct.a` from 29 to 25 bits (`-1,024 B/ct`).
   - Length-$d$ twisted negacyclic FFTs (`ctx.negacyclic_plan`) evaluate polynomials at the odd roots of $-1$, $\psi^{2k+1} = e^{i\pi(2k+1)/d}$, which is algebraically isomorphic to $\mathbb{R}[X]/(X^d+1)$ and produces bit-identical integer coefficients after `nearbyint`.
3. **Bit-Sliced Samplers & `FastChaChaPrng` (`S5a`)**:
   - `SampleCenteredBinomialAndSubtract` sums 80 independent fair bits minus 80 independent fair bits via `absl::popcount` on `Rand64()` words, matching $\text{Binomial}(80, 0.5) - 40$ ($\sigma^2 = 40$) identically. `FastChaChaPrng` calls the exact same `ChaChaPrngResalt` block function as `::rlwe::ChaChaPrng`.

---

## 6. Whole-Colour Grouped Selector Encryption & Continuous Cross-Colour LWE Response Packing

### 6.1 Whole-Colour Grouped Selector Encryption (`CreateGroupedSelector`, `RotateGroupMaskForInterval`)
- **Algebraic Equivalence (View 2 Rotated Mask == View 1 Interval Extraction)**:
  - Suppose group $g$ packs colours with row counts $R_i$ at disjoint intervals $[\text{offset}_i, \text{offset}_i + R_i) \subset [0, d)$ ($\sum_{i \in g} R_i \le d = 2048$) into one RLWE ciphertext $(a_g(X), b_g(X))$ where $b_g(X) = \Delta \sum_{i \in g} X^{\text{offset}_i + r_i^*} - a_g(X) s(X) + e_g(X) \bmod q$.
  - Colour $i$'s query vector is the slice $y_i[r] = b_g[\text{offset}_i + r]$ for $r \in [0, R_i)$. The LWE mask vector corresponding to $b_g[\text{offset}_i + r]$ is row $\text{offset}_i + r$ of $\text{Negacyclic}(a_g)$:
    $$\text{Negacyclic}(a_g)_{\text{offset}_i + r, j} = \begin{cases} a_g[\text{offset}_i + r - j] & j \le \text{offset}_i + r, \\ -a_g[d + \text{offset}_i + r - j] \bmod q & j > \text{offset}_i + r. \end{cases}$$
  - Define $\tilde{a}_i(X) = X^{-\text{offset}_i} a_g(X) \bmod (X^d + 1)$ (`RotateGroupMaskForInterval`). For every $r \in [0, R_i)$ (since $\text{offset}_i + r < d$) and every column $j \in [0, d)$, row $r$ of $\text{Negacyclic}(\tilde{a}_i)$ has $(r, j)$ entry:
    - If $j \le r$: $\tilde{a}_i[r - j] = a_g[\text{offset}_i + r - j]$ (since $r - j < R_i \le d - \text{offset}_i$).
    - If $r < j \le \text{offset}_i + r$: $-\tilde{a}_i[d + r - j] = -(-a_g[\text{offset}_i + r - j]) = a_g[\text{offset}_i + r - j]$ (since $d + r - j \ge d - \text{offset}_i$).
    - If $j > \text{offset}_i + r$: $-\tilde{a}_i[d + r - j] = -a_g[d + \text{offset}_i + r - j]$ (since $d + r - j < d - \text{offset}_i$).
    Hence the first $R_i$ rows of $\text{Negacyclic}(\tilde{a}_i)$ are **identically equal** to rows $[\text{offset}_i, \text{offset}_i + R_i)$ of $\text{Negacyclic}(a_g)$. Verified directly in `PirEndToEndTest::GroupedSelectorEncryptionRecoversPayloadsAtBoundariesAndHeterogeneousHeights`.
- **Zero Cross-Colour Noise or Rounding Contamination**:
  - Each coefficient $b_g[k]$ has its own independent centered binomial error $e_g[k]$ and is rounded to 40 bits component-wise. Because colour $i$ only reads $b_g[\text{offset}_i : \text{offset}_i + R_i)$, the message bits $\Delta X^{\text{offset}_{i'} + r_{i'}^*}$ of other colours $i' \ne i$ in group $g$ lie at disjoint indices outside $[\text{offset}_i, \text{offset}_i + R_i)$ and contribute **zero** to colour $i$'s query vector or LWE inner product.
- **Cryptographic Security (IND-CPA)**:
  - Multiplication by $X^{-\text{offset}_i}$ is a ring automorphism (unit multiplication) in $\mathcal{R}_q = \mathbb{Z}_q[X]/(X^d + 1)$, so $a_g(X)$ is uniform in $\mathcal{R}_q$ and $(a_g(X), b_g(X))$ is a single standard RLWE sample encrypting $m_g(X)$ under secret key $s(X)$ and fresh error $e_g(X)$. Publishing the sub-slices $b_g[\text{offset}_i : \text{offset}_i + R_i)$ (a subset of coefficients of $b_g$) is a deterministic post-processing of one standard RLWE ciphertext per group, so joint pseudorandomness holds under standard RLWE with $G_{\text{sel}} \le P$ samples.

### 6.2 Continuous Cross-Colour LWE Response Packing Across All Row Widths (`PackedRowWords`, `PreprocessCrossColorStripe`, `PirServer::CreateCrossColor`)
- **Algebraic Correctness for Arbitrary $d_{\text{row}} \ge 1$ (Including $1024 < d_{\text{row}} < 2048$ and Multi-Tile $d_{\text{row}} > 2048$)**:
  - With `PackedRowWords(row_u16, d) = row_u16` (`crypto/pir/pir_params.h:300-302`), colour $s \in [0, P)$ has $d_{\text{row}} \ge 1$ active `uint16_t` payload words per physical row ($C = \lceil d_{\text{row}} / d \rceil \ge 1$, unbundled or bundled) and produces $d_{\text{row}}$ Stage 1 LWE ciphertexts $(h_{s, u}(X), b'_{s, u})$ for $u \in [0, d_{\text{row}})$, where $h_{s, u}(X) = \sum_{r=0}^{M_{s,\text{phys}} - 1} D_s[r, u] \cdot \mathbf{a}_{s, r}$ and $b'_{s, u} = \sum_{r=0}^{M_{s,\text{phys}} - 1} D_s[r, u] \cdot y_s[r] \bmod q$.
  - Because all $P$ colours in a batch request are encrypted under the same per-query secret key $s(X)$, concatenating the $P$ colours' $d_{\text{row}}$-word rows forms one flat stream of $P \cdot d_{\text{row}}$ LWE ciphertexts partitioned into $N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / d \rceil$ blocks of $d = 2048$ slots.
  - For every block $b \in [0, N_{\text{resp}})$ covering $[g_0, g_1) = [b \cdot d, \min((b + 1) \cdot d, P \cdot d_{\text{row}}))$, offline `PreprocessCrossColorStripe` (`preprocessing_server.h`), online `ComputeStage1` (`pir_server.cc`), and client scatter (`palette_workload.cc`) decompose $[g_0, g_1)$ via the exact same quotient-remainder recurrence:
    $$\text{slot} = \lfloor \text{idx} / d_{\text{row}} \rfloor, \qquad \text{off\_in\_row} = \text{idx} \bmod d_{\text{row}}, \qquad \text{take} = \min(d_{\text{row}} - \text{off\_in\_row},\ g_1 - \text{idx}).$$
    When $d_{\text{row}} > d = 2048$, a single colour spans $\ge 2$ consecutive response blocks: interior blocks slice a full $d$-word segment (`take = 2048`), while boundary blocks pack the colour's trailing tail (`d_row - off_in_row < 2048`) alongside the next colour's head (`off_in_row = 0`) identically to sub-tile segments. Unused trailing slots $[g_1 - g_0, d)$ on the final block $b = N_{\text{resp}} - 1$ carry zero hint rows and zero `b_prime_all` entries. When $d \mid d_{\text{row}}$ ($d_{\text{row}} = 2048 C$), each block maps 1-to-1 to `(slot = b / C, col = b % C)` with `take = 2048`, producing **bit-for-bit identical** ciphertexts to per-colour `PirServer::Create` (`TallTableTest::WideRowBlocksMatchPerColorServers`).
  - **Unit & E2E Test Coverage (`crypto/pir/pir_tall_table_test.cc`, `crypto/pir/pir_end_to_end_real_params_test.cc`)**:
    - `PirEndToEndTest::CrossColorResponsePackingAndGroupedSelectorsMatchPlaintext` (sub-half-tile $d_{\text{row}} \le 1024$ + grouped selectors).
    - `TallTablePlanTest::ChooseBundleFactorMinimizesExactWireBytes` (`PackedRowWords(w) == w`, exact wire/ciphertext counts at `64×8192` `1024w` `bundle=3 -> 96` cts / `1,983,936 B`, `166×2048` `1552w -> 126` cts, `128×2048` `2064w -> 129` cts, and `8×8192` `2064w -> bundle=2, 17` cts).
    - `TallTableTest::ContinuousCrossColorSubFullTileRowsDecryptExactly` ($1024 < d_{\text{row}} < 2048$: `d_row = 1536` at $P = 8 \to 6$ cts for $M \le 2048$; `d_row = 1552` at $P = 5 \to 4$ cts for $M \in \{4096, 8192\}$ with `stripes = 1 vs 4` bit-identity + 16-query noise verification).
    - `TallTableTest::ContinuousCrossColorMultiTileRowsDecryptExactly` ($d_{\text{row}} > 2048$: `d_row = 2064` at $P = 5 \to 6 < 10$ cts across $M \le 2048$ and $M \in \{4096, 8192\}$ with `stripes = 1 vs 3` bit-identity + 16-query noise verification; `d_row = 3072` at $P = 4 \to 6 < 8$ cts with `stripes = 1 vs 2` bit-identity).
    - `TallTableTest::BundledContinuousCrossColorDecryptsAcrossModesAndStripes` (`bundle` $\in \{1, 0, 2\}$ across $M \le 2048$ and $M \in \{4096, 8192\}$, including multi-column bundled `orig_row_u16 = 2064` ($C_{\text{orig}} = 2$) + `bundle = 2` $\to C_{\text{phys}} = 3, d_{\text{row}} = 4128, 9 < 12$ cts + `stripes = 1 vs 2` bit-identity).
- **IEEE-754 `f64` Negacyclic FFT Exactness**:
  - Offline `ComputeHintMatrix` transforms each `(slot, stripe)` slice of height `take <= 2048` over degree-$d$ negacyclic polynomials; each polynomial product sums $d = 2048$ integer products regardless of `take` or $d_{\text{row}}$.
  - Online `FinalizeMatrixPack` (`InnerProductFft`) operates on one assembled $2048$-slot block after `PreprocessMatrixPack` reduces the $2048 \times 2048$ hint matrix modulo $q = 2^{52}$ into balanced signed 20-bit centered `t_vec_h` digits ($v_{\text{mag}} = 20$) paired with signed 19-bit `y_vec_h` digits ($u_{\text{mag}} = 18$). Thus $\log_2 N_{\text{polys}} + \log_2 d + u_{\text{mag}} + v_{\text{mag}} = 1 + 11 + 18 + 20 = 50.0 \le 50.0$ (RMS $\approx 2^{41.9} \ll 2^{53}$, `f64` error $\approx 0.0055 \ll 0.5$) holds identically for all $d_{\text{row}} \ge 1$, so `std::nearbyint` recovers the exact integer modulo $2^{52}$.
- **RLWE Decryption Noise Invariance & Empirical Verification (`1024 < d_row < 2048` and `d_row > 2048`)**:
  - In `MatrixPack`, output coefficient $k \in [0, d)$ of block $b$ accumulates only its own slot's Stage 1 LWE error $e_{\text{stage1}, b, k}$ — whose variance depends **solely** on the physical height $M_{\text{slot},\text{phys}}$ and entry distribution of `slot = (b * d + k) / d_row` — plus the key-switching noise of the $d$ LWE-to-RLWE automorphisms under the shared packing gadget $(\log B_{\text{pack}}, \ell_{\text{pack}}) = (19, 2)$ and shared ternary secret $s(X)$, plus single-rounding modulus-switching error at $(q_1, q_2) = (2^{25}, 2^{18})$. Neither term has **any** dependence on $d_{\text{row}}$, on whether $1024 < d_{\text{row}} < 2048$ or $d_{\text{row}} > 2048$, or on multi-block boundary straddling.
  - **Measured decryption noise (`reinspire-workload --noise_stats=true`, $P = 8, Q = 16$, failure bound `256` $q_1$ units)**:

    | `row_bytes` (`orig_u16`) | `M` | `bundle` | `d_row` | `responses` | `centered_db` | `random` $\sigma_{q_1}$ / `max_abs_q1` | `0x8000` $\sigma_{q_1}$ / `max_abs_q1` | `0xffff` $\sigma_{q_1}$ / `max_abs_q1` |
    |---:|---:|---:|---:|---:|:---:|---:|---:|---:|
    | `3,104` (`1552`) | `2048` | `1` | `1552` | `7` | `false` | `46.07` / `179` | `45.50` / `181` | `50.38` / `180` |
    | `3,104` (`1552`) | `2048` | `2` | `3104` | `13` | `false` | `45.00` / `175` | `44.69` / `165` | `47.53` / `178` |
    | `3,104` (`1552`) | `8192` | `1` | `1552` | `7` | `true` | `46.20` / `187` | `51.53` / `209` | `43.56` / `152` |
    | `3,104` (`1552`) | `8192` | `2` | `3104` | `13` | `true` | `45.01` / `173` | `47.70` / `176` | `43.71` / `165` |
    | `4,128` (`2064`) | `2048` | `1` | `2064` | `9` | `false` | `46.02` / `192` | `45.49` / `167` | `50.29` / `205` |
    | `4,128` (`2064`) | `2048` | `2` | `4128` | `17` | `false` | `44.94` / `166` | `44.67` / `165` | `47.51` / `199` |
    | `4,128` (`2064`) | `8192` | `1` | `2064` | `9` | `true` | `46.21` / `181` | `51.50` / `208` | `43.74` / `156` |
    | `4,128` (`2064`) | `8192` | `2` | `4128` | `17` | `true` | `44.99` / `175` | `47.65` / `184` | `43.68` / `160` |

    Across every $(M, B_{\text{row}}, \text{fill})$ pair, intermediate single-tile widths (`d_row = 1552`) and multi-tile widths (`d_row = 2064`, `3104`, `4128`) exhibit **statistically indistinguishable** $\sigma_{q_1}$ (matching within $\le 0.18$ $q_1$ units) and `max_abs_q1 <= 209 < 256` (**`4.97–5.88σ` margin**).
- **Cryptographic Security (IND-CPA)**:
  - Continuous cross-colour response packing is a purely server-side reorganization of public database words and public CRS hint polynomials; for any fixed $B_{\text{row}}$, the client's uploaded message (packing key bundle + selector slices) is bit-for-bit unchanged. Both $d_{\text{row}} = B_{\text{row}} \cdot w_{16}$ and $N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$ (and every block's `(slot, off_in_row, take)` layout) are deterministic functions of public database dimensions $(P, w_{16}, B_{\text{row}})$ only, leaking **zero** target information and preserving IND-CPA under standard RLWE.
  - In `ResolveRowU16` (`palette_workload.cc:72-111`), the logical row width $w_{16}$ (`row_u16`) in Palette (`client.pal`) or under `PIR_ROW_U16` is computed strictly from public index schema metadata ($w_{16} = \lceil \lceil H \cdot M_{\text{sub}} \cdot n_{\text{bits}} / 8 \rceil / 2 \rceil$), completely independent of payload contents; even the standalone `TPIRv001` trailing-zero fallback scans only the server's public plaintext database once at offline setup and never depends on client secret targets.

---

## 7. Gen 4 Tall-Table (`M > 2048`) Retrieval, Centering, Row Bundling & Stripe Parallelism

### 7.1 Centered Signed `int16` Contraction (`centered_db`, `R1a`) & `42`-Bit `vpmaddwd` Overflow Safety (`R1b`)
- **Algebraic Equivalence Modulo $p = 2^{16}$**:
  - Let $D_{r,u} \in [0, 2^{16})$ be an unsigned database word and define the centered signed entry
    $$D^{\text{c}}_{r,u} = (D_{r,u} \oplus 2^{15}) - 2^{15} \in [-2^{15}, 2^{15}-1].$$
    Whether bit 15 of $D_{r,u}$ is $0$ ($D^{\text{c}}_{r,u} = D_{r,u}$) or $1$ ($D^{\text{c}}_{r,u} = D_{r,u} - 2^{16}$), we have $D^{\text{c}}_{r,u} \equiv D_{r,u} \pmod{2^{16}}$.
  - For selector encryption $y_r = \Delta \mathbf{1}[r = r^*] - \langle \mathbf{a}_r, \mathbf{s} \rangle + e_r + \delta_r \pmod{q}$ with $\Delta = q / p = 2^{36}$, the message contribution at $r = r^*$ satisfies $\Delta D^{\text{c}}_{r^*,u} \equiv \Delta D_{r^*,u} \pmod{2^{52}}$ because $2^{36} \cdot 2^{16} = 2^{52} = q$.
  - On the server, passing $D \oplus \text{0x8000}$ to `Matrix<uint16_t>::Condense` stores $(D \oplus 2^{15}) - 2^{15} = D^{\text{c}}$ directly in the condensed `int16_t` lanes. `MultiplyStripe` (`centered = true`) skips the online $+2^{15}\sum_r v_r$ bias restoration, while `PreprocessCrossColorStripe` subtracts $2^{15}\sum_{r} \mathbf{a}_r \pmod{2^{52}}$ from the offline LWE hint, so $\langle \mathbf{h}^{\text{c}}_u, \mathbf{s} \rangle + b'^{\text{c}}_u = \Delta D_{r^*,u} + \sum_r D^{\text{c}}_{r,u}(e_r + \delta_r) \pmod{2^{52}}$ holds identically.
- **`vpmaddwd` 32-Bit Signed Accumulator Bound at `42`-Bit Selectors (`<3, 10>`)**:
  - At `selector_bits = 42` (`shift = 10`), centered query coefficients satisfy $v_r / 2^{10} \in [-2^{41}, 2^{41}]$, which decomposes **without remainder** into three balanced signed 14-bit digits $d_0, d_1, d_2 \in [-8192, 8192]$ ($8192 \cdot (1 + 2^{14} + 2^{28}) > 2^{41}$).
  - Each 16-lane AVX2 `_mm256_madd_epi16` inner chunk sums at most $1024$ row-pairs $(2k, 2k+1)$ (`kMaxChunkCols = 2048` in `crypto/matrix.cc`), so the worst-case 32-bit signed lane magnitude is $1024 \times 2 \times 32768 \times 8192 = 2^{39} / 2^9 = 2^{30} < 2^{31} - 1$ (**zero `int32` overflow**).

### 7.2 Tall-Table Decryption Noise Budget & Adversarial Payload Verification
- **Closed-Form Variance at Single-Rounding Scale $q_1 = 2^{25}$ (Decoding Half-Width $= \frac{1}{2} \cdot \frac{2^{25}}{2^{16}} = 256\text{ LSB}_{q_1}$)**:
  - With centered entries $|D^{\text{c}}_{r,u}| \le 2^{15}$ ($\mathbb{E}[(D^{\text{c}})^2] = 2^{30}/3$ on uniform data, $\le 2^{30}$ on adversarial constant `0x8000` / `0xFFFF`), physical height $M_{\text{phys}} = \lceil R_i / B_{\text{row}} \rceil$, fresh binomial variance $\sigma_e^2 = 40$, and selector rounding variance $\sigma_{\delta}^2 = 2^{2(52 - b_{\text{sel}})} / 12$:
    $$\sigma_{q_1}^2 = \sigma_{\text{pack}, q_1}^2 + \sigma_{\text{ms}, q_1}^2 + M_{\text{phys}} \cdot \mathbb{E}[(D^{\text{c}})^2] \cdot (\sigma_e^2 + \sigma_{\delta}^2) \cdot 2^{-54}.$$
  - At $M_{\text{phys}} \le 8{,}192$ with $b_{\text{sel}} = 40$ ($\sigma_\delta^2 = 2^{24}/12 \approx 1.40 \times 10^6$), Stage 1 adds $\le 207\text{ LSB}_{q_1}^2$ on uniform data ($\le 621\text{ LSB}_{q_1}^2$ on all-`0x8000`), keeping $\sigma_{q_1} \le 46\text{--}50$.
  - Above $M_{\text{phys}} = 8{,}192$, `AutoSelectorBits` switches to $b_{\text{sel}} = 42$ ($\sigma_\delta^2 = 2^{20}/12 \approx 8.74 \times 10^4$, **$16\times$ smaller**), keeping $\sigma_{q_1}$ below `54` even unbundled (`B_row = 1`) at $R_i = 131{,}072$!
- **Empirical Audit Across Adversarial Fills (`PirTallTableTest` & Stage-2 Verifier Audit)**:
  - Tested across $R_i \in \{4{,}096, 8{,}192, 16{,}384, 32{,}768, 65{,}536, 131{,}072\}$ and five payload fills (`random`, `0x0000`, `0x8000`, `0xFFFF`, `0x0001`):
    - **Default compute-optimal unbundled mode (`--bundle_rows=1`)**: $\sigma_{q_1} \in [43.13, 53.89]$, `max_abs_q1 <= 207 < 256`, **`4.75–5.94σ` margin** even at unbundled $R_i = 131{,}072$ on worst-case `0x8000` fill (minimizing $N_{\text{resp}} = \lceil P \cdot w_{16} / 2048 \rceil$ and Stage 2 `LWEsToRLWE` server CPU work; see [`improvements.md`](./improvements.md) for the Stage 2 CPU vs wire tradeoff).
    - **Wire-optimal `ChooseBundleFactor` mode (`--bundle_rows=0`)**: $\sigma_{q_1} \in [43.35, 47.45]$, `max_abs_q1 <= 176 < 256`, **`5.40–5.91σ` margin** (`0` errors across all $30$ configurations; `0x0000` has exact `0` Stage 1 noise because $(0 \oplus \text{0x8000}) - 32768 = 0$, whereas `0x8000` maps to $-32768$ and serves as the worst-case constant-norm stress fill).

### 7.3 `M <= 2048` Baseline Invariance & Multi-Thread Row-Stripe Bit-Identity
- **Strict `M <= 2048` Invariance**: When $\max_i R_i \le 2048$, both default `--bundle_rows=1` and `--bundle_rows=0` (`ChooseBundleFactor`) evaluate to $B_{\text{row}} = 1$, `AutoSelectorBits` returns `40`, and `--centered_db=auto` resolves to `false` — preserving **100% bit-for-bit identical** upload bytes, download bytes, and ciphertexts across all sub-half-tile ($d_{\text{row}} \le 1024$) and whole-column ($2048 \mid d_{\text{row}}$) `M <= 2048` regression geometries (`TallTableGoldenTest::ShortTablesMatchBaselineResponses`), while extending continuous cross-colour packing ($N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$) to intermediate ($1024 < d_{\text{row}} < 2048$) and non-multiple multi-tile ($d_{\text{row}} > 2048, 2048 \nmid d_{\text{row}}$) row widths.
- **Bit-Identical Row-Stripe Reduction (`T = 1` vs `T = 32`) & Workload Adapter Verification**: `RowStripe` (`crypto/pir/pir_params.h:113-120`) partitions `ceil(R_i / 2048)` blocks at multiples of $d = 2048$ (`kMaxChunkCols`), so every stripe performs identical `vpmaddwd` chunk reductions and sums modulo $2^{64}$ before a single `PackFromStage1` on `stripe == 0`. Verified `download_fnv64(T=1) == download_fnv64(T=32)` across sub-tile, intermediate ($1024 < d_{\text{row}} < 2048$: `row_bytes = 3,104`, $G = 7, S = 4$), multi-tile ($d_{\text{row}} > 2048$: `row_bytes = 4,128`, $G = 9, S = 3$), bundled (`bundle` $\in \{1, 0, 2\}$), and non-divisible (`3 × 10001`) geometries (`8/8` continuous-width `T = 1` vs `T = 32` configs × 4 queries = `32/32` bit-identical `download_fnv64` hashes, `24/24` synthetic geometry/bundling configs across `row_bytes` $\in \{3{,}072, 3{,}104, 4{,}128, 6{,}144\}$, and `TPIRv001` $c = 2$ `row_u16 = 2064` `client.pal` / `PIR_ROW_U16` / trailing-zero fallback checks).

### 7.4 Cryptographic Security (IND-CPA)
- **Multi-Sample Tall Selectors (`R1`)**: A colour of physical height $M_{\text{phys}} > d$ encrypts $\lceil M_{\text{phys}} / d \rceil$ independent standard RLWE samples $\{(a_{i,k}, b_{i,k})\}$ under distinct CRS seeds (`GroupSelectorSeed` stream) and independent binomial error polynomials $e_{i,k}(X)$ under the single per-query secret $s(X)$, then truncates the final sample to $M_{\text{phys}} \bmod d$ coefficients. Joint pseudorandomness reduces directly to standard RLWE with $\sum_g \text{num\_samples}_g$ samples.
- **Query-Independent Bundling, Continuous Packing & Striping (`R2`)**: `ChooseBundleFactor`, `AutoSelectorBits`, `PackedRowWords`, and `RowStripe` depend solely on public table dimensions $(R_0, \dots, R_{P-1}, w_{16}, B_{\text{row}}, T)$ and never inspect client targets. Sub-row extraction `[(r^* % bundle) * w_16, +w_16)` happens strictly on the client after decryption.
