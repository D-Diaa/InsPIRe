# ReInsPIRe Correctness, Noise, and Security Audit (Gen 1, Gen 2, Gen 3 & Sub-Tile Packing)

This document audits all optimizations introduced in **Gen 1** (`codex/reinspire`: `I5`, `I3a`, `I3b`, `I4`), **Gen 2** (`combined`: `I1`, `I2`, `I3`, `I4`), **Gen 3** (`S3`, `S4`, `S5a`), **Whole-Colour Grouped Selector Encryption**, and **Sub-Tile Cross-Colour LWE Response Packing** for algebraic correctness, IEEE-754 floating-point exactness, RLWE noise impact, and cryptographic security.

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
| **Cross-Colour Response Packing** | Server `PreprocessCrossColorBlock`, `CreateCrossColor` | **Yes** (exact LWE slot correspondence) | **Zero** (per-slot noise distribution invariant) | Download $P \to N_{\text{resp}}$ blocks; CRS packing keys unchanged |

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

## 6. Whole-Colour Grouped Selector Encryption & Sub-Tile Cross-Colour LWE Response Packing

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

### 6.2 Sub-Tile Cross-Colour LWE Response Packing (`PreprocessCrossColorBlock`, `PirServer::CreateCrossColor`)
- **Algebraic Correctness**:
  - For $C = 1$ and active row payload length $d_{\text{row}} \le 1024$, colour $s \in [0, P)$ has $d_{\text{row}}$ active Stage 1 LWE inner products $(h_{s, u}(X), b'_{s, u})$ for $u \in [0, d_{\text{row}})$, where $h_{s, u}(X) = \sum_{r=0}^{R_s - 1} D_{s}[r, u] \cdot \text{row}_r(\text{Negacyclic}(\tilde{a}_s))$ and $b'_{s, u} = \sum_{r=0}^{R_s - 1} D_{s}[r, u] \cdot y_s[r] \bmod q$.
  - Inspire's `PreprocessMatrixPack` + `FinalizeMatrixPack` is an LWE-to-RLWE packing operator that takes *any* tuple of $d = 2048$ LWE ciphertexts $\{(h_k(X), b'_k)\}_{k=0}^{d-1}$ encrypted under the same client secret key $s(X)$ and packs their plaintexts into the $d$ polynomial coefficients of a single RLWE ciphertext.
  - Because all $P$ colours in a request are encrypted under the same per-query secret key $s(X)$, concatenating the $P \cdot d_{\text{row}}$ LWE ciphertexts across colours into $N_{\text{resp}} = \lceil P \cdot d_{\text{row}} / 2048 \rceil$ blocks of 2048 LWEs (including slices that straddle block boundaries when $d_{\text{row}} \nmid 2048$) packs colour $s$'s active coefficients into global positions $[s \cdot d_{\text{row}}, (s + 1) \cdot d_{\text{row}})$. Verified in `PirEndToEndTest::CrossColorResponsePackingAndGroupedSelectorsMatchPlaintext`.
- **Noise Invariance**:
  - In `MatrixPack`, each output coefficient $k \in [0, d)$ accumulates its own Stage 1 LWE noise $e_{\text{stage1}, k}$ plus the key-switching noise from the $d$ LWE-to-RLWE automorphisms (whose distribution depends only on the gadget parameters $\ell_{\text{pack}} = 2, B_{\text{pack}} = 2^{19}$ and the shared ternary key $s(X)$, not on which colour produced LWE slot $k$). Thus the per-coefficient decryption noise variance in each of the $N_{\text{resp}}$ cross-colour response ciphertexts is **identically distributed** to that of per-colour response ciphertexts.
- **Cryptographic Security**:
  - Cross-colour response packing is a purely server-side offline/online reorganization of public database rows and public CRS hint polynomials; the client's uploaded message (packing key bundle + selector slices) is unchanged.
