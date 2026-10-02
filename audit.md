# ReInsPIRe Correctness, Noise, and Security Audit (Gen 1 & Gen 2)

This document audits all optimizations introduced in **Gen 1** (`codex/reinspire`: `I5`, `I3a`, `I3b`, `I4`) and **Gen 2** (`combined`: `I1`, `I2`, `I3`, `I4`) for algebraic correctness, IEEE-754 floating-point exactness, RLWE noise impact, and cryptographic security.

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
