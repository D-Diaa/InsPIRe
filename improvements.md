# ReinsPIRe online-path improvements: I5, I3, I4

Three changes to the Tiptoe workload's online path (`crypto/pir/benchmarking/
tiptoe_workload.cc`). None touches parameters, key material, the hint, the
server file format or the wire format: every byte count is identical to the
previous implementation, and each change is proven bit-identical to the code
it replaces by a unit test against the old path on random and extreme inputs.
All arithmetic stays in portable Highway / plain C++; no raw intrinsics.

| | where | before | after |
|---|---|---|---|
| I5 | server `pack_matrix` | emulated 64×64 `MulAdd` per lane, ~12 GB/s | 32×32 split kernel, ~25 GB/s (DRAM-bound) |
| I3a | server `pack_finalize` | 8 forward + 3 inverse FFTs per `FinalizeMatrixPack` | forward FFTs precomputed (once per server / once per request), 3 inverse |
| I3b | server `inner_product` | emulated 64×64 `MulAdd` per 16-bit field, ~8 GB/s | 32×32 split kernel + next-row-block prefetch, ~17 GB/s cold |
| I4 | client encrypt / decrypt | NTT products per polynomial | exact chunked f64 negacyclic FFT against the cached FFT-domain secret key |

## Caveats at a glance

- **Server memory (I3a).** +512 KiB of precomputed FFTs per partition
  (2 MiB at P=4, 41 MiB at P=82). The harness's peak RSS can grow by much
  more (+36…+250 MB in the table below): glibc keeps the preprocessing
  threads' arenas from being trimmed once the long-lived FFT buffers land in
  them. This is allocator retention, not live data; it disappears with
  tcmalloc/jemalloc. No allocator workaround was added.
- **Per-client setup (I4).** `FftContext::Create` costs ≈0.4 ms and ~100 KiB
  per `PirClient`; the harness builds one client per query, so this is
  inside `keygen`. `PirClient` methods must not be called concurrently on one
  instance (already the case before because of the owned PRNGs).
- **Exactness budget (I4).** `kFftExactProductBits = 44` is a conservative
  analytic bound. It is free at d=2048 with a ternary key (32-bit chunks); at
  much larger d or wider keys it forces more chunks, and `MultFft` refuses
  (`InvalidArgument`) if no exact chunking exists rather than silently
  rounding.
- **Host-tuned prefetch (I3b).** The next-row-block prefetch distance was
  chosen by measurement on an AMD EPYC 7B13 (built `-march=znver2`); another
  microarchitecture may prefer a different distance. Correctness does not
  depend on it.
- **I5 has no tradeoff**, but its kernel is now at the single-core DRAM
  ceiling; further `pack_matrix` gains need multiple threads or a smaller
  packing matrix.
- **Everything else is unchanged:** parameters, noise, hint, server file
  format, wire format (bytes identical on all nine workloads), and the RGSW
  path.

## I5 — packing-matrix kernel (`crypto/matrix.cc`, `CondensedMultiply<int32_t>`)

The online packing step is `combined_pack_matrix_ · y_vec_g`: an int32 matrix
of `C·t·d` rows × `k_gadget·d` columns (8192 × 4096 = 128 MiB at C=4, d=2048,
once per partition) times a uint64 vector, modulo 2⁶⁴. AVX2 has no 64-bit
multiply, so the previous Highway `MulAdd` on emulated u64 lanes was
instruction-bound at ~12 GB/s.

**Idea.** Split the vector once per call, `y = y_lo + 2³²·y_hi` with
`y_lo = int32(y mod 2³²)` (sign-extended by the multiply) and
`y_hi = (y >> 32) + bit31(y)` (the +1 cancels that sign extension). Then,
modulo 2⁶⁴,

```
m·y = m·y_lo + 2³²·(m·y_hi mod 2³²)
```

so each product is one signed 32×32→64 `MulEven` plus one 32×32→32 `Mul`
whose result is shifted into the top word. A condensed lane holds
`(m[2j], m[2j+1])` as its low/high halves, which `MulEven` reads directly and
after a 32-bit shift. Rows are processed four at a time so the split vector is
loaded once per block.

**Result.** The kernel streams the matrix at ~25 GB/s (single-core DRAM
ceiling on this host); `pack_matrix` is ≈2× faster on every shape. No
tradeoff: same memory, same layout, same API.

## I3 — packing finalize and database kernel (server)

### I3a — precomputed FFTs (`crypto/lwes_to_rlwe.cc`, `crypto/polynomial_fft.cc`, `crypto/pir/server/pir_server.cc`)

`FinalizeMatrixPack` ends with `InnerProduct(t_vec_h, y_vec_h)`, a chunked
double-FFT negacyclic product. The forward transforms are redundant:
`t_vec_h` depends only on the preprocessed data and `y_vec_h` is part of the
packing key, identical for every (shard, chunk) of one request.
`Polynomial::ToChunkedFft` produces the digit FFTs exactly as the streaming
`InnerProductFft` did (shared helpers, same digit split, same accumulation
order, same inverse transforms), so the result is bit-identical. The server
transforms every `t_vec_h` once at construction (`PrecomputeTVecHFfts`, used
by both `Create` and `LoadFromBuffer`) and `y_vec_h` once per request. Per
`FinalizeMatrixPack` call this leaves the 3 inverse FFTs; `pack_finalize` is
halved on every shape.

**Tradeoff — memory.** The server keeps `num_digits · 2d` complex doubles per
(shard, chunk): 128 KiB at d=2048, 512 KiB per partition at C·t=4, i.e.
2 MiB for P=4, 20.5 MiB for P=41, 41 MiB for P=82 (measured as live heap).
Peak RSS of the harness can grow by more than that (up to a few hundred MB
on some shapes): the long-lived FFT buffers are allocated in the
preprocessing threads' glibc arenas after large transient allocations, which
prevents those arenas from being trimmed. This is allocator retention, not
live data; it does not occur with a non-arena allocator (tcmalloc/jemalloc)
and no allocator hacks were added to the library.

### I3b — database kernel (`crypto/matrix.cc`, `CondensedMultiply<uint16_t>`)

Same split as I5, unsigned: for `m < 2¹⁶`, `m·y = m·y_lo + 2³²·(m·y_hi mod
2³²)`. A condensed lane holds four 16-bit fields; viewed as uint32 lanes,
`m & 0xFFFF` gives fields (0, 2) and `m >> 16` fields (1, 3), so one `Mul`
against the matching `y_hi` pair covers two fields and `MulEven` consumes
fields 0/1 directly and 2/3 after a 32-bit shift — six multiplies per
32-byte lane group instead of four emulated 64-bit products. The split vector
is laid out as six planes so the inner loop is loads, multiplies and adds;
rows go four at a time.

**Caveat — prefetch.** The condensed rows are short (2 KiB at M=1024) and the
DB matrix is cold after the 128 MiB packing matrix streamed through, so the
hardware prefetcher restarts on every row; each row load is paired with a
`hwy::Prefetch` of the same offset in the next row block (cold 12.6 → 17.4
GB/s in isolation). The distance (one row block) was chosen by measurement on
this host; other microarchitectures may prefer a different one.

## I4 — client arithmetic (`crypto/polynomial_fft.cc`, `crypto/encryption.cc`, `crypto/pir/client/pir_client.cc`)

The client's RLWE encryptions (selector, packing/automorphism keys) and the
response decryption each multiply a polynomial by the ternary secret key. The
secret key is now cached once in a twisted d-point negacyclic f64 FFT domain
(`FftPolynomial`), and `Polynomial::MultFft(const FftPolynomial&, …)`
computes the exact product modulo 2^(8·sizeof(CoeffType)) by splitting the
other operand into chunks and rounding each chunk product.

**Exactness.** With chunk magnitudes ≤ 2ᵘ and 2ᵛ every coefficient of a chunk
product is an integer bounded by 2^(u+v+log₂d); the f64 round trip perturbs
it by at most c·2⁻⁵³ times that bound with c < 2⁸ (Percival's bound for a
radix-2 FFT convolution, ≈170 at d=2¹¹). Rounding therefore recovers the exact
integer whenever `u + v + log₂d ≤ 53 − 8 − 1 = kFftExactProductBits (44)`;
`MultFft` derives the chunk width from this and returns `InvalidArgument` if
no exact chunking exists. At d=2048 with a ternary key this gives 32-bit
chunks: two transforms for a 52-bit operand, one for the 29-bit
modulus-switched ciphertext. Encryption samples are bit-identical to the NTT
path for the same PRNG stream and decryption is bit-identical on random and
extreme inputs (tests in `polynomial_test.cc`, `encryption_test.cc`); the
unchanged upload bytes and server behaviour confirm it end to end. The RGSW
encryption path (not per query) stays on the NTT.

**Tradeoffs.**

- `FftContext::Create` costs ≈0.4 ms and ~100 KiB of tables per client; the
  harness builds a client per query, so this shows up inside `keygen`.
- `PirClient` methods are not concurrently callable on one instance (the
  context's scratch buffers are shared) — already the case because of the
  owned PRNGs.
- The 44-bit budget is the conservative analytic bound; it costs nothing at
  d=2048, but at much larger d or wider keys it would force more chunks.

## Measurements

One pinned core of an AMD EPYC 7B13 (online phase on CPU 13, offline
preprocessing on 8 other cores), `-c opt -march=znver2`, medians over 6 (dev)
or 4 (E0xx) queries after one warm-up. "Upstream" is the `tiptoe` branch
(unmodified ReinsPIRe + harness). Full JSON in the benchmark logs.

### Headline (medians, ms; upstream → final, speedup)

| workload | server | client (query+decode) | roundtrip | offline s | up/down bytes | correct |
|---|---|---|---|---|---|---|
| dev_M1024_P4_C4 | 65.6 → 36.2 (1.81×) | 47.1 → 7.2 (6.56×) | 113.0 → 43.4 (2.60×) | 19.8 → 11.9 | 79896/192608 = | ✓ 6/6 |
| dev_M2048_P4_C4 | 75.4 → 40.1 (1.88×) | 48.4 → 7.2 (6.70×) | 123.9 → 47.3 (2.62×) | 12.7 → 12.2 | 106520/192608 = | ✓ 6/6 |
| dev_M2048_P4_C2 | 37.9 → 22.0 (1.72×) | 34.2 → 5.6 (6.05×) | 72.2 → 27.7 (2.61×) | 6.4 → 6.2 | 106520/96304 = | ✓ 6/6 |
| dev_M512_P4_C8 | 120.4 → 64.5 (1.87×) | 73.6 → 10.1 (7.30×) | 193.8 → 74.5 (2.60×) | 24.1 → 23.4 | 66584/385216 = | ✓ 6/6 |
| dev_M2048_P16_C4 | 294.5 → 165.5 (1.78×) | 159.7 → 22.4 (7.14×) | 455.5 → 187.9 (2.42×) | 42.8 → 41.2 | 266300/770432 = | ✓ 6/6 |
| E035_M2048_P20_C4 | 366.9 → 207.7 (1.77×) | 195.3 → 27.3 (7.16×) | 562.6 → 235.0 (2.39×) | 54.6 → 52.9 | 319560/963040 = | ✓ 4/4 |
| E072_M2048_P41_C4 | 761.8 → 403.6 (1.89×) | 383.9 → 54.0 (7.11×) | 1143.4 → 457.7 (2.50×) | 118.0 → 113.3 | 599175/1974232 = | ✓ 4/4 |
| E071_M2048_P41_C2 | 383.3 → 225.8 (1.70×) | 257.9 → 37.8 (6.82×) | 640.9 → 263.8 (2.43×) | 59.2 → 57.6 | 599175/987116 = | ✓ 4/4 |
| E069_M1024_P82_C4 | 1376.2 → 755.6 (1.82×) | 778.1 → 101.8 (7.64×) | 2156.8 → 857.7 (2.51×) | 223.0 → 213.1 | 599298/3948464 = | ✓ 4/4 |

### Phase breakdown (medians, ms; upstream → final)

| workload | server | inner_product | pack_matrix | pack_finalize | client_query | keygen | selector | client_decode | decrypt | roundtrip |
|---|---|---|---|---|---|---|---|---|---|---|
| dev_M1024_P4_C4 | 65.6 → 36.2 | 8.4 → 5.9 | 43.8 → 22.3 | 10.7 → 5.3 | 20.3 → 4.1 | 10.3 → 1.9 | 9.5 → 1.7 | 26.8 → 3.1 | 24.8 → 1.1 | 113.0 → 43.4 |
| dev_M2048_P4_C4 | 75.4 → 40.1 | 17.8 → 9.5 | 43.9 → 22.2 | 10.7 → 5.3 | 20.6 → 4.1 | 10.3 → 1.8 | 9.6 → 1.6 | 27.8 → 3.1 | 25.8 → 1.1 | 123.9 → 47.3 |
| dev_M2048_P4_C2 | 37.9 → 22.0 | 8.4 → 5.3 | 21.9 → 11.1 | 5.5 → 3.3 | 20.5 → 4.1 | 10.3 → 1.8 | 9.6 → 1.6 | 13.6 → 1.6 | 12.6 → 0.5 | 72.2 → 27.7 |
| dev_M512_P4_C8 | 120.4 → 64.5 | 8.6 → 7.2 | 86.9 → 43.8 | 21.3 → 9.5 | 20.2 → 3.9 | 10.3 → 1.8 | 9.5 → 1.6 | 53.4 → 6.2 | 49.3 → 2.1 | 193.8 → 74.5 |
| dev_M2048_P16_C4 | 294.5 → 165.5 | 67.9 → 44.5 | 174.5 → 88.7 | 42.4 → 21.5 | 50.3 → 10.0 | 10.3 → 1.9 | 38.4 → 6.5 | 109.4 → 12.4 | 101.1 → 4.3 | 455.5 → 187.9 |
| E035_M2048_P20_C4 | 366.9 → 207.7 | 84.0 → 55.2 | 217.6 → 112.2 | 53.1 → 26.9 | 59.7 → 11.9 | 10.2 → 1.8 | 47.4 → 8.1 | 135.6 → 15.4 | 125.6 → 5.3 | 562.6 → 235.0 |
| E072_M2048_P41_C4 | 761.8 → 403.6 | 175.0 → 93.9 | 451.3 → 228.0 | 109.6 → 54.5 | 111.3 → 22.2 | 10.3 → 1.9 | 97.5 → 16.8 | 272.6 → 31.8 | 251.8 → 11.1 | 1143.4 → 457.7 |
| E071_M2048_P41_C2 | 383.3 → 225.8 | 86.8 → 58.3 | 225.3 → 114.3 | 54.4 → 33.7 | 112.0 → 22.1 | 10.3 → 1.9 | 98.1 → 16.7 | 145.9 → 15.7 | 135.5 → 5.4 | 640.9 → 263.8 |
| E069_M1024_P82_C4 | 1376.2 → 755.6 | 171.4 → 109.0 | 946.0 → 487.4 | 216.9 → 110.8 | 209.6 → 38.8 | 10.4 → 1.9 | 195.7 → 33.3 | 568.5 → 63.0 | 527.0 → 21.7 | 2156.8 → 857.7 |

### Peak RSS (MiB, harness process including offline preprocessing; upstream → final; see the I3a memory note)

| workload | max_rss |
|---|---|
| dev_M1024_P4_C4 | 2006 → 2196 |
| dev_M2048_P4_C4 | 2397 → 2433 |
| dev_M2048_P4_C2 | 1600 → 1725 |
| dev_M512_P4_C8 | 3191 → 3273 |
| dev_M2048_P16_C4 | 6122 → 6102 |
| E035_M2048_P20_C4 | 6798 → 6883 |
| E072_M2048_P41_C4 | 10436 → 10636 |
| E071_M2048_P41_C2 | 6216 → 6422 |
| E069_M1024_P82_C4 | 15480 → 15728 |

Remaining online time after these changes: server — `pack_matrix` (DRAM-bound
at ~25 GB/s single-core; the next step would be multi-threading or a smaller
packing matrix), `inner_product`, `pack_finalize` inverse FFTs; client — PRNG
/ error sampling and harness bit (un)packing.
