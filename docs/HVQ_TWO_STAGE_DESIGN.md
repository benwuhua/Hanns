# HVQ Two-Stage Pipeline Design (ExRaBitQ Architecture)

## Problem

Current HVQ brute-force scan scores every vector with full B-bit decoding — a D-dimensional dot product per vector (768 ops for D=768). Result: 3-9 QPS on 1M vectors, vs PQ's 60+ QPS with 96 table lookups per vector.

**Root cause**: Missing the core ExRaBitQ two-stage architecture.

## ExRaBitQ Two-Stage Pipeline

```
Query → Rotate →
  Stage 1: 1-bit FastScan (ALL vectors, 32 at a time)
    → approximate IP via LUT+shuffle (O(D/128) per vector)
    → keep top nprobe*K candidates
  Stage 2: B-bit Rerank (survivors only, ~0.5-5% of vectors)
    → precise score via existing score_code()
    → return top-K
```

**Expected speedup**: If Stage 1 filters 95% of vectors and runs 10-20x faster per vector than B-bit scoring, overall throughput could improve 5-10x.

---

## Stage 1: 1-bit FastScan

### 1-bit Code Generation

The 1-bit code is the simplest quantization of the rotated unit residual:

```rust
// During encode:
// o_hat = rotate(v - centroid) / ||rotate(v - centroid)||
// 1-bit code: b[i] = (o_hat[i] >= 0) ? 1 : 0
// Packed: D/8 bytes per vector
```

This is already a degenerate case of our `fast_quantize` with nbits=1. But for FastScan we need a SEPARATE 1-bit code stored in transposed layout, even when the B-bit code uses nbits=4 or 8.

### FastScan Memory Layout (Faiss-style)

Vectors are grouped into blocks of 32. Within each block, bits are transposed into "nibble groups" for SIMD shuffle processing:

```
Block layout for 32 vectors, D dimensions:
  - D/4 "nibble groups", each group covers 4 consecutive dimensions
  - For nibble group g (dims g*4 .. g*4+3):
    - Vectors 0-15:  4 bits per vector → 16 nibbles → 8 bytes (low half)
    - Vectors 16-31: 4 bits per vector → 16 nibbles → 8 bytes (high half)
    - Total: 16 bytes per nibble group

Total per block: D/4 * 16 = 4D bytes for 32 vectors
Per vector: D/8 bytes (same as naive, just rearranged)

Nibble packing for vector v in group g:
  nibble = bit(g*4+0) | (bit(g*4+1) << 1) | (bit(g*4+2) << 2) | (bit(g*4+3) << 3)
```

### FastScan SIMD Kernel (AVX512)

For each query, precompute D/4 LUTs, each with 16 entries:

```rust
// For nibble group g (dims d0..d3 = g*4 .. g*4+3):
// LUT[nibble] = sum of q_rot[d_j] for each bit j set in nibble
//             - sum of q_rot[d_j] for each bit j NOT set in nibble
// This gives: LUT[nibble] = 2 * (sum of q_rot[d_j] where bit_j=1) - sum(q_rot[d0..d3])
//
// Quantize LUT entries to i8 for SIMD shuffle:
// lut_scale = max(|LUT entries|) / 127.0
// lut_i8[i] = round(LUT[i] / lut_scale)
```

Processing 32 vectors per iteration:

```
// _mm512_shuffle_epi8 operates on 4 independent 128-bit lanes
// Each lane: 16 LUT entries (i8) shuffled by 16 nibble indices
//
// For each nibble group g:
//   1. Load LUT into zmm register (replicate 16-entry table across 4 lanes)
//   2. Load 16 bytes of packed nibbles (32 vectors' nibbles for group g)
//   3. Split into low/high nibbles if needed
//   4. _mm512_shuffle_epi8(lut, nibble_indices) → 64 partial sums (i8)
//   5. Accumulate into i16 accumulators
//   6. Every 128 groups, spill i16 → i32 to avoid overflow
//
// After all groups: reduce i32 accumulators → 32 approximate scores
```

### 1-bit Score Metadata

Each vector also stores `norm_o` and `base_quant_dist_1bit` alongside the 1-bit code. For FastScan, we can either:
- (a) Store these per-vector and multiply after FastScan (accurate but needs scatter/gather)
- (b) **Approximate**: Use average `norm_o` across the dataset, score only by the 1-bit IP. Since Stage 1 is just filtering, approximate ranking is sufficient.

**Recommendation**: Option (b) for maximum speed. The 1-bit FastScan rank order is dominated by the IP term; `norm_o` variation is secondary for filtering.

### Per-vector metadata for Stage 2

For rerank, we need per-vector `norm_o`, `vmax`, `base_quant_dist`. These are already stored in the B-bit code header (12 bytes).

---

## Stage 2: B-bit Rerank

Use existing `score_code()` on survivors only. The current SIMD paths (AVX512 masked for 1-bit, VNNI for 4/8-bit) are already optimized for single-vector scoring.

**No changes needed** to score_code for Stage 2.

---

## New Structs and API

### `HvqIndex` (new struct, in `src/quantization/hvq.rs`)

```rust
pub struct HvqIndex {
    pub quantizer: HvqQuantizer,

    // B-bit codes (existing format: [norm_o|vmax|base_quant_dist|packed_bits])
    pub codes: Vec<u8>,
    pub n: usize,

    // 1-bit FastScan codes (transposed layout)
    // Layout: blocks of 32 vectors, each block = dim/4 * 16 bytes
    pub fastscan_codes: Vec<u8>,
    pub n_blocks: usize,       // ceil(n / 32)
    pub fastscan_block_size: usize, // dim/4 * 16 bytes per block

    // Per-vector norms for rerank (extracted from B-bit code headers)
    // Could be avoided by re-reading from codes, but separate array is cache-friendlier
    pub norms: Vec<f32>,        // norm_o per vector
}
```

### `HvqFastScanState` (new struct for query processing)

```rust
pub struct HvqFastScanState {
    pub q_rot: Vec<f32>,
    pub q_sum: f32,
    pub centroid_score: f32,

    // FastScan LUTs: D/4 groups, each 16 entries (i8)
    pub lut: Vec<i8>,           // D/4 * 16 entries, flattened
    pub lut_scale: f32,         // scale factor for dequantizing LUT results
}
```

### Key Methods

```rust
impl HvqIndex {
    /// Build index from raw vectors
    pub fn build(quantizer: HvqQuantizer, data: &[f32], n: usize) -> Self { ... }

    /// Transpose 1-bit codes into FastScan layout
    fn transpose_to_fastscan(raw_1bit_codes: &[Vec<u8>], dim: usize) -> Vec<u8> { ... }

    /// Two-stage search
    pub fn search(&self, query: &[f32], k: usize, nprobe_factor: usize) -> Vec<(usize, f32)> {
        let q_rot = self.quantizer.rotate_query(query);
        let fastscan_state = self.precompute_fastscan_state(&q_rot);

        // Stage 1: FastScan all blocks
        let n_candidates = k * nprobe_factor; // e.g., nprobe_factor=10 → keep 100 for k=10
        let candidates = self.fastscan_topk(&fastscan_state, n_candidates);

        // Stage 2: Precise B-bit rerank
        let bbit_state = self.quantizer.precompute_query_state(&q_rot);
        let code_size = self.quantizer.code_size_bytes();
        let mut results: Vec<(usize, f32)> = candidates.iter().map(|&(vid, _)| {
            let code = &self.codes[vid * code_size..(vid + 1) * code_size];
            (vid, self.quantizer.score_code(&bbit_state, code))
        }).collect();

        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);
        results
    }

    /// Precompute FastScan LUTs for a query
    fn precompute_fastscan_state(&self, q_rot: &[f32]) -> HvqFastScanState { ... }

    /// FastScan: process all blocks, return top-n approximate candidates
    fn fastscan_topk(&self, state: &HvqFastScanState, n: usize) -> Vec<(usize, f32)> { ... }

    /// AVX512 kernel: score 32 vectors in one block, return 32 i32 raw scores
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw")]
    unsafe fn fastscan_block_avx512(&self, state: &HvqFastScanState, block: &[u8]) -> [i32; 32] { ... }
}
```

---

## FastScan Block Kernel (AVX512) — Detailed

```rust
// Input: one block of 32 vectors in transposed nibble layout
// Output: 32 approximate IP scores (i32)
//
// Registers:
//   lut_reg: 512-bit, loaded from state.lut for current nibble group
//            (16 entries replicated across 4 x 128-bit lanes)
//   acc_lo, acc_hi: 512-bit i16 accumulators (vectors 0-15, 16-31)
//   acc32_lo, acc32_hi: 512-bit i32 accumulators (spill targets)
//
// For each nibble group g = 0..D/4:
//   1. Load 16 bytes from block[g * 16 .. g * 16 + 16]
//      - Bytes 0-7: low 16 vectors' nibbles (packed: 2 nibbles per byte)
//      - Bytes 8-15: high 16 vectors' nibbles
//   2. Actually, the 16 bytes contain 32 nibbles for 32 vectors
//      - Need to split each byte into low/high nibble
//      - low_nibbles = data & 0x0F (for even-indexed vectors in pair)
//      - high_nibbles = (data >> 4) & 0x0F (for odd-indexed vectors in pair)
//
//      Wait — let me reconsider the layout:
//      For 32 vectors, each contributing 1 nibble per group:
//      - 32 nibbles = 16 bytes (2 nibbles per byte)
//      - Byte i contains: nibble of vector 2i (low) and vector 2i+1 (high)
//
//   3. Split into two __m512i of nibbles:
//      - Set up input as __m128i (16 bytes), zero-extend to __m512i
//      - low_nibbles = _mm512_and_si512(data, 0x0F_mask)
//      - high_nibbles = _mm512_srli_epi16(data, 4) & 0x0F_mask
//      - Interleave so we get 32 separate nibble values in correct order
//
//   Actually, simpler approach matching Faiss:
//
//   Layout: 16 bytes per group
//     - First 8 bytes: vectors 0-15, each 1 nibble packed into 4 bits
//       byte j contains: vector 2j (low nibble) + vector 2j+1 (high nibble)
//     - Next 8 bytes: vectors 16-31, same packing
//
//   Processing:
//     lo_data = _mm_loadu_si128(block_ptr)      // 16 bytes
//     // Broadcast to 512-bit for shuffle
//     // Split low/high nibbles
//     // Use vpshufb with LUT
//     // Accumulate
//
// Spill i16 → i32 every 128 iterations to avoid overflow
// (i8 LUT values in [-127,127], summed over D/4 groups:
//  max i16 accumulator = 127 * D/4 = 127 * 192 = 24384 for D=768 → fits i16!)
//
// Final: convert i32 accumulators to f32, multiply by lut_scale
```

### Simplified Faiss-style layout (recommended)

```
For 32 vectors in a block, D dimensions:

Step 1 — Generate 1-bit codes:
  For each vector v: code_v[i] = (o_hat_v[i] >= 0) ? 1 : 0, packed into D/8 bytes

Step 2 — Transpose into nibble groups:
  For nibble group g (dims g*4 .. g*4+3):
    For vectors 0..15:
      nibble_v = bit(g*4+0,v) | (bit(g*4+1,v) << 1) | (bit(g*4+2,v) << 2) | (bit(g*4+3,v) << 3)
      Pack nibble_0 + nibble_1 into byte 0 (v0=low, v1=high)
      Pack nibble_2 + nibble_3 into byte 1 ...
      → 8 bytes for vectors 0-15
    For vectors 16..31:
      Same → 8 bytes
    → 16 bytes per nibble group

  Total block size: D/4 * 16 bytes
  For D=768: 192 * 16 = 3072 bytes per 32 vectors = 96 bytes/vector
  (vs naive 1-bit: 768/8 = 96 bytes/vector — same total, just rearranged!)
```

---

## Build Path Changes

```rust
impl HvqIndex {
    pub fn build(quantizer: HvqQuantizer, data: &[f32], n: usize, nbits: u8) -> Self {
        let dim = quantizer.config.dim;

        // 1. Encode all vectors with B-bit codes (existing path)
        let bbit_quantizer = HvqQuantizer::new(HvqConfig { dim, nbits }, seed);
        bbit_quantizer.train(n, data);
        let codes = bbit_quantizer.encode_batch(n, data, 0);

        // 2. Generate 1-bit codes for each vector
        let mut raw_1bit = Vec::with_capacity(n);
        for i in 0..n {
            let v = &data[i * dim..(i + 1) * dim];
            let v_centered: Vec<f32> = v.iter().zip(bbit_quantizer.centroid.iter())
                .map(|(a, b)| a - b).collect();
            let o = bbit_quantizer.rotate(&v_centered);
            let norm_o = o.iter().map(|x| x * x).sum::<f32>().sqrt();
            let o_hat: Vec<f32> = o.iter().map(|x| x / norm_o.max(1e-12)).collect();

            // 1-bit: sign quantization
            let mut bits = vec![0u8; dim.div_ceil(8)];
            for (j, &val) in o_hat.iter().enumerate() {
                if val >= 0.0 {
                    bits[j / 8] |= 1 << (j % 8);
                }
            }
            raw_1bit.push(bits);
        }

        // 3. Transpose 1-bit codes to FastScan layout
        let fastscan_codes = Self::transpose_to_fastscan(&raw_1bit, dim, n);

        // 4. Extract norms
        let code_size = bbit_quantizer.code_size_bytes();
        let norms: Vec<f32> = (0..n).map(|i| {
            f32::from_le_bytes(codes[i * code_size..i * code_size + 4].try_into().unwrap())
        }).collect();

        HvqIndex { quantizer: bbit_quantizer, codes, n, fastscan_codes, n_blocks, fastscan_block_size, norms }
    }
}
```

---

## LUT Precomputation

```rust
fn precompute_fastscan_state(&self, q_rot: &[f32]) -> HvqFastScanState {
    let dim = self.quantizer.config.dim;
    let n_groups = dim.div_ceil(4);
    let mut lut_f32 = vec![0.0f32; n_groups * 16];

    for g in 0..n_groups {
        let base = g * 4;
        let q = [
            q_rot.get(base).copied().unwrap_or(0.0),
            q_rot.get(base + 1).copied().unwrap_or(0.0),
            q_rot.get(base + 2).copied().unwrap_or(0.0),
            q_rot.get(base + 3).copied().unwrap_or(0.0),
        ];

        // LUT[nibble] = sum of q[j] * sign(bit_j)
        // where sign(1) = +1, sign(0) = -1
        for nibble in 0..16u8 {
            let mut val = 0.0f32;
            for j in 0..4 {
                if nibble & (1 << j) != 0 {
                    val += q[j];
                } else {
                    val -= q[j];
                }
            }
            lut_f32[g * 16 + nibble as usize] = val;
        }
    }

    // Quantize to i8
    let max_val = lut_f32.iter().copied().map(f32::abs).fold(0.0f32, f32::max).max(1e-6);
    let lut_scale = max_val / 127.0;
    let lut: Vec<i8> = lut_f32.iter()
        .map(|&v| (v / lut_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    let q_sum = q_rot.iter().sum();
    let centroid_score = q_rot.iter().zip(self.quantizer.rotated_centroid.iter())
        .map(|(a, b)| a * b).sum();

    HvqFastScanState { q_rot: q_rot.to_vec(), q_sum, centroid_score, lut, lut_scale }
}
```

---

## FastScan Block Kernel

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn fastscan_block_avx512(
    &self,
    state: &HvqFastScanState,
    block: &[u8],  // fastscan_block_size bytes
) -> [i32; 32] {
    use std::arch::x86_64::*;

    let dim = self.quantizer.config.dim;
    let n_groups = dim.div_ceil(4);
    let nibble_mask = _mm512_set1_epi8(0x0F);

    // i16 accumulators for 32 vectors (split into low 16 and high 16)
    // Actually, use two __m512i for 32 x i16:
    let mut acc_lo = _mm512_setzero_si512(); // vectors 0-31 low lanes
    let mut acc_hi = _mm512_setzero_si512(); // vectors 0-31 high lanes
    // Wait — 32 x i16 = 64 bytes = one __m512i
    // So one __m512i holds all 32 i16 accumulators
    let mut acc16 = _mm512_setzero_si512(); // 32 x i16
    let mut acc32_even = _mm512_setzero_si512(); // 16 x i32 for even vectors
    let mut acc32_odd = _mm512_setzero_si512();  // 16 x i32 for odd vectors

    // Actually, let's follow Faiss's approach more carefully:
    //
    // Each nibble group: 16 bytes in block
    //   bytes 0-7:  vectors 0-15, each pair packed into 1 byte
    //   bytes 8-15: vectors 16-31
    //
    // For each group:
    //   1. Load LUT (16 x i8) into a 128-bit register, broadcast to 512-bit
    //   2. Load 16 bytes of nibbles
    //   3. Split into low_nibbles (even vectors) and high_nibbles (odd vectors)
    //   4. Shuffle LUT by low_nibbles → 16 i8 partial sums for even vectors
    //   5. Shuffle LUT by high_nibbles → 16 i8 partial sums for odd vectors
    //   6. Accumulate into i16

    // Simpler: use __m256i for 16 vectors, process vectors 0-15 and 16-31 separately
    // Then combine into 32-element result

    // Even simpler reference implementation first:
    let mut scores = [0i32; 32];

    for g in 0..n_groups {
        let lut_base = g * 16;
        let block_offset = g * 16;

        // Vectors 0-15: 8 bytes, each byte = 2 nibbles
        for v in 0..16 {
            let byte_idx = block_offset + v / 2;
            let nibble = if v % 2 == 0 {
                block[byte_idx] & 0x0F
            } else {
                (block[byte_idx] >> 4) & 0x0F
            };
            scores[v] += state.lut[lut_base + nibble as usize] as i32;
        }

        // Vectors 16-31: next 8 bytes
        for v in 0..16 {
            let byte_idx = block_offset + 8 + v / 2;
            let nibble = if v % 2 == 0 {
                block[byte_idx] & 0x0F
            } else {
                (block[byte_idx] >> 4) & 0x0F
            };
            scores[16 + v] += state.lut[lut_base + nibble as usize] as i32;
        }
    }

    scores
}
```

The above is the **scalar reference**. The SIMD version uses `_mm512_shuffle_epi8`:

```rust
// SIMD version (to be implemented):
// For each nibble group g:
//   lut_128 = _mm_loadu_si128(state.lut[g*16..])     // 16 x i8 LUT
//   lut_512 = _mm512_broadcast_i32x4(lut_128)        // replicate to 4 lanes
//   data_128 = _mm_loadu_si128(block[g*16..])         // 16 bytes of nibble data
//   data_lo = _mm512_and_si512(broadcast(data_128), 0x0F)   // even vectors
//   data_hi = _mm512_srli_epi16(broadcast(data_128), 4) & 0x0F  // odd vectors
//   partial_lo = _mm512_shuffle_epi8(lut_512, data_lo)  // 64 partial sums
//   partial_hi = _mm512_shuffle_epi8(lut_512, data_hi)
//   // Accumulate i8 → i16:
//   acc_lo = _mm512_adds_epi16(acc_lo, _mm512_cvtepi8_epi16(partial_lo_256))
//   acc_hi = _mm512_adds_epi16(acc_hi, _mm512_cvtepi8_epi16(partial_hi_256))
```

---

## Integration with quant_compare.rs

Add a new benchmark section for HVQ two-stage:

```rust
// After existing HVQ benchmark:
if enabled_methods.iter().any(|m| m.eq_ignore_ascii_case("HVQ2")) {
    for tier in &TIERS {
        let hvq_index = HvqIndex::build(&quantizer, &train_data, n_train, tier.hvq_bits);

        // Measure scan QPS with two-stage
        let start = Instant::now();
        let results: Vec<_> = queries.par_iter().map(|q| {
            hvq_index.search(q, TOP_K, 10) // nprobe_factor=10
        }).collect();
        let elapsed = start.elapsed();
        let qps = n_queries as f64 / elapsed.as_secs_f64();

        // Compute recall
        let recall = compute_recall(&results, &gt, TOP_K);

        println!("HVQ2  {:>6}  {:>10}  {:>8.2}  {:>10.4}  {:>10}", ...);
    }
}
```

---

## Implementation Plan (for Codex)

### Phase 1: Data structures + scalar reference (must work first)
1. Add `HvqIndex`, `HvqFastScanState` structs to `hvq.rs`
2. Implement `transpose_to_fastscan()` — convert per-vector 1-bit codes to nibble-transposed block layout
3. Implement `precompute_fastscan_state()` — build D/4 LUTs (i8)
4. Implement `fastscan_block_scalar()` — scalar reference for one block of 32 vectors
5. Implement `fastscan_topk()` — iterate all blocks, maintain top-N heap
6. Implement `search()` — two-stage pipeline
7. Add `HvqIndex::build()` method
8. Tests:
   - `test_fastscan_layout_roundtrip` — transpose then verify nibble extraction matches original bits
   - `test_fastscan_scores_match_bruteforce` — verify Stage 1 ranking roughly matches full B-bit ranking
   - `test_two_stage_recall` — verify two-stage recall >= single-stage at reasonable nprobe_factor

### Phase 2: AVX512 FastScan kernel
1. `fastscan_block_avx512()` with `_mm512_shuffle_epi8`
2. Runtime dispatch (AVX512BW check)
3. Test: `test_fastscan_avx512_matches_scalar`

### Phase 3: Benchmark integration
1. Add `HVQ2` method to `quant_compare.rs`
2. Run on x86, compare against single-stage HVQ

---

## Performance Targets

| Metric | Current (single-stage) | Target (two-stage) |
|--------|----------------------|-------------------|
| 32x scan QPS | 5 | 30-50 |
| 8x scan QPS | 9 | 40-60 |
| 4x scan QPS | 3 | 20-40 |
| Recall@10 (8x) | 0.89 | >= 0.85 (trade recall for speed) |

These are rough targets. The key metric is **QPS improvement with acceptable recall**.

---

## Key Constraints

1. **1-bit code must use the SAME rotation matrix** as the B-bit code — both come from the same `HvqQuantizer`
2. **Padding**: if `n % 32 != 0`, pad the last block with zero codes (they'll get low scores and be filtered)
3. **D must be divisible by 4** for nibble grouping. If not, pad dimensions with 0.
4. **LUT quantization error**: i8 LUT introduces quantization noise. For D=768, error is ~0.5% — acceptable for Stage 1 filtering.
5. **nprobe_factor tuning**: too low → recall drops; too high → no speedup. Start with 10, tune based on benchmarks.

---

## Files to Modify

- `src/quantization/hvq.rs` — add `HvqIndex`, `HvqFastScanState`, all new methods
- `src/quantization/mod.rs` — export `HvqIndex`, `HvqFastScanState`
- `examples/quant_compare.rs` — add `HVQ2` benchmark method
