# Bitset SIMD Optimizations (OPT-01)

## Overview

This document describes the SIMD optimizations added to the bitset implementation for accelerated batch testing and zero-counting operations.

## Implementation Details

### AVX2 Optimizations (x86_64)

**Location:** `src/bitset.rs`

#### `test_batch_avx2()`
- Processes **256 bits (32 bytes)** at once using AVX2 instructions
- Uses `_mm256_loadu_si256()` for unaligned load
- Uses `_mm256_testz_si256()` for efficient zero testing
- Returns `true` if all bits are zero

#### `count_zero_batch_avx2()`
- Counts zero bits in **256-bit (32 bytes)** batches
- Uses `_mm256_cmpeq_epi8()` to compare with zero
- Uses `_mm256_movemask_epi8()` to extract comparison results
- Counts set bits in the mask to determine zero count

### NEON Optimizations (ARM64/Apple Silicon)

#### `test_batch_neon()`
- Processes **128 bits (16 bytes)** at once using NEON instructions
- Uses `vld1q_u8()` for load
- Uses `vceqq_u8()` for comparison with zero
- Uses `vminvq_u8()` for horizontal minimum to check all lanes

#### `count_zero_batch_neon()`
- Counts zero bits in **128-bit (16 bytes)** batches
- Uses `vceqq_u8()` to compare each byte with zero
- Extracts comparison results and counts zero bytes
- Each zero byte contributes 8 zero bits

### Fallback Implementations

For platforms without SIMD support or when SIMD features are not detected:

#### `test_batch_fallback()`
- Sequential byte-by-byte testing
- Checks if any byte is non-zero

#### `count_zero_batch_fallback()`
- Iterates through all bytes
- Uses `count_zeros()` for non-zero bytes
- Counts 8 zeros for zero bytes

### Runtime Dispatch

#### `test_batch_auto()`
- Automatically detects CPU features at runtime
- Dispatches to AVX2/NEON/fallback based on availability
- Zero-cost abstraction when compiled with appropriate target features

#### `count_zero_batch_auto()`
- Same runtime dispatch pattern for zero counting
- Ensures optimal performance across different platforms

### Wrapper Functions

For compatibility with existing code using `&[u64]`:

- `test_batch()` - SIMD-optimized test for u64 slices
- `count_zero_batch()` - SIMD-optimized count for u64 slices
- `test_batch_generic()` - Non-SIMD version for comparison
- `count_zero_batch_generic()` - Non-SIMD version for comparison

## Usage Examples

```rust
use knowhere_rs::bitset::{
    test_batch_auto, 
    count_zero_batch_auto,
    test_batch_fallback,
    count_zero_batch_fallback,
};

// Create test data
let data = vec![0u8; 4096];

// Automatic SIMD dispatch
let all_zero = test_batch_auto(&data, 0);
let zero_count = count_zero_batch_auto(&data);

// Explicit fallback (for comparison/testing)
let all_zero_fallback = test_batch_fallback(&data, 0);
let zero_count_fallback = count_zero_batch_fallback(&data);
```

## Performance Benchmarks

Run benchmarks with:
```bash
cargo test --lib bitset::tests::benchmark_simd_performance -- --nocapture
cargo test --lib bitset::tests::benchmark_avx2_specific_wrapper -- --nocapture  # x86_64
cargo test --lib bitset::tests::benchmark_neon_specific_wrapper -- --nocapture  # ARM64
```

### Expected Performance Gains

- **AVX2 (x86_64):** 4-8x speedup for batch operations on large datasets
- **NEON (ARM64):** 2-4x speedup for batch operations
- Gains are most significant for:
  - Large bitsets (>1KB)
  - Sparse bitsets (mostly zeros)
  - Repeated batch operations

## Testing

All SIMD optimizations include comprehensive tests:

1. **Correctness Tests:**
   - `test_simd_batch_operations` - Basic functionality
   - `test_simd_edge_cases` - Edge cases (empty, single bit, etc.)
   - `test_simd_fallback_consistency` - SIMD vs fallback consistency
   - `test_simd_large_dataset` - Large dataset handling

2. **Platform-Specific Tests:**
   - `test_avx2_intrinsics_wrapper` - AVX2 intrinsic testing
   - `test_neon_intrinsics_wrapper` - NEON intrinsic testing

3. **Performance Tests:**
   - `benchmark_simd_performance` - General SIMD vs fallback
   - `benchmark_avx2_specific_wrapper` - AVX2-specific benchmarks
   - `benchmark_neon_specific_wrapper` - NEON-specific benchmarks

## Architecture Detection

The implementation includes runtime CPU feature detection:

```rust
#[cfg(target_arch = "x86_64")]
if is_x86_feature_detected!("avx2") {
    // AVX2 available
}

#[cfg(target_arch = "aarch64")]
if is_aarch64_feature_detected!("neon") {
    // NEON available
}
```

## Compilation

The SIMD optimizations are automatically enabled based on target architecture:

- **x86_64:** AVX2 intrinsics compiled when target supports it
- **aarch64:** NEON intrinsics compiled when target supports it
- **Other:** Falls back to generic implementation

No special compilation flags are required - the code uses `#[target_feature]` attributes for safe SIMD code generation.

## Future Enhancements

Potential improvements:

1. **AVX-512 Support:** For newer Intel CPUs (Skylake-X, Ice Lake, etc.)
2. **SVE Support:** For ARM Scalable Vector Extension
3. **Bitset Operations:** SIMD-accelerated AND/OR/XOR operations
4. **Find First/Set:** SIMD-accelerated bit scanning

## References

- [Rust std::arch Documentation](https://doc.rust-lang.org/std/arch/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics)

## Implementation Date

2026-02-28

## Author

OpenClaw Builder Agent (OPT-01 Task)
