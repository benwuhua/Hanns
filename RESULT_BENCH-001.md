# BENCH-001: SIFT1M Dataset Loading and Benchmark Test

**Date:** 2026-03-01  
**Status:** ✅ DONE

## Summary

Implemented SIFT1M dataset loading capability and benchmark testing framework for knowhere-rs.

## Files Created/Modified

### New Files
1. **`src/dataset/sift_loader.rs`** (7.2KB)
   - `load_sift1m_complete()` - Load complete SIFT1M dataset (base, query, ground truth)
   - `load_fvecs()` - Load fvecs format files
   - `load_ivecs()` - Load ivecs format files (ground truth)
   - `SiftDataset` struct - Container for loaded dataset

2. **`src/dataset/mod.rs`** (6.2KB)
   - Added `sift_loader` module
   - Re-exports `SiftDataset` and loader functions

3. **`src/benchmark/recall.rs`** (5.3KB)
   - `average_recall_at_k()` - Calculate recall@k metric
   - `calculate_recall()` - Single query recall calculation
   - Comparison logic against ground truth

4. **`src/benchmark/mod.rs`** (3.2KB)
   - Added `recall` module
   - `BenchmarkResult` struct for result tracking
   - `print_table()` for formatted output

5. **`tests/bench_sift1m.rs`** (10.3KB)
   - `test_sift1m_benchmark()` - Full benchmark (default 1000 queries)
   - `test_sift1m_quick()` - Quick test (100 queries)
   - Benchmarks for Flat, HNSW, IVF-Flat indexes
   - Recall@1, Recall@10, Recall@100 metrics

### Modified Files
- `src/lib.rs` - Already exports `benchmark` and `dataset` modules

## Benchmark Results (10K vectors, 128D - Random Data)

| Index    | Build(ms) | Search 100q(ms) | QPS    |
|----------|-----------|-----------------|--------|
| Flat     | 0.44      | 65.36           | 1,530  |
| HNSW     | 566.02    | 3.35            | 29,872 |
| IVF-Flat | 43,189.34 | 66.48           | 1,504  |

**Notes:**
- HNSW shows ~20x QPS improvement over Flat
- IVF-Flat build time is high due to k-means training on full dataset
- Results from `test_performance_comparison_small`

## SIFT1M Benchmark Usage

```bash
# Download SIFT1M dataset
# http://corpus-texmex.irisa.fr/
# - base.fvecs: 1M base vectors
# - query.fvecs: 10K query vectors  
# - groundtruth.ivecs: 10K x 100 ground truth

# Place in ./data/sift/ or set SIFT1M_PATH
export SIFT1M_PATH=/path/to/sift

# Run full benchmark (1000 queries)
cargo test --test bench_sift1m -- --nocapture

# Run quick test (100 queries)
cargo test --test bench_sift1m test_sift1m_quick -- --nocapture

# Custom query count
export SIFT_NUM_QUERIES=500
cargo test --test bench_sift1m -- --nocapture
```

## Expected Output Format

```
=== SIFT1M Benchmark Results ===
Index       Build(ms)  Search(ms)  QPS    Recall@1  Recall@10  Recall@100
Flat        1234.56    567.89      176    1.000     1.000      1.000
HNSW        2345.67    45.67       2190   0.995     0.985      0.970
IVF-Flat    890.12     123.45      810    0.960     0.952      0.940
```

## Comparison with C++ Knowhere

### C++ Implementation (`tests/ut/test_search.cc`)
- Uses Catch2 test framework
- `GenDataSet()` for random data generation
- `GetKNNRecall()` for recall calculation
- JSON-based parameter configuration
- Supports multiple index types via IndexFactory

### Rust Implementation
- Uses standard test framework
- `load_sift1m_complete()` for real dataset loading
- `average_recall_at_k()` for recall calculation
- Struct-based configuration (`IndexConfig`, `IndexParams`)
- Direct index type instantiation

### Key Differences
| Aspect | C++ Knowhere | Rust knowhere-rs |
|--------|--------------|------------------|
| Test Framework | Catch2 | std test |
| Dataset Loading | In-memory gen | File-based (fvecs) |
| Config Format | JSON | Struct |
| Recall Calc | GetKNNRecall() | average_recall_at_k() |
| Index Factory | IndexFactory | Direct construction |

### Advantages of Rust Implementation
- Type-safe configuration (compile-time checks)
- Real dataset loading (not just random data)
- Clear separation of benchmark utilities
- Easy to extend with new index types

## Next Steps

### P0 (High Priority)
- [ ] **BENCH-002**: Add Deep1M and GIST1M dataset support
- [ ] **BENCH-003**: Implement comparison script vs C++ knowhere

### P1 (Medium Priority)
- [ ] **OPT-002**: Optimize IVF k-means training (parallel/mini-batch)
- [ ] **OPT-003**: SIMD acceleration for distance calculation

### P2 (Low Priority)
- [ ] **BENCH-004**: Add memory usage tracking
- [ ] **BENCH-005**: Add throughput benchmark (concurrent queries)

## References
- SIFT1M Dataset: http://corpus-texmex.irisa.fr/
- C++ Knowhere Tests: `/Users/ryan/Code/vectorDB/knowhere/tests/ut/`
- fvecs Format: 4-byte dim + N * dim * 4-byte floats
