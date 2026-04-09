# Knowhere-rs Benchmark Guide

This document describes how to run benchmarks and compare performance with C++ knowhere.

## Quick Start

### Run Rust Benchmark Only

```bash
# Quick test (100 queries)
cargo test --test bench_sift1m test_sift1m_quick -- --nocapture

# Full benchmark (1000 queries)
cargo test --test bench_sift1m test_sift1m_benchmark -- --nocapture

# With JSON output
JSON_OUTPUT=./results.json cargo test --test bench_sift1m test_sift1m_benchmark -- --nocapture
```

### Run All Datasets

```bash
# Use the unified comparison test
DATASET=sift1m NUM_QUERIES=1000 cargo test --test bench_compare test_bench_compare_sift1m -- --nocapture
DATASET=deep1m NUM_QUERIES=1000 cargo test --test bench_compare test_bench_compare_deep1m -- --nocapture
DATASET=gist1m NUM_QUERIES=100 cargo test --test bench_compare test_bench_compare_gist1m -- --nocapture
```

### Run Comparison Script (Rust + C++)

```bash
# Rust only
python3 scripts/compare_benchmark.py --dataset sift1m --rust-only

# With C++ comparison (if available)
python3 scripts/compare_benchmark.py --dataset sift1m

# Generate reports
python3 scripts/compare_benchmark.py --dataset all --report benchmark_report.md --json-report results.json
```

### Use Shell Wrapper

```bash
# Quick run
./scripts/run_benchmark.sh -d sift1m -q 1000

# With reports
./scripts/run_benchmark.sh -d all -j results.json -r report.md --rust-only
```

## Benchmark Tests

### Individual Dataset Tests

| Test File | Dataset | Command |
|-----------|---------|---------|
| `tests/bench_sift1m.rs` | SIFT1M (128D) | `cargo test --test bench_sift1m` |
| `tests/bench_deep1m.rs` | DEEP1M (96D) | `cargo test --test bench_deep1m` |
| `tests/bench_gist1m.rs` | GIST1M (960D) | `cargo test --test bench_gist1m` |

### Unified Comparison Test

`tests/bench_compare.rs` provides a unified interface for all datasets:

```bash
# Run specific dataset
DATASET=sift1m cargo test --test bench_compare test_bench_compare_all -- --nocapture

# Custom query count
DATASET=sift1m NUM_QUERIES=5000 cargo test --test bench_compare test_bench_compare_all -- --nocapture

# JSON output
DATASET=sift1m JSON_OUTPUT=./sift1m_results.json cargo test --test bench_compare test_bench_compare_all -- --nocapture
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SIFT1M_PATH` | Path to SIFT1M dataset | `./data/sift` |
| `DEEP1M_PATH` | Path to DEEP1M dataset | `./data/deep` |
| `GIST1M_PATH` | Path to GIST1M dataset | `./data/gist` |
| `DATASET` | Dataset type for bench_compare | `sift1m` |
| `NUM_QUERIES` | Number of queries to run | `1000` |
| `JSON_OUTPUT` | Path for JSON output | (none) |

## Dataset Download

### SIFT1M
```bash
mkdir -p data/sift
cd data/sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xzf sift.tar.gz
```

### DEEP1M
```bash
mkdir -p data/deep
cd data/deep
# Download from https://github.com/arbabenko/GNOIMI
```

### GIST1M
```bash
mkdir -p data/gist
cd data/gist
wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar xzf gist.tar.gz
```

## Output Format

### Console Output

```
======================================================================
Knowhere-rs Benchmark: SIFT1M
Queries: 1000, Dataset: SIFT1M
======================================================================

=== Benchmark Results ===

Index        Build(ms)   Search(ms)        QPS          R@1         R@10        R@100
------------------------------------------------------------------------------------------
Flat          1234.56       456.78        2190        1.000        1.000        1.000
HNSW          5678.90       123.45        8100        0.985        0.992        0.998
IVF-Flat       890.12       234.56        4263        0.920        0.965        0.985
```

### Markdown Table

```markdown
## SIFT1M Benchmark Results

| Index | Build Time (ms) | Search Time (ms) | QPS | Recall@1 | Recall@10 | Recall@100 |
|-------|-----------------|------------------|-----|----------|-----------|------------|
| Flat | 1234.56 | 456.78 | 2190 | 1.000 | 1.000 | 1.000 |
| HNSW | 5678.90 | 123.45 | 8100 | 0.985 | 0.992 | 0.998 |
| IVF-Flat | 890.12 | 234.56 | 4263 | 0.920 | 0.965 | 0.985 |
```

### JSON Output

```json
{
  "dataset": "SIFT1M",
  "timestamp": "2026-03-01T10:30:00Z",
  "results": [
    {
      "index_name": "Flat",
      "build_time_ms": 1234.56,
      "search_time_ms": 456.78,
      "num_queries": 1000,
      "qps": 2190,
      "recall_at_1": 1.0,
      "recall_at_10": 1.0,
      "recall_at_100": 1.0
    }
  ]
}
```

## Comparison with C++ Knowhere

The `compare_benchmark.py` script automatically:

1. Runs Rust hanns benchmarks
2. Attempts to run C++ knowhere benchmarks (if available)
3. Generates comparison reports with:
   - Performance ratios (Rust QPS / C++ QPS)
   - Status indicators (✅ ≥80%, ⚠️ ≥50%, ❌ <50%)
   - Markdown tables for reports

### C++ Knowhere Setup

If you have C++ knowhere installed:

```bash
# Build knowhere
cd ~/Code/vectorDB/knowhere
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The script will automatically detect it at:
- `~/Code/vectorDB/knowhere`
- `~/Projects/knowhere`
- `/opt/knowhere`

Or set `KNOWHERE_PATH` environment variable.

## Performance Tips

1. **Use release mode**: Always run benchmarks with `--release`
2. **Warm up**: First run may be slower due to cold caches
3. **Consistent queries**: Use same query count for fair comparison
4. **Multiple runs**: Run 3-5 times and average results
5. **System state**: Close other applications during benchmark

## Troubleshooting

### Dataset Not Found
```
Failed to load SIFT1M dataset from ./data/sift
```
→ Set `SIFT1M_PATH` environment variable or download dataset to correct location.

### Out of Memory
→ Reduce `NUM_QUERIES` or use a machine with more RAM.

### Slow Build Times
→ Ensure you're using `--release` flag. Debug builds are 10-100x slower.

### C++ Benchmark Skipped
```
⊘ C++ benchmark skipped: C++ knowhere not found
```
→ This is normal if you only want Rust benchmarks. Use `--rust-only` flag to suppress warning.
