# SIFT1M Benchmark

This benchmark tests the performance of different index types (Flat, HNSW, IVF-Flat) on the SIFT1M dataset.

## Dataset

SIFT1M is a standard benchmark dataset for approximate nearest neighbor search.

**Download:** http://corpus-texmex.irisa.fr/

**Files needed:**
- `base.fvecs` - 1,000,000 base vectors (128 dimensions)
- `query.fvecs` - 10,000 query vectors (128 dimensions)
- `groundtruth.ivecs` - Ground truth nearest neighbors (100 per query)

## Setup

### Option 1: Default location
```bash
mkdir -p ./data/sift
# Download and extract SIFT1M files to ./data/sift/
```

### Option 2: Custom location
```bash
export SIFT1M_PATH=/path/to/your/sift1m
```

## Running the Benchmark

### Full benchmark (1000 queries by default)
```bash
cargo test --test bench_sift1m -- --nocapture
```

### Quick benchmark (100 queries)
```bash
cargo test --test bench_sift1m test_sift1m_quick -- --nocapture
```

### Custom number of queries
```bash
SIFT_NUM_QUERIES=5000 cargo test --test bench_sift1m -- --nocapture
```

## Output Format

```
=== SIFT1M Benchmark Results ===

Index        Build(ms)    Search(ms)        QPS          R@1         R@10        R@100
------------------------------------------------------------------------------------------
Flat          1234.56       567.89        176       1.000       1.000       1.000
HNSW          2345.67        45.67       2190       0.985       0.985       0.990
IVF-Flat       890.12       123.45        810       0.952       0.952       0.970
```

## Metrics

- **Build(ms)**: Time to build the index
- **Search(ms)**: Time to search all queries
- **QPS**: Queries per second
- **R@1**: Recall@1 (fraction of queries where top-1 result is correct)
- **R@10**: Recall@10 (fraction of ground truth neighbors found in top-10)
- **R@100**: Recall@100 (fraction of ground truth neighbors found in top-100)

## Implementation

### Files

- `src/dataset/sift_loader.rs` - SIFT dataset loading (fvecs/ivecs format)
- `src/benchmark/recall.rs` - Recall calculation utilities
- `tests/bench_sift1m.rs` - Benchmark test

### Fvecs Format

The fvecs format is a simple binary format:
- 4-byte dimension (little-endian u32)
- N vectors, each with:
  - 4-byte dimension prefix (little-endian u32)
  - dim × 4-byte floats (little-endian f32)

### Ivecs Format

Same as fvecs, but with i32 integers instead of f32 floats.

## API Usage

```rust
use hanns::dataset::{load_sift1m_complete, SiftDataset};

// Load complete dataset
let dataset: SiftDataset = load_sift1m_complete("./data/sift")?;

// Access components
let base_vectors = dataset.base.vectors();  // 1M x 128D
let query_vectors = dataset.query.vectors(); // 10K x 128D
let ground_truth = &dataset.ground_truth;    // 10K x 100

// Calculate recall
use hanns::benchmark::average_recall_at_k;
let recall = average_recall_at_k(&results, &ground_truth, 10);
```
