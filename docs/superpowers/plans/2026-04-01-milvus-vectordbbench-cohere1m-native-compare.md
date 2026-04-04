# Milvus VectorDBBench Cohere1M Native Compare Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare Milvus integrated with `knowhere-rs` against native Milvus/knowhere on the same VectorDBBench `Cohere 1M` HNSW case and produce a fair, replayable result table.

**Architecture:** Use the existing Milvus integration environment on `hannsdb-x86` and the existing VectorDBBench runner scripts, instead of standalone Rust examples. Freeze one benchmark lane: same host, same Milvus service shape, same dataset, same HNSW params, same query workload, different backend only (`knowhere-rs` vs native knowhere). Persist raw logs, result JSONs, and one normalized comparison summary.

**Tech Stack:** Milvus integration workspace on `hannsdb-x86`, VectorDBBench Python runners, Milvus HNSW case `Performance768D1M`, `knowhere-rs` shared library integration, JSON result extraction, remote shell commands over SSH.

---

## File/Artifact Map

- Reuse remote runner: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Reuse remote runner: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py`
- Reuse remote result root: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/`
- Reuse remote log root: `/data/work/VectorDBBench/logs/`
- Create local summary artifact: `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json`
- Create local benchmark lock test: `tests/bench_milvus_cohere1m_hnsw_compare.rs`
- Optional helper parser if extraction gets noisy: `scripts/parse_milvus_vectordbbench_result.py`

## Fairness Contract

- Same machine: `hannsdb-x86`
- Same dataset case: `Performance768D1M`
- Same index family: `HNSW`
- Same params: `M=16`, `efConstruction=128`, `efSearch=128`
- Same Milvus endpoint shape
- Same VectorDBBench runner version and output schema
- Change only backend target:
  - `milvus-knowhere-rs`
  - `milvus-native-knowhere`

## Chunk 1: Environment and Lane Validation

### Task 1: Freeze remote environment inputs

**Files:**
- Reuse: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Reuse: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py`
- Reuse: `/data/work/milvus-rs-integ/knowhere-rs`

- [ ] **Step 1: Record the exact remote repo revisions**

Run:
```bash
ssh hannsdb-x86 '
  set -e
  cd /data/work/milvus-rs-integ/knowhere-rs && echo RS=$(git rev-parse HEAD)
  cd /data/work/VectorDBBench && echo VDBBENCH=$(git rev-parse HEAD)
'
```
Expected: two commit SHAs printed.

- [ ] **Step 2: Verify the runner scripts are same-param**

Run:
```bash
ssh hannsdb-x86 '
  sed -n "1,120p" /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py
  sed -n "1,120p" /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py
'
```
Expected: both use `Performance768D1M`, `M=16`, `ef-construction=128`, `ef-search=128`.

- [ ] **Step 3: Verify Milvus service is healthy before benchmark**

Run:
```bash
ssh hannsdb-x86 '
  python3 - <<'"'"'PY'"'"'
import urllib.request, sys
url = "http://127.0.0.1:19530"
try:
    with urllib.request.urlopen(url, timeout=3) as r:
        print("milvus_http_status", r.status)
except Exception as e:
    print("milvus_probe_error", e)
    sys.exit(1)
PY
'
```
Expected: service is reachable. If HTTP root is not a health endpoint, at least connection must succeed.

- [ ] **Step 4: Verify current `knowhere-rs` library is the intended build**

Run:
```bash
ssh hannsdb-x86 '
  find /data/work/milvus-rs-integ -maxdepth 3 -name "libknowhere_rs.so" -o -name "libknowhere_rs.dylib"
'
```
Expected: one active integration library path is identified.

- [ ] **Step 5: Commit**

No commit in this task unless helper scripts/docs are changed.

## Chunk 2: Native Baseline Run

### Task 2: Re-run native Milvus/knowhere baseline

**Files:**
- Reuse: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py`
- Output: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*native*.json`
- Log: `/data/work/VectorDBBench/logs/native_cohere1m_*.log`

- [ ] **Step 1: Clear stale assumptions by locating previous result files**

Run:
```bash
ssh hannsdb-x86 '
  ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*native* 2>/dev/null | head
'
```
Expected: prior native result files are listed for reference only.

- [ ] **Step 2: Run the native benchmark lane**

Run:
```bash
ssh hannsdb-x86 '
  set -e
  cd /data/work/VectorDBBench
  python3 run_milvus_hnsw_1m_cohere_native.py | tee /data/work/VectorDBBench/logs/native_cohere1m_$(date +%Y%m%d_%H%M%S).log
'
```
Expected: VectorDBBench completes and writes a fresh native result JSON.

- [ ] **Step 3: Identify the fresh native result artifact**

Run:
```bash
ssh hannsdb-x86 '
  ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*native* | head -1
'
```
Expected: newest native JSON path printed.

- [ ] **Step 4: Commit**

No commit in this task unless local summary/test files are being added.

## Chunk 3: knowhere-rs Integrated Run

### Task 3: Re-run Milvus with `knowhere-rs`

**Files:**
- Reuse: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Output: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*knowhere-rs*.json`
- Log: `/data/work/VectorDBBench/logs/rs_cohere1m_*.log`

- [ ] **Step 1: Verify the rs-integrated Milvus target is active**

Run:
```bash
ssh hannsdb-x86 '
  ps -ef | egrep "milvus|knowhere_rs|standalone" | egrep -v egrep | head -50
'
```
Expected: active Milvus process inventory is visible.

- [ ] **Step 2: Run the `knowhere-rs` benchmark lane**

Run:
```bash
ssh hannsdb-x86 '
  set -e
  cd /data/work/VectorDBBench
  python3 run_milvus_hnsw_1m_cohere_rs.py | tee /data/work/VectorDBBench/logs/rs_cohere1m_$(date +%Y%m%d_%H%M%S).log
'
```
Expected: VectorDBBench completes and writes a fresh `knowhere-rs` result JSON.

- [ ] **Step 3: Identify the fresh rs result artifact**

Run:
```bash
ssh hannsdb-x86 '
  ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*knowhere-rs* | head -1
'
```
Expected: newest rs JSON path printed.

- [ ] **Step 4: Commit**

No commit in this task unless local summary/test files are being added.

## Chunk 4: Result Extraction and Normalization

### Task 4: Normalize native-vs-rs output into one durable local artifact

**Files:**
- Create: `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json`
- Optional Create: `scripts/parse_milvus_vectordbbench_result.py`

- [ ] **Step 1: Inspect the raw result schema**

Run:
```bash
ssh hannsdb-x86 '
  python3 - <<'"'"'PY'"'"'
import json, glob
for path in sorted(glob.glob("/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*cohere1m*.json"))[-4:]:
    data = json.load(open(path))
    print("FILE", path)
    print("TYPE", type(data).__name__)
    if isinstance(data, dict):
        print("KEYS", list(data.keys())[:20])
PY
'
```
Expected: enough schema detail to write a stable parser.

- [ ] **Step 2: Extract the comparison fields**

Fields to normalize:
- backend label
- dataset case
- M
- efConstruction
- efSearch
- insert/build time
- recall / accuracy field
- QPS / latency field
- result JSON path
- log path

- [ ] **Step 3: Write the normalized local artifact**

Target shape:
```json
{
  "benchmark": "milvus-vectordbbench-cohere1m-hnsw-compare",
  "host": "hannsdb-x86",
  "params": {
    "case_type": "Performance768D1M",
    "m": 16,
    "ef_construction": 128,
    "ef_search": 128
  },
  "rows": [
    {
      "backend": "milvus-native-knowhere",
      "qps": 0.0,
      "recall": 0.0,
      "source_result": "..."
    },
    {
      "backend": "milvus-knowhere-rs",
      "qps": 0.0,
      "recall": 0.0,
      "source_result": "..."
    }
  ]
}
```

- [ ] **Step 4: Copy or generate the artifact into the repo**

Run:
```bash
scp hannsdb-x86:/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_... /tmp/
```
Then write the normalized artifact locally.

- [ ] **Step 5: Commit**

```bash
git add benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json
git commit -m "bench(milvus): capture cohere1m rs vs native comparison"
```

## Chunk 5: Benchmark Contract Lock

### Task 5: Add a regression that locks the comparison artifact

**Files:**
- Create: `tests/bench_milvus_cohere1m_hnsw_compare.rs`

- [ ] **Step 1: Write the failing test**

```rust
use std::fs;

#[test]
fn milvus_cohere1m_hnsw_compare_artifact_exists_and_is_same_param() {
    let raw = fs::read_to_string("benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json")
        .expect("comparison artifact must exist");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("artifact must be valid json");
    assert_eq!(v["params"]["case_type"], "Performance768D1M");
    assert_eq!(v["params"]["m"], 16);
    assert_eq!(v["params"]["ef_construction"], 128);
    assert_eq!(v["params"]["ef_search"], 128);
    assert_eq!(v["rows"].as_array().unwrap().len(), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cargo test --test bench_milvus_cohere1m_hnsw_compare -- --nocapture
```
Expected: FAIL because artifact/test does not exist yet.

- [ ] **Step 3: Add the test file and normalized artifact**

Write the minimal test and artifact.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cargo test --test bench_milvus_cohere1m_hnsw_compare -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/bench_milvus_cohere1m_hnsw_compare.rs benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json
git commit -m "test(milvus): lock cohere1m vectordbbench comparison artifact"
```

## Chunk 6: Review and Handoff

### Task 6: Produce the operator-facing comparison table

**Files:**
- Reuse: `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260401.json`

- [ ] **Step 1: Build a concise comparison table**

Required columns:
- backend
- qps
- recall
- qps ratio vs native
- recall delta vs native

- [ ] **Step 2: State fairness caveats explicitly**

Must call out:
- same host
- same case
- same params
- whether Milvus service was restarted between runs
- whether result schema labels exactly match between native and rs runs

- [ ] **Step 3: Record the next action**

Choose one:
- if rs is slower: open search/insert hotspot diagnosis
- if rs is close/faster: freeze benchmark contract and expand to more cases

- [ ] **Step 4: Commit**

Only if docs/artifacts changed in this step.

---

## Suggested Execution Order

1. Chunk 1
2. Chunk 2
3. Chunk 3
4. Chunk 4
5. Chunk 5
6. Chunk 6

## Notes

- Do not mix standalone Rust examples with Milvus integration numbers in the same table.
- Do not compare runs with different `ef_search` or different dataset case labels.
- Do not trust `thread_num` labels alone; rely on VectorDBBench/Milvus case semantics.
- If Milvus integration uses a stale `libknowhere_rs.so`, stop and rebuild that lane before benchmarking.

Plan complete and saved to `docs/superpowers/plans/2026-04-01-milvus-vectordbbench-cohere1m-native-compare.md`. Ready to execute?
