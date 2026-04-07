# Milvus VectorDBBench RS Authority Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy current knowhere-rs main to hannsdb-x86, run VectorDBBench Cohere 1M HNSW with Milvus+RS backend, and produce a valid authority result comparable to the native baseline (QPS=848, recall=0.9558, 2026-04-01).

**Architecture:** Three sequential phases — Sync+Build → RS Benchmark Run → Archive+Compare. The native baseline is already recorded; only the RS side needs a fresh run. If QPS gap is > 2x vs native, a targeted FFI overhead profiling track kicks in as a fourth phase.

**Tech Stack:** Rust (cargo build --release --lib), SSH (`hannsdb-x86`), rsync, VectorDBBench Python runner, Milvus standalone, JSON result files.

---

## Baselines (Do Not Re-Run Native)

| Lane | QPS | Recall | Insert(s) | Optimize(s) | Load(s) | Date |
|------|----:|-------:|----------:|------------:|--------:|------|
| native Milvus/knowhere Cohere 1M | 848.398 | 0.9558 | 179.063 | 339.646 | 518.709 | 2026-04-01 |
| RS (last measured, April 3) | 351 | 0.990 | — | — | — | 2026-04-03 |

**Target:** RS QPS ≥ 848 AND recall ≥ 0.95. If unmet, diagnose via Task 5.

---

## File Map

| File | Action |
|------|--------|
| `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260405.json` | Create: archive result |
| `docs/milvus-vectordbbench-native-vs-rs-status.md` | Modify: append new row to Table A |
| Remote: `/data/work/milvus-rs-integ/knowhere-rs/` | Sync target |
| Remote: `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so` | Build artifact |
| Remote: `/data/work/VectorDBBench/logs/rs_cohere1m_20260405_HHMMSS.log` | Created by runner |
| Remote: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/*.json` | Benchmark result JSON |

---

## Task 1: Sync & Build

**Files:**
- Remote: `/data/work/milvus-rs-integ/knowhere-rs/` (sync target)
- Remote: `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so`

- [ ] **Step 1: Rsync knowhere-rs to hannsdb-x86**

```bash
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='target/' \
  --exclude='benchmark_results/.diskann_cache*' \
  --exclude='data/' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
```

Expected: transfer completes with `sent ... bytes  received ... bytes`.

- [ ] **Step 2: Record SHA deployed**

```bash
git -C /Users/ryan/.openclaw/workspace-builder/knowhere-rs rev-parse HEAD
```

Save this SHA — it goes into the archive JSON in Task 4.

- [ ] **Step 3: Build libknowhere_rs.so on x86**

```bash
ssh hannsdb-x86 "
  cd /data/work/milvus-rs-integ/knowhere-rs
  source \"\$HOME/.cargo/env\" >/dev/null 2>&1 || true
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  cargo build --release --lib 2>&1 | tail -12
"
```

Expected: ends with `Finished release [optimized] target(s) in ...s`. No `error[E...]` lines.

- [ ] **Step 4: Verify .so is fresh**

```bash
ssh hannsdb-x86 "ls -lh /data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so"
```

Expected: file exists, timestamp within the last 10 minutes, size 50–150 MB.

- [ ] **Step 5: Commit — record sync SHA (no code changes here)**

```bash
# Nothing to commit in this task — SHA was recorded in Step 2 above.
echo "Task 1 done. SHA=$(git -C /Users/ryan/.openclaw/workspace-builder/knowhere-rs rev-parse HEAD)"
```

---

## Task 2: Start Milvus

**Files:**
- Remote: `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log`

- [ ] **Step 1: Kill any stale Milvus process**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' 2>/dev/null; sleep 3; echo killed"
```

Expected: `killed`. `no process found` is also fine.

- [ ] **Step 2: Clear stale Milvus data (force fresh segment build)**

This forces Milvus to rebuild all HNSW segments with the new RS binary, so the slab fast-path is active on every segment.

```bash
ssh hannsdb-x86 "rm -rf /data/work/milvus-rs-integ/milvus-var/data/ && echo cleared"
```

Expected: `cleared`.

- [ ] **Step 3: Start Milvus via canonical wrapper**

```bash
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/milvus-src && \
  scripts/knowhere-rs-shim/start_standalone_remote.sh"
```

Expected output ending with:
```
PID=...
LOG=/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log
HEALTHY
```

If `HEALTHY` does not appear within 60s, check the log:

```bash
ssh hannsdb-x86 "tail -30 /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log"
```

---

## Task 3: Run RS VectorDBBench (Cohere 1M)

**Files:**
- Remote: `/data/work/VectorDBBench/logs/rs_cohere1m_20260405_HHMMSS.log`
- Remote: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/*.json`

**Context:** The Cohere 1M runner takes ~60–90 min. Run in nohup. Search validity requires the final large post-compaction segment (>100 MB serialized) to finish HNSW build + save + QueryNode load *before* the first QPS measurement.

- [ ] **Step 1: Start RS VectorDBBench runner**

```bash
LOGFILE="rs_cohere1m_$(date +%Y%m%d_%H%M%S).log"
ssh hannsdb-x86 "cd /data/work/VectorDBBench && \
  nohup .venv/bin/python run_milvus_hnsw_1m_cohere_rs.py \
  > logs/${LOGFILE} 2>&1 & \
  echo \$! > /tmp/rs_bench.pid && \
  echo \"started PID=\$(cat /tmp/rs_bench.pid) log=logs/${LOGFILE}\""
```

Expected: `started PID=<n> log=logs/rs_cohere1m_20260405_HHMMSS.log`. Save the log filename.

- [ ] **Step 2: Confirm runner is alive after 30s**

```bash
ssh hannsdb-x86 "sleep 30 && tail -10 /data/work/VectorDBBench/logs/rs_cohere1m_*.log 2>/dev/null | tail -10"
```

Expected: shows VectorDBBench startup messages, dataset loading, or insert progress. If empty or error, check pid:

```bash
ssh hannsdb-x86 "cat /tmp/rs_bench.pid && kill -0 \$(cat /tmp/rs_bench.pid) 2>/dev/null && echo alive || echo dead"
```

- [ ] **Step 3: Monitor for valid search window (run ~45-60 min in)**

Validity rule: search must start only after the last large post-compaction segment finishes build+save+load.

```bash
# Check Milvus log for large segment build completions
ssh hannsdb-x86 "grep -E 'loadTextIndexes|finish building index|Segment.*load.*done|compaction.*done' \
  /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log | tail -20"
```

```bash
# Check VectorDBBench log for search start
ssh hannsdb-x86 "grep -E 'concurren|qps|recall|search_duration' \
  /data/work/VectorDBBench/logs/rs_cohere1m_*.log 2>/dev/null | tail -10"
```

Valid window confirmed when: search concurrency lines appear *after* the last `loadTextIndexes` span for a large segment. If search overlaps with active segment loading → mark run **invalid search window** and restart from Task 2 Step 1.

- [ ] **Step 4: Wait for benchmark to complete**

```bash
ssh hannsdb-x86 "while kill -0 \$(cat /tmp/rs_bench.pid) 2>/dev/null; do sleep 60; echo 'still running...'; done; echo DONE"
```

This polls every 60s. Total wait ~60–90 min.

- [ ] **Step 5: Extract result JSON**

```bash
# Find newest result file
ssh hannsdb-x86 "ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/ | head -5"
```

Read the result:
```bash
ssh hannsdb-x86 "cat /data/work/VectorDBBench/vectordb_bench/results/Milvus/<NEWEST_FILE>.json"
```

- [ ] **Step 6: Record the 5 numbers**

```
insert_duration  = ___
optimize_duration = ___
load_duration    = ___
qps              = ___
recall           = ___
```

**Gate:**
- **QPS ≥ 848 AND recall ≥ 0.95** → go to **Task 4** (archive)
- **QPS < 848** → go to **Task 5** (FFI diagnosis) before Task 4
- **recall < 0.95** → investigate graph quality (check if parallel build env var leaked in)

---

## Task 4: Archive Results

**Files:**
- Create: `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260405.json`
- Modify: `docs/milvus-vectordbbench-native-vs-rs-status.md`

- [ ] **Step 1: Write archive JSON**

Create `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260405.json`:

```json
{
  "benchmark": "milvus-vectordbbench-cohere1m-hnsw-compare",
  "authority_scope": "remote_x86_only",
  "host": "hannsdb-x86",
  "date": "2026-04-05",
  "rs_git_sha": "<SHA_FROM_TASK_1_STEP_2>",
  "rs_config": {
    "parallel_build": false,
    "batch_cap": null,
    "notes": "serial HNSW build (default), zero-copy bitset via BitsetRef"
  },
  "params": {
    "case_type": "Performance768D1M",
    "db": "Milvus",
    "index": "HNSW",
    "metric_type": "COSINE",
    "top_k": 100,
    "m": 16,
    "ef_construction": 128,
    "ef_search": 128
  },
  "rows": [
    {
      "backend": "milvus-native-knowhere",
      "qps": 848.398,
      "recall": 0.9558,
      "insert_duration": 179.063,
      "optimize_duration": 339.6463,
      "load_duration": 518.7093,
      "search_window": "valid",
      "source_result": "/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_20260401_milvus-native-knowhere.json"
    },
    {
      "backend": "milvus-knowhere-rs",
      "qps": <QPS>,
      "recall": <RECALL>,
      "insert_duration": <INSERT_S>,
      "optimize_duration": <OPTIMIZE_S>,
      "load_duration": <LOAD_S>,
      "search_window": "valid",
      "source_result": "/data/work/VectorDBBench/vectordb_bench/results/Milvus/<NEWEST_FILE>.json",
      "source_log": "/data/work/VectorDBBench/logs/<LOGFILE>"
    }
  ],
  "ratios": {
    "rs_over_native_qps": <QPS / 848.398>,
    "rs_recall_delta": <RECALL - 0.9558>
  },
  "verdict": "<rs_meets_target | rs_below_target>",
  "notes": [
    "RS build: serial (no parallel). Zero-copy bitset path (BitsetRef, no alloc per query).",
    "Native baseline reused from 2026-04-01."
  ]
}
```

Replace all `<...>` with actual measured numbers.

- [ ] **Step 2: Append new row to status doc**

Open `docs/milvus-vectordbbench-native-vs-rs-status.md`, section `## A. Cohere 1M Comparison`. Append a new row:

```markdown
| rs current (20260405) | <INSERT_S> | <OPTIMIZE_S> | <LOAD_S> | <QPS> | <RECALL> | valid |
```

Also update the "Takeaways" paragraph to note the current RS vs native QPS ratio and what commits drove any change.

- [ ] **Step 3: Commit**

```bash
git add benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260405.json
git add docs/milvus-vectordbbench-native-vs-rs-status.md
git commit -m "$(cat <<'EOF'
bench(milvus): Cohere 1M HNSW RS authority run 2026-04-05

RS QPS: <QPS> vs native 848.4 (<RATIO>x), recall: <RECALL>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: FFI Overhead Diagnosis (Trigger: QPS < 848)

**Trigger:** Task 3 QPS < 848. Work through checks in order; fix the first issue found, then re-build + re-run from Task 2.

**Files:**
- `src/ffi.rs` — may instrument or modify
- `src/faiss/hnsw.rs` — check slab activation

- [ ] **Step 1: Check slab activation**

The fast path `search_layer0_bitset_fast` requires `layer0_slab.is_some()`. If Milvus loaded old serialized segments (before slab was built), it will be `None`.

Clearing Milvus data (Task 2 Step 2) should prevent this. If data was NOT cleared, clear it now and restart from Task 2.

Verify slab is live by checking the RS binary has the slab code:

```bash
# Check that search_layer0_bitset_fast exists in the .so
ssh hannsdb-x86 "nm -D /data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so | grep -c bitset_fast"
```

Expected: count > 0. If 0, the symbol was not compiled — check `src/faiss/hnsw.rs` for the function definition.

- [ ] **Step 2: Check ef_search parameter**

Confirm the RS runner passes ef_search=128 (same as native):

```bash
ssh hannsdb-x86 "grep -E 'ef_search|efSearch' /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py"
```

Expected: `ef_search=128`. If different, update the runner to match native's ef_search.

- [ ] **Step 3: Add FFI search timing instrumentation**

If QPS gap is > 1.5x (RS < 565 QPS) and the above checks pass, the bottleneck is in the FFI path itself. Add minimal timing to `knowhere_search_with_bitset` in `src/ffi.rs`:

Add these lines at the start of the unsafe block in `knowhere_search_with_bitset` (around line 2516):

```rust
let _t0 = std::time::Instant::now();
```

And print at the end (inside the HNSW branch, before returning `Box::into_raw`):

```rust
if _t0.elapsed().as_micros() > 5000 {
    eprintln!("[ffi-timing] search_with_bitset HNSW: {}µs", _t0.elapsed().as_micros());
}
```

Rsync + rebuild + restart + run a short benchmark (Task 2 → Task 3). After a few minutes, check Milvus stderr:

```bash
ssh hannsdb-x86 "grep ffi-timing /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log | head -20"
```

If per-search latency is > 1ms on 768D, the bottleneck is inside the search path. If it is < 1ms, the overhead is in the Milvus Go↔C FFI bridging layer outside our control.

- [ ] **Step 4: Check for Rust panics during concurrent search**

```bash
ssh hannsdb-x86 "grep -E 'panic|SIGABRT|BorrowMut|RefCell|already borrowed' \
  /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log | tail -20"
```

Expected: no matches. If panics found, note function name — likely a TLS scratch contention issue in the search path. Fix: ensure `SearchScratch` uses `thread_local!` storage only.

- [ ] **Step 5: After diagnosis fix — rebuild and re-run**

Apply the targeted fix. Rsync to hannsdb-x86:

```bash
rsync -avz --delete \
  --exclude='.git/' --exclude='target/' --exclude='benchmark_results/.diskann_cache*' --exclude='data/' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/

ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && \
  source \"\$HOME/.cargo/env\" >/dev/null 2>&1 || true && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  cargo build --release --lib 2>&1 | tail -5"
```

Then restart from **Task 2, Step 1** (including clearing Milvus data).

- [ ] **Step 6: Commit any code changes from diagnosis**

```bash
git add src/ffi.rs  # or whichever files changed
git commit -m "fix(ffi): <describe the diagnosis fix>"
```

Then return to Task 3.

---

## Iteration Log

Update after each run:

| Date | Phase | Change | Insert(s) | Optimize(s) | Load(s) | QPS | Recall | Valid? |
|------|-------|--------|-----------|-------------|---------|-----|--------|--------|
| 20260405 | P1 | current main | ? | ? | ? | ? | ? | ? |
