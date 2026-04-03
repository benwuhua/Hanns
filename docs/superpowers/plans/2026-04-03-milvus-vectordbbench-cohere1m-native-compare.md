# Milvus VectorDBBench Cohere 1M RS vs Native Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the 4 bitset-search commits to `hannsdb-x86`, run Milvus + knowhere-rs on Cohere 1M HNSW, and iterate until RS achieves QPS ≥ 848 (≥ native) with recall ≥ 0.95.

**Architecture:** Four sequential phases — Sync+Run → Search Diagnosis (if QPS < 848) → Build Optimization (if optimize > 500s) → Final Verification+Archive. Only one variable changes per iteration; each phase has a clear exit condition.

**Tech Stack:** Rust (cargo build), SSH (`hannsdb-x86`), rsync, VectorDBBench (Python venv), Milvus standalone, JSON result files.

---

## File Map

| File | Role |
|------|------|
| `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260403.json` | Create: archive result for this run |
| `docs/milvus-vectordbbench-native-vs-rs-status.md` | Modify: append new valid row to Cohere 1M table |
| Remote: `/data/work/milvus-rs-integ/knowhere-rs/` | Sync target (knowhere-rs checkout on x86) |
| Remote: `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so` | Build artifact |
| Remote: `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log` | Milvus log for validity check |
| Remote: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/*.json` | Benchmark result JSONs |

---

## Task 1: Sync & Build on x86

**Files:**
- Remote: `/data/work/milvus-rs-integ/knowhere-rs/` (sync target)
- Remote: `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so` (build artifact)

- [ ] **Step 1: Rsync local knowhere-rs to hannsdb-x86**

```bash
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='target/' \
  --exclude='benchmark_results/.diskann_cache*' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
```

Expected: transfer completes, ends with `sent ... bytes  received ... bytes`.

- [ ] **Step 2: Verify git SHA on remote matches local**

```bash
# Local SHA of the tip commit (should be the spec commit c251a8a or later)
git rev-parse HEAD

# Remote SHA
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && git log --oneline -1"
```

Expected: both show the same 7-char short SHA. If they differ, the rsync may have synced files but git HEAD won't update (rsync copies files, not git refs — that's fine; git SHA on remote may differ but the *source files* are synced). What matters: the Rust source files match. Verify with:

```bash
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && git diff --stat HEAD"
```

Expected: clean or only untracked local files.

- [ ] **Step 3: Build libknowhere_rs.so on x86**

```bash
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && \
  source \"\$HOME/.cargo/env\" >/dev/null 2>&1 || true && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  cargo build --release --lib 2>&1 | tail -10"
```

Expected: ends with `Compiling knowhere_rs ...` then `Finished release [optimized] target(s) in ...s`. No `error[E...]` lines.

- [ ] **Step 4: Verify the .so artifact exists and is fresh**

```bash
ssh hannsdb-x86 "ls -lh /data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so"
```

Expected: file exists, timestamp is within the last few minutes, size ~50-100 MB.

- [ ] **Step 5: Commit — no local code changes in this task, but record the SHA used**

```bash
# Record which SHA was deployed — save for archive use
git log --oneline -5
```

Note the tip SHA (e.g., `c251a8a`). This is recorded in the archive JSON in Task 5.

---

## Task 2: Phase 1 — First RS Run

**Files:**
- Remote: `/data/work/VectorDBBench/logs/rs_cohere1m_bitset4_20260403_HHMMSS.log` (created)
- Remote: `/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_*_milvus.json` (created by runner)
- Remote: `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log` (read for validity)

**Context:** The RS runner takes ~60–90 minutes for Cohere 1M. Run it in a remote screen session or nohup so SSH disconnect doesn't kill it. Valid search window requires the final large post-compaction segment (>100 MB serialized) to finish HNSW build + save + QueryNode load *before* the first QPS measurement.

- [ ] **Step 1: Kill any stale Milvus process**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' 2>/dev/null; sleep 3; echo done"
```

Expected: `done`. If `pkill` says `no process found`, that's fine.

- [ ] **Step 2: Restart Milvus via canonical wrapper**

```bash
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/milvus-src && \
  scripts/knowhere-rs-shim/start_standalone_remote.sh"
```

Expected output:
```
PID=...
LOG=/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log
HEALTHY
```

If `HEALTHY` does not appear within 60s, check the log:
```bash
ssh hannsdb-x86 "tail -30 /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log"
```

- [ ] **Step 3: Start the RS VectorDBBench runner**

```bash
LOGFILE="rs_cohere1m_bitset4_$(date +%Y%m%d_%H%M%S).log"
ssh hannsdb-x86 "cd /data/work/VectorDBBench && \
  nohup .venv/bin/python run_milvus_hnsw_1m_cohere_rs.py \
  > logs/${LOGFILE} 2>&1 & \
  echo \$! > /tmp/rs_bench.pid && \
  echo 'started PID=' \$(cat /tmp/rs_bench.pid) ' log=logs/${LOGFILE}'"
```

Expected: `started PID=<n> log=logs/rs_cohere1m_bitset4_20260403_HHMMSS.log`

Save the log filename. You'll need it to find the result JSON later.

- [ ] **Step 4: Monitor VectorDBBench log to confirm it is running**

```bash
ssh hannsdb-x86 "sleep 30 && tail -20 /data/work/VectorDBBench/logs/rs_cohere1m_bitset4_*.log 2>/dev/null | head -20"
```

Expected: shows VectorDBBench startup, dataset loading, or insert progress.

- [ ] **Step 5: Monitor Milvus log for valid search window**

The validity rule: search must start *after* the last large post-compaction segment (>100 MB serialized) finishes HNSW build, save, and QueryNode load.

Poll the Milvus log for these markers (run during the benchmark, roughly 30–60 min in):

```bash
# Check for large segment build/load completions
ssh hannsdb-x86 "grep -E 'loadTextIndexes|finish building index|Segment.*load.*done|compaction.*done|build.*HNSW' \
  /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log | tail -20"
```

```bash
# Check for search start (VectorDBBench concurrency sweep)
ssh hannsdb-x86 "grep -E 'concurren|search_duration|qps|recall' \
  /data/work/VectorDBBench/logs/rs_cohere1m_bitset4_*.log 2>/dev/null | tail -10"
```

Valid window confirmed when: search concurrency log lines appear only *after* the last `loadTextIndexes` span for a large segment. If search appears to overlap with an active `loadTextIndexes`, mark the run **invalid search window** and restart from Step 1.

- [ ] **Step 6: Wait for benchmark to complete, then read result JSON**

```bash
# Wait for PID to exit
ssh hannsdb-x86 "while kill -0 \$(cat /tmp/rs_bench.pid) 2>/dev/null; do sleep 60; done; echo 'done'"
```

Then find the result JSON (newest file in results/Milvus/):

```bash
ssh hannsdb-x86 "ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/ | head -3"
```

Read the result file:
```bash
ssh hannsdb-x86 "cat /data/work/VectorDBBench/vectordb_bench/results/Milvus/<NEWEST_FILE>.json"
```

- [ ] **Step 7: Record the 5 numbers**

From the result JSON, note:
```
insert_duration  = ___
optimize_duration = ___
load_duration    = ___
qps              = ___
recall           = ___
```

**Gate:**
- If **QPS ≥ 848 AND recall ≥ 0.95** → skip Tasks 3 and 4, go directly to **Task 5**
- If **QPS < 848** → proceed to **Task 3** (Phase 2: Search QPS Diagnosis)
- If **QPS ≥ 848 but optimize_duration > 500s** → proceed to **Task 4** (Phase 3: Build Optimization)

---

## Task 3: Phase 2 — Search QPS Diagnosis

**Trigger:** Task 2 QPS < 848.

**Files:**
- `src/faiss/hnsw.rs` — May be modified (targeted fix)
- `src/ffi/hnsw_ffi.rs` or equivalent — Check ef_search pass-through

**Context:** The 4 bitset commits (4257def..4f766a7) add `search_layer0_bitset_fast` using `layer0_slab` + TLS scratch + SIMD batch-4. If QPS is still below native, one of four root causes is active. Work through this checklist *in order*, fix the first issue found, then rebuild + re-run Task 2.

- [ ] **Step 1: Check 1 — slab activation**

The fast path `search_layer0_bitset_fast` is only called when `self.layer0_slab.is_some()`. The slab is built in `HnswIndex::build()` and persisted in `serialize`/`deserialize`. If Milvus loaded an old serialized segment from a prior RS run that predates the slab feature, `layer0_slab` will be `None` and the fast path silently falls back to `search_layer_idx_shared`.

Verify:

```bash
# Check if any segments were loaded from cache (pre-bitset RS build)
ssh hannsdb-x86 "ls -lt /data/work/milvus-rs-integ/milvus-var/ 2>/dev/null | head -5"
```

If Milvus data directory contains old serialized segments, force a fresh rebuild by clearing them:

```bash
ssh hannsdb-x86 "rm -rf /data/work/milvus-rs-integ/milvus-var/data/ && echo cleared"
```

Then re-run from Task 2 Step 1 (fresh restart + fresh insert). This forces Milvus to rebuild all segments with the new RS binary.

If slab was not the issue (or data was already fresh), continue to Step 2.

- [ ] **Step 2: Check 2 — q_norm TLS set before query**

In `search_single_with_bitset` (hnsw.rs), the cosine fast path requires `HNSW_COSINE_QUERY_NORM_TLS` to be set to `|q|` before each query. Grep to confirm the TLS set call is in the right place:

```bash
grep -n "HNSW_COSINE_QUERY_NORM_TLS" src/faiss/hnsw.rs | head -20
```

Expected: should show a `.with(|cell| cell.set(...))` call inside `search_with_bitset` or `search_single_with_bitset` *before* the `search_layer0_bitset_fast` call, using the pre-computed `q_norm` from the query.

If the TLS set is missing or placed after the fast path call, fix it:

```rust
// In search_single_with_bitset, before calling search_layer0_bitset_fast:
let q_norm = simd::l2_norm(query);
HNSW_COSINE_QUERY_NORM_TLS.with(|cell| cell.set(q_norm));
```

After any fix: rebuild → restart Milvus → re-run Task 2.

- [ ] **Step 3: Check 3 — concurrent search panics**

Milvus QueryNode searches multi-threaded. Scan the Milvus log for Rust panics during the search phase:

```bash
ssh hannsdb-x86 "grep -E 'panic|SIGABRT|BorrowMut|RefCell|already borrowed' \
  /data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log | tail -20"
```

Expected: no matches. If panics are found, note the line and function name — this indicates a `RefCell::borrow_mut` conflict in the search path (likely in `SearchScratch` or `Layer0OrderedPool`). Fix: ensure TLS scratch uses per-thread storage (`thread_local!`) not shared state.

- [ ] **Step 4: Check 4 — ef_search mismatch**

Confirm Milvus passes `ef_search=128` to the RS FFI (same as native). Check the RS VectorDBBench runner:

```bash
ssh hannsdb-x86 "grep -E 'ef_search|efSearch|ef_search' /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py"
```

Expected: `ef_search=128` (matching native). If the RS runner uses a different ef_search (e.g., a default larger value), that degrades QPS. Fix by aligning the runner config.

- [ ] **Step 5: Apply fix, rebuild, re-run**

After finding the root cause in Steps 1–4, apply the minimal targeted fix. Rebuild:

```bash
# Local fix + rsync
rsync -avz --delete \
  --exclude='.git/' --exclude='target/' --exclude='benchmark_results/.diskann_cache*' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/

ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && \
  source \"\$HOME/.cargo/env\" >/dev/null 2>&1 || true && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  cargo build --release --lib 2>&1 | tail -5"
```

Then restart from **Task 2, Step 1**. Repeat Phase 2 diagnosis if QPS is still below 848 after the fix.

- [ ] **Step 6: Commit any code changes**

```bash
git add src/faiss/hnsw.rs  # or whichever file was changed
git commit -m "fix(hnsw): <describe the diagnosis fix>"
```

---

## Task 4: Phase 3 — Parallel Build Optimization

**Trigger:** optimize_duration > 500s (after QPS ≥ 848 is confirmed).

**Files:**
- Remote: Milvus startup environment only (env-var-only change, no Rust recompile needed)

**Context:** Serial HNSW build on large post-compaction segments (700 MB–1 GB) takes many minutes. Native C++ uses multi-threaded build. The RS FFI has an env-var-gated parallel build path: `FFI_ENABLE_PARALLEL_HNSW_ADD_ENV=1`. Local experiments show `batch_cap=64` gives recall@10=0.9922 (well above 0.95 gate). This is tunable without recompile.

- [ ] **Step 1: Restart Milvus with parallel build env vars (cap=64)**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' 2>/dev/null; sleep 3; echo killed"

ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/milvus-src && \
  FFI_ENABLE_PARALLEL_HNSW_ADD_ENV=1 \
  FFI_PARALLEL_HNSW_ADD_BATCH_CAP=64 \
  scripts/knowhere-rs-shim/start_standalone_remote.sh"
```

Expected: `HEALTHY` with env vars active in the Milvus process.

Verify env vars are visible to the Milvus process:
```bash
ssh hannsdb-x86 "grep -E 'PARALLEL_HNSW|BATCH_CAP' \
  /proc/\$(pgrep -f 'milvus run standalone')/environ 2>/dev/null | tr '\0' '\n' || echo 'check /proc manually'"
```

- [ ] **Step 2: Re-run VectorDBBench (full run)**

Follow Task 2 Steps 3–7 exactly, with the new log filename reflecting the cap:

```bash
LOGFILE="rs_cohere1m_bitset4_cap64_$(date +%Y%m%d_%H%M%S).log"
ssh hannsdb-x86 "cd /data/work/VectorDBBench && \
  nohup .venv/bin/python run_milvus_hnsw_1m_cohere_rs.py \
  > logs/${LOGFILE} 2>&1 & \
  echo \$! > /tmp/rs_bench.pid && echo 'started'"
```

- [ ] **Step 3: Read result and check both gates**

After completion, record numbers as in Task 2 Step 7.

Check both gates:
- **optimize_duration ≤ 500s** — if still > 500s, lower cap to 32 and repeat this task
- **recall ≥ 0.95** — if recall dropped below 0.95, lower cap to 32 and repeat
- **QPS ≥ 848** — must still hold; if not, investigate (cap=64 should not regress QPS vs serial)

If cap=64 fails either gate:
```bash
# Retry with cap=32
ssh hannsdb-x86 "pkill -f 'milvus run standalone' 2>/dev/null; sleep 3 && \
  cd /data/work/milvus-rs-integ/milvus-src && \
  FFI_ENABLE_PARALLEL_HNSW_ADD_ENV=1 \
  FFI_PARALLEL_HNSW_ADD_BATCH_CAP=32 \
  scripts/knowhere-rs-shim/start_standalone_remote.sh"
# Then re-run VectorDBBench (Task 2 Steps 3-7)
```

If cap=64 passes both gates → proceed to Task 5.

---

## Task 5: Phase 4 — Final Verification & Archive

**Trigger:** QPS ≥ 848, recall ≥ 0.95, valid search window confirmed.

**Files:**
- Create: `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260403.json`
- Modify: `docs/milvus-vectordbbench-native-vs-rs-status.md`

- [ ] **Step 1: Run second RS benchmark (reproducibility check)**

Under identical conditions (same env vars, same Milvus restart):

```bash
LOGFILE="rs_cohere1m_bitset4_run2_$(date +%Y%m%d_%H%M%S).log"
ssh hannsdb-x86 "pkill -f 'milvus run standalone' 2>/dev/null; sleep 3 && \
  cd /data/work/milvus-rs-integ/milvus-src && \
  FFI_ENABLE_PARALLEL_HNSW_ADD_ENV=1 \
  FFI_PARALLEL_HNSW_ADD_BATCH_CAP=<CAP_USED> \
  scripts/knowhere-rs-shim/start_standalone_remote.sh && \
  cd /data/work/VectorDBBench && \
  nohup .venv/bin/python run_milvus_hnsw_1m_cohere_rs.py \
  > logs/${LOGFILE} 2>&1 & echo \$! > /tmp/rs_bench_run2.pid && echo started"
```

Replace `<CAP_USED>` with the cap value from Task 4 (or omit the env vars if serial build was already at target).

Wait for completion:
```bash
ssh hannsdb-x86 "while kill -0 \$(cat /tmp/rs_bench_run2.pid) 2>/dev/null; do sleep 60; done; echo done"
```

Read the second result JSON (newest file):
```bash
ssh hannsdb-x86 "ls -lt /data/work/VectorDBBench/vectordb_bench/results/Milvus/ | head -3"
```

Record run 2 numbers: `qps_run2`, `recall_run2`.

- [ ] **Step 2: Verify reproducibility (within 10% QPS)**

```python
# Check in your head or with a calculator:
# |qps_run1 - qps_run2| / qps_run1 <= 0.10
# Example: run1=870, run2=851 → diff=19/870=2.2% ✅
# Example: run1=870, run2=780 → diff=90/870=10.3% ❌ → mark unstable
```

If > 10% variance: mark both runs as `unstable`. Do not archive as valid. Stop here and investigate root cause (likely Milvus didn't fully reload between runs).

If ≤ 10% variance: proceed to Step 3.

- [ ] **Step 3: Write archive JSON**

Create `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260403.json` with the actual numbers from this run. Use the format below, filling in the `?` fields:

```json
{
  "benchmark": "milvus-vectordbbench-cohere1m-hnsw-compare",
  "authority_scope": "remote_x86_only",
  "host": "hannsdb-x86",
  "date": "2026-04-03",
  "rs_git_sha": "<TIP_SHA_FROM_TASK_1>",
  "rs_config": {
    "parallel_build": true,
    "batch_cap": <CAP_USED_OR_NULL>,
    "commits": ["4257def", "e3332d9", "93dedef", "4f766a7"]
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
      "source_result": "/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_20260330_milvus-native-knowhere-hnsw-cohere1m-20260330_milvus.json",
      "source_log": "/data/work/VectorDBBench/logs/native_cohere1m_20260330.log"
    },
    {
      "backend": "milvus-knowhere-rs-bitset4",
      "qps": <QPS_RUN1>,
      "recall": <RECALL_RUN1>,
      "insert_duration": <INSERT_S>,
      "optimize_duration": <OPTIMIZE_S>,
      "load_duration": <LOAD_S>,
      "run2_qps": <QPS_RUN2>,
      "run2_recall": <RECALL_RUN2>,
      "qps_reproducibility_pct": <ABS_DIFF_PCT>,
      "search_window": "valid",
      "source_result": "/data/work/VectorDBBench/vectordb_bench/results/Milvus/<RESULT_JSON_FILENAME>",
      "source_log": "/data/work/VectorDBBench/logs/<LOG_FILENAME>"
    }
  ],
  "ratios": {
    "rs_over_native_qps": <QPS_RUN1 / 848.398>,
    "rs_minus_native_recall": <RECALL_RUN1 - 0.9558>,
    "rs_over_native_optimize_duration": <OPTIMIZE_S / 339.6463>
  },
  "verdict": "rs_meets_target",
  "notes": [
    "4 bitset-search commits: TLS scratch reuse, layer0_slab fast path, SIMD L2 batch-4, SIMD cosine batch-4.",
    "Native baseline reused from result_20260330 (QPS=848.4, recall=0.9558).",
    "Both runs within <REPRODUCIBILITY_PCT>% QPS variance."
  ]
}
```

- [ ] **Step 4: Append new row to status doc**

Open `docs/milvus-vectordbbench-native-vs-rs-status.md`. In section `## A. Cohere 1M Comparison`, append a new row to the table:

```markdown
| rs bitset4 (20260403) | <INSERT_S> | <OPTIMIZE_S> | <LOAD_S> | <QPS> | <RECALL> | valid |
```

Also update the "Takeaways" paragraph below the table to note the bitset-search fast path closed the QPS gap.

- [ ] **Step 5: Commit**

```bash
git add benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_20260403.json
git add docs/milvus-vectordbbench-native-vs-rs-status.md
git commit -m "$(cat <<'EOF'
bench(milvus): RS Cohere 1M HNSW meets native QPS target

bitset-search fast path (4 commits: 4257def..4f766a7)
QPS: <QPS> (≥ native 848.4), recall: <RECALL>, optimize: <OPTIMIZE_S>s

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Iteration Log

Update this table after each run attempt:

| Date | Phase | Change | insert(s) | optimize(s) | load(s) | QPS | Recall | Valid? |
|------|-------|--------|-----------|-------------|---------|-----|--------|--------|
| 20260403 | P1 | bitset-fast-path (4f766a7) | ? | ? | ? | ? | ? | ? |
