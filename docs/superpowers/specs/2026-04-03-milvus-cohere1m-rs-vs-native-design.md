# Milvus Cohere 1M RS vs Native Authority Comparison — Design Spec

**Date:** 2026-04-03

**Goal:** Deploy the 4 bitset search optimization commits to x86, run Milvus + knowhere-rs against Milvus + native knowhere on the Cohere 1M HNSW lane, and iterate until RS achieves QPS ≥ 848 (≥ native) with recall ≥ 0.95.

---

## Scope

This is the Milvus-integrated benchmark track only. Dataset lane:

- `Performance768D1M` — Cohere 1M, 768D, COSINE, HNSW
- Host: `hannsdb-x86`
- Params: M=16, efConstruction=128, efSearch=128

Not in scope: standalone Rust HNSW benchmark, 500K OpenAI lane, DiskANN, IVF.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| RS QPS | ≥ 848 (native baseline) |
| RS recall | ≥ 0.95 |
| RS optimize duration | ≤ 500s (≤ 1.5× native 340s) |
| Search window validity | Search starts only after final large segment finishes build/save/load |
| Reproducibility | Two consecutive RS runs within 10% QPS of each other |

---

## Fairness Contract

The following are fixed across all runs:

- Same host: `hannsdb-x86`
- Same dataset: `Performance768D1M`
- Same index family: HNSW
- Same params: M=16, efConstruction=128, efSearch=128
- Same VectorDBBench checkout and runner scripts
- Same Milvus standalone shape

Only the backend changes between native and RS runs.

**Native baseline:** Reuse `result_20260330_milvus-native-knowhere-hnsw-cohere1m` (QPS=848.4, recall=0.9558, valid). Do not re-run native unless RS shows anomalous results that warrant environment verification.

---

## Validity Rule

A run's `qps/recall` is valid only when search starts **after** the final large post-compaction segment has completed all of:
- HNSW build
- Index save
- QueryNode load

Runs where search overlaps with these background tasks are marked `invalid search window`. Their insert/optimize/load durations remain informative but QPS/recall are discarded.

Enforcement: Monitor Milvus log at `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log` to confirm the final large segment (>100MB serialized) finishes before the first QPS measurement is recorded.

---

## Architecture: 4-Phase Sequential Iteration

```
Phase 1: Sync + First RS Run
           ↓ QPS ≥ 848 → skip to Phase 4
           ↓ QPS < 848 →
Phase 2: Search QPS Diagnosis + Fix
           ↓ optimize >> native →
Phase 3: Build/Optimize Diagnosis + Fix
           ↓
Phase 4: Final Verification + Archive
```

Each phase has a clearly defined exit condition. Only one variable changes per iteration.

---

## Phase 1: Sync & First RS Run

**Input:** Local main branch (4 bitset commits: 4257def..4f766a7)

**Steps:**
1. rsync local `knowhere-rs` to `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs`
2. Build RS library on x86: `CARGO_TARGET_DIR=... cargo build --release --lib`
3. Restart Milvus standalone via `start_standalone_remote.sh`
4. Run `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
5. Monitor Milvus log to confirm valid search window
6. Read result JSON: record insert/optimize/load/QPS/recall

**Exit condition:** One valid RS row with five numbers recorded.

---

## Phase 2: Search QPS Diagnosis

**Trigger:** Phase 1 QPS < 848.

**Diagnosis checklist (in order):**

1. **Slab not active** — Verify `layer0_slab` is enabled in the loaded index. If not, the bitset fast path falls back to the generic shared path (entire optimization ineffective). Check via a small integration test or log injection.

2. **q_norm TLS not set** — Confirm `HNSW_COSINE_QUERY_NORM_TLS` is correctly set before each query in the bitset path. A TLS value of 0.0 causes per-distance recomputation of `sqrt(q·q)`.

3. **Concurrent search TLS conflicts** — Milvus QueryNode searches multi-threaded. TLS scratch is per-thread (safe), but a `RefCell::borrow_mut` panic would silently fall back. Scan Milvus log for panics or errors during search phase.

4. **ef_search mismatch** — Confirm Milvus passes ef_search=128 to RS (same as native). An inflated ef_search would degrade QPS.

**Fix action:** One targeted code change per iteration, rebuild, re-run Phase 1.

**Exit condition:** QPS ≥ 848, or search side confirmed at ceiling and optimize gap is dominant.

---

## Phase 3: Build/Optimize Diagnosis

**Trigger:** optimize duration >> native (> 500s) after search is at target QPS.

**Root cause (already known):** Serial HNSW build on large post-compaction segments. Native C++ uses multi-threaded build; RS FFI defaults to serial.

**Fix options (by preference):**

1. **Parallel build + batch cap (recommended)** — Enable `FFI_ENABLE_PARALLEL_HNSW_ADD_ENV`, set batch cap via `FFI_PARALLEL_HNSW_ADD_BATCH_CAP`. Local experiments show cap=64 gives recall@10=0.9922. This is env-var-only, no recompile to tune.

2. **Size-gated parallel build** — Only use parallel for n > 50K (large segments). Small segments stay serial to preserve graph quality.

3. **Milvus compaction threshold adjustment** — Raise `dataCoord.segment.maxSize` to reduce large segment frequency. Milvus config change, no RS code change.

**Tuning procedure for option 1:**
- Start with cap=64, check recall ≥ 0.95 and optimize < 500s
- If recall < 0.95, lower cap to 32 and re-run
- If optimize still > 500s, raise cap to 128 and re-run

**Exit condition:** optimize ≤ 500s, QPS ≥ 848, recall ≥ 0.95.

---

## Phase 4: Final Verification & Archive

**Trigger:** QPS ≥ 848, recall ≥ 0.95, valid search window.

**Steps:**
1. Run a second RS benchmark under identical conditions (reproducibility check)
2. Confirm two runs within 10% QPS of each other
3. Record final comparison table

**Final comparison table format:**

| Lane | Insert (s) | Optimize (s) | Load (s) | QPS | Recall | Config | Status |
|------|---:|---:|---:|---:|---:|---|---|
| native 1M | 179.1 | 339.6 | 518.7 | 848.4 | 0.9558 | baseline | valid |
| rs (this run) | ? | ? | ? | ≥848 | ≥0.95 | bitset-fast + cap=? | valid |

**Archive artifacts:**
- `benchmark_results/milvus_cohere1m_hnsw_rs_vs_native_YYYYMMDD.json` — raw numbers
- Update `docs/milvus-vectordbbench-native-vs-rs-status.md` — add new valid row
- Git commit recording: code SHA, env vars used, Milvus log path

**If unstable (>10% QPS variance between two runs):** Do not archive as valid. Record as `unstable`, open separate issue, do not proceed to plan.

---

## Remote Paths Reference

| Resource | Path on hannsdb-x86 |
|----------|-------------------|
| knowhere-rs checkout | `/data/work/milvus-rs-integ/knowhere-rs` |
| Rust target dir | `/data/work/milvus-rs-integ/knowhere-rs-target` |
| RS library artifact | `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so` |
| Milvus startup script | `/data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh` |
| Milvus log | `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log` |
| RS runner | `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py` |
| Native runner | `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py` |
| Result JSONs | `/data/work/VectorDBBench/vectordb_bench/results/Milvus/` |
| VectorDBBench logs | `/data/work/VectorDBBench/logs/` |

---

## Iteration Log Template

Each iteration appends one row:

```
Date     | Phase | Change | insert(s) | optimize(s) | load(s) | QPS   | Recall | Valid?
---------|-------|--------|-----------|-------------|---------|-------|--------|-------
20260403 | P1    | bitset-fast-path (4f766a7) | ? | ? | ? | ? | ? | ?
```
