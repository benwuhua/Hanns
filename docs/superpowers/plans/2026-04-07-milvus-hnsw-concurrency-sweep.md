# Milvus HNSW Concurrency Sweep Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map RS vs Native HNSW QPS scaling curve on full Cohere-1M at concurrency=1,2,5,10,20 on hannsdb-x86, to determine where RS exceeds native and what the peak QPS ceiling is.

**Architecture:** Three sequential tasks — (1) sync current RS build to x86, (2) run VectorDBBench concurrency sweep for both RS and native, (3) document results. All work is on hannsdb-x86 via SSH. No code changes.

**Tech Stack:** VectorDBBench, Milvus (RS + native binary), hannsdb-x86 SSH (`hannsdb-x86-hk-proxy`), Cohere-1M dataset already on x86.

---

## Context

### Current State (2026-04-07)

- **RS code**: commit `dc44b8f` (main branch, HNSW_TIMING cleanup done, no diagnostic code)
- **Milvus RS binary**: `/data/work/milvus-rs-integ/milvus-src/bin/milvus` (built with RS shim)
- **Milvus native binary**: `/data/work/milvus-rs-integ/milvus-src/bin/milvus-native` (or needs separate build — check)
- **VectorDBBench**: installed at `/data/work/vectordbbench/` or similar
- **Dataset**: Cohere-1M already loaded at `/data/work/vectordbbench/` datasets

### What We Know

| Metric | RS (R5, c=1) | Native (c=1) |
|--------|-------------|--------------|
| QPS | 349.3 | 350.4 |
| Recall | 0.957 | 0.960 |

- RS scales linearly at c=2 (confirmed with small collection, 1.87×)
- No pathological serialization found in Milvus RS path
- Native uses knowhere thread pool (16 threads) for nq; RS uses single-threaded FFI calls
- Hypothesis: RS may beat native at high concurrency (no shared thread pool contention)

### What We Need

QPS at c = 1, 2, 5, 10, 20 for both RS and native. This closes the "does RS scale?" question definitively.

---

## Task 1: Sync RS Code to x86 and Verify Build

**Files:**
- SSH target: `hannsdb-x86-hk-proxy`
- RS source: `/data/work/milvus-rs-integ/knowhere-rs` (sync from local)
- Build target: `/data/work/milvus-rs-integ/knowhere-rs-target`

- [ ] **Step 1: Sync local RS code to x86**

```bash
ssh hannsdb-x86-hk-proxy "cd /data/work/milvus-rs-integ/knowhere-rs && git fetch origin && git log --oneline -3"
```

Expected: should show commit `dc44b8f` at HEAD. If not, pull:

```bash
ssh hannsdb-x86-hk-proxy "cd /data/work/milvus-rs-integ/knowhere-rs && git pull origin main"
```

- [ ] **Step 2: Rebuild RS shim if needed**

If `dc44b8f` is not HEAD, rebuild:

```bash
ssh hannsdb-x86-hk-proxy "cd /data/work/milvus-rs-integ/knowhere-rs && CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target ~/.cargo/bin/cargo build --release 2>&1 | tail -5"
```

Expected: `Finished release [optimized] target(s) in Xs` or `Finished (0 new) ...` (already up to date).

- [ ] **Step 3: Verify Milvus RS binary exists and links to correct .so**

```bash
ssh hannsdb-x86-hk-proxy "ls -la /data/work/milvus-rs-integ/milvus-src/bin/milvus && ldd /data/work/milvus-rs-integ/milvus-src/bin/milvus | grep knowhere"
```

Expected: binary exists, links to `libknowhere_rs.so` in the RS target dir.

- [ ] **Step 4: Write DONE to status file**

```bash
ssh hannsdb-x86-hk-proxy "echo 'DONE: RS dc44b8f synced and build verified' > /tmp/task1_status.txt"
cat /tmp/task1_status.txt  # verify from local
```

---

## Task 2: RS Concurrency Sweep (c=1,2,5,10,20)

**Files:**
- VectorDBBench working dir: determine via `ls /data/work/` on x86
- Results output: `/tmp/rs_concurrency_sweep.txt`

- [ ] **Step 1: Locate VectorDBBench and confirm Cohere-1M dataset**

```bash
ssh hannsdb-x86-hk-proxy "ls /data/work/ && ls /data/work/vectordbbench/ 2>/dev/null || ls /data/work/VectorDBBench/ 2>/dev/null"
```

Note the exact path to VectorDBBench and confirm `cohere-1m` dataset is present.

- [ ] **Step 2: Start Milvus RS**

```bash
ssh hannsdb-x86-hk-proxy "pkill -f 'bin/milvus' 2>/dev/null; sleep 2"
ssh hannsdb-x86-hk-proxy "nohup /data/work/milvus-rs-integ/milvus-src/bin/milvus run standalone > /tmp/milvus_rs.log 2>&1 &"
sleep 10  # wait for Milvus to start
ssh hannsdb-x86-hk-proxy "ps aux | grep milvus | grep -v grep | head -2"
```

Expected: Milvus process running.

- [ ] **Step 3: Run RS at c=1**

```bash
# Adjust path to VectorDBBench as found in Step 1
ssh hannsdb-x86-hk-proxy "cd /data/work/vectordbbench && python -m vectordb_bench --concurrency 1 --db Milvus --db-config '{...cohere-1m config...}' 2>&1 | tail -5"
```

**NOTE**: The exact VectorDBBench command syntax depends on the installed version. Check `python -m vectordb_bench --help` on x86 first. The key parameters are:
- `--concurrency 1` (or equivalent flag)
- Milvus connection: `host=localhost port=19530`
- Dataset: Cohere-1M (cohere-wiki-embedding-1m)
- Case type: Performance (search-only after initial load, or fresh load)

Record QPS and recall from output. Write to `/tmp/rs_concurrency_sweep.txt`:
```
RS c=1 QPS=<value> recall=<value>
```

- [ ] **Step 4: Run RS at c=2,5,10,20**

Repeat Step 3 for each concurrency level. Append each result:
```
RS c=2 QPS=<value> recall=<value>
RS c=5 QPS=<value> recall=<value>
RS c=10 QPS=<value> recall=<value>
RS c=20 QPS=<value> recall=<value>
```

**Important**: Between each run, keep Milvus running (collection already loaded). Just change `--concurrency` parameter. If collection reload is needed, factor in ~5 min for Cohere-1M load.

- [ ] **Step 5: Write task status**

```bash
ssh hannsdb-x86-hk-proxy "echo \"DONE: RS sweep complete, see /tmp/rs_concurrency_sweep.txt\" >> /tmp/task2_status.txt"
```

---

## Task 3: Native Concurrency Sweep (c=1,2,5,10,20)

**Files:**
- Native Milvus binary: `/data/work/milvus-rs-integ/milvus-src/bin/milvus-native` (or check actual location)
- Results output: `/tmp/native_concurrency_sweep.txt`

- [ ] **Step 1: Locate native Milvus binary**

```bash
ssh hannsdb-x86-hk-proxy "ls /data/work/milvus-rs-integ/milvus-src/bin/"
```

If `milvus-native` exists, use it. If only one `milvus` binary, check how to switch:
- The RS vs native switch might be via env var `KNOWHERE_USE_RS=0/1`
- Or via LD_LIBRARY_PATH pointing to native vs RS .so
- Or there's a separate native-milvus binary

Determine the exact method and document it.

- [ ] **Step 2: Restart Milvus with native backend**

```bash
ssh hannsdb-x86-hk-proxy "pkill -f 'bin/milvus' 2>/dev/null; sleep 3"
# Start with native (use correct binary/env based on Step 1 finding)
ssh hannsdb-x86-hk-proxy "nohup /data/work/milvus-rs-integ/milvus-src/bin/milvus-native run standalone > /tmp/milvus_native.log 2>&1 &"
sleep 10
ssh hannsdb-x86-hk-proxy "ps aux | grep milvus | grep -v grep | head -2"
```

- [ ] **Step 3: Run native at c=1,2,5,10,20**

Same procedure as Task 2, Steps 3-4, but with native Milvus. Collect into `/tmp/native_concurrency_sweep.txt`:
```
Native c=1 QPS=<value> recall=<value>
Native c=2 QPS=<value> recall=<value>
Native c=5 QPS=<value> recall=<value>
Native c=10 QPS=<value> recall=<value>
Native c=20 QPS=<value> recall=<value>
```

- [ ] **Step 4: Write task status**

```bash
# Combine both result files
cat /tmp/rs_concurrency_sweep.txt /tmp/native_concurrency_sweep.txt > /tmp/full_sweep_results.txt
echo "DONE: both RS and native sweep complete" >> /tmp/task3_status.txt
cat /tmp/full_sweep_results.txt
```

Write results to `/tmp/codex_status.txt` in format:
```
DONE:
RS: c1=<qps> c2=<qps> c5=<qps> c10=<qps> c20=<qps>
Native: c1=<qps> c2=<qps> c5=<qps> c10=<qps> c20=<qps>
```

---

## Task 4: Document Results and Update Memory

**Files:**
- Modify: memory file `project_milvus_hnsw_optimization.md` (via Claude, not Codex)

This task is for Claude to do after receiving Task 3 results:

- [ ] **Step 1: Compute RS/Native ratio at each concurrency**

From Task 3 output, compute:
```
c=1:  RS/Native = <ratio>
c=2:  RS/Native = <ratio>
c=5:  RS/Native = <ratio>
c=10: RS/Native = <ratio>
c=20: RS/Native = <ratio>
```

Identify the crossover point (if any) where RS exceeds native.

- [ ] **Step 2: Update project memory with authority numbers**

Add a new section to `~/.claude/projects/.../memory/project_milvus_hnsw_optimization.md`:

```markdown
## Concurrency Sweep Authority Numbers (2026-04-07, Cohere-1M, hannsdb-x86)

| Concurrency | RS QPS | Native QPS | RS/Native |
|-------------|--------|-----------|-----------|
| c=1 | <qps> | <qps> | <ratio> |
| c=2 | <qps> | <qps> | <ratio> |
| c=5 | <qps> | <qps> | <ratio> |
| c=10 | <qps> | <qps> | <ratio> |
| c=20 | <qps> | <qps> | <ratio> |

**Crossover point**: [concurrency at which RS ≥ native, or "RS does not exceed native in this range"]
```

- [ ] **Step 3: Commit result summary to repo**

```bash
git add docs/superpowers/plans/2026-04-07-milvus-hnsw-concurrency-sweep.md
git commit -m "bench(milvus): HNSW concurrency sweep results RS vs native

RS: c1=<> c2=<> c5=<> c10=<> c20=<>
Native: c1=<> c2=<> c5=<> c10=<> c20=<>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
