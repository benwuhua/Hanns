# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## Project Goal

**knowhere-rs is a production-grade Rust replacement for Milvus KnowHere (C++), with absolute performance advantage on core CPU paths.**

Not just parity — leadership. The target is measurably faster than native C++ at equal recall, on real-world authority hardware (x86, not just Apple Silicon).

Current verdicts (2026-03-17):
- **HNSW**: ✅ leading (1.789x faster than native at near-equal recall)
- **DiskANN/AISAQ**: ⚠️ constrained — functional, capability closure in progress
- **IVF-PQ**: ❌ no-go — recall < 0.8 gate
- **Sparse/Binary/ScaNN**: ✅ parity or better

---

## Index Priority Order

When allocating work, follow this order:

```
HNSW > DiskANN/AISAQ > PQ > IVF > Sparse/Other
```

Performance authority: **x86 > ARM**. Mac Apple Silicon is for fast iteration; x86 remote is the authority machine for final numbers.

---

## Session Start Protocol

At the start of each session (when user says "继续" or similar):

1. Read `~/.claude/projects/.../memory/project_diskann_progress.md` — current task status
2. Read `~/.claude/projects/.../memory/code_aisaq_architecture.md` — code structure (saves re-reading 3300 lines)
3. Check `TASK_QUEUE.md` for next task
4. If working with Codex: invoke `claude-codex-collab` skill

**Next task**: AISAQ-CAP-001 (exact rerank stage — post-beam raw float re-sort)

---

## Environment

### Local
- Repo: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- tmux session: `knowhere-rs` — pane 0.0 = Codex, pane 0.1 = Claude

### Remote x86 (Authority Machine)
- SSH: `knowhere-x86-hk-proxy` (configured in `~/.ssh/config`)
- Proxy script: `/Users/ryan/Code/knowhere/scripts/remote/socks5_proxy.py`
- Source: `/data/work/knowhere-rs-src`
- Build cache: `/data/work/knowhere-rs-target`
- Cargo: `~/.cargo/bin/cargo` (1.94.0)
- Run benchmark: `ssh knowhere-x86-hk-proxy "cd /data/work/knowhere-rs-src && CARGO_TARGET_DIR=/data/work/knowhere-rs-target ~/.cargo/bin/cargo run --example benchmark --release 2>&1"`

### Reference Codebases
- Native knowhere C++: `/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc`
- Rust DiskANN reference: `/Users/ryan/Code/DiskANN/` (Microsoft Rust port)

---

## Collaboration: Claude + Codex

**Claude = architect/reviewer/planner. Codex = implementer.**
Claude does NOT write implementation code (except trivial <5-line edits).

### Sending tasks to Codex (tmux pane 0.0)

```bash
cat > /tmp/codex_task.txt << 'EOF'
SUBAGENT TASK — skip all skill loading, brainstorming, and design reviews. Proceed directly to implementation.

**目标**: ...
...
**完成后**: 写入 /tmp/codex_status.txt，格式：DONE: <结果>
EOF

tmux send-keys -t knowhere-rs:0.0 "$(cat /tmp/codex_task.txt)" Tab
tmux send-keys -t knowhere-rs:0.0 "" Enter
sleep 3 && tmux capture-pane -t knowhere-rs:0.0 -p | tail -5  # 确认显示 "• Working"
```

**关键**: Tab 排队 + Enter 执行。只发 Enter = 换行不提交。

### Watcher (background)

```bash
rm -f /tmp/codex_status.txt
until grep -qE "^DONE:|^ERROR:" /tmp/codex_status.txt 2>/dev/null; do sleep 30; done
cat /tmp/codex_status.txt
```

Run with `run_in_background=true`.

### After Codex finishes
1. Read modified files (don't trust description)
2. `cargo build 2>&1 | grep "^error"`
3. `cargo test 2>&1 | tail -5`
4. For perf changes: run benchmark and compare numbers
5. Only Claude commits (Codex sandbox blocks git)

---

## Build Commands

```bash
cargo build                                    # dev build
cargo build --release                          # release (LTO + codegen-units=1)
cargo test                                     # all tests
cargo test test_name -- --nocapture            # specific test
cargo run --example benchmark --release        # local benchmark (Mac, reference only)
```

---

## Key Source Files

| File | Status | Purpose |
|------|--------|---------|
| `src/faiss/hnsw.rs` | ✅ primary | HNSW — production implementation, leading verdict |
| `src/faiss/hnsw_safe.rs` | ✅ active | HNSW native (high-perf, no-lock design) |
| `src/faiss/hnsw_pq.rs` | ✅ active | HNSW-PQ |
| `src/faiss/hnsw_prq.rs` | ✅ active | HNSW-PRQ |
| `src/faiss/hnsw_quantized.rs` | ✅ active | HNSW-SQ |
| `src/faiss/diskann_aisaq.rs` | ✅ primary | PQFlashIndex (~3300 lines), capability closure in progress |
| `src/faiss/diskann.rs` | ✅ active | simplified Vamana graph |
| `src/faiss/ivfpq.rs` | ⚠️ no-go | recall < 0.8, needs fix |
| `src/faiss/ivf.rs` | ⚠️ scaffold | coarse-assignment scaffold only |
| `src/faiss/pq.rs` | ✅ active | PQ encoder, parallel k-means |
| `src/faiss/sparse*.rs` | ✅ active | sparse indexes (parity) |
| `src/faiss/scann.rs` | ✅ active | ScaNN |
| `examples/benchmark.rs` | ✅ primary | main benchmark, always run with --release |

---

## Benchmark Selection

### Dataset
All benchmarks use **synthetic random float32 vectors, dim=128, L2 metric** (seeded RNG). This is NOT SIFT — it's uniform random, sufficient for QPS/build timing but not recall on real distributions. All recall numbers are vs brute-force on the same synthetic data.

### Scale Tiers

| Tier | Scale | When to use | Where to run |
|------|-------|-------------|--------------|
| Dev | 10K | Fast iteration, recall checking | Local Mac |
| Medium | 100K | Config sweep, perf regression | Local Mac |
| **Authority** | **1M** | **Final numbers, PRs, verdicts** | **Remote x86** |

### Which functions to run

- **Dev loop (local)**: `benchmark_diskann()`, `benchmark_pqflash()` — 10K vectors, ~seconds
- **Medium (local)**: `benchmark_diskann_100k()`, `benchmark_pqflash_100k()` — 100K vectors, ~1-2min
- **Authority (x86)**: `benchmark_diskann_1m()`, `benchmark_pqflash_1m()` — 1M vectors, 2-4min

### Running on x86

```bash
# Full benchmark (takes 5-10 min)
ssh knowhere-x86-hk-proxy "cd /data/work/knowhere-rs-src && CARGO_TARGET_DIR=/data/work/knowhere-rs-target ~/.cargo/bin/cargo run --example benchmark --release 2>&1"

# Quick sanity check (10K only, fast)
# Currently not separated — full benchmark.rs runs all tiers in sequence
```

**Important**: After every non-trivial change, run at minimum the 10K dev benchmark locally before sending to x86.

---

## Benchmark Baselines (authority: x86, 2026-03-25, target-cpu=native)

Note: `.cargo/config.toml` sets `target-cpu=native` for x86_64+aarch64. All benchmarks reflect this.
Previous baselines (2026-03-18) shown in parentheses where significantly changed.

| Index | Scale | Build | QPS (x86) | Notes |
|-------|-------|-------|-----------|-------|
| HNSW (10K) | 10K | — | **27,505** | benchmark fn, ef=50 |
| HNSW (1M, M=16) | 1M | **375s** (parallel, 8t) | **6,374** | ef=50, recall=0.125; ef=200: QPS=1945/recall=0.309; was 938s serial |
| DiskANN (1M) | 1M | 1595s | **4,556** | L=128, R=48, recall@10=0.303 (random data) |
| PQFlash NoPQ SIFT-1M | 1M | **355s** | **6,268** | recall@10=0.9943 (SIFT-1M, 2026-03-26, lazy overflow refine) |
| PQFlash NoPQ | 10K | 4.26s | **10,518** | |
| PQFlash PQ32 | 10K | 13.47s | **5,214** | |
| DiskANN 100K | 100K | 117.8s | **8,881** | L=128, R=48 |
| PQFlash NoPQ | 100K | 0.1s | **698,350** | |
| PQFlash PQ32 | 100K | 92.9s | **142,476** | |
| PQFlash NoPQ | 1M | 3.7s | **31,842** | (was 25,444) |
| PQFlash PQ32 | 1M | 210.3s | **101,780** | (was 105,515) |
| PQFlash NoPQ+SQ8 | 1M | 4.3s | **30,472** | SQ8 prefilter path (δ=-4.3% vs NoPQ) |
| IVF-SQ8 nprobe=32 | 1M | 122s | **10,878** (batch) | recall@10=0.236 |
| IVF-SQ8 nprobe=512 | 1M | — | **508** | recall@10=0.907 (0.95 gate not reachable on random 1M) |

---

## Issue Tracker

See `docs/ISSUES.md` — 25/25 issues resolved (2026-03-25).

All P0 and P1 issues closed. Phase 6 open items:
- GAP-MILVUS-001 [P0]: needs external Milvus test environment

- HNSW 1M parallel build: 938s single-thread → add num_threads for ~120s target

---

## Docs Structure

| Path | Purpose |
|------|---------|
| `docs/superpowers/specs/2026-03-18-diskann-aisaq-gap-analysis.md` | Full gap analysis vs native + Rust DiskANN ref |
| `docs/FFI_CAPABILITY_MATRIX.md` | FFI capability per index type |
| `docs/PARITY_AUDIT.md` | Comprehensive parity audit (historical) |
| `docs/diskann_capability_closure_plan.md` | DiskANN closure strategy |
| `docs/AISAQ_DESIGN.md` | AISAQ architecture design |
| `benchmark_results/` | Authority verdict artifacts (HNSW=leading, IVF-PQ=no-go, DiskANN=constrained) |
| `TASK_QUEUE.md` | Current task panel |
| `GAP_ANALYSIS.md` | Gap analysis (historical, superseded by specs/2026-03-18) |
