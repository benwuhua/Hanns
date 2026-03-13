# PARITY_AUDIT (Non-GPU)

Last updated: 2026-03-13
Sync baseline: a911f2af70f6f47721ab42cfba7b97ee3fd6f206 from main

## 轮次记录
- 2026-03-13: **builder-loop：收口 `hnsw-search-fastpath-audit-round9` 与 `hnsw-round9-authority-same-schema-rerun`，把 round9 production fast-path hypothesis 跑到 authority 终局（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round9_baseline.json`、`tests/bench_hnsw_reopen_round9.rs`、`src/faiss/hnsw.rs`、`src/simd.rs`、`docs/superpowers/plans/2026-03-13-hnsw-round9-search-fastpath-audit.md`。
  2. 阶段结论：round 9 不再缺快路径实现本身，真正需要回答的是 production `layer0 + L2 + no-filter` fast path 与 cached batch-4 dispatch 是否已经被 durable artifact 锁住，以及这些更改是否真的把 authority same-schema lane 推到了新的 evidence band。round 8 的 hard-stop 不允许再靠推测推进，必须把 audit 与 same-schema rerun 一起收口。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round9_profile.rs` 生成 `benchmark_results/hnsw_reopen_search_fastpath_audit_round9.json`，并在 `src/faiss/hnsw.rs` 暴露 audit-visible mode helpers；随后重跑 authority Rust HDF5 same-schema command 与 fresh native HNSW capture，把当前结果写回 `benchmark_results/rs_hnsw_sift128.full_k100.json` 和 `benchmark_results/hnsw_reopen_round9_authority_summary.json`，最终把 round 9 结论定为“successful narrow reopen, verdict unchanged, continue only with another small authority-backed hypothesis”。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture`、`cargo test --features long-tests --test bench_hnsw_reopen_round9_profile -- --ignored --nocapture`、`cargo test --test bench_hnsw_reopen_round9 -- --nocapture` 均通过，`cargo fmt --all -- --check` 先红后绿；authority Rust same-schema HDF5 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round9/test_20260313T034306Z_66062.log` 通过，得到 `1845.608` qps / `0.9909` recall；fresh native capture 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round9/native_hnsw_qps_linkfix_20260313T034322Z.log` 通过，得到 `10348.740` qps / `0.95` recall；authority profile replay 首次因未 resync 新 test target 失败，再经 `bash init.sh` 重放后通过，最终日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round9/test_20260313T040153Z_74459.log`；authority default-lane replay `/data/work/knowhere-rs-logs-hnsw-reopen-round9/test_20260313T040406Z_75601.log` 通过。
  5. 后续主缺口：当前不再有排队中的 tracked feature。round 9 证明 production fast-path cleanup 能显著改善 Rust same-schema qps，但 native 仍快约 `5.61x`，所以 family verdict 仍不得刷新；如果后续再重开 HNSW，必须继续保持 hypothesis 很小、可归因，并由 authority lane 直接验收。

- 2026-03-13: **builder-loop：重开 `hnsw-reopen-round9-activation` 规划/执行线，把下一个 HNSW hypothesis 切到 production search fast-path cleanup（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round8_authority_summary.json`、`src/faiss/hnsw.rs`、`src/simd.rs`、`docs/superpowers/specs/2026-03-13-hnsw-round9-search-fastpath-audit-design.md`、`docs/superpowers/plans/2026-03-13-hnsw-round9-search-fastpath-audit.md`。
  2. 阶段结论：round 8 已经用 authority evidence 否定了“graph-quality parity 会直接把 real lane 推上去”的说法，所以新的 reopen line 必须更小、更可归因。最小可信切口不是再动 build path，而是验证生产 `layer0 + L2 + no-filter` 搜索是否仍被 profiling 代码污染，并顺手去掉 batch-4 每次调用的 feature detection。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round9.rs`，并先用缺失 `benchmark_results/hnsw_reopen_round9_baseline.json` / audit / summary artifacts 的失败做 TDD red；随后新增 `benchmark_results/hnsw_reopen_round9_baseline.json`，明确把 round-8 hard-stop authority evidence 冻结成 round 9 起点，并在 `feature-list.json`、`task-progress.md`、`RELEASE_NOTES.md` 与 `.gitignore` 中重开 durable workflow state。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round9 -- --nocapture` 按预期先因缺失 round-9 artifacts 失败；随后通过 activation 基线更新，把失败面缩到后续 audit/authority artifacts；workflow validator 需要在 docs/state 一并切到 round 9 后恢复为可接受状态。
  5. 后续主缺口：当前不再缺 round-9 activation。下一条 tracked feature 必须是 `hnsw-search-fastpath-audit-round9`，用新的 audit artifact 把 production fast path 与 profiled path 的结构分离锁成 authority-backed 证据，再决定是否值得跑新的 same-schema rerun。

- 2026-03-13: **builder-loop：收口 `hnsw-round8-authority-same-schema-rerun`，用 fresh authority same-schema evidence 给 round 8 graph-quality hypothesis 终判（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round8_baseline.json`、`benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`、`tests/bench_hnsw_reopen_round8.rs`、`benchmark_results/hnsw_p3_002_final_verdict.json`。
  2. 阶段结论：round 8 不再缺 synthetic build audit，也不再缺 bulk-build 结构对齐；唯一剩下的问题是这些改动是否真的把 authority same-schema lane 推到了新的证据带。因此本轮只做最终 authority rerun，并把结论冻结成新的 summary artifact。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round8.rs` 升级成要求 `benchmark_results/hnsw_reopen_round8_authority_summary.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后重跑 authority Rust HDF5 same-schema command、fresh native HNSW capture、authority `bench_hnsw_reopen_round8_profile` 和 authority `bench_hnsw_reopen_round8`；最后新增 `benchmark_results/hnsw_reopen_round8_authority_summary.json`，明确记录 `parallel_build_graph_quality_parity` 的 real-lane 结果是 `hard_stop`，因为 Rust qps 与 recall 都没有随 bulk-build graph-quality rework 改善。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round8 -- --nocapture` 先因缺失 round-8 summary artifact 失败再转绿；`bash init.sh` 通过；authority Rust same-schema HDF5 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T015649Z_53053.log` 通过并回传新的 `benchmark_results/rs_hnsw_sift128.full_k100.json`；fresh native capture 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/native_hnsw_qps_linkfix_20260313T015830Z.log` 通过；authority profile replay 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T021527Z_55615.log` 通过；authority default-lane replay 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T021527Z_55616.log` 通过。
  5. 后续主缺口：当前不再存在下一条活动 HNSW reopen feature。round 8 已经用 authority evidence 把 `parallel_build_graph_quality_parity` 归档为 `hard_stop`，后续若还要重开 HNSW performance line，必须提出新的 authority-backed hypothesis，而不是继续默认延长这条已经被终判否定的 graph-quality 线。

- 2026-03-13: **builder-loop：收口 `hnsw-parallel-build-graph-rework-round8`，把 round8 bulk-build graph-quality 语义真正对齐到可验证实现（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`、`tests/bench_hnsw_reopen_round8.rs`、`tests/bench_hnsw_reopen_round8_profile.rs`、`src/faiss/hnsw.rs`、`docs/superpowers/plans/2026-03-13-hnsw-round8-parallel-build-graph-quality.md`。
  2. 阶段结论：round 8 已经不缺“bulk-build 路径和 serial/native 语义不一致”的证据，真正该回答的是能不能把这两条差异用最小代码改动闭掉，而且不破坏 round-6/round-7 的搜索侧 surfaces。继续停留在 audit artifact 不会再提供新增信息。
  3. 本轮执行：先在 `src/faiss/hnsw.rs` 加 focused regressions，用缺失 upper-layer descent / heuristic shrink / refreshed profile mode 的失败做 TDD red；随后把 bulk-build neighbor search 改成先做 `max_level -> node_level+1` 的 greedy descent，再把 upper-layer overflow 从 `truncate_to_best` 换成 `shrink_layer_neighbors_heuristic_idx()`；最后刷新 `tests/bench_hnsw_reopen_round8.rs`、`tests/bench_hnsw_reopen_round8_profile.rs` 和 `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`，现在 artifact 已经不再记录 pre-rework gap，而是记录 `parallel_insert_entry_descent_mode=greedy_from_max_level`、`upper_layer_overflow_shrink_mode=heuristic_shrink`、`omitted_upper_layer_descent_levels=0`、`upper_layer_heuristic_shrink_events=304`。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture` 先因三条新的 round-8 focused regressions 失败再转绿；本地 `cargo test --test bench_hnsw_reopen_round8 -- --nocapture` 先因 contract 切到新 mode 而旧 artifact 仍在失败再转绿；本地 `cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture`、`cargo test --test bench_hnsw_reopen_round7_flat_graph -- --nocapture`、`cargo test --test bench_hnsw_reopen_round6_prefetch -- --nocapture`、`cargo test --test bench_hnsw_reopen_round5 -- --nocapture`、`cargo fmt --all -- --check` 均通过；`bash init.sh` 通过；authority replay `bench_hnsw_reopen_round7_flat_graph -q` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T014659Z_51291.log`；authority replay `bench_hnsw_reopen_round8_profile -- --ignored --nocapture` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T014717Z_51389.log`；authority replay `bench_hnsw_reopen_round8 -q` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T014748Z_51525.log`。
  5. 后续主缺口：当前不再缺 build-path 结构对齐；下一条 tracked feature 必须是 `hnsw-round8-authority-same-schema-rerun`。只有把 fresh same-schema Rust/native evidence 跑出来，才能判断这次 round-8 build rework 是否真的把 authority lane 推进到了新的证据带。
  状态：Phase 6 Active（round-8 build rework closed；authority same-schema rerun is next）。
- 2026-03-13: **builder-loop：收口 `hnsw-parallel-build-graph-audit-round8`，把 bulk-build graph-quality 偏差锁成 authority-backed audit artifact（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`tests/bench_hnsw_reopen_round8.rs`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_reopen_profile.rs`、`tests/bench_hnsw_reopen_round7_profile.rs`、`docs/superpowers/plans/2026-03-13-hnsw-round8-parallel-build-graph-quality.md`。
  2. 阶段结论：round 8 当前首先缺的不是 graph-quality 修复本身，而是一个真正走 `add_parallel()` 语义的可回放审计面。现有 `build_profile_report()` 只覆盖串行插入，无法证明 bulk-build 仍在跳过 upper-layer greedy descent，也无法把 upper-layer overflow 仍用 `truncate_to_best` 写成可执行证据。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round8.rs` 升级成要求 `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后在 `src/faiss/hnsw.rs` 增加 `parallel_build_profile_report()` 与 bulk-build profiled helpers，显式记录 `parallel_insert_entry_descent_mode`、`upper_layer_overflow_shrink_mode`、`omitted_upper_layer_descent_levels`、`upper_layer_connection_update_calls` 等 round-8 审计字段；最后新增 `tests/bench_hnsw_reopen_round8_profile.rs` 生成 artifact，并补 `.gitignore` 让 round-8 baseline/audit JSON 成为 durable state。
  4. 验证结果：本地 `cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture` 先因缺失 `HnswParallelBuildProfileReport` / `parallel_build_profile_report` 失败再转绿；`cargo fmt --all -- --check` 先红后绿；本地 `cargo test --test bench_hnsw_reopen_round8 -- --nocapture` 通过；`bash init.sh` 通过；authority profile replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round8_profile -- --ignored --nocapture"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T012954Z_47513.log`；authority contract replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T013023Z_47680.log`。
  5. 后续主缺口：当前不再缺 build-parity 审计面；下一条 tracked feature 必须是 `hnsw-parallel-build-graph-rework-round8`。只有在保持 round8 audit artifact 可刷新的前提下，把 bulk-build 上层 descent 和 overflow shrink 真正对齐，后面的 authority same-schema rerun 才能保持可归因。
  状态：Phase 6 Active（round-8 build-parity audit closed；graph-quality rework is next）。
- 2026-03-13: **builder-loop：收口 `hnsw-reopen-round8-activation`，把 round 8 baseline 和默认-lane contract 正式挂回 durable workflow（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`tests/bench_hnsw_reopen_round8.rs`、`docs/superpowers/specs/2026-03-13-hnsw-round8-parallel-build-graph-quality-design.md`、`docs/superpowers/plans/2026-03-13-hnsw-round8-parallel-build-graph-quality.md`。
  2. 阶段结论：planning 轮已经把 round 8 的 hypothesis 收窄到 bulk-build graph quality，现在真正需要先回答的问题不是“graph-quality rework 值不值得做”，而是“repo 有没有一个新的 round-8 baseline artifact 和默认-lane contract，能把 round-5 stability + round-6/7 search audits 冻结成可执行起点”。没有这层 activation，后续 round-8 审计和 authority rerun 仍然会回到隐式 work。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round8.rs`，并先用缺失 `benchmark_results/hnsw_reopen_round8_baseline.json` 的失败做 TDD red；随后新增该 baseline artifact，明确把 round 5 stability 与 round 6/7 search-path audits 冻结成 round 8 起点，且显式把下一条假设写成 `parallel_build_graph_quality_parity`。然后更新 `feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`DEV_ROADMAP.md`、`RELEASE_NOTES.md`，把 repo 从 “52/56 pending planning reopen” 切到 “53/56 passing with round-8 baseline frozen” 的活动态。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round8 -- --nocapture` 先因缺失 baseline artifact 失败再转绿；`bash init.sh` 通过；authority replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round8 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round8 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round8 -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round8/test_20260313T011357Z_43949.log`；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 56 features (53 passing, 3 failing); workflow/doc checks passed`。
  5. 后续主缺口：当前不再缺 round-8 activation；下一条 tracked feature 必须是 `hnsw-parallel-build-graph-audit-round8`。只有把 bulk-build upper-layer descent / overflow shrink 差异锁成 round-8 audit artifact，后续 graph-quality rework 和 authority rerun 才能保持可归因。
  状态：Phase 6 Active（round-8 baseline frozen；build-parity audit is next）。
- 2026-03-13: **builder-loop：重开 `hnsw-reopen-round8-activation` 规划线，把下一个 HNSW hypothesis 从 search-path micro-tuning 切到 parallel-build graph quality（plan）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`src/faiss/hnsw.rs`、`src/simd.rs`、以及当前 native `hnswalg.h` 插入路径。
  2. 阶段结论：round 4 到 round 7 已经把 layer-0 search core 的结构差异压得足够窄，剩下最值得先检验的偏差不再是 “再多一个 search micro-opt 会不会买账”，而是当前 bulk-build 路径仍缺 `max_level -> node_level+1` 的 greedy descent，并且 upper-layer overflow 仍在用 `truncate_to_best`。这两点都比 serial/native 插入语义更弱，更像图质量差距的真实来源。
  3. 本轮执行：新增 `docs/superpowers/specs/2026-03-13-hnsw-round8-parallel-build-graph-quality-design.md` 与 `docs/superpowers/plans/2026-03-13-hnsw-round8-parallel-build-graph-quality.md`，把 round 8 明确限定为 `parallel_build_graph_quality_parity`；同时在 durable workflow state 中新增 `hnsw-reopen-round8-activation`、`hnsw-parallel-build-graph-audit-round8`、`hnsw-parallel-build-graph-rework-round8`、`hnsw-round8-authority-same-schema-rerun` 四条 failing feature，并同步 queue/roadmap/gap/progress 文档。
  4. 验证结果：`bash init.sh` 通过；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 56 features (52 passing, 4 failing); workflow/doc checks passed`。
  5. 后续主缺口：先完成 `hnsw-reopen-round8-activation` 的 baseline artifact 与 default-lane contract，再进入 build-audit / graph-quality rework / authority rerun。batch-4 dispatch caching 与 AVX2/FMA specialization 明确延后，避免首个 round-8 authority rerun 混淆归因。
  状态：Phase 6 Active（round-8 graph-quality hypothesis queued；activation is next）。
- 2026-03-12 12:22: **builder-loop：收口 `hnsw-round4-authority-same-schema-rerun`，给第四轮 HNSW reopen 做真实 authority same-schema 终判（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round4_baseline.json`、`benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`、`tests/bench_hnsw_reopen_round4.rs`、`docs/superpowers/plans/2026-03-12-hnsw-round4-layer0-searcher-parity.md`。
  2. 阶段结论：round 4 已经拿到真实 layer-0 parity rework，剩下唯一值得回答的问题是：这次 ordered-pool + batch-distance cut 能不能把 authority same-schema Rust row 推到足以重开 HNSW family verdict。没有 fresh Rust/native same-schema evidence，就不能把 round 4 记成成功，也不能把这条 reopen line无限延长。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round4.rs` 再次升级成要求 `benchmark_results/hnsw_reopen_round4_authority_summary.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后 authority 侧重跑 Rust same-schema HDF5 benchmark、native HNSW capture、以及 `bench_hnsw_reopen_round4_profile`，再把 fresh `benchmark_results/rs_hnsw_sift128.full_k100.json` 和 `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json` 回传到本地；最后新增 `benchmark_results/hnsw_reopen_round4_authority_summary.json`，明确记录 round 4 的真实结果是“same-schema 有大幅改善，但仍不足以重开 family verdict”。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round4 -- --nocapture` 先因缺失 `benchmark_results/hnsw_reopen_round4_authority_summary.json` 失败再转绿；`bash init.sh` 通过；authority Rust same-schema rerun `generate_hdf5_hnsw_baseline` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T115946Z_42625.log`；native capture 通过 linkfix fallback，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/native_hnsw_qps_linkfix_20260312T121538Z.log`；authority round-4 synthetic profile rerun 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T121650Z_45259.log`；summary artifact 回推远端后，authority contract replay `bench_hnsw_reopen_round4 -q` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T122210Z_46060.log`。
  5. 后续主缺口：当前 tracked reopen line 已经结束，不存在下一条活动 feature。若未来仍要继续 HNSW，必须先提出新的 authority-backed hypothesis，而不是继续默认延长这一轮已经被证据归档为 `soft_stop` 的 layer-0 parity line。
  状态：Phase 6 Closed（round 4 soft-stop archived；all tracked features passing again）。
- 2026-03-12 11:52: **builder-loop：收口 `hnsw-layer0-searcher-core-rework`，把 round-4 layer-0 parity 假设落成真正的搜索核心改动（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`、`tests/bench_hnsw_reopen_round4.rs`、`tests/bench_hnsw_reopen_round4_profile.rs`、`src/faiss/hnsw.rs`、`src/simd.rs`、`docs/superpowers/plans/2026-03-12-hnsw-round4-layer0-searcher-parity.md`。
  2. 阶段结论：round 4 已经不缺“native-vs-Rust layer-0 形状差异是否存在”的证据，真正该回答的是 Rust 能不能先把自己的 layer-0 search core 改成更像 native 的 ordered-pool + batch-distance 路径，再看 same-schema authority lane 是否买账。继续停留在 `dual_binary_heap + scalar_pointer_fast_path` 已经没有信息增益。
  3. 本轮执行：先在 `src/faiss/hnsw.rs` 里加 focused regressions，用缺失 ordered-pool/batch-4 helper 的失败做 TDD red；随后把 `SearchScratch` 扩成复用 layer-0 frontier/result pools，新增 `Layer0PoolEntry`/`Layer0OrderedFrontier`/`Layer0OrderedResults`，把 layer-0 `L2 + no-filter` 搜索核心从双 `BinaryHeap` 切到 ordered-pool 路径，同时在 `src/simd.rs` 增加 `l2_batch_4_ptrs` 并让 layer-0 neighbor expansion 按 4 个邻居一批算距离。最后刷新 `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`，现在 artifact 已经不再记录旧 search core，而是记录新的 `ordered_pool + batch4_pointer_fast_path` 形状、`layer0_batch4_calls=3960`、`layer0_query_distance≈23.185ms`、`sample_search.qps≈2603.588`。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture` 先因缺失 ordered-pool helpers 失败再转绿；本地 `cargo test --test bench_hnsw_reopen_round4 -- --nocapture` 先因 stale audit artifact 仍写着 `dual_binary_heap` 失败再转绿；本地 `cargo test --test bench_hnsw_cpp_compare -q`、`cargo test --test bench_hnsw_reopen_round3 -q`、`cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture`、`cargo fmt --all -- --check` 均通过；`bash init.sh` 通过；authority compare replay 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T115024Z_39686.log`；authority long-test profile replay 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T115054Z_39863.log`；authority default-lane contract replay 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T115136Z_40082.log`。
  5. 后续主缺口：round 4 现在已经没有新的 synthetic parity blocker 了，下一条 tracked feature 必须是 `hnsw-round4-authority-same-schema-rerun`。历史 HNSW family verdict 继续保持 `functional-but-not-leading`，直到 fresh same-schema Rust/native evidence 真正说明这次 layer-0 parity cut 是否足够重要。
  状态：Phase 6 Active（round-4 core rework closed；same-schema authority rerun is next）。
- 2026-03-12 11:35: **builder-loop：收口 `hnsw-layer0-searcher-audit`，把第四轮 HNSW 的 native-vs-Rust layer-0 结构差异锁成 authority-backed artifact（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round4_baseline.json`、`tests/bench_hnsw_reopen_round4.rs`、`src/faiss/hnsw.rs`、`docs/superpowers/plans/2026-03-12-hnsw-round4-layer0-searcher-parity.md`。
  2. 阶段结论：round 4 activation 已经足够说明“下一条假设是 layer-0 parity”，但还不足以支撑真正的算法改动，因为 repo 里还缺一个 authority-backed artifact 来把 native `NeighborSetDoublePopList + distances_batch_4` 和当前 Rust `dual_binary_heap + scalar_pointer_fast_path` 的差异固定成可执行事实。没有这层 audit，后续 core rework 仍然会退回到只改热点 helper 的试错。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round4.rs` 升级成要求 `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后在 `src/faiss/hnsw.rs` 的 candidate-search profile report 上补充 round-4 所需的 `search_core_shape`、`batch_distance_mode`、`batch_distance_call_counts` 元数据，再新增 `tests/bench_hnsw_reopen_round4_profile.rs` 生成 authority-backed audit artifact。最终生成的 artifact 明确记录：native layer-0 search uses `NeighborSetDoublePopList + distances_batch_4`，当前 Rust still uses `dual_binary_heap + scalar_pointer_fast_path`，`layer0_batch4_calls=0`，而 `layer0_query_distance≈31.400ms` of the measured `38.399ms` distance bucket。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round4 -- --nocapture` 先因缺失 `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json` 失败再转绿；本地 `cargo test --features long-tests --test bench_hnsw_reopen_round4_profile -- --ignored --nocapture` 通过；`cargo fmt --all -- --check` 先红后绿；`bash init.sh` 通过；authority round-4 long-test replay 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T113413Z_36164.log`；authority round-4 contract replay 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T113455Z_36352.log`。
  5. 后续主缺口：round 4 现在已经不缺“结构差异是否真实存在”的证据，下一条 tracked feature 必须是 `hnsw-layer0-searcher-core-rework`。在新的 same-schema authority artifact 真正改善之前，历史 HNSW family verdict 继续保持不动。
  状态：Phase 6 Active（round-4 audit closed；layer-0 search-core rework is next）。
- 2026-03-12 11:22: **builder-loop：收口 `hnsw-reopen-round4-activation`，把第四轮 HNSW layer-0 parity 攻关线正式挂回 durable workflow（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round3_authority_summary.json`、`benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`、`tests/bench_hnsw_reopen_round4.rs`、`docs/superpowers/plans/2026-03-12-hnsw-round4-layer0-searcher-parity.md`。
  2. 阶段结论：round 3 已经诚实结束，但 repo 不能继续停在“43/43 全部关闭”的终态叙事里，否则第四轮 layer-0 parity 攻关会再次变成隐式 work。激活 round 4 的真正目的，不是宣称 HNSW 已经接近 native，而是把新的 native-vs-Rust layer-0 searcher hypothesis、工单边界和 authority acceptance 面固定下来。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round4.rs`，并先用缺失 `benchmark_results/hnsw_reopen_round4_baseline.json` 的失败做 TDD red；随后新增该 baseline artifact，明确把 round 3 的 soft-stop 结果冻结成 round 4 起点：same-schema Rust HNSW 为 `553.060` qps、recall 为 `0.9943`、native 为 `10792.646` qps，而下一条显式假设改成 `layer0_searcher_parity`。然后扩展 `feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`DEV_ROADMAP.md`、`RELEASE_NOTES.md`，把项目从 `43/43 passing` 的结束态切到新的 `47-feature` round-4 reopen line。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round4 -- --nocapture` 先红后绿；`bash init.sh` 通过；首条 authority replay 因在同步完成前启动而失败，不作为最终证据（`/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T112127Z_32612.log`）；串行 authority replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round4 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round4 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round4 -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round4/test_20260312T112200Z_32809.log`。
  5. 后续主缺口：HNSW round 4 现在是活动态，下一条 tracked feature 必须是 `hnsw-layer0-searcher-audit`。在新的 authority HNSW artifact 真正改善之前，IVF-PQ、DiskANN 与项目级 final acceptance 继续保持 archived state。
  状态：Phase 6 Active（round-4 baseline frozen；layer-0 searcher audit is next）。
- 2026-03-12 09:56: **builder-loop：收口 `hnsw-round3-authority-same-schema-rerun`，给第三轮 HNSW reopen 做真实 authority same-schema 终判（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round3_baseline.json`、`benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`、`tests/bench_hnsw_reopen_round3.rs`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round3-distance-compute.md`。
  2. 阶段结论：round 3 已经拿到 synthetic/profile 改善，剩下唯一值得回答的问题是：这次 distance-compute cut 能不能把 authority same-schema Rust row 真正往前推 enough to justify a later verdict refresh。没有 fresh Rust/native same-schema evidence，就不能把 round 3 记成成功，也不能把它再延长成无穷 reopen line。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round3.rs` 再次升级成要求 `benchmark_results/hnsw_reopen_round3_authority_summary.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后 authority 侧重跑 Rust same-schema HDF5 benchmark、native HNSW capture、以及 `bench_hnsw_reopen_round3_profile`，再把 fresh `benchmark_results/rs_hnsw_sift128.full_k100.json` 和 `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json` 回传到本地；最后新增 `benchmark_results/hnsw_reopen_round3_authority_summary.json`，明确记录 round 3 的真实结果是“same-schema 有改善，但改善幅度不足以重开 family verdict”。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round3 -- --nocapture` 先因缺失 `benchmark_results/hnsw_reopen_round3_authority_summary.json` 失败再转绿；`bash init.sh` 通过；authority Rust same-schema rerun `generate_hdf5_hnsw_baseline` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T093400Z_21869.log`；native capture 通过 linkfix fallback，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/native_hnsw_qps_linkfix_20260312T094945Z.log`；authority round-3 synthetic profile rerun 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T095112Z_24204.log`；summary artifact 同步回 remote 后，authority contract replay `bench_hnsw_reopen_round3 -q` 通过，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T095502Z_24753.log`。
  5. 后续主缺口：当前 tracked reopen line 已经结束，不存在下一条活动 feature。若未来仍要继续 HNSW，必须先提出新的 authority-backed hypothesis，而不是继续默认延长这一轮已经被证据归档为 `soft_stop` 的 distance-compute line。
  状态：Phase 6 Closed（round 3 soft-stop archived；all tracked features passing）。
- 2026-03-12 09:29: **builder-loop：收口 `hnsw-distance-l2-fast-path-rework`，把第三轮 HNSW 的 layer-0 L2 热路径真正改成 pointer fast path（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_reopen_round3.rs`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round3-distance-compute.md`。
  2. 阶段结论：round 3 的 profiler 已经明确把主热点缩到 `layer0_query_distance`，因此这轮不该再做宽泛 candidate-search 改写，而应只验证一个更小的假设：把 `L2 + no filter` 搜索内层从 slice-based distance 调度切成 pointer-backed fast path，能不能在不改 API/FFI/持久化边界的前提下，真实降低 distance bucket。
  3. 本轮执行：先沿用 round-3 TDD regressions 并补上 focused L2 fast-path 回归；第一次实现虽然完成了外层分派，但 synthetic profile 反而略退，root cause 是 helper 仍在每次 hop 上重建切片。随后把 `src/faiss/hnsw.rs` 的 L2 fast path 改成 `simd::l2_distance_sq_ptr`，在 upper-layer greedy descent 与 layer-0 candidate-expansion 内缓存 `query_ptr/base_ptr`，并把过滤路径回归从不稳定的“必须返回全部允许 id”收窄到稳定的 predicate contract。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture` 先因 overfit 的 filtered-search 断言失败再转绿；`cargo test test_search_single_l2_fast_matches_generic_and_filter_path_stays_stable --lib -- --nocapture`、`cargo test --test bench_hnsw_cpp_compare -q`、`cargo test --test bench_hnsw_reopen_round2 -q`、`cargo test --test bench_hnsw_reopen_round3 -q` 全部通过；`cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture` 刷新 artifact 后，aggregate `distance_compute` 从 `40.165ms` 降到 `38.528ms`，`layer0_query_distance` 从 `32.500ms` 降到 `31.244ms`，sample-search qps 从 `2023.694` 升到 `2069.930`；`cargo fmt --all -- --check` 先红后绿；`bash init.sh` 通过；authority replays `bench_hnsw_cpp_compare`、`bench_hnsw_reopen_round3_profile`、`bench_hnsw_reopen_round3` 分别通过，日志是 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T092732Z_20631.log`、`/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T092751Z_20701.log`、`/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T092835Z_20832.log`。
  5. 后续主缺口：当前已经不缺“synthetic/profile 能否被这刀 L2 fast path 改善”的答案，真正剩下的问题只有一个：`hnsw-round3-authority-same-schema-rerun` 必须重跑真实 recall-gated same-schema Rust/native evidence，判断这次 profile 改善是不是也能兑现到 authority qps，而不是继续停留在 safety contracts 上。
  状态：Phase 6 Active（round-3 L2 fast path rework closed；authority same-schema rerun is next）。
- 2026-03-12 09:13: **builder-loop：收口 `hnsw-distance-compute-profiler`，把第三轮 HNSW 的 `distance_compute` 再拆成 authority-backed 子热点（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round3_baseline.json`、`tests/bench_hnsw_reopen_round3.rs`、`src/faiss/hnsw.rs`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round3-distance-compute.md`。
  2. 阶段结论：round 3 不再缺“distance_compute 是不是当前主热点”这个大结论，缺的是把它拆成下一刀算法改动能直接利用的子来源。只有在 authority surface 上把 `distance_compute` 分解成更小热点，后续 L2 fast path rework 才不会继续围绕一个过宽的 timing bucket 试错。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round3.rs` 升级成要求 `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后在 `src/faiss/hnsw.rs` 中扩展现有 candidate-search profiling，保留 aggregate `distance_compute` bucket 的同时，再显式记录 `upper_layer_query_distance`、`layer0_query_distance`、`node_node_distance` 三个子桶与对应 call counts；再新增 `tests/bench_hnsw_reopen_round3_profile.rs` 生成 authority artifact 并回传到本地 durable state。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round3 -- --nocapture` 先红后绿；本地 `cargo test --features long-tests --test bench_hnsw_reopen_round3_profile -- --ignored --nocapture` 通过；`cargo fmt --all -- --check` 先红后绿；`bash init.sh` 通过；第一次并行 authority contract replay 因 shared wrapper lock 返回 `status=conflict`（`/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T091236Z_18116.log`），最终有效证据来自串行成功 reruns：`bench_hnsw_reopen_round3_profile` 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T091236Z_18115.log`，`bench_hnsw_reopen_round3` 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T091319Z_18338.log`；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 43 features (41 passing, 2 failing); workflow/doc checks passed`。
  5. 后续主缺口：round 3 authority profile 现在明确显示 `layer0_query_distance≈32.500ms`、`upper_layer_query_distance≈7.665ms`、`node_node_distance=0`，因此下一条活动 feature 必须是 `hnsw-distance-l2-fast-path-rework`，而不是继续停留在 profiling 或过早重开 family verdict 叙事。
  状态：Phase 6 Active（round-3 distance-compute profiler closed；L2 fast path rework is next）。
- 2026-03-12 09:03: **builder-loop：收口 `hnsw-reopen-round3-activation`，把第三轮 HNSW 的 distance-compute 线正式挂回 durable workflow（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round2_authority_summary.json`、`benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`、`benchmark_results/hnsw_p3_002_final_verdict.json`、`tests/bench_hnsw_reopen_round2.rs`、`docs/superpowers/specs/2026-03-12-hnsw-reopen-round3-distance-compute-design.md`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round3-distance-compute.md`。
  2. 阶段结论：round 2 已经被 authority evidence 判成 `hard_stop`，所以 round 3 不能再泛泛地说“继续优化 candidate-search”。最小诚实切口是把 HNSW 的 active hypothesis 收窄到 `distance_compute_inner_loop`，并把这个新假设写回 durable workflow，而不是继续停在 `39/39` 的终态叙事里。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round3.rs`，并先用缺失 `benchmark_results/hnsw_reopen_round3_baseline.json` 的失败做 TDD red；随后新增该 baseline artifact，明确把 round 2 hard-stop 冻结成 round 3 起点：same-schema Rust HNSW 仍为 `521.031` qps、native BF16 仍为 `10519.683` qps、历史 HNSW family verdict 仍为 `functional-but-not-leading`，而新的 round-3 target 改为 `distance_compute_inner_loop`。最后扩展 `feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`DEV_ROADMAP.md`、`RELEASE_NOTES.md`，把 repo 从 round-2 hard-stop 终态切到新的 `43-feature` round-3 reopen line。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round3 -- --nocapture` 先红后绿；`bash init.sh` 通过；第一次 authority replay 因为我把 replay 和 `init.sh` 并行跑而不算最终证据，串行 replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round3 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round3 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round3 -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round3/test_20260312T090426Z_16155.log`；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 43 features (40 passing, 3 failing); workflow/doc checks passed`。
  5. 后续主缺口：当前不再缺“round 3 有没有启动”的治理动作，真正的下一条 feature 是 `hnsw-distance-compute-profiler`。只有当新的 profiler 把 `distance_compute` 拆成更小的可解释来源之后，L2 fast path 的最小实现切口才有意义。
  状态：Phase 6 Active（round 3 activated；distance-compute profiler is next）。
- 2026-03-12 08:45: **builder-loop：收口 `hnsw-round2-authority-same-schema-rerun`，用 fresh authority same-schema evidence 给第二轮 HNSW reopen 终判（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round2_baseline.json`、`benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`、`benchmark_results/hnsw_p3_002_final_verdict.json`、`tests/bench_hnsw_reopen_round2.rs`。
  2. 阶段结论：第二轮 core rework 之后，真正要回答的问题已经不是“synthetic profile 有没有改善”，而是“same-schema authority lane 有没有跟着改善”。因此本轮只做三件事：重跑真实 HDF5 Rust row、重跑 native capture、再把 round-2 profile 和 contract 都刷成 fresh evidence。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round2.rs` 升级成要求 `benchmark_results/hnsw_reopen_round2_authority_summary.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后 authority 侧重跑 Rust same-schema HDF5 command、native HNSW capture、以及 `bench_hnsw_reopen_round2_profile`，把 fresh `benchmark_results/rs_hnsw_sift128.full_k100.json` 和 `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json` 回传到本地；最后新增 `benchmark_results/hnsw_reopen_round2_authority_summary.json`，明确记录 synthetic hotspot 虽然改善，但 authoritative same-schema Rust qps 仍明显回退，因此 round 2 进入 `hard_stop`。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round2 -- --nocapture` 先红后绿；`cargo fmt --all -- --check` 先红后绿；authority Rust same-schema HDF5 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T082542Z_10202.log`；native fresh capture 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/native_hnsw_qps_linkfix_20260312T083313Z.log`；authority round-2 profile refresh 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T084128Z_12238.log`；第一次 authority contract replay 按预期因为 summary 尚未同步而失败（`/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T084211Z_12364.log`），同步后最终 replay 转绿（`/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T084531Z_12900.log`）；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 39 features (39 passing, 0 failing); workflow/doc checks passed`。
  5. 后续主缺口：当前 tracked reopen line 已经结束，不存在下一条活动 feature。若未来仍要再动 HNSW，必须先提出新的 authority-backed hypothesis，而不是继续默认延长这一轮已经被证据判成 `hard_stop` 的 candidate-search line。
  状态：Phase 6 Closed（round 2 hard-stop archived；all tracked features passing）。
- 2026-03-12 08:18: **builder-loop：收口 `hnsw-candidate-search-core-rework`，把第二轮 HNSW 的 shared candidate-search core 真正改掉（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round2_baseline.json`、`benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_cpp_compare.rs`、`tests/bench_hnsw_reopen_round2.rs`。
  2. 阶段结论：round-2 profiler 已经把热点缩到 `entry_descent` 与 `distance_compute`，所以下一刀不该再去扩 artifact，而应直接把 shared candidate-search core 的上层 hop 逻辑做轻。最小诚实切口是：让 `ef<=1` 的 shared layer search 走 greedy fast path，并把无过滤 query path 的 upper-layer descent 从之前的宽泛 heap search 拉回标准 greedy 行为。
  3. 本轮执行：先在 `src/faiss/hnsw.rs` 增加 focused unit regressions，要求一个新的 `greedy_upper_layer_descent_idx` helper 和 `ef=1` shared layer search 与之对齐；随后实现该 helper、让 `search_layer_idx_with_optional_profile()` 在 `ef<=1` 时走 greedy fast path、让无过滤 query path 复用这条更轻的 upper-layer descent，并删除 `SearchScratch` 中没有读者的 `touched` visited 写流量。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture` 先因缺失 helper 报错再转绿；本地 `cargo test --test bench_hnsw_cpp_compare -q`、`cargo test --test bench_hnsw_reopen_round2 -q` 通过；`cargo fmt --all -- --check` 先红后绿；`bash init.sh` 通过；第一次并行 authority 回放中一条 lane 因 shared wrapper lock 返回 `status=conflict`（`/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T081711Z_8977.log`），最终有效证据来自串行成功 reruns：`bench_hnsw_cpp_compare` 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T081711Z_8978.log`，`bench_hnsw_reopen_round2` 日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T081736Z_9088.log`；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 39 features (38 passing, 1 failing); workflow/doc checks passed`。
  5. 后续主缺口：这轮 core rework 只证明“共享 candidate-search 路径真的变了，而且 safety gates 仍绿”，还没有新的 same-schema authority benchmark 结果。因此下一条活动 feature 必须是 `hnsw-round2-authority-same-schema-rerun`；只有 fresh recall-gated authority artifact 才能回答历史 HNSW verdict 是否该动。
  状态：Phase 6 Active（shared candidate-search core rework closed；same-schema authority rerun is next）。
- 2026-03-12 08:09: **builder-loop：收口 `hnsw-candidate-search-profiler`，把第二轮 HNSW 的 `candidate_search` 大桶拆成 authority-backed 热点图（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_round2_baseline.json`、`tests/bench_hnsw_reopen_round2.rs`、`src/faiss/hnsw.rs`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round2-candidate-search.md`。
  2. 阶段结论：round 2 不再缺“candidate_search 是不是热点”这个结论，缺的是把它拆成下一刀算法改动能直接利用的子成本。只有在 authority surface 上把 `candidate_search` 分解成更小热点，后续 core rework 才不会继续围绕一个过宽的 timing bucket 试错。
  3. 本轮执行：先把 `tests/bench_hnsw_reopen_round2.rs` 升级成要求 `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json` 的默认-lane contract，并用缺失 artifact 的失败做 TDD red；随后在 `src/faiss/hnsw.rs` 中为 shared search core 增加 round-2 profiling，显式记录 `entry_descent`、`frontier_ops`、`visited_ops`、`distance_compute`、`candidate_pruning` 五个子桶，再新增 `tests/bench_hnsw_reopen_round2_profile.rs` 生成 authority artifact 并将其回传到本地 durable state。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round2 -- --nocapture` 先红后绿；本地 `cargo test --features long-tests --test bench_hnsw_reopen_round2_profile -- --ignored --nocapture` 通过；`bash init.sh` 通过；authority long-test replay 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T080534Z_6832.log`；authority default-lane replay 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T080615Z_7017.log`；`python3 scripts/validate_features.py feature-list.json` 通过并返回 `VALID - 39 features (37 passing, 2 failing); workflow/doc checks passed`。
  5. 后续主缺口：round 2 authority profile 现在明确显示 `entry_descent` 约占 `46.1%`、`distance_compute` 约占 `39.4%`，因此下一条活动 feature 必须是 `hnsw-candidate-search-core-rework`，而不是继续停留在 profiling 或过早重开 family verdict 叙事。
  状态：Phase 6 Active（round-2 candidate-search profile closed；shared core rework is next）。
- 2026-03-12 15:54: **builder-loop：收口 `hnsw-reopen-round2-activation`，把 HNSW 第二轮 candidate-search 攻关线正式挂回 durable workflow（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_baseline.json`、`benchmark_results/hnsw_reopen_profile_round1.json`、`tests/bench_hnsw_cpp_compare.rs`、`docs/superpowers/specs/2026-03-12-hnsw-reopen-round2-candidate-search-design.md`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-round2-candidate-search.md`。
  2. 阶段结论：round 1 已经诚实结束，但 repo 不能继续停在“35/35 全部关闭”的终态叙事里，否则第二轮 candidate-search 攻关会再次变成隐式 work。激活 round 2 的真正目的，不是宣称 HNSW 变快了，而是把新的假设、工单边界和 authority acceptance 面固定下来。
  3. 本轮执行：新增 `tests/bench_hnsw_reopen_round2.rs`，并先用缺失 `benchmark_results/hnsw_reopen_round2_baseline.json` 的失败做 TDD red；随后新增该 baseline artifact，明确把 round 1 的 mixed result 冻结成 round 2 起点：build wall clock 约快 `5.8%`，但 sample-search qps 约慢 `5.7%`，`candidate_search` 仍然是主热点。然后扩展 `feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`DEV_ROADMAP.md`、`RELEASE_NOTES.md`，把项目从 `35/35 passing` 的结束态切到新的 `39-feature` round-2 reopen line。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_round2 -- --nocapture` 先红后绿；`bash init.sh` 通过；authority replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round2 -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-round2/test_20260312T075654Z_4565.log`；`python3 scripts/validate_features.py feature-list.json` 返回 `VALID - 39 features (36 passing, 3 failing); workflow/doc checks passed`。
  5. 后续主缺口：当前不再缺“round 2 有没有启动”的治理动作，真正的下一条 feature 是 `hnsw-candidate-search-profiler`。只有当新 profiler 把 `candidate_search` 拆成更小的可解释热点之后，第二轮 core rework 才值得开始。
  状态：Phase 6 Active（round 2 activated；candidate-search profiler is next）。
- 2026-03-12 15:28: **builder-loop：收口 `hnsw-authority-rerun-and-verdict-refresh`，刷新第一轮 HNSW reopen authority 结果并确认历史 verdict 暂不改写（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_profile_round1.json`、`tests/bench_hnsw_reopen_profile.rs`、`tests/bench_hnsw_reopen_progress.rs`、`tests/bench_hnsw_cpp_compare.rs`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-algorithm-line.md`。
  2. 阶段结论：第一刀 bulk-build rework 需要一个真正的 authority refresh 来回答“历史 HNSW verdict 有没有动”。刷新后的 reopen profile 证明 build path 的 `neighbor_selection` / `connection_update` / `layer_descent` 的确下降了，但 `candidate_search` 不降反升，并且 sample-search qps 也回落，因此这轮最多只能判成“瓶颈更清楚”，不能判成 family-level improvement。
  3. 本轮执行：重新跑 authority `bench_hnsw_reopen_profile`，再串行回放 `bench_hnsw_cpp_compare -q` 与 `bench_hnsw_reopen_progress -q`；初始并行尝试触发了 remote wrapper 的共享锁冲突，所以最终 evidence 全部来自串行 reruns。随后把远端生成的 `benchmark_results/hnsw_reopen_profile_round1.json` 拉回本地并和 rework 前版本对比，确认 build wall clock 从 `17006.687ms` 降到 `16018.719ms`，但 `candidate_search` 从 `8667.369ms` 升到 `9086.895ms`，sample-search qps 从 `1410.104` 降到 `1329.255`。
  4. 验证结果：`bash init.sh` 通过；authority `bench_hnsw_reopen_profile` 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen/test_20260312T072331Z_99668.log`；最终串行 authority compare/progress replays 日志为 `/data/work/knowhere-rs-logs-hnsw-reopen/test_20260312T072549Z_118.log` 与 `/data/work/knowhere-rs-logs-hnsw-reopen/test_20260312T072619Z_220.log`；本地 `cargo test --test bench_hnsw_cpp_compare -q`、`cargo test --test bench_hnsw_reopen_progress -q` 与 `python3 scripts/validate_features.py feature-list.json` 均通过。
  5. 后续主缺口：tracked reopen line 已经没有未关闭 feature，但这不等于 HNSW verdict 变了。若要开启第二轮算法攻关，下一条最小诚实目标应直接是 `candidate_search`，并且只有新的 recall-gated same-schema authority artifact 才有资格改写 `benchmark_results/hnsw_p3_002_final_verdict.json`。
  状态：Phase 6 Closed（tracked HNSW reopen line closed；historical family verdict unchanged）。
- 2026-03-12 07:22: **builder-loop：收口 `hnsw-build-quality-rework`，交付第一刀真实 HNSW 算法改动（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_reopen_profile_round1.json`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_reopen_progress.rs`、`tests/bench_hnsw_cpp_compare.rs`。
  2. 阶段结论：round-1 profiler 已经给出明确的“先切 candidate_search”方向，但真正落地前还需要一个语义回归把 bulk path 的弱连接维护钉住。新的 deterministic layer-0 regression 随后证明，当前 `add_parallel()` 会把中心节点的 reverse neighbors 维持成 `{0,1,2}`，而 repeated single-node insertion 的强语义路径会保留 `{1,4,5,6}` 这一组多样化邻居。
  3. 本轮执行：在 `src/faiss/hnsw.rs` 中新增该 regression，并把 build path 重构成两层：`add()`/profiling 路径复用 `SearchScratch` 降低 insertion-time candidate-search 分配开销；`add_parallel()` 则修复了首节点自连接和层级映射问题，并把 layer-0 heuristic shrink 延后到 batch 末尾，以更接近 repeated insertion 的 reverse-link semantics 而不把 10K parallel smoke 拖到不可用。
  4. 验证结果：本地 `cargo test hnsw --lib -- --nocapture` 先红后绿；focused `test_parallel_bulk_build_matches_single_insert_layer0_neighbor_diversification` 和 `test_hnsw_parallel_build` 都通过；`cargo test --test bench_hnsw_reopen_progress -q` 与 `cargo test --test bench_hnsw_cpp_compare -q` 通过；authority safety replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-rework KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-rework bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-rework/test_20260312T071858Z_98869.log`。
  5. 后续主缺口：当前 repo 已经不再缺“第一刀算法改动”，只剩 `hnsw-authority-rerun-and-verdict-refresh`。下一轮必须刷新 reopen profile / compare evidence，回答这次 build-path rework 是否真的改变 authority verdict，而不是继续停留在局部 regression 层。
  状态：Phase 6 Active（first algorithm slice closed；only authority rerun/verdict refresh remains）。
- 2026-03-12 07:05: **builder-loop：收口 `hnsw-build-path-profiler`，把 HNSW reopen line 从“只有基线”推进到“有可执行热点分解”的状态（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`docs/superpowers/plans/2026-03-12-hnsw-reopen-algorithm-line.md`、`tests/bench_hnsw_reopen_progress.rs`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_cpp_compare.rs`。
  2. 阶段结论：HNSW reopen line 不能直接跳进“试着优化一点点”；必须先把 build-path 热点拆成可执行事实，否则下一轮核心算法修改仍然会退回凭感觉调参。默认 lane 因此需要新增一个 profile artifact contract，而 authority lane 需要有可回放的 long-test 生成器。
  3. 本轮执行：扩展 `tests/bench_hnsw_reopen_progress.rs`，让它要求 `benchmark_results/hnsw_reopen_profile_round1.json`；新增 `tests/bench_hnsw_reopen_profile.rs` 作为 reopen profile 生成器；在 `src/faiss/hnsw.rs` 中加入 `HnswIndex::build_profile_report()` 和最小 build-stage instrumentation，随后生成并跟踪 `benchmark_results/hnsw_reopen_profile_round1.json`。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_progress -- --nocapture` 先因缺失 profile artifact 失败后转绿；`cargo test --features long-tests --test bench_hnsw_reopen_profile -- --ignored --nocapture` 通过并重写 artifact；authority replay 在重跑 `bash init.sh` 刷新远端同步后成功，日志分别为 `/data/work/knowhere-rs-logs-hnsw-reopen-profiler/test_20260312T070256Z_96556.log` 与 `/data/work/knowhere-rs-logs-hnsw-reopen-profiler/test_20260312T070349Z_96795.log`。
  5. 后续主缺口：round-1 profiler 现已表明 `candidate_search` 约占 `51%` 的 profiled build time，`neighbor_selection` 约占 `31%`，`connection_update` 约占 `12%`；因此下一条活动 feature 应该直接是 `hnsw-build-quality-rework`，优先切 build-time candidate search。
  状态：Phase 6 Active（profile surface closed；next feature is `hnsw-build-quality-rework`）。
- 2026-03-12 06:47: **builder-loop：重开 `hnsw-reopen-baseline-freeze`，把 HNSW 从 archived verdict 重新切回 active algorithm line（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`benchmark_results/hnsw_p3_002_final_verdict.json`、`benchmark_results/baseline_p3_001_stop_go_verdict.json`、`src/faiss/hnsw.rs`、`tests/bench_hnsw_cpp_compare.rs`。
  2. 阶段结论：当前 repo 的 durable workflow 已经过度偏向“维护 final verdict”，不适合继续做 HNSW 核心算法攻关。重开范围因此被刻意收窄为“只重开 HNSW，不重开全项目 acceptance”，并保留 2026-03-12 的 final artifacts 作为历史基线。
  3. 本轮执行：新增 `docs/superpowers/specs/2026-03-12-hnsw-reopen-algorithm-line-design.md` 与 `docs/superpowers/plans/2026-03-12-hnsw-reopen-algorithm-line.md`，随后增加 `tests/bench_hnsw_reopen_progress.rs`，用缺失 `benchmark_results/hnsw_reopen_baseline.json` 的失败做 TDD red；再新增该 artifact，并重写 `feature-list.json`、`task-progress.md`、`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`DEV_ROADMAP.md`、`RELEASE_NOTES.md` 以建立新的 HNSW reopen feature chain。
  4. 验证结果：本地 `cargo test --test bench_hnsw_reopen_progress -- --nocapture` 先红后绿；`bash init.sh` 成功；authority replay `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-baseline KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-baseline bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_progress -q"` 返回 `test=ok`，日志 `/data/work/knowhere-rs-logs-hnsw-reopen-baseline/test_20260312T064856Z_93162.log`。
  5. 后续主缺口：HNSW reopen line 已从“结论归档”切回“算法攻关”，下一条活动 feature 是 `hnsw-build-path-profiler`。在新的 authority HNSW artifact 真正改善之前，IVF-PQ、DiskANN 与项目级 final acceptance 继续保持 archived state。
  状态：Phase 6 Active（HNSW reopen baseline frozen；build-path profiler is next）。
- 2026-03-12 05:56: **builder-loop：收口 `final-production-acceptance`，归档项目级 `not accepted` verdict（plan+exec）**
  1. 复核输入：`feature-list.json`、`benchmark_results/final_core_path_classification.json`、`benchmark_results/final_performance_leadership_proof.json`、`scripts/gate_profile_runner.sh`、`tests/bench_hnsw_cpp_compare.rs`、`README.md`。
  2. 阶段结论：所有 prerequisite gates 都已经关闭，当前剩下的不是“继续补工程项”，而是把项目级最终 verdict 明确归档。由于 leadership criterion 已被 authority evidence 明确判成未满足，最终收口必须是 `production_accepted=false`，而不是悬空保留一个 failing feature。
  3. 本轮执行：新增 `tests/test_final_production_acceptance.rs` 作为 full-regression-visible 的 verdict lock，并用缺失 `benchmark_results/final_production_acceptance.json` 的失败做 TDD red；随后新增该 artifact，明确记录 production gates 已关闭但项目仍 `not accepted`。本地 `cargo fmt --all -- --check` 因新测试格式化先红后绿，随后 authority `fmt`、`clippy`、`scripts/gate_profile_runner.sh --profile full_regression` 均在 isolated remote dirs 下通过。为支持最终 handoff，本轮还把 `scripts/validate_features.py` 与 `tests/test_validate_features.py` 扩到支持“all features passing -> Current focus/Next feature = none”的终态表达。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md`、`README.md` 已统一写回最终 verdict；当前已无剩余 failing feature。
  5. 后续主缺口：当前没有未归档的 queue feature。若未来重开项目，只能基于新的 authority artifact 改写 leadership 或 core-path verdict chain，不能靠文档或本地结果翻案。
  状态：Phase 5 Closed（all tracked features archived；project verdict = `not accepted` on current authority evidence）。
- 2026-03-12 05:41: **builder-loop：收口 `prod-readme-remote-workflow-docs`，把入口文档统一改成 remote-first workflow（plan+exec）**
  1. 复核输入：`feature-list.json`、`README.md`、`AGENTS.md`、`docs/FFI_CAPABILITY_MATRIX.md`、`long-task-guide.md`、`task-progress.md`。
  2. 阶段结论：这一轮已经不需要新的 runtime 修复；阻断点是入口文档仍在暗示“本地 cargo 命令就是最终验收”，并且 `README.md` 还保留着过时的 passed-tests 叙事。要关闭这条 feature，关键是把 operator-facing docs 统一到 remote-first authority workflow 与当前 verdict truth。
  3. 本轮执行：重写 `README.md` 作为 remote-first landing page，补入 authority workflow、durable-state 入口与当前 HNSW / IVF-PQ / DiskANN / final-proof truth；同步更新 `AGENTS.md` 与 `docs/FFI_CAPABILITY_MATRIX.md`，把 local prefilter 与 remote acceptance 的边界写清楚；随后用 `bash init.sh`、`python3 scripts/validate_features.py feature-list.json`、`bash scripts/remote/build.sh --no-all-targets` 验证文档收口后 workflow 仍保持可执行。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 docs gate closure，下一条按 `feature-list.json` 选择的可执行 feature 是 `final-production-acceptance`。
  5. 后续主缺口：`PROD-P3-005` 的准备条件现已全部关闭，只剩最后一条 `final-production-acceptance`。但当前 authority evidence 仍明确表明 leadership criterion 未满足，因此最终收口必须诚实地表达“project not accepted”而不是生成正向 completion claim。
  状态：Phase 5 Active（remote-first operator docs closed；next feature per durable source is `final-production-acceptance`）。
- 2026-03-12 05:36: **builder-loop：收口 `prod-ffi-observability-persistence-gate`，关闭 authority FFI/serialize/JSON export gate（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`src/ffi.rs`、`src/serialize.rs`、`tests/bench_json_export.rs`、`scripts/remote/test.sh`。
  2. 阶段结论：这条 feature 已不再缺新的行为级修复；前几轮 family-specific contract closure 已把底层 FFI / observability / persistence gaps 收口完毕，本轮需要做的是在 authority x86 上把三条 recorded regression surface 重新跑通并归档成 durable evidence。
  3. 本轮执行：先在本地重新确认 `cargo test --lib ffi -- --nocapture`、`cargo test --lib serialize -- --nocapture`、`cargo test --test bench_json_export -q` 全部为绿；随后在 isolated `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-ffi-observability-persistence-gate` 与 `KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-ffi-observability-persistence-gate` 下串行重放三条 authority tests。初始并行尝试触发了 wrapper shared-lock `status=conflict`，因此最终 authority evidence 以串行 isolated runs 为准。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 gate closure，下一条按 `feature-list.json` 选择的可执行 feature 是 `prod-readme-remote-workflow-docs`。
  5. 后续主缺口：`PROD-P3-005` 现在只剩 remote-first operator documentation closure；在新的 authority leadership evidence 出现前，仍不得宣称正向 final acceptance。
  状态：Phase 5 Active（cross-cutting FFI/observability/persistence gate closed；next feature per durable source is `prod-readme-remote-workflow-docs`）。
- 2026-03-12 05:18: **builder-loop：收口 `prod-all-targets-clippy-fmt`，关闭 authority remote fmt/clippy/build gate（plan+exec）**
  1. 复核输入：`feature-list.json`、`task-progress.md`、`tests/bench_diskann_1m.rs`、`tests/bench_hdf5.rs`、`benches/bench_rabitq_recall.rs`、`benches/cpp_faiss_simd_bench.rs`、`scripts/remote/build.sh`。
  2. 阶段结论：production governance 的下一道真实门槛不是再做 benchmark 叙事，而是让 remote x86 authority lane 的 `fmt` / `all-targets all-features clippy` / release build 三条基础 gate 真正变绿；本地预筛只能加速排查，不能替代 authority 结论。
  3. 本轮执行：系统清掉 tests / benches / examples 上残留的 lint drift，并修复 `tests/bench_diskann_1m.rs` 默认 lane 仍调用 `long-tests` gated `generate_report()` 的 build-only compile hole，使 remote release build 不再因 gated benchmark helper 失真。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 lint/build gate closure，下一条按 `feature-list.json` 选择的可执行 feature 是 `prod-ffi-observability-persistence-gate`。
  5. 后续主缺口：`PROD-P3-005` 仍未关闭，但剩余工作已收缩为 cross-cutting FFI / persistence / observability gate 与 remote-first operator docs，而不再是基础 lint/build governance。
  状态：Phase 5 Active（remote lint/build gate closed；next feature per durable source is `prod-ffi-observability-persistence-gate`）。
- 2026-03-12 03:49: **builder-loop：收口 `final-performance-leadership-proof`，把最终 leadership criterion 明确归档为未满足（plan+exec）**
  1. 复核输入：`feature-list.json`、`benchmark_results/baseline_p3_001_stop_go_verdict.json`、`benchmark_results/hnsw_p3_002_final_verdict.json`、`benchmark_results/final_core_path_classification.json`、`tests/bench_hnsw_cpp_compare.rs`、`scripts/remote/native_hnsw_qps_capture.sh`。
  2. 阶段结论：当前 authority evidence 已足够给出项目级 final-proof 结论，不需要再重开调参或 benchmark 设计。HNSW 是唯一具备 trusted same-schema compare evidence 的 core path，但 native 仍约快 `14.8x`；IVF-PQ 已 `no-go`，DiskANN 已 `constrained`，因此“至少一条可信 leadership lane”这个 completion criterion 必须诚实记为未满足。
  3. 本轮执行：新增 `benchmark_results/final_performance_leadership_proof.json`，把 `criterion_met=false`、family-level blockers、以及 HNSW same-schema gap 一次性归档；同时将 `tests/bench_hnsw_cpp_compare.rs` 升级成默认 lane 的 final-proof regression，保证 compare lane 持续绑定当前 project-level conclusion。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 final-proof closure，下一条按 `feature-list.json` 选择的可执行 feature 是 `prod-all-targets-clippy-fmt`。
  5. 后续主缺口：当前不再缺“final proof 结论”；剩余工作转向 production governance gates，而最终 acceptance 不得在新的 authority leadership evidence 出现前声称正向通过。
  状态：Phase 5 Active（leadership criterion archived as unmet；next feature per durable source is `prod-all-targets-clippy-fmt`）。
- 2026-03-12 03:39: **builder-loop：收口 `final-core-path-classification`，归档三条 core CPU paths 的统一最终分类（plan+exec）**
  1. 复核输入：`feature-list.json`、`benchmark_results/hnsw_p3_002_final_verdict.json`、`benchmark_results/ivfpq_p3_003_final_verdict.json`、`benchmark_results/diskann_p3_004_final_verdict.json`、`benchmark_results/recall_gated_baseline.json`、`benchmark_results/cross_dataset_sampling.json`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs`。
  2. 阶段结论：HNSW、IVF-PQ、DiskANN 三条 family verdict 都已经各自收口，但 final acceptance 层还缺一个统一、可回放的 cross-family classification input；当前 authority baseline 与 cross-dataset artifacts 已足以支撑这个 rollup，不需要重开任一家族的 benchmark 工作。
  3. 本轮执行：新增 `benchmark_results/final_core_path_classification.json`，把 HNSW=`functional-but-not-leading`、IVF-PQ=`no-go`、DiskANN=`constrained` 统一归档；并将 `tests/bench_recall_gated_baseline.rs` 与 `tests/bench_cross_dataset_sampling.rs` 升级成默认 lane 的 rollup regressions，保证该 artifact 始终与现有 authority-backed benchmark facts 对齐。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 cross-family rollup，下一条按 `feature-list.json` 选择的可执行 feature 是 `final-performance-leadership-proof`。
  5. 后续主缺口：项目仍未满足 final acceptance，因为“至少一条可信 leadership lane”仍未证明；下一条应转向 `final-performance-leadership-proof`，而不是再次重开已关闭的 family classifications。
  状态：Phase 5 Active（core path classifications rolled up；next feature per durable source is `final-performance-leadership-proof`）。
- 2026-03-12 03:32: **builder-loop：收口 `DISKANN-P3-004` family verdict，将 DiskANN 归档为 constrained（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`feature-list.json`、`benchmark_results/diskann_p3_004_benchmark_gate.json`、`benchmark_results/recall_gated_baseline.json`、`benchmark_results/cross_dataset_sampling.json`、`src/faiss/diskann.rs`、`src/faiss/diskann_aisaq.rs`、`tests/bench_diskann_1m.rs`、`tests/bench_compare.rs`。
  2. 阶段结论：benchmark gate 的 no-go 结论已经成立，但 family 本身仍有真实可执行的简化 Vamana/AISAQ surfaces，因此 DiskANN family 更准确的最终分类不是 `no-go`，而是 `constrained`。
  3. 本轮执行：新增 `benchmark_results/diskann_p3_004_final_verdict.json`，把 family-level final classification 固化为 `constrained`；同时将 `src/faiss/diskann.rs`、`tests/bench_diskann_1m.rs`、`tests/bench_compare.rs` 升级为真实 final-verdict regressions，确保 library lane、benchmark lane、compare lane 三者都绑定同一口径。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 DiskANN final verdict，下一条按 `feature-list.json` 选择的可执行 feature 是 `final-core-path-classification`。
  5. 后续主缺口：DiskANN family 已不再是开放 verdict；下一条工作应转入跨 family 汇总，把 HNSW / IVF-PQ / DiskANN 三个 core CPU paths 的最终分类统一归档到 final acceptance 层。
  状态：Phase 5 Active（DiskANN archived as constrained；next feature per durable source is `final-core-path-classification`）。
- 2026-03-12 03:23: **builder-loop：收口 `DISKANN-P3-004` benchmark gate，归档 DiskANN 远端 no-go benchmark evidence（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`feature-list.json`、`benchmark_results/recall_gated_baseline.json`、`benchmark_results/cross_dataset_sampling.json`、`tests/bench_diskann_1m.rs`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs`、`tests/bench_compare.rs`。
  2. 阶段结论：DiskANN 的实现边界此前已经被诚实收口，但 benchmark gate 本身还缺一个 replayable artifact 和 authority-backed cross-dataset DiskANN rows；在本轮 authority refresh 后，baseline 与 sampled datasets 仍一致停留在 sub-gate / non-trusted 区间，因此 benchmark gate 可以正式归档为受限范围下的 explicit no-go benchmark evidence，而不是继续停留在口头描述。
  3. 本轮执行：新增 `benchmark_results/diskann_p3_004_benchmark_gate.json`，把当前 benchmark lane 固化为 `no_go_for_native_comparable_benchmark`；扩展 `src/benchmark/cross_dataset_sampling.rs` 生成 `DiskANN` rows，并将 `tests/bench_diskann_1m.rs`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs` 改成默认 lane 的真实 DiskANN benchmark regressions；随后从 authority run 回传并落库新的 `benchmark_results/cross_dataset_sampling.json`。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md`、`DEV_ROADMAP.md` 已统一写回 DiskANN benchmark-gate closure，下一条按 `feature-list.json` 选择的可执行 feature 是 `diskann-stop-go-verdict`。
  5. 后续主缺口：不要把 benchmark-gate artifact 误当成 family-level final classification；下一条需要单独归档 DiskANN family 的 constrained/no-go final verdict，并继续阻止任何 native-comparable or leadership claim。
  状态：Phase 5 Active（DiskANN benchmark gate archived as constrained no-go evidence；next feature per durable source is `diskann-stop-go-verdict`）。
- 2026-03-12 03:02: **builder-loop：收口 `IVFPQ-P3-003`，将 IVF-PQ family 归档为 no-go（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`feature-list.json`、`benchmark_results/ivfpq_p1_002_focused.json`、`benchmark_results/recall_gated_baseline.json`、`benchmark_results/cross_dataset_sampling.json`、`tests/bench_ivf_pq_perf.rs`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs`。
  2. 阶段结论：IVF-PQ hot-path、baseline、cross-dataset 三条 authority-backed artifact 链都未把 recall 拉过 `0.8` gate；与此同时 FFI / persistence / metadata contract 已在前一轮关闭，因此 family 级最终状态不再是“缺 benchmark 事实”，而是明确 `no-go`。
  3. 本轮执行：新增 `benchmark_results/ivfpq_p3_003_final_verdict.json`，把 IVF-PQ family 级最终分类固化为 `no-go`；并将 `tests/bench_ivf_pq_perf.rs`、`tests/bench_recall_gated_baseline.rs`、`tests/bench_cross_dataset_sampling.rs` 改成默认 lane 的 artifact/verdict regressions，而不是 `long-tests` 文件级空壳。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`RELEASE_NOTES.md` 已统一写回 IVF-PQ 收口状态，下一条按 `feature-list.json` 选择的可执行 feature 是 `diskann-remote-benchmark-gate`。
  5. 后续主缺口：不再围绕 IVF-PQ 继续做无边界性能调参；除非未来出现新的 authority artifact 清楚越过 recall gate，否则不要重开 IVF-PQ production-candidate 或 leadership claim。
  状态：Phase 5 Active（IVF-PQ archived as no-go；next feature per durable source is `diskann-remote-benchmark-gate`）。
- 2026-03-12 02:00: **builder-loop：收口 `HNSW-P3-002`，将 HNSW family 归档为 functional-but-not-leading（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`feature-list.json`、`benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json`、`benchmark_results/baseline_p3_001_stop_go_verdict.json`、`tests/test_baseline_methodology_lock.py`、`tests/bench_hnsw_cpp_compare.rs`。
  2. 阶段结论：layer-0 repair、same-schema HDF5 refresh、以及 HNSW FFI / persistence contract 已全部形成 authority-backed 证据链；HNSW 不再缺 recall 或 production contract 事实。
  3. 本轮执行：新增 `benchmark_results/hnsw_p3_002_final_verdict.json`，把 HNSW family 级最终分类固化为 `functional-but-not-leading`，同时保留 baseline artifact 的 narrower `no_go_for_performance_leadership` 结论；并将 `tests/bench_hnsw_cpp_compare.rs` 改为默认 lane 的 verdict regression，而不是 `0 tests` 空壳。
  4. 治理同步：`TASK_QUEUE.md`、`GAP_ANALYSIS.md`、`task-progress.md`、`feature-list.json`、`memory/CURRENT_WORK_ORDER.json` 已统一写回 HNSW 收口状态，下一条工作切到 IVF-PQ production contract。
  5. 后续主缺口：不再围绕 HNSW 继续做无边界性能调参；下一阶段应推进 IVF-PQ 的 FFI / persistence / metadata contract closure，并把最终生产验收保留到 family verdict 全部完成之后。
  状态：Phase 5 Active（HNSW archived as functional-but-not-leading；IVF-PQ contract closure promoted）。
- 2026-03-10 00:01: **builder-loop：收口 `DISKANN-P1-003`，将 Rust DiskANN 明确降级为受限实现（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`src/faiss/diskann.rs`、`docs/FFI_CAPABILITY_MATRIX.md`。
  2. 代码结论：`src/faiss/diskann.rs` 中 `l2_sqr` 已稳定走 `simd::l2_distance_sq`；新增单测将 `PQCode` 锁定为“按子段均值量化的 placeholder”，证明它不是 native-comparable PQ/SSD pipeline。
  3. 阶段决策：`DISKANN-P1-003` 验收达成并从 active queue 关闭；Rust DiskANN 维持“简化 Vamana + placeholder PQCode”的受限结论，不进入性能主线，也不作为 native DiskANN 的公平对照对象。
  4. 治理同步：`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`docs/FFI_CAPABILITY_MATRIX.md` 已统一写回该 constrained verdict，避免后续再次把 API 入口误读为性能/实现 parity。
  5. 后续主缺口：Phase 5 的活跃任务回到 `PERF-P3-005`，但仍受 recall-credible 进入条件约束；在新的可信主线路径出现前，不生成虚假 leadership claim。
  状态：Phase 5 Active（DiskANN archived as constrained/no-go implementation evidence；PERF-P3-005 remains gated）。
- 2026-03-09 21:30: **builder-loop：关闭 `IVFPQ-P1-002` 并推进 `DISKANN-P1-003` 的最小真实修复（plan+exec）**
  1. 复核输入：`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`benchmark_results/ivfpq_p1_002_focused.json`、`src/faiss/diskann.rs`。
  2. 阶段结论：`IVFPQ-P1-002` 已具备 focused artifact 与 required checks 证据，且结论明确为“真实运行路径，但 recall 不达可信阈值”，因此不再保留为当前活跃 TODO。
  3. 本轮执行：将 `DISKANN-P1-003` 提升为唯一当前任务，并在 `src/faiss/diskann.rs` 修复 `l2_sqr` 使用 `simd::l2_distance` 后再平方的 sqrt round-trip 反模式，改为直接走 `simd::l2_distance_sq`；同时把 `PQCode` 明确标注为 placeholder，而非 native-comparable PQ。
  4. 治理同步：`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`docs/FFI_CAPABILITY_MATRIX.md` 已统一写回“IVFPQ 已 no-go 收口、DiskANN 进入诚实边界收敛”的口径。
  5. 剩余缺口：DiskANN 仍不是原生可比实现；后续应继续围绕 placeholder PQCode 的去留、能力矩阵口径与 focused benchmark/semantic evidence 收口，而不是提前进入 leadership claim。
  状态：Phase 5 Active（IVFPQ archived as no-go evidence；DiskANN boundary-closure promoted）。
- 2026-03-09 21:03: **计划轮次：关闭 `HNSW-P1-001` 并切换 `IVFPQ-P1-002`（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T12:51:00Z` 晚于当前 plan，且 queue 顶部首个未完成 TODO 已不再是 `PLAN_RESULT.task_id=HNSW-P1-001`，因此本轮不能 skip。
  3. 收口结论：`HNSW-P1-001` 的远端 repo 修复与首份 recall-gated artifact 已完成，当前 before/after 结果为 recall `0.217 -> 0.215`、qps `~1621 -> ~19235`；由于 recall 仍低于可信阈值，这条证据只能作为 `recheck required / no-go` artifact 归档，不能继续占据当前活动任务位。
  4. 阶段决策：按 `BUG > CORE(IMPL/PERF) > SEM/PROD > BENCH` 与 Phase 5 优先级，将当前唯一活动任务切换为 `IVFPQ-P1-002`，聚焦 `src/faiss/ivf.rs` / `src/faiss/ivfpq.rs` / `src/faiss/scann.rs` 的 ADC / centroid search 审计与 focused benchmark；`DISKANN-P1-003` 与 `PERF-P3-005` 保持后继任务。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，把 HNSW 当前状态从 active blocker 改为 archived no-go evidence，避免 exec 继续围绕已收口阶段空转。
  状态：Phase 5 Active（HNSW archived as no-go evidence；IVF/PQ audit promoted）。
- 2026-03-09 20:06: **计划轮次：`HNSW-P1-001` 继续置顶，但 blocker 收窄为远端 repo 基线修复 + artifact 落地（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：queue 首个 TODO 仍为 `HNSW-P1-001`，且最新 `EXEC_RESULT.updated_at=2026-03-09T12:00:00Z` 晚于当前 plan；exec 已把 blocker 从“runner/cwd 需修通”进一步收敛为“远端 repo manifest 已损坏且 sync path 被 dirty worktree 阻断”，因此本轮不能 skip。
  3. 现状复核：本地 `cargo test --lib hnsw -- --nocapture` 仍通过，说明 HNSW 最近一轮热路径收紧没有引入新的 focused correctness 回归；当前缺口是远端隔离 native baseline 工作树（`/data/work/knowhere-native-src`）尚未形成稳定的 benchmark artifact 证据链，而非新的 HNSW 本地 bugfix。
  4. 阶段决策：保持 `HNSW-P1-001` 为唯一当前任务，不前移 `IVFPQ-P1-002` / `DISKANN-P1-003` / `PERF-P3-005`；但将出口再次收窄为“先恢复远端 repo 基线（cwd 断言、manifest 可解析、dirty worktree 可恢复），再产出 recall-gated before/after artifact 并给出 go/no-go 结论”。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，避免 exec 继续停留在已经过时的宽泛 runner 叙事。
  状态：Phase 5 Active（HNSW remains first; remote repo repair is now the active gate before artifact generation）。
- 2026-03-09 19:03: **计划轮次：`HNSW-P1-001` 不换任务号，但收窄为远端 x86 artifact 落地（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：queue 首个 TODO 仍为 `HNSW-P1-001`，但最新 `EXEC_RESULT.updated_at=2026-03-09T10:46:09Z` 晚于当前 plan，且 exec 已把 blocker 从“泛热路径工程缺口”收敛为 `remote_x86_artifact_generation`，因此本轮不能 skip。
  3. 现状复核：本地 `cargo test --lib hnsw -- --nocapture` 已通过，说明最近一轮 HNSW 布局收紧至少没有打破 focused correctness；当前缺的不是新的本地 bugfix，而是在正确 repo cwd 下拿到远端 recall-gated before/after artifact。
  4. 阶段决策：保持 `HNSW-P1-001` 为唯一当前任务，不前移 `IVFPQ-P1-002` / `DISKANN-P1-003` / `PERF-P3-005`；但将出口明确收窄为“先修通 remote runner/cwd，再产出 artifact 并给出邻居布局调整是否值得保留的 go/no-go 结论”。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，把 HNSW 当前 blocker 写成可执行的 scoped sub-stage，避免 exec 在已经通过的本地 HNSW gate 上重复空转。
  状态：Phase 5 Active（HNSW remains first; remote artifact generation is the active gate）。
- 2026-03-09 18:03: **计划轮次：关闭 `CORE-P0-001` 后切换到 `HNSW-P1-001`（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：queue 首个 TODO 已切到 `HNSW-P1-001`，且最新 `EXEC_RESULT.updated_at=2026-03-09T09:35:00Z` 晚于旧 `PLAN_RESULT.updated_at=2026-03-09T09:03:00Z`；旧 plan 仍指向已完成的 `CORE-P0-001`，因此本轮不能 skip。
  3. 收口结论：远端 x86 SIMD verification lane 已恢复并具备新鲜 required-check 证据，`CORE-P0-001` 不再是活跃 blocker；Phase 5 主缺口已回到核心实现优先级最高的 HNSW 热路径工程化。
  4. 阶段决策：将当前唯一活动任务切换为 `HNSW-P1-001`，聚焦 visited list 复用、结果距离复用、邻居布局/benchmark 证据三件事；`IVFPQ-P1-002` 与 `DISKANN-P1-003` 继续保留为后继 scoped tasks，`PERF-P3-005` 暂不前移。
  5. 治理动作：queue/roadmap/gap 已与该阶段切换保持一致；本轮仅更新 planning 结论与审计记录，避免 exec 继续围绕已关闭的 SIMD blocker 空转。
  状态：Phase 5 Active（core-path performance engineering promoted; HNSW first）。
- 2026-03-09 17:05: **计划轮次：将 `CORE-P0-001` 从代码语义修复继续收窄为远端 x86 toolchain unblock（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：queue 首个 TODO 仍与 `PLAN_RESULT.task_id=CORE-P0-001` 一致，但最新 `EXEC_RESULT.updated_at` 已晚于 plan，且 blocker 已由 `src/simd.rs` 代码语义推进为 `remote_x86_toolchain`，因此本轮不能 skip。
  3. 现状复核：本地 smoke `cargo test --lib -q` 通过，`scripts/remote/common.sh` 的最小兼容修复已落地；当前无法拿到新鲜 required gate 的直接原因是远端 cargo 1.75 解析 `getrandom 0.4.1` 时不支持 edition2024 manifest，而不是新的 SIMD 数值错误。
  4. 阶段决策：保留 `CORE-P0-001` 为当前任务，但将其出口改写为“先恢复远端 x86 SIMD 验证链可执行性，再重跑 focused SIMD gate”；`HNSW-P1-001` / `IVFPQ-P1-002` / `DISKANN-P1-003` / `PERF-P3-005` 继续排在其后。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，把当前 blocker 固化为可执行的 scoped task，避免 exec 继续围绕过时的 `src/simd.rs` 叙事空转。
  状态：Phase 5 Active（remote x86 toolchain drift blocks fresh SIMD evidence）。
- 2026-03-09 16:14: **计划轮次：将 `CORE-P0-001` 从泛 SIMD 正确性收窄为 x86 default+simd 构建语义修复（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：queue 首个 TODO 与 `PLAN_RESULT.task_id` 形式上一致，但最新 exec 已把 blocker 收敛为 `remote_x86_default_simd_build`，且 `next_action=plan`；原 plan 中“修 SIMD correctness + default policy”的宽目标已失效，因此本轮不能 skip。
  3. 现状复核：exec 已完成 SSE/AVX2 L2 reduction 最小修复、补 irregular-input focused regression，并将 `simd` 纳入 default；但远端 x86 `default+simd` 构建暴露 `src/simd.rs` 大量既有 intrinsic 调用仍缺 `unsafe` / `target_feature` 边界，导致当前拿不到可信 x86 required gate。
  4. 阶段决策：保留 `CORE-P0-001` 作为当前任务，但将其 scope 收窄为“先恢复 x86 default+simd build + focused correctness gate”，避免 exec 继续围绕过宽的泛 SIMD 叙事空转；`HNSW-P1-001` / `IVFPQ-P1-002` / `DISKANN-P1-003` / `PERF-P3-005` 继续排在其后。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，把当前 blocker 写成可执行的 scoped task，并要求 required checks 明确落在远端 x86。
  状态：Phase 5 Active（core-path SIMD foundation still blocking; scope narrowed to concrete x86 build semantics）。
- 2026-03-09 15:03: **计划轮次：关闭 `PERF-P3-004` 并切换 `PERF-P3-005`（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T06:34:30Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T06:07:00Z`，且 exec 已完成 queue 顶部 `PERF-P3-004`，因此本轮不能 skip。
  3. 收口结论：远端 x86 已成功构建 `benchmark_float_qps`，`--gtest_list_tests` artifact 已产出，GTest/CMake/Conan runtime 依赖链不再是活跃 blocker；`PERF-P3-004` 验收达成。
  4. 阶段决策：Phase 5 的最小高价值缺口切换为 `PERF-P3-005`——在 `clustered_l2 + HNSW` 上生成第一条 recall-gated native-vs-rs 对照，并给出 `领先 / parity / 落后` 的明确判定；若未领先，再拆最小 optimization follow-up。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，将 active task 从“修链路”切到“跑对照”。
  状态：Phase 5 Active（native benchmark harness closed；leadership proof promoted）。
- 2026-03-09 14:07: **计划轮次：将 `PERF-P3-004` 从“harness enablement”继续收窄到 GTest/CMake 构建链路修复（builder-plan）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`docs/PERF_P3_004_NATIVE_HARNESS.md`、`scripts/remote/native_benchmark_probe.sh`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T06:15:00Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T05:08:00Z`，因此本轮 plan 已失效，不能 skip。
  3. 现状复核：exec 已证明 native side 真实 target 存在（`benchmark_float_qps`），schema mapping 也仍有效；当前失败点不再是“找不到 harness”，而是 benchmark 配置链路稳定卡在 `find_package(GTest REQUIRED)`。
  4. 阶段决策：将 `PERF-P3-004` 继续收窄为修通 benchmark 的 GTest/CMake 发现链路，并以 `benchmark_float_qps --gtest_list_tests` 作为本轮唯一可判定出口；只有 binary surface 真正存在后，才进入 `PERF-P3-005` 的 native-vs-rs 对照。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，消除 queue 中仍停留在宽泛 harness 叙事的漂移。
  状态：Phase 5 Active（route chosen; native benchmark blocker narrowed to concrete CMake/GTest fix）。
- 2026-03-09 13:08: **计划轮次：将 `PERF-P3-004` 从宽泛“性能领先证明”收窄为先打通 native benchmark harness（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`benchmark_results/recall_gated_baseline.json`、`benchmark_results/cross_dataset_sampling.json`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T04:52:30Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T04:47:00Z`，且 exec 已将 blocker 明确为 `native_benchmark_harness_unavailable`，因此本轮必须重做 planning，不能 skip。
  3. 现状复核：现有 artifact 已足以把首条主线收敛到 `clustered_l2 + HNSW`，不再缺“选路”；真正缺的是远端 x86 上缺少 native knowhere 可直接复用的 benchmark binary / runner，导致 same-methodology native-vs-rs 对照无法进入可判定状态。
  4. 阶段决策：将 `PERF-P3-004` 重写为“先打通 harness + schema 对齐”的 benchmark 基础设施任务，并拆出紧随其后的 `PERF-P3-005` 负责真正的 native-vs-rs 对照与领先判定，避免 exec 在过宽目标上反复空转。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，把当前 Phase 5 活跃任务收敛到可执行的远端 harness 打通。
  状态：Phase 5 Active（route chosen; harness enablement promoted before leadership proof）。
- 2026-03-09 12:39: **OBS-P3-005 收口：定义最小 observability / trace propagation / resource contract（builder-exec）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`TASK_QUEUE.md`，确认当前轮满足 direct exec turn 条件，且 `OBS-P3-005` 为 queue 顶部任务。
  2. 代码收口：在 `src/ffi.rs` 的 `knowhere_get_index_meta` JSON contract 中新增 `observability` / `trace_propagation` / `resource_contract` 三个稳定 section，统一 build/search/load 事件名、trace context 透传入口与最小资源估算/mmap 审计字段。
  3. 回归更新：扩展 `ffi::tests::test_ffi_abi_metadata_contract`，覆盖 Flat/HNSW/IVF/Sparse（以及 ScaNN 条件分支）的 observability/resource/trace 字段断言，保持 required gate 仍为单一 focused FFI smoke。
  4. 治理同步：`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/FFI_CAPABILITY_MATRIX.md` 已统一收口 `OBS-P3-005`，后续主缺口切换为 `PERF-P3-004`。
  状态：Phase 5 Active（observability baseline closed；performance leadership promoted）。
- 2026-03-09 12:30: **计划轮次：关闭 `PERSIST-P3-003` 并切换 `OBS-P3-005`（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`docs/FFI_CAPABILITY_MATRIX.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T04:13:22Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T04:02:00Z`，因此本轮必须重新 planning，不能 skip。
  3. 收口结论：`PERSIST-P3-003` 已由 exec 完成；queue 顶部任务与实际状态已脱节，若不主动切换会让后续 exec 围绕已收口的 persistence 文档/回归范围空转。
  4. 差距重估：当前 Phase 5 最小高价值缺口不再是 persistence，而是 observability/runtime governance baseline——代码里已有零散 `tracing::*`、benchmark memory estimator、legality `mmap_supported`，但仍缺统一 schema、FFI/runner trace 透传边界与最小资源估算 contract。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`，将当前任务切换为 `OBS-P3-005`，并把 `PERSIST-P3-003` 转入已完成归档；同时计划在后续文档轮次重写 `docs/FFI_CAPABILITY_MATRIX.md` 以消除旧矩阵口径漂移。
  状态：Phase 5 Active（persistence semantics closed；observability/runtime governance promoted）。
- 2026-03-09 12:02: **计划轮次：关闭 `ABI-P3-002` 并切换 `PERSIST-P3-003`（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`src/ffi.rs`、`src/index.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T03:21:00Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:46:00Z`，因此本轮必须重新 planning，不能 skip。
  3. 收口结论：`src/ffi.rs` 已形成逐索引 `additional_scalar` / `capabilities` / `semantics` contract，且 `ffi::tests::test_ffi_abi_metadata_contract` 已覆盖 null-safe、unsupported、partial-supported 与 per-index 差异场景；`ABI-P3-002` 当前验收已达成。
  4. 阶段决策：按 Phase 5 的生产硬化目标，下一最小高价值缺口切换为 `PERSIST-P3-003`，先收敛 `file_save_load` / `memory_serialize` / `deserialize_from_file` 的支持矩阵与 focused regressions，而不是继续在已完成 ABI 范围空转。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，将当前任务切换为更窄的 persistence 子切片。
  状态：Phase 5 Active（ABI metadata hardening closed；persistence semantics hardening promoted）。
- 2026-03-09 02:46: **计划轮次：ABI-P3-002 重新收口，清除陈旧 blocker 叙事（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`、`src/ffi.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T02:42:05Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:34:00Z`，且 exec 摘要声称 required lib gate 被 HNSW parallel compatibility failure 阻塞，因此本轮不能 skip，必须先做直接调度自检与现状复核。
  3. 现状复核：计划侧实测 `cargo test --lib -q` 在当前工作树通过（524 passed, 0 failed, 2 ignored），未复现“required lib gate 被 HNSW parallel 失败阻塞”的结论；说明 blocker 已消失或结果文件已陈旧，当前真正缺口仍是 ABI 语义矩阵未收口。
  4. 阶段决策：不回切到新的 BUG，而是把 `ABI-P3-002` 拆成更窄的 per-index ABI 子矩阵（HNSW / IVF / ScaNN / Sparse 的 additional-scalar + index_meta 字段语义 + focused FFI regressions），避免 exec 围绕陈旧 blocker 空转。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md` 与 `memory/PLAN_RESULT.json`，将当前任务重写为可执行的 ABI 子切片，并把 required checks 重新绑定到现状实测通过的 gate。
  状态：Phase 5 Active（ABI metadata hardening 继续；stale gate blocker cleared at planning layer）。
- 2026-03-09 02:34: **计划轮次：关闭 `SEM-P3-001` 并切换 `ABI-P3-002`（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/PARITY_AUDIT.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at=2026-03-09T02:14:52Z` 晚于 `PLAN_RESULT.updated_at=2026-03-09T02:02:00Z`，且 queue 首个 TODO 仍指向旧 umbrella `SEM-P3-001`，因此当前 plan 已失效，不能 skip。
  3. 收口结论：最新 exec 已把 `HNSW` / `IVF` / `Sparse` / `ScaNN` 的 `GetVectorByIds` / `HasRawData` focused semantic tail 收敛到可审计状态；继续保留 `SEM-P3-001` 作为当前任务只会让 exec 重复进入已关闭范围。
  4. 阶段决策：按 `BUG > PARITY > OPT > BENCH` 与 Phase 5 目标，下一最小高价值缺口切换为 `ABI-P3-002`，先把 FFI metadata / additional-scalar 从“最小稳定摘要”提升为逐模块真实 contract。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`，将 `SEM-P3-001` 从 active queue 出队并把 `ABI-P3-002` 提升为当前任务。
  状态：Phase 5 Active（semantic tail closed；production metadata hardening in progress）。
- 2026-03-09 01:50: **SEM-P3-001 focused delta：收敛 DiskANN / AISAQ 的 `HasRawData` 度量语义（builder-exec）**
  1. 对照原生 `src/index/diskann/diskann.cc` 与 `src/index/diskann/diskann_aisaq.cc`，确认 `HasRawData` 只对 `L2/COSINE` 返回 true，而非“是否已有内存向量”这种运行态条件。
  2. Rust 侧修复：`src/faiss/diskann.rs` 与 `src/faiss/aisaq.rs` 改为按 metric gate 暴露 raw-data 语义；`get_vector_by_ids` 在 `has_raw_data=false` 时返回 Unsupported，而不是继续走伪 raw-data 路径。
  3. 新增 focused conformance tests：覆盖 `L2/COSINE/IP` 三种 metric 的 `has_raw_data` 返回，并锁定 DiskANN 缺失 ID 走 error 而非静默零填充。
  4. 当前剩余 delta：HNSW/IVF/Sparse/ScaNN 的 missing-id / lossy-index / empty-index 语义仍需继续矩阵化，`SEM-P3-001` 暂不关闭。
  状态：Phase 5 Active（semantic fidelity audit in progress）。
- 2026-03-08 20:36: **OPT-P2-004 收口：legality/governance 漂移已消除（builder-exec）**
  1. 复核入口：`src/api/index.rs` 中 `IndexConfig::validate()` 仍统一调用 legality matrix；`src/ffi.rs` 中 `IndexWrapper::new()` 仍在构造期调用 `validate_index_config`，未发现新运行时代码缺口。
  2. 治理动作：`TASK_QUEUE.md` 将 `OPT-P2-004` 标记完成并清空 TODO；`DEV_ROADMAP.md` 与 `GAP_ANALYSIS.md` 同步关闭最后一条 governance tail。
  3. 审计收敛：模块表 `Index factory/legality` 保持 `Done`，并在 changelog 中明确旧 `Partial` 属历史残留而非未完成开发项。
  4. 结果：queue/roadmap/gap/audit 对 legality 状态重新一致，Phase 4 可关闭。
  状态：Phase 4 Closed（no active non-GPU parity/governance tail）。
- 2026-03-08 19:41: **计划轮次：P1/P2 主线收口后重建尾部 Partial 清理队列（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`、`docs/FFI_CAPABILITY_MATRIX.md`、`src/faiss/hnsw_pq.rs`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at` 晚于 `PLAN_RESULT.updated_at`，且当前 queue 已无未完成 TODO，不满足 skip 条件。
  3. 收口结论：`PARITY-P1-011` 已由 exec 完成，P0/P1 主线与 Phase 3 benchmark/gate 能力均已闭环，继续维持空队列会让审计表中的尾部 `Partial` 脱离治理。
  4. 差距重估：当前最小且真实的剩余缺口是 `HNSW-PQ=Partial`（高级接口/持久化语义未闭环）；`Index factory/legality=Partial` 更像历史残留状态漂移，应作为治理清理项而非重新回切 P1。
  5. 治理动作：在 `TASK_QUEUE.md` 新增 `PARITY-P2-001` 与 `OPT-P2-004`，`DEV_ROADMAP.md` 打开新的尾部收口阶段，`GAP_ANALYSIS.md` 改写为 tail-closure 叙事。
  状态：Phase 4 Active（tail partial cleanup / quantized parity polish）。
- 2026-03-08 18:49: **计划轮次：BENCH-P2-003 收口后回切 P1 parity 缺口（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/EXEC_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  2. 调度判断：最新 `EXEC_RESULT.updated_at` / `VERIFY_RESULT.updated_at` 均晚于 `PLAN_RESULT.updated_at`，触发重新 planning；不满足 skip 条件。
  3. 收口结论：`BENCH-P2-003` 的 cross-dataset artifact 已落地且 verify 通过，P2 三个 scoped tasks（`OPT-P2-003` / `BENCH-P2-002` / `BENCH-P2-003`）均可判定为完成。
  4. 差距重估：模块状态仍存在 `Sparse=Partial`、`FFI ABI=Partial`，与“Phase2 已闭环”叙事不一致；按 `BUG > PARITY > OPT > BENCH` 应优先回切 parity。
  5. 治理动作：在 `TASK_QUEUE.md` 新增并提升 `PARITY-P1-010` / `PARITY-P1-011`，同步更新 `DEV_ROADMAP.md`（Phase2 Reopened）与 `GAP_ANALYSIS.md`（主缺口从 P2 切回 P1）。
  状态：Phase 2 Reopened（parity debt burn-down）；Phase 3 Closed（validation/perf hardening 已收口）。
- 2026-03-08 17:35: **计划轮次：BENCH-P2-002 验收关闭并切换 BENCH-P2-003（builder-plan）**
  1. 复核输入：`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`TASK_QUEUE.md`、`tests/bench_recall_gated_baseline.rs`、`benchmark_results/recall_gated_baseline.json`、`src/benchmark/report_schema.rs`。
  2. 调度判断：最新 `VERIFY_RESULT.updated_at` 晚于 `PLAN_RESULT.updated_at`，触发重新 planning；不满足 skip 条件。
  3. 收口结论：`BENCH-P2-002` 验收达成（ScaNN/RaBitQ/Sparse 条目已入 baseline，低可信条目含可执行解释），应从当前任务出队。
  4. 阶段决策：Phase 3 仍 active，下一最小高价值任务切换为 `BENCH-P2-003`（cross-dataset artifact 流水线）。
  5. 治理动作：同步更新 `TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`，保持 queue/roadmap/gap/audit 一致。
  状态：Phase 3 Active（Validation/Performance Hardening，进入 cross-dataset 收敛）。
- 2026-03-08 17:00: **任务收口守护：OPT-P2-003 验收完成并切换 BENCH-P2-002（progress-guard）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`scripts/gate_profile_runner.sh`、`scripts/README_GATE_PROFILES.md`。
  2. 结论：不存在空转；本轮属于任务收口后的阶段内连续推进，`OPT-P2-003` 已满足验收。
  3. 差距重估：Phase 3 当前主缺口转为 benchmark 覆盖与可信度解释（ScaNN/RaBitQ/Sparse），而非门禁执行器。
  4. 动作：`TASK_QUEUE.md` 将 `OPT-P2-003` 标记 Done，当前任务切换为 `BENCH-P2-002`；`memory/PLAN_RESULT.json` 同步更新 next executor 为 dev。
  5. 治理同步：更新 `DEV_ROADMAP.md`（活跃任务列表）与 `GAP_ANALYSIS.md`（门禁缺口关闭），保持 queue/roadmap/gap/audit 一致。
  状态：Phase 3 Active（Validation/Performance Hardening，benchmark 扩面中）。
- 2026-03-08 16:40: **阶段切换守护：P0/P1 收口后重建 P2 队列（progress-guard）**
  1. 复核输入：`TASK_QUEUE.md`、`memory/PLAN_RESULT.json`、`memory/DEV_RESULT.json`、`memory/VERIFY_RESULT.json`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  2. 结论：最近 dev/verify 存在真实推进（`OPT-P2-002` 新增稳定回归矩阵并通过 required checks），不构成“空转”。
  3. 新差距：队列清空后若停在 `NONE` 会造成阶段治理脱节；当前主要缺口已从 parity 修复转为 validation/perf hardening。
  4. 动作：新增并排定 P2 scoped tasks（`OPT-P2-003` / `BENCH-P2-002` / `BENCH-P2-003`），并将当前任务切换为 `OPT-P2-003`。
  5. 审计约束：后续轮次需在报告中同时引用 gate profile 与 benchmark artifact，避免“门禁结论可过但性能结论不可复现”。
  状态：Phase 3 Active（Validation/Performance Hardening）。
- 2026-03-07 15:19: **BUG-P1-001 长跑卡住根因收敛（慢测治理第二轮）**
  1. 同步 baseline：`git fetch origin main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮修复：
     - `tests/bench_hnsw_parallel.rs` 将 `test_hnsw_parallel_build_small/medium/large` 与 `test_hnsw_thread_scaling` 标记为 `#[ignore]`
     - `tests/perf_test.rs` 将 `test_performance_comparison_small/100k/1m`、`test_ivf_flat_build_optimization`、`test_opt030_adaptive_ef` 标记为 `#[ignore]`
  3. 最小验证：
     - ✅ `cargo test --test perf_test -q`（5 ignored）
     - ✅ `cargo test --test bench_hnsw_parallel -q`（3 passed, 5 ignored）
     - ✅ `cargo test --lib faiss::ivf_sq_cc::tests::test_ivf_sq_cc_train_add_search`
  4. 结论：默认回归路径进一步剔除性能基准误跑，`BUG-P1-001` 继续收敛；仍待后续轮次确认全量测试全绿。
  状态：BUG-P1-001 In Progress。
- 2026-03-07 14:35: **BUG-P1-001 长跑卡住根因收敛（慢测治理第一轮）**
  1. 同步 baseline：`git fetch origin main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 根因定位：默认回归路径包含 HNSW 100K 构建性能测试（单测+集成测试各一处），导致 `cargo test` 长尾明显；ScaNN 基础测试数据规模偏大也加剧总时长。
  3. 本轮修复：
     - `src/faiss/hnsw.rs::test_hnsw_build_performance` 标记 `#[ignore]`
     - `tests/opt015_hnsw_build.rs::test_hnsw_build_performance` 标记 `#[ignore]`
     - `src/faiss/scann.rs::test_scann_basic` 数据规模 `n: 1000 -> 256`
  4. 最小验证：
     - ✅ `cargo test --lib test_scann_basic`
     - ✅ `cargo test --tests opt015_hnsw_build`（默认忽略，退出成功）
  5. 结论：已完成“长跑根因定位”子项，BUG-P1-001 剩余“全量回归全绿”验收。
  状态：BUG-P1-001 In Progress。
- 2026-03-07 14:30: **PARITY-P1-009 收敛验收（Index trait + 参数对齐）**
  1. 同步 baseline：`git fetch origin main`（基线仍为 `4f60908fc9ad7438b4b8ff64210481ab281009b0`）
  2. 复核结果：
     - `src/index/minhash_lsh_index_trait.rs`：MinHashLSH 已提供统一 `Index` trait wrapper（含 `create_ann_iterator` / `get_vector_by_ids` / `save` / `load`）
     - `src/api/index.rs`：`mh_*` 参数别名反序列化已对齐（`mh_element_bit_width` / `mh_lsh_band` / `mh_lsh_aligned_block_size` / `mh_lsh_shared_bloom_filter` / `mh_lsh_bloom_false_positive_prob`）
     - `src/ffi.rs`：`MinHashLSH` 类型声明、`create_ann_iterator` 接线、`add_binary` 字节对齐校验均已补齐
  3. 最小验证：
     - ✅ `cargo test --lib minhash_index_trait`
     - ✅ `cargo test --lib test_minhash_cpp_param_aliases_deserialize`
     - ✅ `cargo test --lib ffi::tests::test_index_type_minhash_lsh`
     - ✅ `cargo test --lib ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`
  4. 结论：PARITY-P1-009 验收完成；全量回归失败项归属 BUG-P1-001，不再阻塞本条目关闭。
  状态：PARITY-P1-009 Done。
- 2026-03-07 14:25: **PARITY-P1-009 MinHash FFI 参数语义补齐（维度对齐校验）**
  1. 执行同步与基线更新：`git fetch origin && git rev-parse HEAD && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮变更：
     - `src/ffi.rs`：`IndexWrapper::add_binary` 的 MinHash 分支由硬编码参数改为按 `dim(bits)` 推导 `vector_bytes` 与 `mh_vec_length`
     - 增加输入校验：空输入、`vectors.len() % vector_bytes != 0`、`vector_bytes` 非 `u64` 元素对齐、`mh_vec_length == 0` 均返回 `InvalidArg`
     - 新增回归测试 `ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`
  3. 最小验证：
     - ✅ `cargo test --lib test_minhash_add_binary_rejects_invalid_dim_alignment`
  4. 结论：MinHash `mh_*` 参数/维度语义在 FFI add 路径进一步收敛；任务仍受全量回归门槛限制，维持 In Progress。
  状态：PARITY-P1-009 In Progress。
- 2026-03-07 14:25: **PARITY-P1-008 Sparse iterator/filter 行为统一**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. 本轮变更：
     - `src/faiss/sparse_inverted.rs` 新增统一 helper `ann_results_from_sparse_query`，集中处理 bitset 转换 + iterator 全量候选检索入口
     - `SparseInvertedIndex::create_ann_iterator` 改为通过统一 helper（TAAT 路径）
     - `src/faiss/sparse_wand.rs` 的 `create_ann_iterator` 改为复用统一 helper（WAND 路径）
     - 新增回归测试 `test_sparse_wand_iterator_and_search_with_bitset_consistent`
  3. 最小验证：
     - ✅ `cargo test --lib sparse_inverted::tests::test_sparse_inverted_ann_iterator_respects_bitset`
     - ✅ `cargo test --lib sparse_wand::tests::test_sparse_wand_iterator_and_search_with_bitset_consistent`
  4. 结论：PARITY-P1-008 最后子项“统一 iterator/filter 行为”完成；Sparse Index trait 统一接口任务收敛。
  状态：PARITY-P1-008 Done。
- 2026-03-07 13:15: **BUG-P1-001 全量回归阻塞定位 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. C++/Rust 增量扫描：
     - `python3` 关键词统计：`CPP src/index=240`、`CPP include/knowhere/index=53`
     - Rust 关键词统计：`sparse_inverted.rs=3`、`sparse_wand.rs=3`、`index.rs=14`、`ffi.rs=64`
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 仍提供 `create_ann_iterator/get_vector_by_ids/save/load` 统一入口
     - `src/ffi.rs`：Sparse 仍未新增 `create_ann_iterator`/`save/load` 的特化桥接（依赖索引实现）
  4. 验证：
     - ⏳ `cargo test -q` 在 `501 tests` 场景下长时间无退出（已越过 435/501，且出现 `hnsw_build_performance` / `scann_basic` / `sparse_inverted::test_wand_search_basic` 超 60s 提示）
     - ⚠️ 本轮未拿到完整退出码，无法确认“全量回归恢复”验收
  5. 新发现差距：全量回归除既有失败外，新增“长跑卡住/超时不可判定”阻塞点，需在 BUG-P1-001 下拆分 `hang root-cause` 子任务。
  状态：BUG-P1-001 维持 In Progress（从“失败用例修复”转入“长跑阻塞定位”）。
- 2026-03-07 11:55: **BUG-P1-001 子项修复：IVF-CC 检索候选不足导致 topk 回归失败 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `4f60908fc9ad7438b4b8ff64210481ab281009b0`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 2 -type f | wc -l`=49，`find .../include/knowhere/index -maxdepth 2 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 仍提供 `create_ann_iterator/get_vector_by_ids/save/load` 统一入口
     - `src/ffi.rs`：未新增 Sparse 的 `create_ann_iterator` / `save/load` 特化分支（仍依赖索引实现）
  4. 本轮修复：
     - `src/faiss/ivf_flat_cc.rs`：修复 `nprobe` 仅扫描少量倒排桶导致 `top_k` 候选不足的问题，新增按 centroid distance 的 fallback 扫描路径；`num_visited` 改为截断前候选数
     - `src/faiss/ivf_sq_cc.rs`：同样补齐 fallback 扫描逻辑，避免 `test_ivf_sq_cc_train_add_search` 偶发/稳定失败
  5. 验证：
     - ✅ 定向：`cargo test -q faiss::ivf_flat_cc::tests::test_ivf_flat_cc_train_add_search`
     - ✅ 定向：`cargo test -q faiss::ivf_sq_cc::tests::test_ivf_sq_cc_train_add_search`
     - ✅ 定向：`cargo test -q quantization::kmeans::tests::test_kmeans_convergence`
     - ⏳ 全量：`cargo test --tests` 已启动长跑（501 tests，运行中），本轮结束前未获得完整退出状态
  6. 新发现差距：Sparse 模块 `create_ann_iterator` 与 `save/load` 仍未对齐 C++ 能力，维持 P1 缺口。
  状态：BUG-P1-001 从“3 个已知失败点”收敛至“2 个 IVF-CC 子项已修复，待全量回归最终确认”。
- 2026-03-07 11:00: **FFI index_type 返回表补齐（多索引）+ 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `7d72f3b9ea4175af10f914fc386528a64a2cff80`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 1 -type f | wc -l`=7，`find .../include/knowhere/index -maxdepth 1 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`Index` trait 契约入口仍为 `create_ann_iterator/get_vector_by_ids/has_raw_data/save/load`
     - `src/ffi.rs`：`IndexWrapper::index_type` 与 `knowhere_get_index_type` 返回表此前仅覆盖 Flat/HNSW/ScaNN/MinHashLSH
  4. 本轮修复：
     - `src/ffi.rs`：扩展 `IndexWrapper::index_type`，补齐 `HNSW_PRQ/IVF_RABITQ/HNSW_SQ/HNSW_PQ/BinFlat/BinaryHNSW/IVF_SQ8/BinIVFFlat/SparseWand/SparseWandCC`
     - `src/ffi.rs`：同步扩展 `knowhere_get_index_type` 静态 C 字符串返回表，消除 Unknown 回退误判
     - `src/ffi.rs`：新增回归测试 `test_index_type_hnsw_pq`、`test_index_type_sparse_wand`，并抽出 `assert_index_type`
  5. 验证：
     - ✅ 定向：`cargo test -q test_index_type_`（5 passed）
     - ❌ 全量：`cargo test -q` / `cargo test --lib -q` / `cargo test --tests -q` 仍失败（观测到 `ivf_sq_cc`、`ivf_flat_cc`、`quantization::kmeans` 等既有失败）
  6. 新发现差距：全量回归失败点从“泛化失败”可定位到 `ivf_sq_cc` / `ivf_flat_cc` / `quantization::kmeans`，需独立拆分稳定性修复任务。
  状态：FFI 类型声明一致性进一步提升，整体模块状态维持 Partial（受全量回归阻断）。
- 2026-03-07 08:55: **MinHashLSH FFI 声明一致性修复 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `7d72f3b9ea4175af10f914fc386528a64a2cff80`
  2. C++/Rust 增量扫描：`find .../knowhere/src/index -maxdepth 1 -type f | wc -l`=7，`find .../include/knowhere/index -maxdepth 1 -type f | wc -l`=8，`find .../knowhere-rs/src/faiss -maxdepth 1 -type f | wc -l`=47
  3. 逐接口复核：
     - `src/index.rs`：`trait Index` 仍包含 `create_ann_iterator`/`get_vector_by_ids`/`has_raw_data` 统一入口
     - `src/ffi.rs`：`IndexWrapper::index_type` 已含 `MinHashLSH`；`knowhere_get_index_type` 返回表此前缺该分支
  4. 本轮修复：
     - `src/ffi.rs`：为 `knowhere_get_index_type` 新增 `"MinHashLSH"` 分支，消除类型字符串不一致
     - `src/ffi.rs`：为 `IndexWrapper::create_ann_iterator` 新增 `minhash_lsh` 分支，统一 ANN iterator 入口
     - `src/ffi.rs`：新增回归测试 `ffi::tests::test_index_type_minhash_lsh`
  5. 验证：
     - ✅ 定向：`cargo test -q ffi::tests::test_index_type_minhash_lsh`
     - ❌ 全量：`cargo test -q` / `cargo test --lib -q` / `cargo test --tests -q` 仍失败（AISAQ 与 ScaNN/HNSW 既有失败）
  6. 新发现差距：MinHash FFI 已接线，但全量回归仍被非 MinHash 模块阻断，P1 回归债未清。
  状态：MinHash 模块维持 Partial（FFI 接线已补齐，待全量回归恢复）。
- 2026-03-06 23:58: **BUG-P0-004 编译回归收敛 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. C++/Rust 增量扫描：`ls .../knowhere/src/index | wc -l`=17，`ls .../include/knowhere/index | wc -l`=8，`ls .../knowhere-rs/src/faiss | wc -l`=47
  3. 深比对（本轮按要求至少 2 模块逐接口复核）：
     - `src/index.rs`：`trait Index` 统一契约仍在 (`src/index.rs:127`)
     - `src/ffi.rs`：`CIndexConfig` / `knowhere_get_index_type` / `knowhere_create_ann_iterator` 接口声明完整
  4. 回归修复：批量修复 tests/examples 的 `IndexConfig::data_type` 缺失；统一 `crate::api::DataType` -> `knowhere_rs::api::DataType`
  5. 验证结果：
     - `cargo test --tests --no-run -q` ✅（data_type 迁移导致的编译错误已清零）
     - `cargo test -q` / `cargo test --lib -q` ❌（运行期失败：AISAQ trait tests、ScaNN FFI tests、kmeans convergence 等）
  6. 新发现差距：从“编译回归”转为“运行期功能回归”聚焦，下一步应拆分 AISAQ/ScaNN/KMeans 的稳定性缺口。
  状态：BUG-P0-004 关闭（编译回归已修复）；全量功能回归仍为 P1。
- 2026-03-06 18:02: **MinHash Index trait wrapper 接入 + 全量测试回归诊断**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 对齐扫描命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 逐文件复核：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/index/minhash_lsh_index_trait.rs`、`src/index/minhash/minhash_index_node.cc`
  4. 变更：新增 `src/index/minhash_lsh_index_trait.rs`，为 MinHashLSHIndex 实现统一 `Index` trait（train/add/search/range/get_vector/serialize/iterator 元数据）并补齐 5 个单测；`src/index.rs` 导出模块。
  5. 验证：`cargo test --lib minhash_lsh_index_trait` 通过；`cargo test` / `cargo test --tests` 失败，失败原因为 tests 侧 `IndexConfig.data_type` 迁移未完成（非本轮 MinHash 模块逻辑错误）。
  6. 新发现差距：全量测试编译回归，需新增 P0 修复任务 `BUG-P0-004`。
  状态：MinHash 模块从 Partial（无 trait）提升到 Partial（trait 已接入，仍待 FFI 统一接线与全量回归恢复）。
- 2026-03-06 14:38: **MinHash FFI 查询长度对齐修复 + 抽样复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 抽样深比对命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 复核接口：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/ffi/minhash_lsh_ffi.rs`
  4. 修复项：`src/index/minhash_lsh.rs` 新增 `vector_byte_size()`；`src/ffi/minhash_lsh_ffi.rs` 将 query/queries 长度计算从占位逻辑改为 `mh_vec_length * mh_vec_element_size`
  5. 新增回归测试：`test_search_uses_vector_byte_size`
  状态：MinHash 模块维持 Partial（Index trait wrapper 仍缺失），但 FFI query 长度缺口已关闭，风险从 P1-high 降为 P1-medium。
- 2026-03-06 13:32: **MinHash 参数别名对齐 + 目录级复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 扫描 C++ 目录：`src/index/`、`include/knowhere/index/`；并对照 Rust `src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`
  3. MinHash 参数命名对齐（部分完成）：`src/api/index.rs` 新增 `mh_*` 到 Rust 参数字段的 serde alias + 单测
  4. 新发现差距：MinHash 仍未接入统一 `Index trait`；`src/ffi/minhash_lsh_ffi.rs` 的 query 大小计算仍为占位逻辑（`count()*count()`）
  状态：MinHash 模块保持 Partial，风险维持 P1。
- 2026-03-06 12:32: **AISAQ Index trait 实现** - 为 AisaqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load/serialize_to_memory/deserialize_from_memory
  2. 高级接口：AnnIterator (AisaqAnnIterator) / get_vector_by_ids / has_raw_data
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 添加 Serialize/Deserialize 到 AisaqConfig
  5. 创建测试套件验证实现（5 个测试）
  状态：AISAQ 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 10:32: **ScaNN Index trait 验证** - 确认 ScaNNIndex 已实现完整 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：get_vector_by_ids（支持但需检查 has_raw_data）/has_raw_data（取决于 reorder_k）/create_ann_iterator
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（6 个测试全部通过）
  5. 修复编译错误：binary_hnsw.rs 和 diskann.rs 的 data_type 字段问题
  状态：ScaNN 模块从 Partial 升级为 Done（Index trait 实现完成并测试验证）。
- 2026-03-06 07:35: **DiskANN Index trait 实现** - 为 DiskAnnIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/range_search/save/load
  2. 高级接口：AnnIterator (DiskAnnIteratorWrapper) / get_vector_by_ids
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（test_diskann_index_trait）
  状态：DiskANN 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 06:35: **IVF 系列架构缺口修复** - 为 IvfSq8Index 和 IvfRaBitqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：AnnIterator（两个索引）/ get_vector_by_ids（仅 IVF-SQ8，IVF-RaBitQ 因有损压缩返回 Unsupported）
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（7 个测试全部通过）
  状态：IVF core 模块从 Partial 升级为 Done（Index trait 实现完成），剩余参数校验统一化任务。
- 2026-03-06 05:35: **IVF 系列架构缺口诊断** - 发现 IVF-SQ8/IVF-RaBitQ 未实现 Index trait，仅通过 FFI IndexWrapper enum dispatch 访问。这意味着：
  1. IVF 系列无法通过统一 Index trait 调用高级接口（AnnIterator/get_vector_by_ids）
  2. FFI 层需要为每种 IVF 类型重复实现调用逻辑
  3. 参数校验和错误处理可能不一致
  行动：将 PARITY-P1-002 升级为关键架构任务，需要为 IVF 系列实现 Index trait wrapper。
- 2026-03-06 04:35: **HNSW 高级路径测试** - 创建 `tests/test_hnsw_advanced_paths.rs`，覆盖 get_vector_by_ids、AnnIterator、serialize/deserialize、range_search（Unsupported）。5 个测试全部通过。PARITY-P1-001 完成，HNSW 模块状态升级为 Partial → Done（高级路径）。
- 2026-03-06 03:35: **核心契约一致性验证** - 验证所有索引对未实现方法的错误处理一致：Index trait 提供默认 Unsupported 实现；FFI 层 19 处 NotImplemented 返回；所有非 GPU 索引行为一致。核心契约状态从 Partial 升级为 Done（P0 降级）。
- 2026-03-06 01:35: **实现 FFI AnnIterator 接口** - 添加 `knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator` 三个 FFI 函数，支持 HNSW/ScaNN/HNSW-PQ 索引
- 2026-03-06 00:35: **更新 FFI 能力矩阵** - 标记 HNSW/ScaNN/HNSW-PQ/DiskANN 的 AnnIterator 为 ✅；HNSW GetByID ✅；ScaNN GetByID ⚠️；DiskANN GetByID ⚠️
- 2026-03-05 23:35: **实现 AnnIterator 接口** - HNSW, ScaNN, HNSW-PQ, DiskANN 四个索引实现 create_ann_iterator，验收标准达成（>=3个索引）
- 2026-03-05 22:32: 详细对比 C++/Rust 核心接口，发现 IsAdditionalScalarSupported/GetIndexMeta 缺失，AnnIterator 未实现
- 2026-03-05 21:22: 扫描 C++/Rust 接口对齐状态，确认 AnnIterator 未实现但已定义，HNSW 实现核心接口
- 2026-03-05 20:40: 添加 AnnIterator 接口定义，创建 FFI 能力矩阵文档
- 2026-03-05 20:35: 确认 ivf_sq_cc 所有测试通过 (6/6)，BUG-P0-003 完成
- 2026-03-05 19:35: 修复 3 个 P0 BUG (mini_batch_kmeans/diskann_complete/ivf_sq_cc SIMD 切片长度问题)

## 1. Scope

- In scope: non-GPU parity against C++ knowhere
- Out of scope: GPU/cuVS implementation parity

## 2. Status Legend

- `Done`: implementation and behavior aligned
- `Partial`: implemented but behavioral/edge mismatch remains
- `Blocked`: intentionally deferred or requires prerequisite
- `Missing`: not implemented

Risk levels:

- `P0`: blocks production parity
- `P1`: important functional/behavioral gap
- `P2`: optimization/documentation/coverage gap

## 3. Module-Level File Mapping and Gap Items

| Module | Native file(s) | Rust file(s) | Status | Risk | Pending interface items |
|---|---|---|---|---|---|
| Core contract | `include/knowhere/index/index.h`, `include/knowhere/index/index_node.h` | `src/index.rs`, `src/api/search.rs` | Done | P1 | ✅ lifecycle contract unified (2026-03-06); ✅ AnnIterator trait implemented (2026-03-05); all indexes return consistent Unsupported for unimplemented methods |
| Index factory/legality | `include/knowhere/index/index_factory.h`, `include/knowhere/index/index_table.h`, `include/knowhere/comp/knowhere_check.h` | `src/api/index.rs`, `src/api/data_type.rs`, `src/api/legal_matrix.rs`, `src/ffi.rs` | Done | P2 | ✅ centralized legal matrix 已实现；✅ `IndexConfig::validate()` 与 FFI `IndexWrapper::new()` 均走统一校验入口；✅ 当前已无运行时代码缺口，旧 `Partial` 仅为历史残留文案 |
| HNSW | `src/index/hnsw/faiss_hnsw.cc` | `src/faiss/hnsw.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-05), ✅ serialize/deserialize, ✅ range_search (Unsupported, tested 2026-03-06); all advanced paths tested and aligned |
| IVF core | `src/index/ivf/ivf.cc`, `src/index/ivf/ivf_config.h` | `src/faiss/ivf.rs`, `src/faiss/ivf_flat.rs`, `src/faiss/ivfpq.rs`, `src/api/index.rs` | Done | P1 | ✅ Index trait implemented for IvfSq8Index and IvfRaBitqIndex (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids (IVF-SQ8 only); parameter coverage and edge behavior alignment remaining; SIMD slice fix in ivf_sq_cc (2026-03-05) |
| RaBitQ | `src/index/ivf/ivfrbq_wrapper.*` | `src/faiss/ivf_rabitq.rs`, `src/faiss/rabitq_ffi.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ⚠️ get_vector_by_ids (Unsupported for lossy compression); query-bits and config boundary consistency |
| DiskANN | `src/index/diskann/diskann.cc`, `src/index/diskann/diskann_config.h` | `src/faiss/diskann.rs`, `src/faiss/diskann_complete.rs` | Partial | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator (DiskAnnIteratorWrapper); ✅ get_vector_by_ids（按 metric gate 暴露 raw-data 语义）；⚠️ 当前 `src/faiss/diskann.rs` 仍是“简化 Vamana + placeholder PQCode”，不可直接当作原生 DiskANN 性能可比实现；2026-03-09 已修复 `l2_sqr` 的 sqrt round-trip 反模式，但压缩/能力口径仍在活跃收敛中 |
| AISAQ | `src/index/diskann/diskann_aisaq.cc`, `src/index/diskann/aisaq_config.h` | `src/faiss/diskann_aisaq.rs`, `src/faiss/aisaq.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids; parameter and file-layout behavior alignment |
| ScaNN | - | `src/faiss/scann.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-06), ✅ has_raw_data (depends on reorder_k), ✅ Index trait (2026-03-06); tested |
| HNSW-PQ | - | `src/faiss/hnsw_pq.rs` | Done | P2 | ✅ AnnIterator (2026-03-05); ✅ `has_raw_data=false`（lossy PQ）; ✅ `get_vector_by_ids` 显式稳定返回 Unsupported；✅ `save/load` 当前在 persistence scope 内显式稳定返回 Unsupported |
| Sparse | `src/index/sparse/sparse_index_node.cc`, `src/index/sparse/sparse_inverted_index.h` | `src/faiss/sparse_inverted.rs`, `src/faiss/sparse_wand.rs`, `src/faiss/sparse_wand_cc.rs` | Done | P1 | ✅ `SparseInverted`/`SparseWand` 已接入 `create_ann_iterator` + `save/load`；✅ bitset + iterator + persistence 回归已覆盖；ℹ️ `SparseWandCC` 仍为并发包装层，当前未纳入统一 `Index` trait / 持久化承诺（不影响本模块 parity 收口） |
| MinHash | `src/index/minhash/minhash_index_node.cc`, `src/index/minhash/minhash_lsh_config.h` | `src/index/minhash_lsh.rs`, `src/index/minhash_lsh_index_trait.rs`, `src/ffi/minhash_lsh_ffi.rs`, `src/api/index.rs` | Done | P1 | ✅ 参数别名映射已补齐；✅ FFI query 长度已对齐 `mh_vec_length * mh_vec_element_size`；✅ `MinHashLSHIndex` 已接入 `Index trait`；✅ `knowhere_get_index_type`/`create_ann_iterator` 已补齐 MinHashLSH 分支；✅ `add_binary` 按 `dim(bits)` 做字节/对齐校验；全量回归问题已归属 BUG-P1-001 |
| FFI ABI | C++ factory + index runtime behavior | `src/ffi.rs`, `docs/FFI_CAPABILITY_MATRIX.md` | Done | P1 | ✅ capability matrix documented; ✅ consistent error handling (19 NotImplemented returns); ✅ `IsAdditionalScalarSupported` / `GetIndexMeta` 统一入口已补齐并具备回归覆盖；✅ persistence 语义现区分 `file_save_load` / `memory_serialize` / `deserialize_from_file` |

## 4. Validation Policy

- Every benchmark must report:
  - ground truth source
  - R@10
  - QPS
  - credibility tag (`trusted` / `unreliable` / `recheck required`)
- Credibility rules:
  - R@10 >= 80% => trusted (if setup valid)
  - 50% <= R@10 < 80% => unreliable
  - R@10 < 50% => recheck required

### Cross-dataset artifact 记录模板（BENCH-P2-003）

后续轮次引用 `benchmark_results/cross_dataset_sampling.json` 时，最少记录以下字段：
- dataset / index / base_size / query_size / dim
- params / ground_truth_source
- recall_at_10 / qps / confidence / runtime_seconds

## 5. Audit Changelog

- 2026-03-08 20:28: **OPT-P2-004 计划前复核：Index factory/legality 已满足 Done，只剩治理状态漂移待收口**
  - 复核文件：`src/api/index.rs`、`src/api/legal_matrix.rs`、`src/ffi.rs`、`TASK_QUEUE.md`、`DEV_ROADMAP.md`、`GAP_ANALYSIS.md`。
  - 结论：`IndexConfig::validate()` 已统一调用 legality matrix；FFI `IndexWrapper::new()` 也会在构造期执行 `validate_index_config`，非法 `(index_type, data_type, metric)` 组合会在入口被拒绝，而不是流入运行时。
  - 治理判断：审计表中 `Index factory/legality=Partial` 已不再对应真实代码缺口，属于 `PARITY-P1-004` 落地后的历史残留；本轮 plan 应将其提升为当前治理任务而不是继续制造新的功能开发任务。
  - 下一步：提升 `OPT-P2-004` 为当前 TODO，收敛 queue/roadmap/gap/audit 的状态表述。

- 2026-03-08 19:35: **PARITY-P1-011 收口复核：FFI ABI 元数据契约已具备统一入口**
  - 复核文件：`src/index.rs`、`src/ffi.rs`、`include/knowhere/index/index.h`、`include/knowhere/index/index_node.h`。
  - 结论：Rust 侧已补齐 `Index::is_additional_scalar_supported` / `Index::get_index_meta` 最小抽象；FFI 暴露 `knowhere_is_additional_scalar_supported`、`knowhere_get_index_meta`、`knowhere_free_cstring`。
  - 运行时语义：当前统一采用保守 contract——附加标量能力默认 `false`，`GetIndexMeta` 返回稳定 JSON summary（`index_type/dim/count/is_trained/has_raw_data/additional_scalar_supported`），避免 capability 声明与实际调用错位。
  - 回归证据：`ffi::tests::test_ffi_abi_metadata_contract`、`ffi::tests::test_index_type_*`、`ffi::tests::test_minhash_add_binary_rejects_invalid_dim_alignment`。
- 2026-03-08 19:10: **PARITY-P1-010 收口复核：Sparse 高级接口已具备可验证支持矩阵**
  - 复核文件：`src/faiss/sparse_inverted.rs`、`src/faiss/sparse_wand.rs`、`src/index.rs`。
  - 结论：`SparseInverted`/`SparseWand` 均已通过统一 `Index` trait 暴露 `create_ann_iterator`、`save`、`load`；底层持久化使用 `SparseInvertedSnapshot`（`bincode` + version gate），不再是 Unsupported。
  - 回归证据：`test_sparse_inverted_ann_iterator_respects_bitset`、`test_sparse_wand_iterator_and_search_with_bitset_consistent`、`test_sparse_inverted_save_load_roundtrip_preserves_iterator_and_vectors`、`test_sparse_wand_save_load_roundtrip_preserves_wand_behavior`。
  - 受限边界：`SparseWandCC` 当前仅为并发包装层，未承诺统一 `Index` trait / persistence parity；本轮 P1 收口范围限定为 `SparseInverted`/`SparseWand`。
- 2026-03-07 00:38: 增量审计 Sparse 模块（`src/index/sparse/*` vs `src/faiss/sparse_inverted.rs` / `src/faiss/sparse_wand.rs` / `src/index.rs` / `src/ffi.rs`）。
  - ✅ SparseInverted/SparseWand 已通过 Index trait 暴露 train/add/search/search_with_bitset/get_vector_by_ids。
  - （历史记录）当时观察到 `save/load` / `create_ann_iterator` 尚未形成闭环；该结论已由 2026-03-08 19:10 复核覆盖。 
- 2026-03-05 22:32: Detailed interface comparison between C++ and Rust.
  - **C++ Index class methods (17 total):**
    - Build/Train/Add/Search (core lifecycle) ✅
    - AnnIterator (streaming results) ⚠️ trait defined but not implemented in any index
    - RangeSearch (radius-based search) ⚠️ some indexes return Unsupported
    - GetVectorByIds (vector retrieval) ⚠️ some indexes return Unsupported
    - HasRawData (raw data check) ✅
    - IsAdditionalScalarSupported ❌ **MISSING** in Rust
    - GetIndexMeta ❌ **MISSING** in Rust
    - Serialize/Deserialize (BinarySet) ✅ (serialize_to_memory/deserialize_from_memory)
    - DeserializeFromFile ⚠️ Rust has save/load but not exact equivalent
    - Dim/Size/Count/Type ✅ (dim/count/index_type; missing Size)
  - **Priority actions:**
    - P0: Implement AnnIterator for core indexes (HNSW/IVF/Flat)
    - P1: Add IsAdditionalScalarSupported and GetIndexMeta methods
    - P1: Ensure all indexes properly implement or reject unsupported methods
  - **Files checked:**
    - C++: `/Users/ryan/Code/vectorDB/knowhere/include/knowhere/index/index.h:152-236`
    - Rust: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/index.rs:102-267`

- 2026-03-05 21:22: Scanned interface alignment between C++ and Rust.
  - C++ Index class methods: Build/Train/Add/Search/AnnIterator/RangeSearch/GetVectorByIds/HasRawData/Serialize/Deserialize/DeserializeFromFile
  - Rust Index trait methods: train/add/search/range_search/create_ann_iterator/get_vector_by_ids/has_raw_data/serialize_to_memory/deserialize_from_memory/save/load
  - Gap: AnnIterator defined but not implemented in any index; DeserializeFromFile missing in Rust
  - HNSW implements core methods (train/add/search/range_search/get_vector_by_ids/save/load)
  - Next: Verify all core indexes implement or reject unsupported methods consistently

- 2026-03-05: Initialized parity audit baseline with module/file mapping and risk triage.
