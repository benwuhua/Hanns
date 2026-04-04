# ISSUE-AISAQ-VAMANA-DUPLICATION
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
当前仓库存在两套独立维护的 Vamana 图构建/剪枝实现：
- AISAQ 路径：`src/faiss/diskann_aisaq.rs`
- 通用 DiskANN 路径：`src/faiss/diskann.rs`

两套代码都实现了构建搜索、robust-prune、反向边更新、refine pass，但未共享核心图算法抽象，形成持续性架构债（重复修复、行为漂移、性能优化无法复用）。

## 两套实现对比（引用行号）
1. 构建主流程重复但分叉
- AISAQ：`add_with_ids()` 内部直接执行 `vamana_build_search + link_back_with_limit + vamana_refine_pass`（`src/faiss/diskann_aisaq.rs:970-1162`，重点见 `1093-1096`, `1122-1123`, `1141-1146`）。
- DiskANN：`build_vamana_graph()` 内部执行 `gather_build_candidates + prune_neighbors + reverse edges + refine_graph`（`src/faiss/diskann.rs:1306-1424`，重点见 `1361-1373`, `1401-1411`, `1418-1419`）。

2. 并行化设计分叉
- AISAQ：批次并行后再串行回写/link-back；且大规模时 `link_back_with_limit` 直接 guard-return（`node_ids.len() > 100_000`）避免增量反向边维护（`src/faiss/diskann_aisaq.rs:2672-2678`）。
- DiskANN：并行构建分支使用 `RwLock<Vec<_>>` 做每节点并发写入并在线 prune（`src/faiss/diskann.rs:1333-1375`）。

3. prune 质量与参数策略分叉
- AISAQ：`robust_prune_scored()` + `run_robust_prune_pass()` 两阶段 alpha（`src/faiss/diskann_aisaq.rs:2573-2665`），并在 link-back 路径使用固定 `alpha=1.2`（`2700`, `2737`）。
- DiskANN：`prune_neighbors()` + `run_prune_pass()` 两阶段 alpha，是否饱和由配置控制（`src/faiss/diskann.rs:1766-1845`，重点见 `1788-1801`）。

4. 代码规模与维护面重复
- `diskann_aisaq.rs` 4432 行，`diskann.rs` 5164 行（`wc -l`）。
- 两份超大文件均包含 Vamana build/search/prune/refine 逻辑，导致同类 bug/perf 优化需要双份落地。

## 影响
- 功能漂移：一侧修复（例如反向边、refine pass、并行策略）不会自动同步到另一侧。
- 性能不可复用：Vamana 热路径优化需要在两套代码重复实现与验证。
- 质量风险：相同算法概念在两处语义不一致，难以保证 recall/QPS 行为稳定。
- 研发效率下降：review、benchmark、调参都要双轨维护。

## 建议架构方向
抽取共享 `VamanaGraph` 核心层：
- 统一承载：`build/search/prune/link_back/refine` 与并行写策略（lock/flat arrays）。
- 统一参数：`R/L/alpha/saturate/build_batch_size/refine_passes` 行为语义一致。
- AISAQ 仅保留上层职责：PQ 编码、SQ8 prefilter、SSD/flash I/O、save/load 编排。
- `diskann.rs` 与 `diskann_aisaq.rs` 作为 thin wrapper 组合同一图核心，避免算法重复实现。
