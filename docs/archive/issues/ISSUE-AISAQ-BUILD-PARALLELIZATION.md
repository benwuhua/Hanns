# ISSUE-AISAQ-BUILD-PARALLELIZATION
**日期**: 2026-03-24 | **严重程度**: P0
## 问题
AISAQ 构建并行化是“并行算候选 + 串行写图/回边”，并使用图快照（`graph_size_snapshot`）做 batch 内候选搜索；这与 native 的 multi_insert 两阶段并行更新模型（候选生成 + 分区回边提交）存在核心架构差距。

## Native 对比（引用文件+行号）
- AISAQ `add_with_ids` 并行候选但串行写入：`src/faiss/diskann_aisaq.rs:1003-1075`
- AISAQ 串行回边更新：`src/faiss/diskann_aisaq.rs:1065-1067`
- AISAQ 后续 refine 修复：`src/faiss/diskann_aisaq.rs:1133-1147`, `2748-2799`
- Native `multi_insert` 设计说明与并行图更新：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:935-941`
- Native 并行候选生成与任务调度：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:941-1010`
- Native 并行 backedge 分区提交：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:1059-1106`

## 影响
- Batch 快照导致候选基于“旧图”生成，后写入阶段难以获得 native 同等构图质量。
- 串行写图与串行回边在大规模下成为瓶颈，build 时间与图质量需靠额外 refine pass 补偿。
- 架构层面难以同时做到“高吞吐构建 + 高召回稳定”。

## 建议方向
- 对齐 native：改为 multi_insert 风格两阶段并行（候选生成与回边提交分离）。
- 引入分区写入/局部锁策略，避免全局串行写图。
- 将 refine 从“补丁”变为可配置的正式 build phase，并定义每 phase 的质量目标。
