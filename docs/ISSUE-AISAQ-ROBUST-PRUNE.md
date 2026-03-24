# ISSUE-AISAQ-ROBUST-PRUNE
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
AISAQ 当前 `robust_prune_scored` 是简化版两段 occlusion + 兜底填充，且回边更新在大图规模被 guard 截断（`>100_000` 直接 return）。这与 native DiskANN 的 prune 语义完整性不一致，容易在高规模下出现邻接质量退化与连通性不稳定。

## Native 对比（引用文件+行号）
- AISAQ 简化 prune 主体：`src/faiss/diskann_aisaq.rs:2573-2665`
- AISAQ 回边 guard：`src/faiss/diskann_aisaq.rs:2667-2678`
- Native prune 主流程：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:2628-2661`
- Native list prune 与失败/重试语义：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:2683-2738`
- Native occlusion/saturate 选项：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:2770-2834`

## 影响
- 构图阶段边选择质量与 native 偏差变大，特别在 100K+ 规模更敏感。
- 回边被 guard 截断后，图修复依赖后处理 pass，结果受 batch/数据分布影响较大。
- 召回率和稳定性在大规模场景更容易抖动。

## 建议方向
- 对齐 native 的 prune 语义：统一 `robust_prune`/`robust_prune_list` 行为与选项模型（含 saturate/force-saturate）。
- 将回边策略从“硬 guard + 事后补救”迁移为“可控限流 + 分阶段一致 prune”。
- 增加 1M 规模下构图质量回归指标（度分布、连通分量、入口可达率）作为 CI gate。
