# Builder 任务归档

本文件承接原 `TASK_QUEUE.md` 的历史明细，主队列只保留当前大任务面板。

## 归档原则

- 已完成的 `P0 / P1 / P2` 任务不再留在主队列中
- 详细历史以 git 历史和专项文档为准：
  - `DEV_ROADMAP.md`
  - `GAP_ANALYSIS.md`
  - `docs/PARITY_AUDIT.md`
  - `benchmark_results/*`

## 已归档阶段

- `P0`: 核心编译/契约回归恢复
- `P1`: parity closure（HNSW / IVF / DiskANN / ScaNN / Sparse / MinHash）
- `P2`: benchmark / regression / artifact groundwork

## 当前使用方式

- 若只需要知道“现在该做什么”，只看 `TASK_QUEUE.md`
- 若需要追溯“之前做过什么”，再看本文件和相关设计/审计文档
