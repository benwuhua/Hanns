# ISSUE-AISAQ-BEAM-TERMINATION
**日期**: 2026-03-24 | **严重程度**: P1
## 问题
AISAQ 的 beam 搜索主循环主要依赖 `max_visit` 和可选 `search_io_limit` 停止，缺少基于当前 top-k 上界/前沿下界的早停机制；在高 nprobe/L 或长尾 query 下容易出现无效扩展。

## Native 对比（引用文件+行号）
- AISAQ 搜索主循环与停止条件：`src/faiss/diskann_aisaq.rs:1634-1644`, `1635-1639`, `1718-1723`
- AISAQ rerank 在主循环后统一进行：`src/faiss/diskann_aisaq.rs:1736-1757`
- Native 搜索循环支持 `terminate_early()`：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:2049`
- Native 查询访问决策支持 `Terminate`：`/Users/ryan/Code/DiskANN/diskann/src/graph/index.rs:68-77`, `79-90`

## 影响
- 增加不必要的节点展开和距离计算，QPS 受损，I/O 压力增大。
- 在同等召回下需要更高访问预算，整体成本高于 native。

## 建议方向
- 引入 frontier-bound 早停：当 frontier 最优候选已劣于当前 top-k 最差且无潜在改进时提前退出。
- 将终止策略抽象为可配置策略（预算型、质量型、混合型），并与 `search_io_limit` 协同。
- 在 profile 中新增“有效扩展率/早停触发率”指标，作为迭代优化依据。
