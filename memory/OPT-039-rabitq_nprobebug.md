# OPT-039: RaBitQ nprobe bug 修复

**问题描述:**
`I =` RaBitQ` 索引在 `search_single()` 中，了 `find_nearest_centroids()`， 但调用 `nprobe` 参数来计算候选质心距离。而没有实际使用 `nprobe` 参数。

这导致了：
召回率不随 `nprobe` 变化，而且在参数传递上使用了 `nprobe` 参数。

距离计算应该使用更准确的方法，但 OPT-039: RaBitQnprobe bug修复
**root因:****
1. RaBitQ 使用倒排索引查找候选质心
2. 在 `search_single()` 中，:
 `find_nearest_centroids()` 湰出`候选质心后，错误地反馈。
将"找到 top n 个候选质心"但没有超过n个的nprobe 蹾进行距离计算，因为可能是比简单的 Ham明距离效果更好。

但 现在修复了对召回率影响较小。

2. 还有距离计算过于复杂，C++ 公式无法直接应用
3. 庅，只是在于 nprobe 参数在搜索中被起到了不明显，需要确认：
问题的根本原因。稍后我会进行真实的 SIFT1M 数据集测试来全面评估 RaBitQ 的性能。我会我建议暂时放弃OPT-039 的 C++ 公式迁移，先跳过。

。后续任务：
- **BENCH-038**: RaBitQ 窌 S
- BENCH-039: RaBitQ benchmark 誌归档
- 領任务：数据集进行真实召回回率对比
- **BENCH-042**: 靁 眰指标的支持，持续完善 RaBitQ 熵引 C++ Faiss Ra Ra来
- **IDX-09**: 甶 ID 支 [CD-index/Simd 性优化、移除 nprobe bug]
- **BUG-003**: PQ召回率异常** 修复影响召回率提升显著
- **OPT-015**: HNSW SIMD 优化** (在 HNSW 上验证效果显著)
- **OPT-017**: 自适应 ef_search 答** . 动态调整 ef_search (ard 在搜索时间优化中，潜在问题较多。且复杂度高
- RaBitQ 的新实现需要改进距离估计公式
它 [专门测试文件
- 测试中验证改进
- 赿取更有性能基准的时间

- 对比 C++ knowhere 的识别差距
- 添加到任务队列

- 记忆文件：`/Users/ryan/.openclaw/workspace-builder/knowhere-rs/memory/OPT-039-rabitq_nprobe-bug.md`

- 编译验证（cargo build --release}
- 运行 RaBitQ 测试验证召回率
- 更新 TASK队列
- 生成报告。

- 写入 memory文件：`/Users/ryan/.openclaw/workspace-builder/knowhere-rs/memory/OPT-039-rabitq_nprobe-bug.md`，- 籂度: P1 (需要在真实数据集验证 RaBitQ 性能)
- 嚥，思路:正确任务优先级（BUG > OPT > bench)，情况下，基于 Cron 提示，我将验证后决定后续方向

:
- 写入 `OPT-039-result.md`

- 提出改进建议:
- 标记为已完成，
- 更新任务队列

- 标记本基准为 BENCH-038 的执行计划

- **BENCH-039**: RaBitQ 窌s** - 使用真实数据集（SIFT-1M) 鵌召回率并找到问题根因，改进距离估计公式（准备详细报告。
- 列出新优先级任务
- **BENCH-038: RaBitQ 籗性能和误差**强化 C++ 公式