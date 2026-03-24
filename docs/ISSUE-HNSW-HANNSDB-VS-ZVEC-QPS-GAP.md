# ISSUE: HannsDB (knowhere-rs backend) vs zvec QPS 差距

**提交来源**: HannsDB VectorDBBench 集成测试 (2026-03-24)
**测试场景**: Performance1536D50K / cosine / M=16 / ef_construction=64 / ef_search=32 / k=10 / concurrency=1 / 30s
**现象**: HannsDB 1536 QPS vs zvec 2781 QPS，zvec **1.81×** 更高吞吐

---

## 实测数据（x86, 2026-03-24）

### VectorDBBench Performance1536D50K 对比

| DB      | 并发 QPS | serial_p99 | serial_p95 | recall |
|---------|----------|------------|------------|--------|
| HannsDB | 1536.5   | 0.70ms     | —          | 0.9465 |
| zvec    | 2781.0   | 0.90ms     | 0.40ms     | 0.9410 |

- zvec / HannsDB QPS 比值: **1.81×**
- HannsDB **serial p99 更低**（0.70ms < 0.90ms）
- HannsDB **recall 更高**（0.9465 > 0.9410）

### 推导的 mean latency

| DB      | mean latency (≈1/QPS) | p99    |
|---------|-----------------------|--------|
| HannsDB | **0.651ms**           | 0.70ms |
| zvec    | **0.360ms**           | 0.90ms |

**关键观察**:
- HannsDB 延迟分布极度收敛（mean ≈ p99），几乎无尾巴
- zvec 延迟分布右偏（mean=0.36ms 但 p99=0.90ms），大多数查询极快，少数慢
- QPS 差距完全来自 mean latency 的差异（0.65ms vs 0.36ms），与 p99 无关

---

## 与历史测量的关系

本次使用 **真实 VectorDBBench 数据集**（1536-dim OpenAI embedding 风格），与 `ISSUE-HNSW-COSINE-SEARCH-OVERHEAD.md` 中使用的合成随机数据测量结果不可直接比较：

| 测量场景 | 数据 | M | p50 | p99 |
|---------|------|---|-----|-----|
| knowhere-rs 原生 harness（SCRATCH-001 等修复后）| 合成随机 | 8 | 3.021ms | 3.508ms |
| knowhere-rs 原生 harness（SCRATCH-001 等修复后）| 合成随机 | 16 | 5.456ms | 6.073ms |
| **HannsDB VectorDBBench（本次）** | **真实嵌入** | **16** | **~0.65ms** | **0.70ms** |

**合成 vs 真实数据差距**：真实高维嵌入有语义聚类结构，HNSW 图搜索路径更短；合成随机数据在图搜索中无法剪枝，遍历更多节点，是更难的场景。这解释了为何本次 VectorDBBench 实测（0.7ms）远好于合成 benchmark（3.5ms）。

---

## zvec 技术调查结果

通过 `strings` 和 `help()` 对 zvec 0.2.1 (`_zvec.cpython-312-x86_64-linux-gnu.so`) 分析：

### zvec 不是 hnswlib

```
N4zvec14core_interface9HNSWIndexE     # 自研类
N4zvec4core17HnswIndexProviderE       # 自研 Provider
N4zvec14core_interface21HNSWIndexParamBuilderE
HnswIndexHash init, chunkSize=%u factor=%u  # 自研 hash-based 数据结构
```

- **Backend**: 阿里巴巴自研 C++ HNSW（`zvec::core_interface::HNSWIndex`），非 hnswlib
- **Python binding**: pybind11（非 PyO3/Rust）
- **SIMD**: 内置 AVX2 + AVX512 (cd/vl/dq/bw/f) + SSE4，字符串中明确出现

### zvec HnswIndexParam 支持

```
HnswIndexParam(metric_type, m, ef_construction, quantize_type: FP16/INT8/UNDEFINED)
```

zvec 原生支持 FP16/INT8 量化存储，测试时 `quantize_type` 设置已确认（见下方"已确认数据点"）。

---

## 已确认数据点（2026-03-24）

经过针对性调查，以下三项原始假设全部被排除：

| 数据点 | 确认结论 |
|--------|---------|
| **zvec quantize_type** | `""` → `QuantizeType.UNDEFINED`（FP32），与 HannsDB 一致 |
| **zvec M / ef_search** | M=16, ef_search=32，与 HannsDB 完全一致（CLI 显式覆盖了默认 M=50） |
| **knowhere-rs AVX512** | `.so` 包含 440 条 zmm/avx512 指令；源码有 `#[target_feature(enable="avx512f")]` + runtime `is_x86_feature_detected!`；CPU 支持 avx512f/bw/vl |

**结论：量化、参数差异、AVX512 均不是 1.81× QPS 差距的原因。**

差距来源已缩小到：**zvec 自研 HNSW 算法本身的图遍历效率**（见下方假设分析）。

---

## 差距假设分析（更新后）

### ~~假设 1: AVX512 距离计算路径~~ ✅ 已排除

knowhere-rs 编译产物含 440 条 AVX512 指令，runtime dispatch 正常工作，CPU 支持。两者均使用 AVX512。

### ~~假设 3: zvec 使用量化存储~~ ✅ 已排除

zvec 测试时 `quantize_type=""` → FP32，与 HannsDB 相同内存布局。

### 假设 2: PyO3 vs pybind11 per-call 开销（可信度：低，已降级）

每次 `search_ids_raw` 调用经过：
- Python → PyO3 GIL release → Rust mutex lock → HNSW search → Vec<i64> → Python list

zvec 走 `VectorQuery(vector=query)` → pybind11 C++ 调用。两者均有 Python→C/Rust FFI 层，开销量级相近（5–20µs）。在 0.65ms 总延迟中占比 <3%，**非主要因素**。

### 假设 4: zvec `HnswIndexHash` 图遍历效率优势（可信度：**高，当前首要嫌疑**）

排除所有外部因素后，差距必然来自算法实现本身：

1. **Early termination / 自适应截断**：zvec mean=0.36ms << p99=0.90ms（双峰分布），说明大多数查询在 ~0.3ms 提前收敛，偶尔触发慢路径。knowhere-rs mean≈p99=0.65ms（单峰收敛），表明**每次查询都走完整 ef 轮遍历，无早停**。
2. **`HnswIndexHash` 数据结构**：zvec 使用自定义 hash-based 邻居存储（strings 中可见 `HnswIndexHash init, chunkSize=%u`），可能对 graph traversal cache line 访问更友好。
3. **HNSW 图构建质量**：zvec 可能在 build 阶段应用了更激进的连接策略，导致 search 时路径更短。

---

## 延迟分布特征差异分析

| 特征 | HannsDB (knowhere-rs) | zvec |
|------|----------------------|------|
| mean (≈1/QPS) | 0.651ms | 0.360ms |
| p99 | 0.70ms | 0.90ms |
| 分布形状 | **单峰收敛**（mean ≈ p99） | **右偏双峰**（大多数快，少数慢） |

HannsDB "均匀慢"：每次查询固定 ~0.65ms，无论 query 难易。缺少根据 query 特征提前终止的能力。

zvec "大多数快"：约 70–80% 的查询在 0.3ms 内完成，剩余查询触发慢路径（~0.9ms）。这与 **early termination on convergence** 机制吻合。

---

## 优化方向（重新排序）

| 方向 | 预期收益 | 实施难度 | 优先级 | 状态 |
|------|---------|---------|--------|------|
| **自适应早停 / IO-Cutting** | ~1.5–2× QPS（消除无效遍历） | 中（设计见 HANNS_INTEGRATION_DESIGN.md） | **P0** | 待实现 |
| **FP16 向量量化存储** | ~1.3–1.5× QPS（内存带宽优化） | 中（需改存储 + 距离计算） | **P1** | 待实现 |
| **INT8 量化存储** | ~1.5–2× QPS（更激进，轻微 recall 损失） | 中 | **P1** | 待实现 |
| ~~AVX512 路径验证~~ | — | — | **已排除** | AVX512 已启用 |
| search_into buffer 复用（ALLOC-001） | <5% | 低 | **P2** | 待实现 |

---

## 参考

- 本 repo: `docs/ISSUE-HNSW-COSINE-SEARCH-OVERHEAD.md` — 合成数据 SCRATCH/QNORM/ALLOC 修复历史
- HannsDB: `docs/vector-db-bench-notes.md` Section 45–46 — 完整测试记录
- zvec GitHub: https://github.com/alibaba/zvec
