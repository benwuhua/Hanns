# DiskANN Milvus 集成轮次

**数据集**：合成归一化 float32，1,000,000 × 768D，IP metric
**机器**：hannsdb-x86
**参考 raw**：`[[../benchmark_results/diskann_milvus_rs_2026-04-07]]`，`[[../benchmark_results/diskann_milvus_rs_2026-04-08]]`，`[[../benchmark_results/diskann_native_milvus_2026-04-07]]`

---

## 轮次总表

| Round | Commits | Serial QPS | c=80 QPS | 关键变更 |
|-------|---------|-----------|---------|---------|
| R0 (stub) | `0697e6f` | — | — | FFI stub，未接真实 DiskANN |
| **R1 (FFI wired)** | `52bdf81`…`bc5ac75` | **2.4** | **13.1** | FFI 全部接通；serial QPS = 2.4（等同 FLAT）|
| **R2 (fix)** | `72e1cd8` | **11.2** | **12.6** | materialize_storage() fix → serial +4.7× |

---

## R1 详情（FFI Wiring，2026-04-07）

**诊断验证（100K×768D）**：
| Index | Build | Serial QPS |
|-------|-------|-----------|
| RS DiskANN (max_degree=56, sl=100) | 5.0s | 36.6 |
| Native FLAT (brute-force) | 2.2s | 11.2 |
| **DiskANN speedup** | — | **3.27×** |

3.27× 快于 FLAT 确认 Vamana graph search 生效（非暴力搜索）。

**R1 问题分析**：
1M 有 ~25 个 sealed segment（BATCH=10K 插入）
```
serial QPS = 1/(25 × 16ms + 10ms) = 1/410ms ≈ 2.4  ← 根因：每 segment 16ms
```
c=20/c=80 QPS 正常（批处理摊还 H 开销）。

根因：`load()` 创建 DiskStorage → PageCache 路径 → 每 segment +2ms → 25 segment × 2ms = 50ms 损失。
→ 详见 [[decisions/materialize-storage]]

---

## R2 详情（materialize_storage fix，2026-04-08）

**修复**：`PQFlashIndex::load()` 末尾加 `if pq_code_size == 0 { materialize_storage()?; }`

**结果（1M×768D, hannsdb-x86）**：

| 指标 | R1 | R2 | Δ |
|------|----|----|---|
| Insert | 157.9s | 156.3s | ≈ same |
| Build | 16.1s | 17.1s | ≈ same |
| Serial QPS | 2.4 | **11.2** | **+4.7×** |
| c=1 QPS | 2.5 | **10.8** | **+4.3×** |
| c=20 QPS | 12.7 | 12.2 | ≈ same |
| c=80 QPS | 13.1 | 12.6 | ≈ same |

**100K per-segment（R2 vs native）**：

| | RS R2 | Native | Δ |
|-|-------|--------|---|
| Serial QPS | **41.3** | 36.6 | **RS +13%** ✅ |

---

## RS R2 vs Native（1M×768D 完整对比）

| 指标 | Native | RS R2 | R2 vs native |
|------|-------:|------:|-------------|
| Insert | 151.4s | 156.3s | +3% 慢 |
| Build | 16.09s | 17.1s | +6% 慢 |
| Serial QPS | **11.44** | **11.2** | **−2%（parity）** ✅ |
| c=1 QPS | **11.10** | **10.8** | **−3%（parity）** ✅ |
| c=20 QPS | 12.22 | 12.2 | ≈ parity |
| c=80 QPS | 12.93 | 12.6 | ≈ parity |
| Recall | 1.000 | 1.000 | 完全一致 |

**结论**：R1→R2，serial/c=1 差距从 4.75× → **2%（parity）**。

---

## 参数对齐修复（2026-04-08，R2 后）

`pq_code_budget_gb` / `build_dram_budget_gb` / `disk_pq_dims` / `beamwidth` 参数通道打通。
Shim fallback 修正（max_degree 56→48，search_list_size 100→128，beamwidth 改为 value_or(8)）。
NoPQ 行为不变；PQ disk 模式现在可从 Milvus 侧触发。

---

## 相关页面

- [[decisions/materialize-storage]] — load() 路径 bug 完整分析
- [[benchmarks/authority-numbers]] — 权威数字汇总
- [[concepts/diskann]] — PQFlashIndex 架构
- [[concepts/milvus-integration]] — segment dispatch 模型
