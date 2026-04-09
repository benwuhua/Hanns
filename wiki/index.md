# hanns Wiki — Index

> LLM 维护。最后更新：2026-04-09。

**项目目标**：Rust 实现的 Milvus KnowHere，在核心 CPU 路径上全面超越原生 C++。

---

## 核心裁决（Current Verdicts）

| Index | 状态 | 关键指标 |
|-------|------|---------|
| HNSW | ✅ **LEADING** | Milvus c=80: 1042 QPS vs native ~800 (+30%); standalone batch: 17,814 vs native 15,918 (+11.9%) |
| HNSW-PCA-SQ / HNSW-PCA-USQ | 🔄 **集成中** | FFI 接通，待 Milvus 验证 |
| DiskANN (Milvus) | ✅ **PARITY** | serial 11.2 vs native 11.4 QPS (−2%，误差范围内) |
| DiskANN-PCA-USQ | 🔄 **集成中** | FFI 接通，待 Milvus 验证 |
| AISAQ (standalone) | ✅ **LEADING** | NoPQ 6,062 QPS, recall 0.9941 (SIFT-1M); build 244s vs native 1595s (6.5x faster) |
| IVF-Flat | ✅ **LEADING** | batch 3,429 QPS vs native 8T 721 QPS (4.76x) |
| IVF-SQ8 | ✅ **LEADING** | batch 11,717 vs native 8T 8,278 QPS (1.42x) |
| IVF-PQ | ⚠️ recall 上限 | 由量化字节数 m 决定；非 bug |
| ScaNN | ✅ parity | recall 0.969 @reorder_k=1600 |
| Sparse/WAND | ✅ parity | — |

---

## Concepts

- [[concepts/hnsw]] — HNSW 算法 + RS 实现 + 优化历史
- [[concepts/diskann]] — PQFlashIndex 架构 + NoPQ/PQ 路径 + 参数说明
- [[concepts/milvus-integration]] — Milvus segment dispatch + CGO executor + shim ABI
- [[concepts/vdbbench]] — VectorDBBench 使用指南：CLI 命令、数据集、RS vs Native 对比流程

> **注**：IVF 系列（Flat/SQ8/PQ）的概念页尚未建立。权威数字见 [[benchmarks/authority-numbers]] §IVF。

## Benchmarks

- [[benchmarks/authority-numbers]] — 所有 index 的权威数字总表
- [[benchmarks/hnsw-milvus-rounds]] — HNSW Milvus 集成优化 R0–R8 完整历史
- [[benchmarks/diskann-milvus-rounds]] — DiskANN Milvus 集成 R1–R2 历史

## Decisions（经验教训）

- [[decisions/optimization-log]] — 优化尝试全记录（成功 + 失败）
- [[decisions/rayon-private-pool]] — 为什么不能用 rayon 全局池
- [[decisions/prefetch-l1-thrash]] — prefetch 为什么让 QPS 减半
- [[decisions/materialize-storage]] — load() 后必须 materialize_storage() 的根因

## Machines

- [[machines/hannsdb-x86]] — 权威 benchmark 机器：路径、Milvus 启动、构建命令

---

## 快速导航

- 最新 benchmark 结果：`benchmark_results/` (raw) → [[benchmarks/authority-numbers]] (compiled)
- 当前任务：`CLAUDE.md` §Next task
- 环境配置：[[machines/hannsdb-x86]]
- 代码结构：`src/faiss/` — HNSW (`hnsw.rs`), DiskANN (`diskann_aisaq.rs`), FFI (`ffi.rs`)
- PCA 变体：`hnsw_pca_sq.rs`, `hnsw_pca_usq.rs`, `diskann_pca_usq.rs`
