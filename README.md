# Hanns

**Hanns** 是一个纯自研的高性能 Rust 近似最近邻（ANN）检索库，面向 CPU 向量检索场景。

不是对任何已有库的移植或封装——每一行核心代码均从零开始自主实现，并在真实工业数据集上持续以 C++ FAISS/KnowWhere 为基准进行性能对标。

---

## 核心优势

- **纯 Rust 实现**：无 C/C++ 依赖，无 unsafe FFI 包装。完整的类型安全、内存安全保证。
- **多算法覆盖**：HNSW、IVF-Flat、IVF-SQ8、IVF-USQ、IVF-PQ、DiskANN/AISAQ (flash)、ScaNN、Sparse WAND、Binary。
- **SIMD 深度优化**：关键距离计算路径覆盖 AVX2、AVX-512、AVX512VNNI，在 x86 上充分发挥现代 CPU 指令集能力。
- **x86 权威测试**：所有性能数字均来自 x86 服务器（非 Apple Silicon），使用真实工业数据集（SIFT-1M、Cohere Wikipedia-1M）。
- **领先而非追平**：在主力索引上已实现对 KnowWhere C++ native 的量化性能超越。

---

## 性能对比（SIFT-1M，x86 authority，2026-03）

> 全部使用 `target-cpu=native` 编译，batch parallel 查询模式。
> 对比基准：KnowWhere C++ native（8 线程）。

### HNSW（M=16, ef_construction=200）

| ef | Recall@10 | Hanns batch QPS | Native 8T QPS | 倍率 |
|----|-----------|-----------------|---------------|------|
| 60 | **0.972** | **17,814** | 15,918 | **+11.9% ✅** |
| 138 | **0.995** | **17,910** | 15,918 | **+12.5% ✅** |

> 根因：修复 `search_layer_idx_shared` 中无条件 `Instant::now()` 调用（22.57% 时间占比），改为 profiling 按需采样。效果：15,043 → 17,319 QPS（+15.1%）。

### IVF-Flat（nlist=1024）

| nprobe | Recall@10 | Hanns batch QPS | Native 8T QPS | 倍率 |
|--------|-----------|-----------------|---------------|------|
| 32 | **0.978** | **3,429** | 721 | **+5.23x ✅** |
| 32 | 0.978 | 2,339 (serial) | 341 (1T) | **+6.9x ✅** |

### IVF-SQ8（nlist=1024，fused AVX2 decode_dot）

| nprobe | Recall@10 | Hanns batch QPS | Native 8T QPS | 倍率 |
|--------|-----------|-----------------|---------------|------|
| 32 | **0.958** | **11,717** | 8,278 | **+1.42x ✅** |

> SQ8 优化亮点：AVX2 FMA 融合 u8 decode + dot product，zero-alloc one-pass 路径。

### IVF-USQ（Unit Sphere Quantizer，768D Cohere 1M）

| 精度档 | nprobe | Recall@10 | QPS |
|--------|--------|-----------|-----|
| 4-bit (8x 压缩) | 10 | 0.833 | **3,706** |
| 8-bit (4x 压缩) | 10 | 0.905 | **2,980** |
| 8-bit (4x 压缩) | 32 | **0.968** | 1,011 |

> USQ 使用 AVX512VNNI `_mm512_dpbusd_epi32` 做整数点积，全面替代旧版 HVQ/ExRaBitQ 实现（QPS +3~222%）。

### AISAQ / DiskANN Flash（SIFT-1M，mmap 模式）

| 模式 | Recall@10 | QPS |
|------|-----------|-----|
| NoPQ L=64 | **0.979** | **5,806** |
| PQ32 | 0.768 | **10,227** |

---

## 索引一览

| 索引 | 状态 | 典型 Recall@10 | 特性 |
|------|------|----------------|------|
| **HNSW** | ✅ 领先 | 0.972 (SIFT-1M) | 图遍历；cosine TLS scratch 零分配 |
| **HNSW-SQ** | ✅ 可用 | 0.992 | 整数预计算 ADC 路径 |
| **IVF-Flat** | ✅ 领先 | 0.978 (SIFT-1M) | 5x+ 快于 native 8T |
| **IVF-SQ8** | ✅ 领先 | 0.958 (SIFT-1M) | 1.42x 快于 native 8T；AVX2 fused decode |
| **IVF-USQ** | ✅ 可用 | 0.905~0.968 (Cohere) | AVX512VNNI；1/4/8-bit 三档统一量化器 |
| **IVF-PQ** | ✅ 可用 | 受 m 字节数限制 | m=32: 0.720 (synthetic) |
| **AISAQ (DiskANN Flash)** | ✅ 可用 | 0.979 NoPQ (SIFT-1M) | 按需 pread 磁盘模式；Vamana 图构建 |
| **ScaNN** | ✅ 可用 | 0.969 | reorder_k=1600 超过 0.95 门限 |
| **Sparse / WAND** | ✅ 可用 | 1.0 | 稀疏向量检索 |
| **Binary** | ✅ 完整 | — | Hamming 距离 |

---

## 量化子系统

```
src/quantization/
  usq/           — UsqQuantizer：QR 正交旋转 + 1/4/8-bit 统一量化
    config.rs    — UsqConfig { dim, nbits, seed }
    rotator.rs   — QR decomposition 正交旋转矩阵
    quantizer.rs — 训练 + SIMD 评分
    layout.rs    — SoA 存储 + fastscan 转置
    fastscan.rs  — AVX512 快速扫描 + topk
    searcher.rs  — 两阶段粗排 + 精排
  pq/            — Product Quantizer：并行 k-means
  sq/            — Scalar Quantizer：SQ8/SQ4
```

---

## 构建

```bash
# 开发构建
cargo build

# Release（启用 LTO + codegen-units=1 + target-cpu=native）
cargo build --release

# 运行所有测试
cargo test

# 运行 benchmark（本地快速预筛）
cargo run --example benchmark --release
```

**注意**：`.cargo/config.toml` 在 x86_64/aarch64 上自动启用 `target-cpu=native`，充分利用 AVX2/AVX-512 等指令集。

---

## 目录结构

```
src/
  faiss/         — 核心索引实现（HNSW, IVF-*, AISAQ, ScaNN, Sparse, Binary）
  quantization/  — 量化子系统（USQ, PQ, SQ）
  ffi/           — FFI 接口层
tests/           — 集成测试 + 回归测试
benches/         — Criterion 微基准
examples/        — 完整 benchmark 示例
docs/            — 设计文档、性能审计、对比报告
benchmark_results/ — 权威 verdict 归档（JSON）
scripts/remote/  — x86 远程构建/测试脚本
```

---

## 测试数据集

- **SIFT-1M**：标准 128-dim L2 ANN 基准，100万向量
- **Cohere Wikipedia-1M**：768-dim IP 真实嵌入向量，100万条维基百科段落
- **SimpleWiki-OpenAI-260K**：3072-dim，OpenAI text-embedding-3-large

---

## 性能权威机器

所有最终性能数字均在 x86 服务器上产出（非本地 Apple Silicon）。本地 Mac 仅用于快速迭代预筛，不作为最终 acceptance 依据。
