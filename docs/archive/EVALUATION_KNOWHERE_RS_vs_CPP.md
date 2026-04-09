# hanns vs knowhere (C++) 全面评估报告

**日期**: 2026-03-20
**目标**: 评估 hanns 作为 knowhere C++ 的生产级平替方案的当前差距，并制定后续计划

---

## 一、总体评估结论

hanns 已在 **HNSW 核心路径上取得绝对性能优势**（x86 权威基准 2.099x），但要实现**全面生产级平替**，在索引覆盖度、GPU 支持、真实数据验证、生态集成等方面仍存在显著差距。

| 维度 | 完成度 | 说明 |
|------|--------|------|
| HNSW 索引 | ★★★★★ | **领先** — 2.099x 性能优势，功能完备 |
| IVF 系列 | ★★★☆☆ | IVF-Flat/SQ8 可用，IVF-PQ 召回率不达标 |
| DiskANN | ★★★☆☆ | 功能可用但简化实现，非原生可比 |
| 稀疏索引 | ★★★★☆ | 达到对等，功能完备 |
| GPU 索引 | ☆☆☆☆☆ | **完全缺失** — C++ 有 cuVS/CAGRA/GPU-IVF |
| 量化方法 | ★★★★☆ | PQ/SQ/RaBitQ/PRQ/OPQ 全覆盖 |
| 生态集成 | ★★★☆☆ | FFI/Python/JNI 有框架，但非 Milvus 实战验证 |
| 监控可观测 | ★★☆☆☆ | 缺少 Prometheus/OpenTelemetry 集成 |

---

## 二、索引类型逐项对比

### 2.1 密集向量索引

| 索引类型 | C++ knowhere | Rust hanns | 差距评估 |
|----------|-------------|-----------------|----------|
| **Flat** | ✅ fp32/fp16/bf16/int8/bin | ✅ 完整 | 对等 |
| **HNSW** | ✅ 成熟，基于 hnswlib | ✅ **领先** 2.099x | **Rust 领先** |
| **HNSW-SQ** | ✅ | ✅ 已修复 | 对等 |
| **HNSW-PQ** | ✅ | ✅ (has_raw_data=false) | 对等 |
| **HNSW-PRQ** | ✅ | ✅ | 对等 |
| **IVF-Flat** | ✅ + CC 并发版 | ✅ + CC 并发版 | 对等 |
| **IVF-PQ** | ✅ 成熟，Faiss 级 | ❌ recall < 0.8 | **关键差距** |
| **IVF-SQ8** | ✅ | ✅ 已优化 (397 QPS) | 对等 |
| **IVF-RaBitQ** | ✅ | ✅ (refine_k=500 达标) | 对等（有折衷） |
| **IVF-OPQ** | ✅ | ⚠️ recall 不达标 | 差距 |
| **ScaNN** | ✅ (需 AVX2) | ✅ 0.969 recall | 对等 |
| **DiskANN** | ✅ 完整 SSD 分页 | ⚠️ 简化版，内存驻留 | **显著差距** |
| **AISAQ** | ✅ 完整 SSD + 缓存 | ⚠️ 功能闭环中 | 差距 |
| **PageANN** | ✅ | ❌ 未实现 | 缺失 |
| **ANNOY** | ❌ | ✅ (6722 行) | Rust 额外 |

### 2.2 GPU 索引（C++ 独有）

| GPU 索引类型 | C++ knowhere | Rust hanns | 说明 |
|-------------|-------------|-----------------|------|
| GPU_FAISS_FLAT | ✅ | ❌ | CUDA |
| GPU_FAISS_IVF_FLAT | ✅ | ❌ | CUDA |
| GPU_FAISS_IVF_PQ | ✅ | ❌ | CUDA |
| GPU_FAISS_IVF_SQ8 | ✅ | ❌ | CUDA |
| GPU_CUVS_BRUTE_FORCE | ✅ | ❌ | cuVS |
| GPU_CUVS_IVF_FLAT | ✅ | ❌ | cuVS |
| GPU_CUVS_IVF_PQ | ✅ | ❌ | cuVS |
| GPU_CUVS_CAGRA | ✅ | ❌ | cuVS |

**GPU 索引是最大的功能缺口**。knowhere C++ 通过 RAPIDS cuVS 提供了 8 种 GPU 索引，在大规模场景中 GPU 加速是核心竞争力。

### 2.3 稀疏向量索引

| 索引类型 | C++ | Rust | 状态 |
|----------|-----|------|------|
| Sparse Inverted | ✅ | ✅ | 对等 |
| Sparse Inverted CC | ✅ | ✅ | 对等 |
| Sparse WAND | ✅ | ✅ | 对等 |
| Sparse WAND CC | ✅ | ✅ | 对等 |

### 2.4 二进制向量索引

| 索引类型 | C++ | Rust | 状态 |
|----------|-----|------|------|
| BinFlat | ✅ | ✅ | 对等 |
| BinIvfFlat | ✅ | ✅ | 对等 |
| BinaryHnsw | — | ✅ | Rust 额外 |

### 2.5 特殊索引

| 索引类型 | C++ | Rust | 状态 |
|----------|-----|------|------|
| MinHash LSH | ✅ | ✅ | 对等 |
| Cardinal Tiered | ✅ (企业版) | ❌ | 缺失（企业特性） |

---

## 三、距离度量对比

| 度量类型 | C++ | Rust | 说明 |
|----------|-----|------|------|
| L2 | ✅ | ✅ | 对等 |
| IP (内积) | ✅ | ✅ | 对等 |
| COSINE | ✅ | ✅ | 对等 |
| Hamming | ✅ | ✅ | 对等 |
| Jaccard | ✅ | ✅ | 对等 |
| MH-Jaccard | ✅ | ✅ | 对等 |
| **BM25** | ✅ | ❌ | **缺失** — 全文搜索评分 |
| **Substructure** | ✅ | ❌ | 缺失 — 分子相似性 |
| **Superstructure** | ✅ | ❌ | 缺失 — 分子相似性 |
| **MAX_SIM 系列** | ✅ (6 种) | ❌ | **缺失** — 多向量检索 |
| **DTW 系列** | ✅ (6 种) | ❌ | **缺失** — 动态时间规整 |

**差距**: C++ 支持 ~20 种度量，Rust 支持 6 种。缺少的 BM25、MAX_SIM、DTW 是 Milvus 2.x 的重要特性。

---

## 四、SIMD/硬件加速对比

| 指令集 | C++ | Rust | 说明 |
|--------|-----|------|------|
| SSE/SSE2/SSE4 | ✅ | ✅ | 对等 |
| AVX2 | ✅ | ✅ | 对等 |
| AVX-512 | ✅ (F/DQ/BW/VL/ICX) | ✅ (F/BW) | 基本对等 |
| ARM NEON | ✅ | ✅ | 对等 |
| **ARM SVE** | ✅ | ❌ | 缺失 |
| **RISC-V RVV** | ✅ | ❌ | 缺失 |
| **PowerPC VSX** | ✅ | ❌ | 缺失 |

knowhere C++ 的 SIMD 覆盖范围更广（6 种架构 vs 4 种），但 ARM SVE/RISC-V/PowerPC 是边缘场景。核心 x86/ARM 路径对等。

---

## 五、搜索特性对比

| 特性 | C++ | Rust | 说明 |
|------|-----|------|------|
| Top-K 搜索 | ✅ | ✅ | 对等 |
| Range 搜索 | ✅ | ✅ | 对等 |
| Bitset 过滤 | ✅ | ✅ | 对等 |
| AnnIterator | ✅ | ✅ | 对等 |
| 中断/取消 | ✅ | ✅ | 对等 |
| 按 ID 获取向量 | ✅ | ✅ | 对等 |
| 序列化/反序列化 | ✅ | ✅ | 对等 |
| **Embedding List 搜索** | ✅ | ❌ | **缺失** — 多向量检索 |
| **Materialized View** | ✅ | ❌ | **缺失** — 标量字段过滤优化 |
| **Refinement 策略** | ✅ (SQ4/6/8/FP16/BF16/FP32) | 部分 | 差距 |
| **Mmap 支持** | ✅ | ✅ (memmap2) | 对等 |
| **Lazy Load** | ✅ | ❌ | 缺失 |

---

## 六、生态与集成对比

| 维度 | C++ | Rust | 差距 |
|------|-----|------|------|
| **Milvus 集成** | ✅ 生产验证 | ❌ 未集成 | **核心差距** |
| FFI (C ABI) | ✅ 稳定 | ✅ 有框架 (ffi.rs 7318 行) | 接近但未实战验证 |
| Python 绑定 | ✅ SWIG | ✅ PyO3 | 对等 |
| Java 绑定 | — | ✅ JNI | Rust 额外 |
| **Prometheus 监控** | ✅ | ❌ | 缺失 |
| **OpenTelemetry 追踪** | ✅ | ❌ | 缺失 |
| **Conan 包管理** | ✅ | N/A (Cargo) | 不同生态 |
| Index Factory 模式 | ✅ 成熟 | ✅ 有实现 | 对等 |

---

## 七、性能对比（权威 x86 基准）

### 7.1 HNSW（核心路径 — Rust 领先）

| 配置 | Rust QPS | C++ QPS | 比率 | Recall |
|------|---------|---------|------|--------|
| ef=60, 1M SIFT | **33,406** | 15,918 | **2.099x** | 0.953 vs 0.952 |
| ef=138, 1M SIFT | **17,251** | 15,918 | 1.08x | 0.987 vs 0.952 |
| ef=50, 10K | **28,641** | — | — | — |

### 7.2 其他索引

| 索引 | Rust QPS (x86) | 状态 | 说明 |
|------|---------------|------|------|
| PQFlash NoPQ 1M | 9,648 | 可用 | 无 C++ 对比数据 |
| PQFlash PQ32 1M | 8,002 | 可用 | 无 C++ 对比数据 |
| IVF-SQ8 100K | 397 | 达标 | recall=0.985 |
| ScaNN 100K | 28 | 达标 | recall=0.969 |
| IVF-PQ | — | **NO-GO** | recall < 0.8 |

---

## 八、差距分级总结

### P0 — 生产级平替阻断项

| # | 差距 | 影响 | 难度 | 说明 |
|---|------|------|------|------|
| 1 | **IVF-PQ 召回率 < 0.8** | 高 | 高 | Milvus 核心索引类型，必须修复。当前在随机数据上 recall ~0.62，需 SIFT-1M 真实数据验证并修复量化误差过大问题 |
| 2 | **DiskANN 非原生可比** | 高 | 高 | 简化版 Vamana 无 SSD 分页，大规模数据场景核心能力缺失 |
| 3 | **Milvus 集成验证** | 高 | 高 | FFI 层未经 Milvus 实战验证，可能存在语义不一致 |

### P1 — 功能完备性差距

| # | 差距 | 影响 | 难度 | 说明 |
|---|------|------|------|------|
| 4 | **GPU 索引完全缺失** | 高 | 极高 | 8 种 GPU 索引 (cuVS/CAGRA) 为零，大规模场景竞争力弱 |
| 5 | **BM25 度量缺失** | 中 | 中 | 全文检索场景必要 |
| 6 | **Embedding List / MAX_SIM / DTW** | 中 | 中 | Milvus 多向量检索特性 |
| 7 | **Materialized View** | 中 | 中 | 标量+向量混合过滤优化 |
| 8 | **Prometheus / OTel 可观测** | 中 | 低 | 生产监控必需 |

### P2 — 优化与加固

| # | 差距 | 影响 | 难度 | 说明 |
|---|------|------|------|------|
| 9 | IVF-OPQ 召回率不达标 | 低 | 中 | 需修复 OPQ 旋转矩阵训练 |
| 10 | DiskANN AISAQ IO 优化 | 中 | 高 | io_uring 异步 IO 路径未实现 |
| 11 | ARM SVE / RISC-V SIMD | 低 | 中 | 边缘架构支持 |
| 12 | Lazy Load 特性 | 低 | 低 | 延迟加载索引 |
| 13 | PageANN 索引 | 低 | 中 | 页面优化的 DiskANN 变体 |

---

## 九、后续计划

### 阶段一：P0 关键阻断项修复（4-6 周）

**目标**: 消除生产级平替的核心阻断

#### 1.1 IVF-PQ 召回率修复（2 周）
- [ ] 使用 SIFT-1M / GIST-1M 真实数据集替代随机数据进行基准测试
- [ ] 分析 PQ 量化误差 vs 向量距离方差的比值，定位根因
- [ ] 参考 Faiss IVF-PQ 实现对比：
  - PQ codebook 训练质量（k-means 收敛）
  - 残差计算正确性
  - ADC (Asymmetric Distance Computation) 查表实现
- [ ] 目标: recall@10 >= 0.90 (nprobe=64, SIFT-1M)
- [ ] 建立自动化回归测试

#### 1.2 DiskANN 完整实现（3-4 周）
- [ ] 实现完整 Vamana 图构建算法（RobustPrune）
- [ ] 实现 SSD 分页机制（sector-aligned read）
- [ ] 实现 PQ 压缩 + SSD 布局（与 C++ DiskANN 对齐）
- [ ] 实现频率感知缓存（BFS-based cache warmup）
- [ ] io_uring 异步 IO 路径（feature gate）
- [ ] 基准: 与 C++ DiskANN 在 SIFT-1M 上做正式对比

#### 1.3 Milvus FFI 集成验证（2 周）
- [ ] 对齐 C++ knowhere 的完整 FFI ABI 合约
- [ ] 编写 Milvus segment 层的 drop-in 替换测试
- [ ] 验证序列化格式兼容性（二进制格式 round-trip）
- [ ] 错误码和异常语义对齐

### 阶段二：功能完备性补齐（6-8 周）

**目标**: 覆盖 Milvus 2.x 全部非 GPU 特性

#### 2.1 度量类型补齐（2 周）
- [ ] BM25 评分实现（稀疏索引路径）
- [ ] MAX_SIM 系列（6 种：COSINE/IP/L2/Hamming/Jaccard + base）
- [ ] DTW 系列（6 种动态时间规整变体）
- [ ] Substructure / Superstructure（分子相似性，优先级低）

#### 2.2 Embedding List 多向量检索（2 周）
- [ ] SearchEmbList() 接口实现
- [ ] CalcDistByIDs() 暴力距离计算
- [ ] 多阶段检索管线集成

#### 2.3 Materialized View（1 周）
- [ ] 标量字段元数据集成
- [ ] 基于 MV 的过滤搜索优化路径

#### 2.4 可观测性（1 周）
- [ ] Prometheus metrics 导出（prometheus-client crate）
- [ ] OpenTelemetry 追踪集成（tracing-opentelemetry）
- [ ] 操作上下文跟踪

#### 2.5 其他搜索特性（1-2 周）
- [ ] Refinement 策略完善（SQ4/SQ6/FP16/BF16/FP32 全类型）
- [ ] Lazy Load 延迟加载机制
- [ ] 完善 Index Factory 的参数验证

### 阶段三：GPU 索引（12-16 周，可选/并行）

**目标**: 填补 GPU 能力空白（需评估 ROI）

#### 3.1 技术选型
- **方案 A**: 通过 FFI 调用 cuVS C API（工作量小，但依赖 C++ 生态）
- **方案 B**: 基于 Rust CUDA 绑定（cudarc/rustacuda）原生实现（工作量大，纯 Rust）
- **方案 C**: 暂不支持 GPU，明确定位为 "CPU-only 高性能替代"

#### 3.2 优先级排序（若选择实现）
1. GPU_CUVS_BRUTE_FORCE — 最简单，验证 GPU 通路
2. GPU_CUVS_IVF_FLAT — 最常用 GPU 索引
3. GPU_CUVS_CAGRA — 最新最快 GPU 索引
4. 其他 GPU 变体

### 阶段四：生产加固与性能冲刺（持续）

- [ ] 全索引类型的 SIFT-1M / GIST-1M / Deep-1M 权威基准
- [ ] 内存安全审计（unsafe 代码 review）
- [ ] 并发安全验证（多线程搜索 + 写入）
- [ ] 故障注入测试（OOM、磁盘满、进程崩溃恢复）
- [ ] 性能回归 CI（每次提交自动基准测试）
- [ ] IVF 系列 QPS 优化（目标接近 C++ Faiss 水平）

---

## 十、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| IVF-PQ 在真实数据上仍不达标 | 中 | 高 | 可 FFI 回退到 Faiss C++ 的 PQ 实现 |
| DiskANN SSD 性能无法达到 C++ | 中 | 高 | io_uring 是关键，可能需要 unsafe 优化 |
| GPU 索引 ROI 不高 | 高 | 中 | 明确定位 CPU-only，GPU 走 FFI 桥接 |
| Milvus 集成存在未知兼容性 | 中 | 极高 | 尽早做 drop-in 替换测试 |
| 内存安全问题 | 低 | 高 | cargo miri + AddressSanitizer + fuzzing |

---

## 十一、建议策略

### 短期（1-2 月）: 修复 P0，建立真实数据基准
优先修复 IVF-PQ 和 DiskANN，用 SIFT-1M 建立全索引基准线。

### 中期（3-4 月）: 功能补齐，Milvus 集成
补齐度量类型和搜索特性，开始 Milvus drop-in 替换测试。

### 长期（5-6 月+）: GPU 决策，生产部署
根据实际需求决定 GPU 策略。完成生产加固，进入灰度部署。

### 定位建议
hanns 最合理的定位是 **"CPU-first 高性能向量搜索引擎"**：
- 在 HNSW 这一最常用索引上已有 **2x 性能优势**
- CPU 路径全覆盖后可作为 Milvus 的 **非 GPU 节点默认引擎**
- GPU 场景通过 FFI 桥接 C++ knowhere，无需重新实现

---

*本报告基于 2026-03-20 两仓库代码状态生成*
