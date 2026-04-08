# 权威数字总表

> 来源：hannsdb-x86（Milvus 集成）和 knowhere-x86-hk-proxy（standalone benchmark）。
> Mac Apple Silicon 数字仅供本地开发参考，不作为权威。

---

## HNSW — Standalone（SIFT-1M, x86, M=16, ef_c=200）

| ef | recall@10 | 单线程 QPS | batch QPS | vs native 8T (15,918) |
|----|-----------|------------|-----------|----------------------|
| 60 | 0.9720 | 7,482 | **17,319** | **+8.8%** ✅ |
| 138 | 0.9945 | 3,578 | **17,517** | **+10.1%** ✅ |

build: 561s (parallel, M=16, ef_c=200)

## HNSW — Milvus（Cohere-1M, hannsdb-x86, R8, 2026-04-07）

| 指标 | native | RS R8 | RS vs native |
|------|--------|-------|-------------|
| Insert | 304.6s | 336.8s | +10.6% 慢 |
| Optimize | 854.2s | **336.9s** | **2.53× 快** ✅ |
| Load | 1158.9s | **673.7s** | **1.72× 快** ✅ |
| QPS (c=20) | ~500+ | **1051** | **~2× 快** ✅ |
| QPS (c=80) | ~800+ | **1042** | **~1.3× 快** ✅ |
| Recall | 0.960 | 0.957 | parity |

---

## DiskANN / AISAQ — Standalone（SIFT-1M, x86）

| 模式 | QPS | recall@10 | build | 备注 |
|------|-----|-----------|-------|------|
| NoPQ 内存 | **6,062** | **0.9941** | 244s | vs native build 1595s (6.5× faster) |
| NoPQ disk/mmap 冷 | 401 | 0.9941 | 244s | — |
| NoPQ disk/io_uring | 399 | 0.9941 | 244s | ≈ mmap（per-beam 无额外收益）|
| PQ32 disk V3 group=8 | **1,063** | 0.9114 | 642s | 236B/node hot sector |
| PQ32 内存（cache_all） | 2,074 | 0.9146 | ~200s | 非 disk |

## DiskANN — Milvus（1M×768D, hannsdb-x86）

| 指标 | native | RS R1（修复前）| RS R2（修复后）| R2 vs native |
|------|-------:|---------------:|---------------:|-------------:|
| Insert | 151.4s | 157.9s | 156.3s | +3% 慢 |
| Build | 16.09s | 16.1s | 17.1s | +6% 慢 |
| Serial QPS | **11.44** | 2.4 | **11.2** | **−2%（parity）** ✅ |
| c=1 QPS | **11.10** | 2.5 | **10.8** | **−3%（parity）** ✅ |
| c=20 QPS | 12.22 | 12.7 | 12.2 | ≈ parity |
| c=80 QPS | 12.93 | 13.1 | 12.6 | ≈ parity |
| Recall | 1.000 | 1.000 | 1.000 | 完全一致 |

---

## IVF-Flat — Standalone（SIFT-1M, x86, nlist=1024）

| nprobe | recall@10 | QPS (serial) | QPS (batch) | vs native |
|--------|-----------|-------------|------------|-----------|
| 32 | 0.978 | **2,339** | **3,429** | serial 6.9×; batch **4.76× > native 8T** ✅ |

build: train 112.7s + add 9.5s = 122.2s

## IVF 系列 — Build Time + Recall + QPS 完整对比（100K×768D, hannsdb-x86, 2026-04-08）

**数据集**：合成归一化 float32，100K × 768D，IP metric，nlist=1024
**方法**：每个 index 独立 drop + rebuild，RS 和 Native 各建一次（BYPASS env var），测 build time / load time / recall@10 / c=1 QPS

| Index | Mode | Build(s) | Load(s) | Recall@10 | c=1 QPS |
|-------|------|---------|---------|-----------|---------|
| IVF_FLAT (nprobe=32) | RS | 5.5 | 1.8 | 1.000 | 39.4 |
| IVF_FLAT (nprobe=32) | Native | 3.0 | 1.6 | 1.000 | 40.6 |
| IVF_SQ8 (nprobe=32) | RS | 6.0 | 1.6 | 1.000 | 40.8 |
| IVF_SQ8 (nprobe=32) | Native | 3.0 | 2.5 | 1.000 | 37.6 |
| IVF_PQ m=32 (nprobe=64) | RS | 7.0 | 2.0 | 1.000 | 38.8 |
| IVF_PQ m=32 (nprobe=64) | Native | 3.5 | 2.2 | 1.000 | 38.4 |

**结论**：
- **Recall 完全 parity**（100K self-query 场景，nprobe 足够）
- **Search QPS parity**（c=1 误差范围内）
- **Build time：RS 约 1.8–2× 慢于 native** ⚠️ — k-means 训练阶段差距，待优化

注：Recall=1.000 是 self-query（query ∈ index）结果。IVF-PQ 对外部 query 的实际 recall = 0.815（见下表）。

---

## IVF-PQ — Milvus 并发 QPS（100K×768D, hannsdb-x86, 2026-04-08）

**数据集**：合成归一化 float32，100K × 768D，IP metric
**Index**：IVF_PQ，nlist=1024，m=32，nbits=8，nprobe=64
**Recall@10**：0.815（IVF-PQ 有损压缩权衡，m=32 即 24 dims/sub-quantizer）
**结论**：RS 与 native **parity ✅（<1%）**。上限 ~191 QPS（Milvus dispatch ceiling，同 IVF-Flat 量级）。

### RS vs Native 对比（nprobe=64, m=32, 100K×768D）

| Concurrency | RS QPS | Native QPS | Delta |
|-------------|--------|------------|-------|
| c=1  | 38.9 | 39.0 | parity |
| c=20 | 180.9 | 180.8 | parity |
| c=80 | **190.4** | **191.8** | **parity ✅** |

---

## IVF-Flat — Milvus 并发 QPS（100K×768D, hannsdb-x86, 2026-04-08）

**数据集**：合成归一化 float32，100K × 768D，IP metric（Cohere-1M shape）
**Index**：IVF_FLAT，nlist=1024，nprobe=32
**结论**：RS 与 native **parity ✅（<1% 差异）**。上限 ~184 QPS（高于 IVF-SQ8 ~140，因无 SQ8 量化延迟）。Milvus dispatch ceiling 主导，非 RS 计算瓶颈。

### RS vs Native 对比（nprobe=32, 100K×768D）

| Concurrency | RS QPS | Native QPS | Delta |
|-------------|--------|------------|-------|
| c=1  | 38.4 | 38.2 | parity |
| c=20 | 173.6 | 174.2 | parity |
| c=80 | **184.4** | **183.2** | **+0.7% parity ✅** |

RS 天花板 = 184.4 QPS，Native 天花板 = 183.2 QPS（测量噪声范围内）。

注：standalone batch parallel 4.76× faster than native 8T — 该优势在 Milvus 中不体现（nq=1 per FFI call，无批处理）。

---

## IVF-SQ8 — Milvus 并发 QPS（100K×768D, hannsdb-x86, 2026-04-08）

**数据集**：合成归一化 float32，100K × 768D，IP metric（Cohere-1M shape）
**Index**：IVF_SQ8，nlist=1024
**结论**：c=10 起达到 ~140 QPS 平台期，与 nprobe 无关 — Milvus dispatch overhead 主导，非 RS 计算瓶颈。**RS 与 native 完全 parity（误差 <1%）**。

### RS vs Native 对比（nprobe=8, 100K×768D）

| Concurrency | RS QPS | Native QPS | Delta |
|-------------|--------|------------|-------|
| c=1  | 40.3 | 38.7 | parity |
| c=5  | 129.7 | 130.6 | parity |
| c=10 | 134.2 | 135.2 | parity |
| c=20 | 139.0 | 138.8 | parity |
| c=40 | 138.6 | 140.9 | parity |
| c=80 | **139.5** | **141.1** | **parity ✅** |

RS 天花板 = 139.5 QPS，Native 天花板 = 141.1 QPS（测量噪声范围内）。

### RS 完整 QPS 表

| Concurrency | nprobe=8 | nprobe=32 | nprobe=128 |
|-------------|----------|-----------|------------|
| c=1  | 40.3 | 16.3 | 16.3 |
| c=5  | 129.7 | 61.2 | 69.8 |
| c=10 | 134.2 | 135.5 | 136.6 |
| c=20 | 139.0 | 138.2 | 138.7 |
| c=40 | 138.6 | 138.4 | 137.8 |
| c=80 | **139.5** | 135.1 | 135.9 |

**关键对比**：

| | HNSW R8 | IVF-SQ8 RS | IVF-SQ8 Native |
|---|---------|------------|----------------|
| c=1 serial | 111 QPS | 40 QPS | 39 QPS |
| c=20 | **1051 QPS** | 139 QPS | 139 QPS |
| c=80 | **1042 QPS** | 140 QPS | 141 QPS |
| 机制 | HNSW_NQ_POOL 批处理（80 query→1 FFI）| 每 query 独立 FFI call | 每 query 独立 FFI call |
| RS vs Native Gap | — | **parity ✅** | — |
| vs HNSW Gap | — | **7.5× 差距** | **7.5× 差距** |

**结论**：RS 无额外 per-query overhead。140 QPS 天花板是 Milvus dispatch ceiling，RS 和 native 共享同一上限。
**下一步**：实现 IVF_NQ_POOL nq 批处理（参照 HNSW R7），预计突破 500+ QPS @ c=80

---

## IVF-SQ8 — Standalone（SIFT-1M, x86, nlist=1024）

| nprobe | recall@10 | QPS (serial) | QPS (batch) | vs native 8T |
|--------|-----------|-------------|------------|-------------|
| 32 | 0.958 | 5,538 | **11,717** | **1.42× 快** ✅ |

native 8T: 8,278 QPS @ recall=0.952

---

## Cohere-1M Authority（x86, IVF, decode_dot fused）

| Index | nprobe | QPS | recall@10 |
|-------|--------|-----|-----------|
| IVF-Flat | 32 | 339 | 0.798 |
| IVF-SQ8 | 32 | **605** | 0.805 |
| IVF-SQ8 | 4 | 2,897 | 0.530 |

IVF-SQ8 > IVF-Flat：nprobe=32 时 1.78× 更快，recall 相近。

---

## HNSW Cosine Serial Latency（x86, ef_c=200, 1000 queries）

| 配置 | p50 | p99 |
|------|-----|-----|
| M=8, 50K/1536/ef=32 | 3021µs | 3508µs |
| M=16, 50K/1536/ef=32 | 5456µs | 6073µs |

修复前 HannsDB p99=110ms → 修复后 M=8 p99=3.5ms（**31×** 改善）。

---

## 注：x86 benchmark 环境

→ 详见 [[machines/hannsdb-x86]] 和 [[machines/knowhere-x86]]
