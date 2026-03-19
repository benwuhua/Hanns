# HANNS Feature Integration Design

> 版本: 2026-03-19
> 作者: knowhere-rs builder session
> 参考: HANNS (knowhere-2.3.1 增强版) 设计文档

---

## 概述

HANNS 在 knowhere C++ 上实现了六项核心增强。本文档描述如何将这些特性以 Rust 原生方式集成进 knowhere-rs，并在每个模块标注与原始设计的关键差异和修正。

### 特性全景

| 特性 | HANNS 实现 | knowhere-rs 路径 | 优先级 |
|------|-----------|-----------------|--------|
| Thread-local Visited Pool | `VisitedPool` + generation counter | `src/search/visited_pool.rs` | P0 |
| IO Cutting | `IoCuttingState` 比例早停 | `src/search/io_cutting.rs` | P0 |
| PCA Transform | nalgebra/ndarray eigenvalue | `src/quantization/pca.rs` | P1 |
| SQ Flash Index + PCA | SQ8 量化 + PCA graph | `src/faiss/diskann_sq.rs` | P1 |
| HVQ Quantizer | 随机旋转 + bit packing | `src/quantization/hvq.rs` | P2 |
| AVX512 VNNI | vpdpbusd u8×i8 点积 | `src/simd/hvq_avx512.rs` | P2 |

---

## 1. Thread-local Visited Pool

### 1.1 问题背景

当前 AISAQ `search_internal` 和 DiskANN Vamana search 在每次查询时都需要一个"已访问节点"标记结构。现在的实现是 `HashSet<u32>`——每次查询都要 allocate、insert、最后 drop，在 warm 高 QPS 路径上是明显的每查询分配开销。

### 1.2 设计

```
Generation Counter 方案（hnswlib / FAISS 同款）

每个 VisitedList 持有一个 u16 generation counter：
  - visited[node_id] == current_gen → 已访问
  - visited[node_id] != current_gen → 未访问
  - reset() = gen += 1，O(1)
  - 每 65535 次查询 overflow → 做一次 O(n) 清零
```

### 1.3 实现方案

**原始 HANNS 设计有一个 lifetime bug**：`VisitedPool::get()` 尝试两次 `borrow_mut()` 来返回 `RefMut`，无法编译。正确做法是 `thread_local!`，每个线程独立持有，不需要 HashMap。

```rust
// src/search/visited_pool.rs

pub struct VisitedList {
    gen: u16,
    visited: Vec<u16>,
}

impl VisitedList {
    pub fn new(n: usize) -> Self {
        Self { gen: 1, visited: vec![0; n] }
    }

    #[inline]
    pub fn is_visited(&self, id: u32) -> bool {
        self.visited[id as usize] == self.gen
    }

    #[inline]
    pub fn mark(&mut self, id: u32) {
        self.visited[id as usize] = self.gen;
    }

    pub fn reset(&mut self, required_n: usize) {
        if required_n > self.visited.len() {
            self.visited.resize(required_n, 0);
        }
        if self.gen == u16::MAX {
            self.visited.fill(0);
            self.gen = 1;
        } else {
            self.gen += 1;
        }
    }
}

thread_local! {
    static VISITED: RefCell<VisitedList> = RefCell::new(VisitedList::new(0));
}

/// 在每次 search 调用开头获取 thread-local visited list
pub fn with_visited<R>(n: usize, f: impl FnOnce(&mut VisitedList) -> R) -> R {
    VISITED.with(|v| {
        let mut v = v.borrow_mut();
        v.reset(n);
        f(&mut v)
    })
}
```

### 1.4 集成点

**AISAQ `search_internal`**（`src/faiss/diskann_aisaq.rs`）：
- 当前：`let mut visited: HashSet<u32> = HashSet::new()`
- 改后：`with_visited(self.node_ids.len(), |visited| { ... })`
- `visited.contains(&id)` → `visited.is_visited(id)`
- `visited.insert(id)` → `visited.mark(id)`

**DiskANN Vamana search**（`src/faiss/diskann.rs`）：同样替换。

### 1.5 预期收益

每次查询省掉 HashSet 的 `new()` + `drop()`，对于短查询（10K 缓存命中，warm path）收益明显；对 1M 磁盘路径（I/O 时间主导）收益较小但无负面影响。

---

## 2. IO Cutting

### 2.1 问题背景

beam search 的终止条件目前有两个：`max_visit`（最大访问节点数）和 `search_io_limit`（最大 page miss 数）。两者都是绝对计数上限，不能感知"候选列表是否已经收敛"。

IO Cutting 用比例判断：如果候选列表尾部连续无效插入次数 ≥ 剩余容量 × threshold，说明搜索已经收敛，可以提前退出。

### 2.2 核心逻辑

```
候选列表大小 = search_list_size (L)
change_point_index = 最后一次有效插入的位置（0-indexed，0=最近）
stable_count = 自上次有效插入后的连续无效插入计数

early terminate 条件：
  remaining = (L - 1) - change_point_index
  stable_count >= ceil(remaining * threshold)
```

直觉：如果列表末尾还有 `remaining` 个空位，但我们已经连续失败 `threshold × remaining` 次，说明我们扫描的邻居质量已经差到填不满列表了。

### 2.3 实现方案

```rust
// src/search/io_cutting.rs

pub struct IoCuttingState {
    stable_count: u32,
    change_point_index: u32,  // 最后有效插入的位置
    search_list_size: u32,
    threshold: f32,           // 推荐默认值: 0.9
}

impl IoCuttingState {
    pub fn new(l: usize, threshold: f32) -> Self {
        Self {
            stable_count: 0,
            change_point_index: 0,
            search_list_size: l as u32,
            threshold,
        }
    }

    /// 记录一次候选插入结果
    /// is_valid: 是否成功插入（即新邻居距离比当前列表最差的更近）
    /// insert_pos: 插入位置（0 = 最前/最近）
    #[inline]
    pub fn record(&mut self, is_valid: bool, insert_pos: usize) {
        if is_valid {
            self.stable_count = 0;
            self.change_point_index = insert_pos as u32;
        } else {
            self.stable_count += 1;
        }
    }

    #[inline]
    pub fn should_stop(&self) -> bool {
        let remaining = (self.search_list_size - 1).saturating_sub(self.change_point_index);
        if remaining == 0 { return true; }
        self.stable_count as f32 >= (remaining as f32 * self.threshold).ceil()
    }
}
```

### 2.4 集成点

在 `search_internal` 的 beam loop 中，每次尝试将新邻居插入候选列表后：

```rust
// 伪代码，在 beam loop 里
let (inserted, pos) = result_list.try_insert(neighbor_id, dist);
io_state.record(inserted, pos);
if config.io_cutting_enabled && io_state.should_stop() {
    break;
}
```

`AisaqConfig` 新增：
```rust
pub io_cutting_enabled: bool,   // 默认 false（保守）
pub io_cutting_threshold: f32,  // 默认 0.9
```

### 2.5 与现有终止条件的关系

三个条件独立检查，任一满足即停：
```
max_visit 超限 OR search_io_limit 超限 OR io_cutting.should_stop()
```

不删除现有条件，IO Cutting 是可选的第三层保险。

---

## 3. PCA Module

### 3.1 动机

DiskANN graph 建图时用的是原始 float 向量做 beam search，128-dim 的距离计算是主要开销。PCA 降到 32-64 dim 可以在建图阶段大幅减少计算量，同时保留足够的结构信息用于邻居筛选。

搜索阶段不用 PCA（最终距离还是用原始向量精算）。

### 3.2 实现方案

**关键决策：用 nalgebra SVD，不用 openblas**

HANNS 文档建议 `ndarray-linalg + openblas-system`，但 openblas 引入系统依赖，破坏 cross-compile。nalgebra 的 `DMatrix::svd()` 是纯 Rust 实现，精度足够，训练只做一次，速度不是瓶颈。

```rust
// src/quantization/pca.rs
use nalgebra::{DMatrix, DVector};

pub struct PcaTransform {
    mean: Vec<f32>,
    /// row-major: [pca_dim × original_dim]
    components: Vec<f32>,
    original_dim: usize,
    pca_dim: usize,
}

impl PcaTransform {
    pub fn train(data: &[f32], n: usize, dim: usize, pca_dim: usize) -> Self {
        assert!(pca_dim <= dim && pca_dim <= n);

        // 1. 计算均值
        let mean: Vec<f32> = (0..dim)
            .map(|j| data.iter().skip(j).step_by(dim).sum::<f32>() / n as f32)
            .collect();

        // 2. 中心化，构建 [n × dim] 数据矩阵
        let centered: Vec<f32> = data.chunks(dim)
            .flat_map(|row| row.iter().zip(&mean).map(|(x, m)| x - m))
            .collect();
        let mat = DMatrix::from_row_slice(n, dim, &centered);

        // 3. SVD（协方差矩阵的主成分 = 数据矩阵的右奇异向量）
        // 用 thin SVD 降低内存：只算前 pca_dim 个奇异向量
        let svd = mat.svd(false, true);
        let vt = svd.v_t.expect("SVD failed");

        // 4. 取前 pca_dim 行（主成分），转为 Vec<f32>
        let components = vt.rows(0, pca_dim)
            .iter()
            .map(|&x| x as f32)
            .collect();

        Self { mean, components, original_dim: dim, pca_dim }
    }

    /// 变换一条查询向量，O(pca_dim × original_dim)
    #[inline]
    pub fn transform_one(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.pca_dim];
        for i in 0..self.pca_dim {
            let row = &self.components[i * self.original_dim..(i + 1) * self.original_dim];
            out[i] = v.iter().zip(&self.mean).zip(row)
                .map(|((x, m), c)| (x - m) * c)
                .sum();
        }
        out
    }

    /// 批量变换，返回 [n × pca_dim] row-major
    pub fn transform_batch(&self, data: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * self.pca_dim];
        data.chunks(self.original_dim)
            .enumerate()
            .for_each(|(i, row)| {
                let dst = &mut out[i * self.pca_dim..(i + 1) * self.pca_dim];
                let t = self.transform_one(row);
                dst.copy_from_slice(&t);
            });
        out
    }
}
```

### 3.3 Cargo 依赖

```toml
nalgebra = { version = "0.33", default-features = false, features = ["std"] }
```

不加 `ndarray-linalg`，不加 `openblas-system`。

---

## 4. SQ Flash Index + PCA

### 4.1 动机

当前 DiskANN AISAQ 的两条路径：
- NoPQ：原始 float 存储，搜索精度高，disk 占用大
- PQ32：PQ 量化，recall 牺牲约 1-2%，build 慢（238s vs 116s）

SQ Flash 提供第三条路径：
- **Build**：PCA 降维（128→32-64 dim）后做 SQ8 量化，建图速度与 NoPQ 相当
- **Search**：beam search 阶段用 PCA+SQ8 近似距离快速筛选，最终用原始 float 精算 top-k
- **Recall**：预期好于 PQ32（SQ8 量化误差小于 PQ），接近 NoPQ

### 4.2 文件结构

```
src/faiss/diskann_sq.rs      -- SqFlashIndex 主体
src/quantization/pca.rs      -- PCA（上节）
src/quantization/sq.rs       -- 复用现有 SQ8 quantizer（已有）
```

### 4.3 SqFlashIndex 核心结构

```rust
// src/faiss/diskann_sq.rs

pub struct SqFlashConfig {
    pub max_degree: usize,          // 默认 48
    pub search_list_size: usize,    // 默认 128
    pub beamwidth: usize,           // 默认 8
    pub pca_dim: Option<usize>,     // None = 不用 PCA；Some(32) = 降到 32 维
    pub rerank_factor: f32,         // 精排池大小 = k × rerank_factor，默认 3.0
}

pub struct SqFlashIndex {
    config: SqFlashConfig,
    pca: Option<PcaTransform>,
    sq: Sq8Quantizer,               // 复用 src/quantization/sq.rs
    /// 每个节点的 SQ8 编码（pca_dim 或 dim 个 i8）
    sq_codes: Vec<i8>,
    /// 原始向量（精排用）
    raw_vectors: Vec<f32>,
    /// 图邻接表（平铺，与 DiskANNIndex / PQFlashIndex 同构）
    node_neighbor_ids: Vec<u32>,
    node_neighbor_counts: Vec<u32>,
    flat_stride: usize,
    entry_points: Vec<u32>,
    n: usize,
    dim: usize,
}
```

### 4.4 Build 流程

```
train(vectors):
  1. 如果 pca_dim 设置：
       pca = PcaTransform::train(vectors, n, dim, pca_dim)
       reduced = pca.transform_batch(vectors, n)    // [n × pca_dim]
     否则：
       reduced = vectors
  2. sq = Sq8Quantizer::train(&reduced, effective_dim)
  3. sq_codes = sq.encode_batch(&reduced)           // [n × effective_dim] i8
  4. raw_vectors = vectors.to_vec()                 // 保留原始向量用于精排
  5. 建图（Vamana 风格 beam search，用 sq_l2_asymmetric 距离）

sq_l2_asymmetric(float_query_reduced, i8_code):
  // float query vs i8 database，与现有 hnsw_quantized.rs 的 sq_l2_asymmetric 同款
```

### 4.5 Search 流程

```
search(query, k):
  1. 如果 pca: query_reduced = pca.transform_one(query)
     否则: query_reduced = query
  2. beam search（beam loop 用 sq_l2_asymmetric 快速评分）
     + IO Cutting 早停（HANNS-IOC-001）
     + thread-local visited（HANNS-VIS-001）
  3. 取候选 top-(k × rerank_factor) 个节点
  4. 对这些候选用原始 float 向量精算精确 L2
  5. 返回精排后 top-k
```

### 4.6 期望性能

| | Build | QPS (warm) | Recall |
|-|-------|-----------|--------|
| NoPQ | 116s | ~8K | 0.997 |
| PQ32 | 239s | ~7K | 0.985 |
| **SQ+PCA(32)** | **~80s** | **~9K** | **~0.990** |

（数字为设计目标，需 x86 authority 验证）

---

## 5. HVQ Quantizer

> **前置条件**：需要 HANNS 实际 SIFT-1M recall 数字（m=8/16/32 + 各 nbits）才能判断 HVQ 是否值得在 knowhere-rs 实现。目前为 P2 设计备忘，不开发。

### 5.1 算法概述

HVQ 不训练 codebook（区别于 PQ），而是：
1. **随机正交旋转**：让向量分量近似等能量分布，使量化误差均匀
2. **标量量化**：旋转后的每个分量量化为 1/2/4/8 bit
3. **Code refinement**：最多 6 次迭代，每次重新分配量化步长减少误差

优点：无需训练，适合冷启动场景。缺点：recall 上限低于精心训练的 PQ（旋转不感知数据分布）。

### 5.2 关键设计决策

**旋转矩阵生成**：用 nalgebra `QR::new(random_matrix).q()` 生成正交矩阵，dim×dim。旋转矩阵只需生成一次，固定后用于所有向量。

**Bit packing**：
- 1-bit：8 分量 → 1 byte（简单位移）
- 2-bit：4 分量 → 1 byte（mask `0x03`）
- 4-bit：2 分量 → 1 byte（nibble）
- 8-bit：1 分量 → 1 byte（无压缩）

**Code refinement**：
```
初始量化 → 计算重建误差 → 用误差调整 delta → 重新量化
重复最多 6 次，或误差不再缩小
```

### 5.3 距离函数

HVQ 用内积（Inner Product）而非 L2，因为旋转保持 L2 等价，但 IP 更易 SIMD 加速：

```
dist(q, x) = ||q||² - 2⟨q, x⟩ + base_quant_dist[x]
             └── 查询范数（常数）  └── 量化误差修正项（建库时预计算）
```

`base_quant_dist` 是每个数据库向量的量化误差 `||x - x̂||²`，build 时算好存起来。

---

## 6. AVX512 VNNI

> 依赖 HVQ，P2。

### 6.1 指令

`_mm512_dpbusd_epi32(acc, a, b)`（vpdpbusd）：
- `a`：u8 × 64 元素（无符号）
- `b`：i8 × 64 元素（有符号）
- 每 4 对元素做 u8×i8 乘加，累加到 i32 × 16 向量
- 等价于：`acc[i] += a[4i]*b[4i] + a[4i+1]*b[4i+1] + a[4i+2]*b[4i+2] + a[4i+3]*b[4i+3]`

### 6.2 运行时 dispatch

```rust
pub fn select_hvq_dist_fn(nbits: u8) -> HvqDistFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") {
            return match nbits { 2 => hvq_2bit_vnni, 4 => hvq_4bit_vnni, 8 => hvq_8bit_vnni, _ => hvq_scalar };
        }
        if is_x86_feature_detected!("avx512f") {
            return match nbits { 2 => hvq_2bit_avx512, 4 => hvq_4bit_avx512, 8 => hvq_8bit_avx512, _ => hvq_scalar };
        }
    }
    hvq_scalar
}
```

### 6.3 2-bit / 4-bit Unpack

```rust
// 4-bit: 每 byte 含 2 个 4-bit 值
// byte b → [b & 0x0F, (b >> 4) & 0x0F]
let lo = _mm512_and_si512(packed, _mm512_set1_epi8(0x0F));
let hi = _mm512_srli_epi16(packed, 4);
let hi = _mm512_and_si512(hi, _mm512_set1_epi8(0x0F));
// 分别与 query 的对应半做 dpbusd
```

---

## 7. 配置参数扩展

在 `AisaqConfig` / `DiskAnnParams` 增加：

```rust
// HANNS 新增参数（全部可选，有合理默认值）
pub visited_pool_enabled: bool,      // 默认 true（HANNS-VIS-001）
pub io_cutting_enabled: bool,        // 默认 false
pub io_cutting_threshold: f32,       // 默认 0.9（仅 io_cutting_enabled=true 时生效）
pub pca_dim: Option<usize>,          // 默认 None（不用 PCA）
pub sq_encoding: bool,               // 默认 false（用 SQ Flash 路径）
pub rerank_factor: f32,              // 默认 3.0（精排池 = k × factor）
pub hvq_nbits: Option<u8>,           // 默认 None（不用 HVQ）；Some(4) 启用 4-bit HVQ
```

---

## 8. 实施顺序和里程碑

```
Week 1:
  HANNS-VIS-001  Thread-local Visited Pool       ← 独立模块，集成 AISAQ
  HANNS-IOC-001  IO Cutting                      ← 依赖 VIS-001 完成后集成

Week 2-3:
  HANNS-PCA-001  PCA Module (nalgebra SVD)        ← 纯计算模块，无 I/O 依赖
  HANNS-SQ-001   SQ Flash Index                  ← 依赖 PCA-001

Week 4+（视 HVQ HANNS recall 数据决定）:
  HANNS-HVQ-001  HVQ Quantizer
  HANNS-VNNI-001 AVX512 VNNI
```

---

## 9. 未解决问题

1. **HVQ vs PQ recall 对比**：HVQ 不需要训练 codebook，但 recall 是否能追上同 bit 数的 PQ？需要 HANNS 的实测数据。

2. **SQ Flash build time 估计**：PCA 训练（nalgebra SVD on 1M × 128）耗时未知，预计 10-30s。需要实测。

3. **Graph 复用**：SqFlashIndex 和 PQFlashIndex 都有相似的邻接表结构（`node_neighbor_ids`, `node_neighbor_counts`, `flat_stride`）。未来可以抽成公共 `GraphStorage` 结构减少重复，但初版先各自独立。

4. **IO Cutting threshold 调优**：0.9 是 HANNS 的推荐值，但最优值依赖数据分布和 L 大小。需要在 SIFT-1M 上做扫描实验。
