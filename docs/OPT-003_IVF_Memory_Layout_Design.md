# OPT-003: IVF 内存布局重构设计文档

**版本**: v1.0
**日期**: 2026-03-03
**优先级**: P0 (生产阻塞)
**工作量**: 5-7 天
**目标**: QPS 从 2-5% C++ → 50%+ C++

---

## 1. 问题分析

### 1.1 当前实现

```rust
// src/faiss/ivf_flat.rs
pub struct IvfFlatIndex {
    // 当前内存布局 - 高开销
    inverted_lists: HashMap<usize, Vec<(i64, Vec<f32>)>>,
    //                     ^^^^^^   ^^^^^^^^^^^
    //                     簇 ID    (向量ID, 单独分配的向量)
}
```

**性能问题**：
1. **HashMap 查找开销**: 每次 `search()` 需要多次 HashMap 查找
2. **大量小对象**: 每个 `(i64, Vec<f32>)` 单独分配
3. **缓存不友好**: 内存不连续，cache miss 高
4. **碎片化**: 高频 `push()` 导致频繁重新分配

### 1.2 C++ 参考实现

```cpp
// faiss/IndexIVF.h
struct InvertedLists {
    // 连续内存布局
    std::vector<std::vector<idx_t>> ids;      // ID 数组
    std::vector<std::vector<uint8_t>> codes;   // 编码数据
    // 优势：连续内存，缓存友好
};
```

### 1.3 性能差距

| 指标 | C++ | Rust (当前) | 差距 |
|-----|-----|------------|------|
| IVF-Flat QPS | ~5,000 | ~100-250 | **2-5%** |
| IVF-PQ QPS | ~8,000 | ~160-400 | **2-5%** |
| Cache miss | ~5% | ~40% | **8x** |

---

## 2. 优化方案

### 2.1 方案 A: 简单数组替换（推荐）

```rust
/// 优化后的 IVF-Flat 内存布局
pub struct IvfFlatIndexOptimized {
    dim: usize,
    nlist: usize,
    
    /// 质心 (连续存储)
    centroids: Vec<f32>,  // [nlist * dim]
    
    /// 倒排列表 (Vec 替代 HashMap)
    invlist_ids: Vec<Vec<i64>>,          // [nlist] 个 ID 列表
    invlist_vectors: Vec<Vec<f32>>,      // [nlist] 个向量列表
    // 优势：
    // - HashMap → Vec: O(1) 索引，无哈希开销
    // - 连续存储: 缓存友好
    // - 预分配: 减少动态分配
}
```

**改动范围**：
- `src/faiss/ivf_flat.rs`: ~200 行
- `src/faiss/ivfpq.rs`: ~180 行

**预期收益**：
- HashMap 查找开销: -80%
- Cache miss: -60%
- QPS 提升: **3-5x**

### 2.2 方案 B: 完全扁平化（激进）

```rust
/// 完全扁平化的内存布局
pub struct IvfFlatIndexFlat {
    dim: usize,
    nlist: usize,
    
    /// 质心
    centroids: Vec<f32>,
    
    /// 倒排列表元数据
    invlist_offsets: Vec<usize>,  // 每个列表的起始偏移
    invlist_lengths: Vec<usize>,  // 每个列表的长度
    
    /// 扁平化存储
    all_ids: Vec<i64>,      // 所有 ID 连续存储
    all_vectors: Vec<f32>,  // 所有向量连续存储
}
```

**优势**：
- 单次分配，零碎片
- 最佳缓存利用
- 内存占用更小

**劣势**：
- 删除操作复杂
- 需要预知大小或动态扩容

### 2.3 方案 C: 混合方案（平衡）

```rust
/// 混合方案：列表级连续 + 全局 ID 索引
pub struct IvfFlatIndexHybrid {
    dim: usize,
    nlist: usize,
    
    centroids: Vec<f32>,
    
    /// 列表级连续存储
    invlists: Vec<InvertedList>,
    
    /// 全局 ID → (cluster_id, local_idx) 索引
    id_to_location: HashMap<i64, (usize, usize)>,
}

struct InvertedList {
    ids: Vec<i64>,
    vectors: Vec<f32>,  // [len * dim]
}
```

---

## 3. 推荐实现路线

### 3.1 Phase 1: HashMap → Vec（1-2 天）

**目标**: 快速收益，最小改动

```rust
// Step 1: 替换 HashMap 为 Vec
pub struct IvfFlatIndex {
    // 旧
    // inverted_lists: HashMap<usize, Vec<(i64, Vec<f32>)>>,
    
    // 新
    invlist_ids: Vec<Vec<i64>>,
    invlist_vectors: Vec<Vec<f32>>,
}
```

**改动文件**：
- `src/faiss/ivf_flat.rs`
  - `new()`: 初始化 Vec 而非 HashMap
  - `add()`: 直接索引 `invlist_ids[cluster_id]`
  - `search()`: 直接索引，移除 HashMap 查找
- `src/faiss/ivfpq.rs`
  - 类似改动

**验证**：
```bash
cargo test --release --test perf_test -- test_ivf_flat --nocapture
# 预期: QPS +100-200%
```

### 3.2 Phase 2: 向量连续存储（2-3 天）

**目标**: 消除小对象分配

```rust
// Step 2: 向量连续存储
pub struct IvfFlatIndex {
    invlist_ids: Vec<Vec<i64>>,
    // 不再存储 Vec<(i64, Vec<f32>)>
    // 改为两个独立的 Vec
    invlist_vectors: Vec<Vec<f32>>,  // 每个簇的向量连续存储
}
```

**改动**：
- `add()`: 批量追加向量
- `search()`: 连续内存访问

**验证**：
```bash
cargo test --release --test perf_test -- test_ivf_flat --nocapture
# 预期: QPS +200-300%
```

### 3.3 Phase 3: 预分配策略（1 天）

**目标**: 减少动态分配

```rust
pub fn add(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize> {
    let n = vectors.len() / self.dim;
    
    // 预估每个簇的大小，预分配
    let estimated_per_cluster = n / self.nlist * 2;
    for i in 0..self.nlist {
        self.invlist_ids[i].reserve(estimated_per_cluster);
        self.invlist_vectors[i].reserve(estimated_per_cluster * self.dim);
    }
    
    // ... 添加向量
}
```

**验证**：
```bash
# 内存分配次数减少
cargo test --release --test perf_test -- test_ivf_flat --nocapture
```

### 3.4 Phase 4: 性能基准测试（1 天）

**目标**: 验证优化效果

```bash
# SIFT1M 测试
cargo test --release --test perf_test -- test_ivf_performance_sift1m --nocapture

# 对比 C++
# 预期: QPS 从 2-5% → 50%+ C++
```

---

## 4. 风险评估

### 4.1 兼容性风险

| API | 影响 | 缓解 |
|-----|------|------|
| `train()` | 无影响 | - |
| `add()` | 内部实现变化 | 接口不变 |
| `search()` | 内部实现变化 | 接口不变 |
| `save/load()` | **需要更新** | 增加版本号 |

**缓解策略**：
- 保持公共 API 不变
- 序列化格式向前兼容

### 4.2 回归风险

**测试覆盖**：
- 现有测试: 191 个
- 需要新增:
  - IVF-Flat 并发测试
  - IVF-PQ 大规模测试
  - 边界条件测试

**回归测试**：
```bash
cargo test
cargo test --release --test perf_test
```

---

## 5. 成功标准

### 5.1 性能目标

| 指标 | 当前 | 目标 | 验证方式 |
|-----|------|------|---------|
| IVF-Flat QPS | 100-250 | **2,500+** | perf_test |
| IVF-PQ QPS | 160-400 | **4,000+** | perf_test |
| vs C++ | 2-5% | **50%+** | 对比测试 |

### 5.2 质量目标

- 测试通过: 100%
- 无回归: 召回率保持 90%+
- 内存安全: Valgrind/ASAN clean

---

## 6. 时间表

| 阶段 | 工作内容 | 时间 | 产出 |
|-----|---------|------|------|
| Phase 1 | HashMap → Vec | 1-2 天 | QPS +100-200% |
| Phase 2 | 向量连续存储 | 2-3 天 | QPS +200-300% |
| Phase 3 | 预分配策略 | 1 天 | 减少分配 |
| Phase 4 | 性能基准 | 1 天 | 验证报告 |
| **总计** | - | **5-7 天** | **50%+ C++ QPS** |

---

## 7. 附录

### 7.1 参考代码

**C++ faiss/IndexIVF.h**:
```cpp
struct InvertedLists {
    size_t nlist;
    size_t code_size;
    
    virtual size_t list_size(size_t list_no) const = 0;
    virtual const idx_t* get_ids(size_t list_no) const = 0;
    virtual const uint8_t* get_codes(size_t list_no) const = 0;
};
```

**Rust 目标实现**:
```rust
pub struct InvertedLists {
    nlist: usize,
    code_size: usize,
    
    ids: Vec<Vec<i64>>,
    codes: Vec<Vec<u8>>,
}

impl InvertedLists {
    fn list_size(&self, list_no: usize) -> usize {
        self.ids[list_no].len()
    }
    
    fn get_ids(&self, list_no: usize) -> &[i64] {
        &self.ids[list_no]
    }
    
    fn get_codes(&self, list_no: usize) -> &[u8] {
        &self.codes[list_no]
    }
}
```

### 7.2 性能测试命令

```bash
# 小规模快速测试
cargo test --release --test perf_test -- test_ivf_flat_small --nocapture

# SIFT1M 完整测试
cargo test --release --test perf_test -- test_ivf_performance_sift1m --nocapture

# 对比 C++ (需要 C++ 环境)
cd /path/to/knowhere
./build/bin/benchmark_ivf --index IVFFlat --dataset SIFT1M
```

---

## 8. 更新日志

### 2026-03-03 (v1.0)
- 初始设计文档
- 确定优化方案（HashMap → Vec）
- 制定实现路线图
