# HNSW 图修复功能实现 (OPT-017)

## 概述

实现了 C++ knowhere HNSW 的图修复功能，用于查找并修复构建完成后不可达的向量，提高图的连通性。

## 实现的功能

### 1. `find_unreachable_vectors(&self) -> Vec<usize>`

从 entry_point 开始，在每一层执行 BFS 遍历，查找不可达的向量。

**算法流程：**
- 从 entry_point 开始，对每一层（从 max_level 到 0）执行 BFS
- 标记所有访问到的节点
- 返回在某一層存在但未被访问到的节点索引列表

**参考 C++ 实现：**
```cpp
std::vector<tableint> findUnreachableVectors() {
    // BFS from entry_point at each layer
    // Return indices of unreachable vectors
}
```

### 2. `repair_graph_connectivity(&mut self, unreachable_idx: usize, level: usize)`

修复指定向量在指定层的连通性。

**算法流程：**
- 从 entry_point 开始贪婪搜索，找到最近的邻居节点
- 在对应层添加边连接到最近的 neighbors
- 提高图的连通性

### 3. `find_and_repair_unreachable(&mut self) -> usize`

 convenience 方法，组合查找和修复操作。

**返回：** 修复的不可达向量数量

### 4. `build_with_repair(&mut self, vectors: &[f32], ids: Option<&[i64]>) -> Result<usize>`

构建 HNSW 索引并自动执行图修复。

**流程：**
1. 训练索引（如果尚未训练）
2. 添加所有向量
3. 查找不可达向量
4. 修复所有不可达向量在所有层的连通性
5. 打印修复的向量数量

## 使用示例

```rust
use knowhere_rs::api::{IndexConfig, IndexType, MetricType};
use knowhere_rs::faiss::hnsw::HnswIndex;

// 创建索引配置
let config = IndexConfig {
    index_type: IndexType::Hnsw,
    metric_type: MetricType::L2,
    dim: 4,
    params: IndexParams {
        m: Some(16),
        ef_construction: Some(200),
        ..Default::default()
    },
};

let mut index = HnswIndex::new(&config)?;

// 准备向量数据
let vectors = vec![
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    // ... more vectors
];

// 构建并自动修复图
let count = index.build_with_repair(&vectors, None)?;
println!("Built index with {} vectors", count);

// 或者手动控制查找和修复
index.train(&vectors)?;
index.add(&vectors, None)?;

let unreachable = index.find_unreachable_vectors();
println!("Found {} unreachable vectors", unreachable.len());

for &idx in &unreachable {
    let node_info = &index.node_info[idx];
    for level in 0..=node_info.max_layer {
        index.repair_graph_connectivity(idx, level);
    }
}
```

## 测试

实现了 5 个单元测试：

1. **test_find_unreachable_vectors_empty_index** - 测试空索引
2. **test_find_unreachable_vectors_single_vector** - 测试单个向量（应该是可达的）
3. **test_build_with_repair** - 测试完整的构建 + 修复流程
4. **test_repair_graph_connectivity_manual** - 测试手动修复
5. **test_graph_connectivity_after_repair** - 测试修复后的图连通性
6. **test_unreachable_detection_multilayer** - 测试多层结构中的不可达检测

所有测试均通过：
```
running 15 tests
test faiss::hnsw::tests::test_find_unreachable_vectors_empty_index ... ok
test faiss::hnsw::tests::test_find_unreachable_vectors_single_vector ... ok
test faiss::hnsw::tests::test_hnsw_search_with_bitset_all_filtered ... ok
test faiss::hnsw::tests::test_hnsw_index_trait_with_bitset ... ok
test faiss::hnsw::tests::test_hnsw_cosine_metric ... ok
test faiss::hnsw::tests::test_hnsw ... ok
test faiss::hnsw::tests::test_hnsw_ip_metric ... ok
test faiss::hnsw::tests::test_hnsw_search_with_filter ... ok
test faiss::hnsw::tests::test_hnsw_search_with_bitset ... ok
test faiss::hnsw::tests::test_random_level_distribution ... ok
test faiss::hnsw::tests::test_repair_graph_connectivity_manual ... ok
test faiss::hnsw::tests::test_build_with_repair ... ok
test faiss::hnsw::tests::test_unreachable_detection_multilayer ... ok
test faiss::hnsw::tests::test_multilayer_structure ... ok
test faiss::hnsw::tests::test_graph_connectivity_after_repair ... ok

test result: ok. 15 passed; 0 failed
```

## 实现细节

### 数据结构
- 使用 `VecDeque` 进行 BFS 遍历
- 使用 `HashSet` 跟踪访问状态
- 使用 `BinaryHeap` 进行最近邻搜索

### 并行化
当前实现使用顺序修复，未来可以优化为：
- 使用 `rayon` 并行修复不同层的不可达向量
- 注意需要处理可变借用的问题

### 与 C++ 实现的差异
1. Rust 版本使用 `&mut self` 进行修复，而 C++ 使用裸指针
2. Rust 版本将查找和修复分离为独立的方法，提供更灵活的使用方式
3. 当前实现使用顺序修复而非并行（可以后续优化）

## 文件位置
- 实现：`/Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss/hnsw.rs`
- 测试：同文件中的 `#[cfg(test)] mod tests`
- 文档：`/Users/ryan/.openclaw/workspace-builder/knowhere-rs/HNSW_GRAPH_REPAIR.md`

## 后续优化
1. 使用 rayon 并行修复不可达向量
2. 添加更详细的日志和统计信息
3. 优化修复策略（例如只修复一定比例的最不可达向量）
4. 添加性能基准测试
