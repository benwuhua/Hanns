# BENCH-005: 吞吐量基准测试实现

## 任务概述
添加并发查询的吞吐量基准测试，用于 QPS 压力测试和延迟分布分析。

## 完成时间
2026-03-01

## 改动文件

### 1. tests/bench_throughput.rs (新建，13.6KB)
实现并发吞吐量基准测试框架：
- **核心功能**:
  - 使用 Rayon 进行并行查询
  - 测试不同并发度下的 QPS (1, 2, 4, 8, 16 线程)
  - 支持 Flat, HNSW, IVF-Flat 三种索引类型
  - 使用 hdrhistogram 记录延迟分布 (P50, P90, P99)

- **测试函数**:
  - `test_throughput_flat()` - Flat 索引吞吐量测试
  - `test_throughput_hnsw()` - HNSW 索引吞吐量测试
  - `test_throughput_ivf_flat()` - IVF-Flat 索引吞吐量测试
  - `test_throughput_all_indexes()` - 综合对比测试

- **输出格式**:
```
╔══════════════════════════════════════════════════════════════╗
║  knowhere-rs 吞吐量基准测试 (10K 向量，128 维)               ║
╚══════════════════════════════════════════════════════════════╝

索引类型：Flat
并发度 | QPS     | P50(ms) | P90(ms) | P99(ms)
--------|---------|---------|---------|--------
1       | 19      | 53.279  | 59.039  | 61.343
2       | 38      | 56.575  | 63.263  | 67.519
4       | 73      | 60.607  | 71.999  | 76.287
```

### 2. Cargo.toml (修改)
添加 dev-dependency:
```toml
hdrhistogram = "7.5"
```

## 测试结果

### 快速验证 (10K 向量，128 维，100 查询)
```
=== 吞吐量基准测试 - Flat 索引 (10K 向量，128 维) ===

并发度                 QPS      P50(ms)      P90(ms)      P99(ms)
--------------------------------------------------------------
1                    19       53.279       59.039       61.343
2                    38       56.575       63.263       67.519
4                    73       60.607       71.999       76.287
```

**观察**:
- 并发度提升带来线性 QPS 增长 (1→2→4: 19→38→73)
- 延迟随并发度略有增加 (P50: 53ms→56ms→60ms)
- P99 延迟增长较明显，表明存在尾延迟问题

## 使用方法

### 运行单个索引测试
```bash
# Flat 索引
cargo test --test bench_throughput test_throughput_flat -- --nocapture

# HNSW 索引
cargo test --test bench_throughput test_throughput_hnsw -- --nocapture

# IVF-Flat 索引
cargo test --test bench_throughput test_throughput_ivf_flat -- --nocapture
```

### 运行综合测试
```bash
cargo test --test bench_throughput test_throughput_all_indexes -- --nocapture
```

### 修改测试规模
编辑 `tests/bench_throughput.rs`:
```rust
let n = 100_000;        // 向量数量 (10K, 100K, 1M)
let dim = 128;          // 维度
let num_queries = 1000; // 查询数量
let concurrencies = vec![1, 2, 4, 8, 16]; // 并发度
```

## 性能分析

### 可扩展性
- **并发度 1→2**: QPS 提升 ~100% (理想线性)
- **并发度 2→4**: QPS 提升 ~92% (接近线性)
- **并发度 4→8**: 预计 QPS 提升 ~85-90% (开始饱和)
- **并发度 8→16**: 预计 QPS 提升 ~70-80% (GIL/锁竞争)

### 延迟分布
- **P50**: 典型查询延迟
- **P90**: 90% 查询的延迟上限
- **P99**: 尾延迟，反映最慢的 1% 查询

### 优化建议
1. **减少锁竞争**: 考虑使用无锁数据结构或更细粒度的锁
2. **批量查询**: 支持批量查询以减少单次查询开销
3. **查询队列**: 实现查询优先级队列，优化高优先级查询

## 与 C++ knowhere 对比

### C++ 实现参考
- 位置：`/Users/ryan/Code/vectorDB/knowhere/tests/ut/test_search.cc`
- 使用 Catch2 测试框架
- 支持并发搜索测试

### 差异分析
| 特性 | C++ knowhere | Rust knowhere-rs |
|------|--------------|------------------|
| 测试框架 | Catch2 | Rust std::test |
| 并发模型 | std::thread | Rayon |
| 延迟统计 | 手动计算 | hdrhistogram |
| 输出格式 | 简单打印 | 格式化表格 |

### 可借鉴点
1. **BruteForce 验证**: C++ 使用暴力搜索计算 Ground Truth 验证召回率
2. **参数生成器**: 使用 GENERATE 宏自动生成多组参数测试
3. **内存映射测试**: 支持 MMAP 索引测试

## 后续优化方向

### BENCH-006: 添加召回率验证
- 在吞吐量测试中同时验证召回率
- 确保 QPS 提升不牺牲搜索质量

### BENCH-007: 添加内存使用跟踪
- 监控不同并发度下的内存占用
- 分析内存带宽瓶颈

### BENCH-008: 添加混合负载测试
- 同时测试构建 + 查询性能
- 模拟真实生产环境负载

## 技术细节

### 并发实现
```rust
let results: Vec<Vec<f64>> = (0..concurrency).into_par_iter().map(|thread_id| {
    // 每个线程处理一部分查询
    let start_idx = thread_id * queries_per_thread;
    let end_idx = ...;
    
    for i in start_idx..end_idx {
        // 执行查询并记录延迟
        let latency = q_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency);
    }
    latencies
}).collect();
```

### 延迟统计
```rust
let mut hist = Histogram::<u64>::new(3).unwrap();
for lat in latencies {
    let _ = hist.record((lat * 1000.0) as u64); // 转换为微秒
}
let p50 = hist.value_at_percentile(50.0) as f64 / 1000.0;
```

## 总结
BENCH-005 成功实现了并发吞吐量基准测试框架，支持：
- ✅ 多索引类型对比 (Flat/HNSW/IVF-Flat)
- ✅ 多并发度测试 (1/2/4/8/16 线程)
- ✅ 延迟分布统计 (P50/P90/P99)
- ✅ 格式化输出报告

为后续性能优化提供了可靠的基准测试工具。
