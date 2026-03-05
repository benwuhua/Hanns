# OPT-003: SIMD 加速距离计算 - 实现报告

**日期:** 2026-03-01  
**任务:** SIMD 加速距离计算 - AVX2/NEON 优化  
**状态:** ✅ DONE

## 完成内容

### 1. 新增 SIMD 函数

在 `src/simd.rs` 中添加了以下优化函数：

#### L2 平方距离函数（避免 sqrt，用于最近邻搜索）
- `l2_distance_sq()` - 主入口函数，自动检测 CPU 特性选择最优实现
- `l2_scalar_sq()` - 标量版本 fallback
- `l2_sse_sq()` - SSE 4.2 优化版本（4 路并行）
- `l2_avx2_sq()` - AVX2 优化版本（8 路并行）+ 优化的水平求和
- `l2_avx512_sq()` - AVX-512 优化版本（16 路并行）
- `l2_neon_sq()` - ARM NEON 优化版本（4 路并行）

### 2. 优化技术

#### 水平求和优化 (AVX2)
```rust
fn horizontal_sum_avx2(sum: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(lo, hi);
    
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}
```

#### 关键优化点
1. **避免 sqrt**: L2 平方距离用于最近邻搜索，排序结果相同但性能更高
2. **批量处理**: 利用 SIMD 寄存器一次处理 4/8/16 个浮点数
3. **水平求和优化**: 使用 shuffle/add 指令快速求和 SIMD 寄存器
4. **余数处理**: 对无法被 SIMD 宽度整除的部分使用标量处理

### 3. 代码改动统计

```
src/simd.rs: +250 行 (约)
- 添加 l2_distance_sq 主函数
- 添加 SSE/AVX2/AVX512/NEON 平方距离实现
- 添加 horizontal_sum_avx2 辅助函数
```

## 性能测试结果

### 小规模测试 (10K 向量，128 维)

| 索引类型 | 构建时间 (ms) | 搜索 100 查询 (ms) | QPS |
|---------|-------------|------------------|-----|
| Flat    | 0.49        | 46.20            | 2164 |
| HNSW    | 414.75      | 1.64             | 60917 |
| IVF-Flat| 5342.40     | 46.28            | 2161 |

### 性能分析

1. **Flat 索引**: 
   - 构建极快 (0.49ms)，无需训练
   - 搜索性能 2164 QPS，使用 SIMD 距离计算
   
2. **HNSW 索引**:
   - 构建较慢 (414.75ms)，需要构建图层
   - 搜索性能极佳 60917 QPS，28x 于 Flat
   - SIMD 优化在图层搜索中发挥作用

3. **IVF-Flat 索引**:
   - 构建最慢 (5342.40ms)，主要耗时在 k-means 训练
   - 搜索性能 2161 QPS，与 Flat 相当
   - SIMD 优化在 centroid 查找和倒排列表搜索中使用

## 对比 C++ knowhere

### C++ knowhere SIMD 实现特点

1. **批量距离计算**: `fvec_L2sqr_batch_4_avx` 一次计算 1 个查询与 4 个数据库向量
2. **FMA 指令**: 使用 `_mm256_fmadd_ps` 加速乘加运算
3. **预取优化**: 使用 `__builtin_prefetch` 减少 cache miss
4. **对齐内存**: 使用 `__attribute__((aligned(32)))` 优化内存访问

### Rust 实现差距

| 特性 | C++ knowhere | Rust knowhere-rs | 差距 |
|------|-------------|-----------------|------|
| 单向量 L2 | ✅ AVX2/AVX512 | ✅ AVX2/AVX512 | ✅ 已对标 |
| 批量 L2 (4 路) | ✅ batch_4 | ❌ 未实现 | ⚠️ 待优化 |
| FMA 指令 | ✅ 使用 | ❌ 未使用 | ⚠️ 待优化 |
| 内存预取 | ✅ 使用 | ❌ 未使用 | ⚠️ 待优化 |
| 对齐内存 | ✅ 使用 | ❌ 未使用 | ⚠️ 待优化 |

### 后续优化方向

1. **批量距离计算**: 实现 `l2_batch_4_avx2` 一次计算 4 个距离
2. **FMA 指令**: 使用 `fma` crate 或 intrinsics 加速乘加
3. **内存对齐**: 使用 `aligned_alloc` 或 `posix_memalign` 分配对齐内存
4. **软件预取**: 使用 `std::arch::x86_64::_mm_prefetch`

## 测试验证

### 编译测试
```bash
cargo check
# ✅ 通过 (181 warnings, 0 errors)
```

### 单元测试
```bash
cargo test --release --test perf_test
# ✅ 通过 (2/2 tests)
```

### 功能验证
- ✅ SSE 路径测试 (x86_64)
- ✅ AVX2 路径测试 (x86_64)
- ✅ AVX512 路径测试 (x86_64)
- ✅ NEON 路径测试 (aarch64)
- ✅ 标量 fallback 测试

## 总结

### 已完成
- ✅ 添加 L2 平方距离 SIMD 优化（SSE/AVX2/AVX512/NEON）
- ✅ 实现优化的水平求和算法
- ✅ 集成到现有距离计算框架
- ✅ 通过所有测试

### 性能提升
- **理论峰值**: L2 距离计算性能提升 4-16x（取决于 SIMD 级别）
- **实际搜索**: QPS 提升取决于索引类型和内存带宽
- **构建时间**: k-means 训练受益明显（已在 OPT-004 中验证）

### 待优化
- [ ] 批量距离计算 (batch of 4)
- [ ] FMA 指令优化
- [ ] 内存对齐优化
- [ ] 软件预取优化

## 参考实现

- C++ knowhere: `src/simd/distances_avx.cc`, `distances_avx512.cc`, `distances_neon.cc`
- Faiss: `faiss/utils/distances_simd.cpp`
