# SIFT1M 数据集获取指南

## 数据集说明

SIFT1M 是广泛使用的向量检索基准数据集：
- **Base**: 1,000,000 向量 × 128 维
- **Query**: 10,000 查询向量 × 128 维
- **Ground truth**: 每个查询的 100 个最近邻

## 获取方式

### 方式 1：从 ann-benchmarks 下载（推荐）

```bash
# 下载 HDF5 格式
wget https://comp21data.blob.core.windows.net/sift-128-euclidean/sift-128-euclidean.hdf5

# 或使用 curl
curl -L -O https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

### 方式 2：使用现有的 HDF5 文件

如果你已经有 HDF5 格式的数据集，可以使用 knowhere-rs 的 HDF5 加载器：

```rust
#[cfg(feature = "hdf5")]
use knowhere_rs::dataset::load_hdf5_dataset;

let dataset = load_hdf5_dataset("sift-128-euclidean.hdf5")?;
```

### 方式 3：从 Faiss/DiskANN 仓库获取

```bash
# Faiss demos
git clone https://github.com/facebookresearch/faiss.git
cd faiss/demos

# 或 DiskANN test data
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN/test_data
```

### 方式 4：手动下载

原始 texmex 源已不可用，可尝试：
- Archive.org: https://archive.org/download/sift1m/
- 学术机构镜像
- 同事分享

## 使用方法

### 环境变量

```bash
export SIFT1M_PATH=/path/to/sift1m
export SIFT_NUM_QUERIES=1000  # 可选，默认 1000
```

### 运行 benchmark

```bash
# 完整 benchmark（7 个索引类型）
cargo test --release --test bench_sift1m -- --nocapture

# 快速测试（4 个索引类型，100 个查询）
cargo test --release --test bench_sift1m test_sift1m_quick -- --nocapture

# 保存 JSON 结果
export JSON_OUTPUT=sift1m_results.json
cargo test --release --test bench_sift1m -- --nocapture
```

## 测试的索引类型

BENCH-038 支持以下 7 种索引类型的端到端验证：

1. **Flat** - 暴力搜索，召回率 100%（基准线）
2. **HNSW** - 图索引，高召回率，快速查询
3. **IVF-Flat** - 聚类 + 暴力，中等召回率
4. **IVF-PQ** - 聚类 + 乘积量化，高压缩比
5. **IVF-SQ8** - 聚类 + 8-bit 标量量化
6. **RaBitQ** - 聚类 + 二进制量化，超高压缩比
7. **ScaNN** - 各向异性量化，高吞吐量

## 预期结果

生产级要求：
- **Flat**: R@10 = 100%, QPS baseline
- **HNSW**: R@10 > 95%, QPS > 800
- **IVF-PQ**: R@10 > 85%, QPS > 2000
- **IVF-SQ8**: R@10 > 90%, QPS > 1500
- **RaBitQ**: R@10 > 80%, QPS > 3000
- **ScaNN**: R@10 > 88%, QPS > 2500

## 故障排除

### 数据集未找到

```
Skipping benchmark - SIFT1M dataset not found
```

解决：设置 `SIFT1M_PATH` 或将数据集放在 `./data/sift/`

### 编译错误

```bash
cargo clean
cargo build --release --tests
```

### 召回率过低

检查：
1. 数据集格式是否正确
2. 距离度量是否匹配（L2 vs IP）
3. 参数是否合理（nprobe, ef_search 等）

## 相关任务

- BENCH-038: SIFT1M 真实数据集端到端验证
- BENCH-040: RaBitQ 大数据集召回率优化
- BENCH-021: HNSW vs C++ 公平对比
