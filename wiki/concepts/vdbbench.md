# VectorDBBench 使用指南

**一句话定义**：VectorDBBench（VDBBench）是 Zilliz 开源的向量数据库 benchmark 工具，支持 Milvus/HannsDB 等数据库的性能对比。

---

## 环境位置

| 项目 | 路径 |
|------|------|
| VDBBench 源码 | `/data/work/VectorDBBench/` |
| Python venv | `/data/work/VectorDBBench/.venv/bin/python3` |
| 数据集本地目录 | `/data/work/datasets/` |
| Cohere-1M 数据 | `/data/work/datasets/wikipedia-cohere-1m/`（base.fbin, query.fbin, gt.ibin）|
| SIFT-1M 数据 | `/data/work/datasets/sift-1m/`（base.fbin, query.fbin, gt.ibin）|
| 结果输出目录 | `/data/work/VectorDBBench/vectordb_bench/results/` |
| 配置文件目录 | `/data/work/VectorDBBench/vectordb_bench/config-files/` |

---

## 安装

```bash
cd /data/work/VectorDBBench
SETUPTOOLS_SCM_PRETEND_VERSION=v0.1.0 .venv/bin/pip install -e '.[milvus]'
```

---

## 运行方式

### 方式 1：CLI（推荐）

VDBBench 用 click CLI 提供命令行接口。

**HNSW benchmark**：
```bash
cd /data/work/VectorDBBench
DATASET_LOCAL_DIR=/data/work/datasets \
.venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSW \
  --uri http://localhost:19530 \
  --db-label milvus-rs-hnsw \
  --m 16 \
  --ef-construction 200 \
  --ef-search 128 \
  --case-id Performance768D1M \
  2>&1 | tee /tmp/vdb_hnsw_rs.log
```

**HNSW-SQ benchmark**：
```bash
DATASET_LOCAL_DIR=/data/work/datasets \
.venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSWSQ \
  --uri http://localhost:19530 \
  --db-label milvus-rs-hnsw-sq \
  --m 16 --ef-construction 200 --ef-search 128 \
  --sq-type SQ8 --refine true --refine-type FP32 --refine-k 1.0 \
  2>&1 | tee /tmp/vdb_hnsw_sq_rs.log
```

**IVF-SQ8 benchmark**：
```bash
DATASET_LOCAL_DIR=/data/work/datasets \
.venv/bin/python3 -m vectordb_bench.cli.cli MilvusIVFSQ8 \
  --uri http://localhost:19530 \
  --db-label milvus-rs-ivfsq8 \
  --nlist 1024 --nprobe 64 \
  2>&1 | tee /tmp/vdb_ivfsq8_rs.log
```

**DiskANN benchmark**：
```bash
DATASET_LOCAL_DIR=/data/work/datasets \
.venv/bin/python3 -m vectordb_bench.cli.cli MilvusDISKANN \
  --uri http://localhost:19530 \
  --db-label milvus-rs-diskann \
  --search-list 200 \
  2>&1 | tee /tmp/vdb_diskann_rs.log
```

### 方式 2：Streamlit 前端（需要 X 转发）

```bash
.venv/bin/pip install streamlit
.venv/bin/python3 -m vectordb_bench
```

### 方式 3：Python 脚本（适合自定义索引）

```python
import sys
sys.path.insert(0, '/data/work/VectorDBBench')

from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import TaskConfig, DB
from vectordb_bench.backend.clients.milvus.config import HNSWConfig, MilvusConfig
from pydantic import SecretStr

task = TaskConfig(
    db=DB.Milvus,
    db_config=MilvusConfig(uri=SecretStr("http://localhost:19530")),
    db_case_config=HNSWConfig(M=16, efConstruction=200, ef=128),
)
benchmark_runner.run([task], task_label="milvus-rs-hnsw")
```

---

## CLI 支持的 Milvus 命令

| 命令 | 索引类型 | 关键参数 |
|------|---------|---------|
| `MilvusHNSW` | HNSW | `--m`, `--ef-construction`, `--ef-search` |
| `MilvusHNSWSQ` | HNSW+SQ | `--sq-type`, `--refine`, `--refine-type` |
| `MilvusHNSWPQ` | HNSW+PQ | `--nbits`, `--refine`, `--refine-type` |
| `MilvusHNSWPRQ` | HNSW+PRQ | `--nrq`, `--nbits`, `--refine` |
| `MilvusIVFFlat` | IVF-Flat | `--nlist`, `--nprobe` |
| `MilvusIVFSQ8` | IVF-SQ8 | `--nlist`, `--nprobe` |
| `MilvusIVFRabitQ` | IVF-RaBitQ | `--nlist`, `--nprobe`, `--rbq-bits-query` |
| `MilvusDISKANN` | DiskANN | `--search-list` |
| `MilvusAutoIndex` | AutoIndex | — |
| `MilvusFlat` | Flat | — |

---

## 通用 CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--uri` | Milvus 连接 URI | 必填 |
| `--db-label` | 测试标签 | — |
| `--case-id` | 测试 case（数据集+规模）| 必填 |
| `--drop-old` | 重建 collection | True |

---

## CaseType 枚举（数据集规模）

| CaseType | 含义 | 维度 | 向量数 |
|----------|------|------|--------|
| `Performance768D1M` | Cohere-1M | 768 | 1M |
| `Performance768D10M` | Cohere-10M | 768 | 10M |
| `Performance1536D500K` | OpenAI-500K | 1536 | 500K |
| `Performance1024D1M` | 1024D-1M | 1024 | 1M |

---

## RS vs Native 对比流程

1. **RS 版**：默认路由（不带 BYPASS env var）
2. **Native 版**：重启 Milvus 带 `HANNS_HNSW_BYPASS=1` 等环境变量
3. 对比同一 case 下 Load time / QPS / Recall

```bash
# RS
ssh hannsdb-x86 'cd /data/work/VectorDBBench && DATASET_LOCAL_DIR=/data/work/datasets .venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSW --uri http://localhost:19530 --m 16 --ef-construction 200 --ef-search 128 --case-id Performance768D1M --db-label rs-hnsw 2>&1 | tee /tmp/vdb_rs.log'

# Native（需要重启 Milvus + BYPASS）
ssh hannsdb-x86 'pkill -f "milvus run" || true; sleep 5'
ssh hannsdb-x86 'HANNS_HNSW_BYPASS=1 bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh'
sleep 30

ssh hannsdb-x86 'cd /data/work/VectorDBBench && DATASET_LOCAL_DIR=/data/work/datasets .venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSW --uri http://localhost:19530 --m 16 --ef-construction 200 --ef-search 128 --case-id Performance768D1M --db-label native-hnsw 2>&1 | tee /tmp/vdb_native.log'
```

---

## 数据集路径配置

VDBBench 默认数据集目录：`/tmp/vectordb_bench/dataset`

通过环境变量覆盖：
```bash
export DATASET_LOCAL_DIR=/data/work/datasets
```

已有数据集：
- `wikipedia-cohere-1m/` — Cohere 768D 1M（base.fbin + query.fbin + gt.ibin）
- `sift-1m/` — SIFT 128D 1M
- `simplewiki-openai/` — OpenAI 1536D

---

## 注意事项

- **VDBBench 没有 `.git`**：安装时需要 `SETUPTOOLS_SCM_PRETEND_VERSION=v0.1.0`
- **streamlit 需要单独安装**：`.venv/bin/pip install streamlit`
- **数据集目录**：必须设 `DATASET_LOCAL_DIR=/data/work/datasets`，否则会从 S3 下载
- **Milvus 必须在运行**：`curl -s http://127.0.0.1:9091/healthz` 确认
- **每个 case 会重建 collection**：`--drop-old=True`（默认行为）
- **pydantic 必须 <2**：VDBBench 使用 pydantic v1 API（`@validator` 的 `field`/`config` 参数）

---

## HannsDB Standalone VDBB

HannsDB 作为嵌入式数据库（不依赖 Milvus），可直接用 VDBBench 对比性能。

### 客户端代码

`VectorDBBench/vectordb_bench/backend/clients/hannsdb/`：
- `cli.py` — click 命令注册（命令名：`hannsdb`，小写）
- `config.py` — `HannsdbConfig`（path 参数）+ `HannsdbHNSWIndexConfig`（M/ef_construction/ef_search）
- `hannsdb.py` — 实际的 insert/optimize/search 逻辑
  - 使用 `search_ids_raw()` 快速路径（numpy batch）for non-filtered queries
  - 使用 `collection.query()` for filtered queries

### 运行命令

```bash
cd /data/work/VectorDBBench
PYTHONPATH=. python3 -c "
import vectordb_bench.backend.clients.hannsdb.cli
from vectordb_bench.cli.cli import cli
import sys
sys.argv = [
    'vectordbbench',
    'hannsdb',
    '--path', '/tmp/hannsdb-vdbb-1536d50k',
    '--case-type', 'Performance1536D50K',
    '--k', '100',
    '--m', '16',
    '--ef-construction', '64',
    '--ef-search', '32',
    '--skip-search-concurrent',
    '--db-label', 'hannsdb-x86-hnsw-k100',
]
cli()
"
```

### 权威结果（2026-04-10）

| 机器 | k | Load(s) | Optimize(s) | p99(ms) | p95(ms) | Recall | NDCG |
|------|---|---------|-------------|---------|---------|--------|------|
| x86 (94.74.108.167) | 100 | 148.0 | 87.9 | 1.8 | 1.7 | 0.9756 | 0.9801 |
| ARM64 MacBook | 100 | 215.2 | — | 1.4 | 1.0 | 0.9756 | 0.9801 |
| ARM64 MacBook | 10 | 218.8 | — | 0.5 | 0.3 | 0.9441 | — |

---

## Zvec VDBB

Zvec VDBB 客户端在 `VectorDBBench/vectordb_bench/backend/clients/zvec/`：
- `cli.py` — click 命令注册（命令名：`zvec`，小写）
- 默认 HNSW 参数：M=50, ef_construction=500, ef_search=300
- 构建需要 CMake ≥3.26 且 thirdparty 子模块完整

### 运行命令

```bash
cd /data/work/VectorDBBench
PYTHONPATH=. python3 -c "
import vectordb_bench.backend.clients.zvec.cli
from vectordb_bench.cli.cli import cli
import sys
sys.argv = [
    'vectordbbench',
    'zvec',
    '--path', '/tmp/zvec-vdbb-1536d50k',
    '--case-type', 'Performance1536D50K',
    '--k', '100',
    '--m', '16',
    '--ef-construction', '64',
    '--ef-search', '32',
    '--skip-search-concurrent',
    '--db-label', 'zvec-x86-hnsw-k100',
]
cli()
"
```

### Zvec 权威结果（2026-04-10, x86）

| k | Load(s) | p99(ms) | p95(ms) | Recall | NDCG |
|---|---------|---------|---------|--------|------|
| 100 | 13.8 | 2.0 | 1.3 | 0.9286 | 0.941 |

### HannsDB vs Zvec 对比（同机器, HNSW M=16 ef=32 k=100）

| 指标 | HannsDB | Zvec | 分析 |
|------|---------|------|------|
| Load | 148.0s | 13.8s | Zvec 10.7× 快（C++ 原生 + Arrow 列存）|
| p99 | 1.8ms | 2.0ms | 持平 |
| p95 | 1.7ms | 1.3ms | 持平 |
| Recall@100 | **0.9756** | 0.9286 | HannsDB +5.1%（HNSW 图质量更高）|
| NDCG@100 | **0.9801** | 0.941 | HannsDB +4.2% |

---

## 相关页面

- [[machines/hannsdb-x86]] — Milvus 部署环境
- [[concepts/milvus-integration]] — RS 集成架构
- [[benchmarks/hnsw-milvus-rounds]] — HNSW 优化历史
