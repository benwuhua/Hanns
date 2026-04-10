# knowhere-x86-hk-proxy

**角色**：x86 benchmark 辅助机器（HannsDB standalone + Zvec VDBB 对比）

---

## 硬件

| 项目 | 值 |
|------|-----|
| SSH alias | `knowhere-x86-hk-proxy` |
| IP | 94.74.108.167 |
| 用户 | root |
| 连接方式 | SOCKS5 proxy |
| Python | 3.12（系统） |

---

## 关键路径

| 用途 | 路径 |
|------|------|
| HannsDB 源码 | `/data/work/HannsDB/` |
| Hanns 库（knowhere-rs） | `/data/work/knowhere-rs/` |
| Zvec 源码 | `/data/work/zvec/` |
| VDBBench | `/data/work/VectorDBBench/` |
| VDBBench 数据集缓存 | `/tmp/vectordb_bench/dataset/` |
| HannsDB Python 绑定 | 已 `pip install` 到系统 Python |
| VDBBench 结果 | `/data/work/VectorDBBench/vectordb_bench/results/` |

---

## HannsDB 构建

```bash
ssh knowhere-x86-hk-proxy 'cd /data/work/HannsDB && \
  cargo build --release --features knowhere-backend 2>&1 | tail -5'
```

### Python 绑定

```bash
ssh knowhere-x86-hk-proxy 'cd /data/work/HannsDB && \
  pip3 install --break-system-packages -e . --no-build-isolation 2>&1 | tail -10'
```

---

## VDBBench 运行

### HannsDB HNSW benchmark

```bash
ssh knowhere-x86-hk-proxy 'cd /data/work/VectorDBBench && \
  PYTHONPATH=. python3 -c "
import vectordb_bench.backend.clients.hannsdb.cli
from vectordb_bench.cli.cli import cli
import sys
sys.argv = [
    \"vectordbbench\",
    \"hannsdb\",
    \"--path\", \"/tmp/hannsdb-vdbb-1536d50k\",
    \"--case-type\", \"Performance1536D50K\",
    \"--k\", \"100\",
    \"--m\", \"16\",
    \"--ef-construction\", \"64\",
    \"--ef-search\", \"32\",
    \"--skip-search-concurrent\",
    \"--db-label\", \"hannsdb-x86-hnsw-k100\",
]
cli()
"'
```

### Zvec benchmark（待构建完成）

```bash
# 类似方式，用 zvec CLI command
```

---

## 已安装的 Python 依赖

VDBBench 运行所需的关键包（系统 Python 3.12）：

- `pydantic<2`（VDBBench 不兼容 pydantic v2）
- `environs`
- `polars`
- `hdrhistogram`（提供 `hdrh` 模块）
- `s3fs`
- `click`
- `numpy`

---

## 注意事项

- **pydantic 必须 <2**：VDBBench 使用 pydantic v1 API（`@validator` 的 `field`/`config` 参数），pydantic v2 会报错
- **数据集 rsync**：先在本地缓存 `/tmp/vectordb_bench/dataset/openai/openai_small_50k/`，然后 rsync 到远程
- **HannsDB 依赖 knowhere-rs**：需确保 `/data/work/knowhere-rs/` 存在且 HannsDB `Cargo.toml` 指向正确路径
- **SSH 通过 proxy**：需配置 `~/.ssh/config` 的 SOCKS5 proxy

---

## 相关页面

- [[machines/hannsdb-x86]] — Milvus 集成权威机器（189.1.218.159）
- [[concepts/vdbbench]] — VDBBench 使用指南
