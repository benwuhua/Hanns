# hannsdb-x86

**角色**：Milvus RS 集成测试的权威 benchmark 机器（x86）。所有 Milvus 集成数字以此为准。

---

## 硬件

| 项目 | 值 |
|------|-----|
| SSH alias | `hannsdb-x86` |
| IP | 189.1.218.159 |
| CPU cores | 16 |
| CGO_EXECUTOR_SLOTS | 32（= ceil(16 × 2.0)）|
| HNSW_NQ_POOL threads | 4（= (16-32).max(4)）|

---

## 关键路径

| 用途 | 路径 |
|------|------|
| hanns 源码 | `/data/work/milvus-rs-integ/hanns/` |
| hanns build cache | `/data/work/milvus-rs-integ/hanns-target/` |
| Milvus 源码 | `/data/work/milvus-rs-integ/milvus-src/` |
| Milvus 二进制 | `/data/work/milvus-rs-integ/milvus-src/bin/milvus` |
| RS shim 源码 | `milvus-src/internal/core/thirdparty/knowhere-rs-shim/` |
| shim ABI header | `knowhere-rs-shim/src/cabi_bridge.hpp` |
| DiskANN shim | `knowhere-rs-shim/src/diskann_rust_node.cpp` |
| HNSW shim | `knowhere-rs-shim/src/hnsw_rust_node.cpp` |
| SIFT-1M 数据 | `/data/work/datasets/sift-1m/`（base.fbin, query.fbin, gt.ibin）|
| VectorDBBench venv | `/data/work/VectorDBBench/.venv/bin/python3` |

---

## 常用命令

### hanns 构建

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/hanns && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/hanns-target \
  ~/.cargo/bin/cargo build --release 2>&1 | tail -5'
```

### Milvus shim 重建

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/cmake_build && make -j8 knowhere 2>&1 | tail -20'
```

### Milvus 重启（保留数据）

```bash
ssh hannsdb-x86 'pkill -f "milvus run" || true; sleep 3'
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ && \
  MILVUS_RS_RESET_RUNTIME_STATE=false \
  bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh
sleep 30 && ssh hannsdb-x86 'curl -s http://127.0.0.1:9091/healthz'
```

### Milvus 重启（清除数据）

去掉 `MILVUS_RS_RESET_RUNTIME_STATE=false`，或设为 `true`。

### rsync 代码（⚠️ 必须 exclude sift data）

```bash
rsync -az --exclude=target --exclude='data/' \
  /Users/ryan/Code/vectorDB/Hanns/ \
  hannsdb-x86:/data/work/milvus-rs-integ/hanns/
```

---

## 注意事项

- **rsync 务必加 `--exclude='data/'`**，否则会删除 SIFT-1M 数据（路径 `/data/work/hanns-src/data/sift`，实际在 `/data/work/datasets/sift-1m/`）
- **Milvus 重启时 etcd 配置**：必须用 `start_standalone_remote.sh` 脚本，该脚本包含 `ETCD_USE_EMBED=true` 等环境变量。直接 pkill + 重启 milvus 会导致 etcd "context deadline exceeded" 崩溃
- **Collection 数据会在 Milvus 重启后丢失**（如果 etcd 被覆盖）：每次 benchmark 前确认 Milvus 状态，或重建 collection
- **Milvus 集成需要 Cargo 1.94.0**：`~/.cargo/bin/cargo`（不是系统 cargo）

---

## shim git 仓库

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim && git log --oneline -5'
```

---

## 相关页面

- [[concepts/milvus-integration]] — Milvus 集成架构
- [[concepts/hnsw]] — HNSW Milvus 集成
- [[concepts/diskann]] — DiskANN Milvus 集成
