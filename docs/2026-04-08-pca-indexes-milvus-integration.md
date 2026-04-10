# PCA Indexes Milvus Integration 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 HNSW-PCA-SQ、HNSW-PCA-USQ、DiskANN-PCA-USQ 三个索引集成到 Milvus FFI，在 hannsdb-x86 用 VDBBench 跑 benchmark。

**Architecture:** 在 `src/ffi.rs` 的 `IndexWrapper` 中添加三个新 Option 字段，扩展 `CIndexType` 枚举，在 train/add/search 路由中添加新分支。CIndexConfig 新增 `pca_dim` 字段传递 PCA 维度。

**Tech Stack:** Rust FFI (`#[repr(C)]`), C++ shim (hannsdb-x86), Python VDBBench

---

### Task 1: FFI — 扩展 CIndexType + CIndexConfig

**Files:**
- Modify: `src/ffi.rs:76-97` (CIndexType enum)
- Modify: `src/ffi.rs:112-163` (CIndexConfig struct + Default)

**Step 1: 添加三个新枚举值**

```rust
pub enum CIndexType {
    // ... existing ...
    DiskAnn = 19,
    HnswPcaSq = 20,
    HnswPcaUsq = 21,
    DiskAnnPcaUsq = 22,
}
```

**Step 2: CIndexConfig 添加 pca_dim 字段**

在 `pq_nbits` 之后追加：
```rust
pub pca_dim: usize,  // PCA target dimensionality (0 = no PCA)
```

Default impl 追加：`pca_dim: 0,`

**Step 3: 构建验证**
```bash
cargo build 2>&1 | grep "^error"
```

---

### Task 2: FFI — 扩展 IndexWrapper

**Files:**
- Modify: `src/ffi.rs:311-331` (IndexWrapper struct)

**Step 1: 添加三个 Option 字段**

```rust
struct IndexWrapper {
    // ... existing ...
    diskann: Option<crate::faiss::diskann_aisaq::PQFlashIndex>,
    hnsw_pca_sq: Option<crate::faiss::HnswPcaSqIndex>,
    hnsw_pca_usq: Option<crate::faiss::HnswPcaUsqIndex>,
    diskann_pca_usq: Option<crate::faiss::DiskAnnPcaUsqIndex>,
    dim: usize,
    nprobe: usize,
}
```

**Step 2: 更新所有 IndexWrapper 构造位置**

所有 `Some(Self { ... })` 块必须添加：
```rust
hnsw_pca_sq: None,
hnsw_pca_usq: None,
diskann_pca_usq: None,
```

搜索所有 `diskann: None,` + `dim,` 的组合，在其间插入三个新字段。

**Step 3: 构建验证**

---

### Task 3: FFI — 添加 build/train/add/search 路由

**Files:**
- Modify: `src/ffi.rs` — 4 个函数各加分支

**Step 1: IndexWrapper::new() — 三个新 build 分支**

HnswPcaSq:
```rust
CIndexType::HnswPcaSq => {
    let pca_dim = if config.pca_dim > 0 { config.pca_dim } else { dim / 2 };
    let idx = crate::faiss::HnswPcaSqIndex::new(crate::faiss::HnswPcaSqConfig {
        dim,
        pca_dim,
        m: config.ef_construction.min(32).max(4),
        ef_construction: config.ef_construction.max(50),
        ef_search: config.ef_search.max(10),
    });
    // ... Self { hnsw_pca_sq: Some(idx), ... }
}
```

HnswPcaUsq 和 DiskAnnPcaUsq 类似。

**Step 2: train() — 三个新分支**

```rust
} else if let Some(ref mut idx) = self.hnsw_pca_sq {
    idx.train(vectors).map_err(|_| CError::Internal)?;
    Ok(())
} else if let Some(ref mut idx) = self.hnsw_pca_usq {
    idx.train(vectors).map_err(|_| CError::Internal)?;
    Ok(())
} else if let Some(ref mut idx) = self.diskann_pca_usq {
    idx.build(vectors, vectors.len() / self.dim, self.dim).map_err(|_| CError::Internal)
}
```

**Step 3: add() — 三个新分支**

**Step 4: search() — 三个新分支**

**Step 5: cargo test 验证**

---

### Task 4: 远端集成 — rsync + build + Milvus restart

**Step 1: rsync 到 hannsdb-x86**
```bash
rsync -az --exclude=target --exclude='data/' \
  /Users/ryan/Code/vectorDB/Hanns/ \
  hannsdb-x86:/data/work/milvus-rs-integ/hanns/
```

**Step 2: 远端编译 hanns**
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/hanns && CARGO_TARGET_DIR=/data/work/milvus-rs-integ/hanns-target ~/.cargo/bin/cargo build --release 2>&1 | tail -10'
```

**Step 3: 重建 Milvus shim**
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/cmake_build && make -j8 knowhere 2>&1 | tail -20'
```

**Step 4: 重启 Milvus**
```bash
ssh hannsdb-x86 'pkill -f "milvus run" || true; sleep 5'
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ && bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh'
sleep 30 && ssh hannsdb-x86 'curl -s http://127.0.0.1:9091/healthz'
```

---

### Task 5: VDBBench — 跑 HNSW benchmark 作为 baseline

用已有的 HNSW 跑一个 Cohere-1M benchmark 确认环境正常：

```bash
ssh hannsdb-x86 'cd /data/work/VectorDBBench && DATASET_LOCAL_DIR=/data/work/datasets .venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSW --uri http://localhost:19530 --m 16 --ef-construction 200 --ef-search 128 --case-id Performance768D1M --db-label rs-hnsw-baseline 2>&1 | tee /tmp/vdb_hnsw_baseline.log'
```

---

### Task 6: VDBBench — 跑 HNSW-SQ + IVF-SQ8 benchmark

```bash
# HNSW-SQ
ssh hannsdb-x86 'cd /data/work/VectorDBBench && DATASET_LOCAL_DIR=/data/work/datasets .venv/bin/python3 -m vectordb_bench.cli.cli MilvusHNSWSQ --uri http://localhost:19530 --m 16 --ef-construction 200 --ef-search 128 --sq-type SQ8 --refine true --refine-type FP32 --refine-k 1.0 --case-id Performance768D1M --db-label rs-hnsw-sq 2>&1 | tee /tmp/vdb_hnsw_sq.log'

# IVF-SQ8
ssh hannsdb-x86 'cd /data/work/VectorDBBench && DATASET_LOCAL_DIR=/data/work/datasets .venv/bin/python3 -m vectordb_bench.cli.cli MilvusIVFSQ8 --uri http://localhost:19530 --nlist 1024 --nprobe 64 --case-id Performance768D1M --db-label rs-ivfsq8 2>&1 | tee /tmp/vdb_ivfsq8.log'
```

---

### Task 7: 记录结果到 wiki

提取 Load time / QPS / Recall，追加到 `wiki/benchmarks/authority-numbers.md` 和 `wiki/log.md`。
