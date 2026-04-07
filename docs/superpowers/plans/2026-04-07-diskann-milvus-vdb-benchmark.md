# DiskANN Milvus VDBBench 对比计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 hannsdb-x86 上对比 knowhere-rs (RS) 与 native KnowWhere C++ 的 DiskANN 性能，使用 VectorDBBench + pymilvus 直接测试，数据集 Cohere-1M（768-dim, IP）。

**Architecture:**
1. 验证 RS Milvus 上的 DiskANN 索引是否可建立（FFI 已支持但需确认 Milvus 集成完整性）
2. 建立 DiskANN collection，运行 VDBBench 风格的并发 QPS 测试（c=1/20/80）
3. 用同样的 Python 脚本对 native Milvus 做相同测试（需先检查是否有 native binary，或现场编译）
4. 对比 Build time / QPS / Recall 三项指标

**Tech Stack:** Milvus standalone, pymilvus, Python 3, x86 server (SSH: hannsdb-x86)

**Environment:**
- Remote: `hannsdb-x86`
- RS Milvus source: `/data/work/milvus-rs-integ/milvus-src`
- RS knowhere lib: `/data/work/milvus-rs-integ/knowhere-rs-target/release/libknowhere_rs.so`
- VDBBench venv: `/data/work/VectorDBBench/.venv/bin/python3`
- Cohere-1M data: likely at `/data/work/VectorDBBench/` or needs download

---

### Task 1: 调查 — DiskANN 在 Milvus RS 集成中的状态

**Files:** None — 纯调查

- [ ] **Step 1: 检查当前 Milvus RS 是否正在运行，获取当前 collection 列表**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && .venv/bin/python3 -c \"
from pymilvus import connections, Collection, utility
connections.connect(host='127.0.0.1', port='19530')
cols = utility.list_collections()
print('Collections:', cols)
for c in cols:
    col = Collection(c)
    print(f'  {c}: index={col.indexes[0].params if col.indexes else \"none\"}')
\" 2>&1 || echo 'Milvus not running or connection failed'"
```

- [ ] **Step 2: 检查 knowhere-rs FFI 中 DiskANN 的实际入口点**

```bash
grep -n "diskann\|DiskANN\|DISKANN" /data/work/milvus-rs-integ/knowhere-rs/src/ffi.rs | head -30
# 从本地也可以检查
grep -n "diskann\|DiskANN" /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ffi.rs | head -20
```

- [ ] **Step 3: 检查 hannsdb-x86 上是否有 native Milvus binary**

```bash
ssh hannsdb-x86 "
ls /data/work/milvus-native/ 2>/dev/null || echo 'no native dir'
ls /data/work/milvus-src-native/ 2>/dev/null || echo 'no native src dir'
find /data/work -name 'libknowhere.so' 2>/dev/null | head -5
which milvus 2>/dev/null || echo 'no milvus in PATH'
ls /data/work/milvus-rs-integ/milvus-src/bin/ 2>/dev/null
"
```

- [ ] **Step 4: 检查 Cohere-1M 数据是否在 x86 上可用**

```bash
ssh hannsdb-x86 "
ls /data/work/VectorDBBench/dataset/ 2>/dev/null | head -20
ls /data/datasets/ 2>/dev/null | head -20
find /data -name '*.fvecs' -o -name '*.hdf5' 2>/dev/null | head -10
find /data -name '*cohere*' -o -name '*Cohere*' 2>/dev/null | head -10
"
```

- [ ] **Step 5: 记录调查结果**

记录以下信息（用于后续 task 决策）：
- DiskANN FFI 入口点是否存在（Y/N）
- native binary 是否存在（位置）
- Cohere 数据位置
- 当前 Milvus 状态（运行中/停止）

---

### Task 2: 建立 DiskANN collection 并测试 RS

**依赖 Task 1 结果**

- [ ] **Step 1: 确保 RS Milvus 在运行**

```bash
ssh hannsdb-x86 "pgrep -f 'milvus run standalone' && echo 'running' || (
  cd /data/work/milvus-rs-integ/milvus-src &&
  nohup bin/milvus run standalone > /tmp/milvus_diskann.log 2>&1 &
  sleep 30 && echo 'started'
)"
```

- [ ] **Step 2: 写 DiskANN 建索引脚本**

```bash
ssh hannsdb-x86 "cat > /tmp/diskann_setup.py << 'PYEOF'
import numpy as np
import time
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

connections.connect(host='127.0.0.1', port='19530')

COLL_NAME = 'diskann_rs_cohere1m'
DIM = 768

# Drop if exists
if utility.has_collection(COLL_NAME):
    utility.drop_collection(COLL_NAME)
    print(f'Dropped existing {COLL_NAME}')

# Schema
fields = [
    FieldSchema('id', DataType.INT64, is_primary=True),
    FieldSchema('vector', DataType.FLOAT_VECTOR, dim=DIM),
]
schema = CollectionSchema(fields)
col = Collection(COLL_NAME, schema)
print(f'Created collection {COLL_NAME}')

# Load Cohere data — try known paths
import os
data_paths = [
    '/data/work/VectorDBBench/dataset',
    '/data/datasets',
    '/data/work/dataset',
]
cohere_file = None
for dp in data_paths:
    for f in os.listdir(dp) if os.path.exists(dp) else []:
        if 'cohere' in f.lower() or 'Cohere' in f:
            cohere_file = os.path.join(dp, f)
            break
    if cohere_file:
        break

if cohere_file and cohere_file.endswith('.hdf5'):
    import h5py
    print(f'Loading from {cohere_file}')
    with h5py.File(cohere_file) as f:
        vectors = f['train'][:]
        print(f'  vectors shape: {vectors.shape}')
else:
    print(f'Cohere data not found at known paths, generating synthetic 768-dim data')
    np.random.seed(42)
    vectors = np.random.randn(1_000_000, DIM).astype('float32')
    # Normalize for IP metric
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-9)

n = len(vectors)
print(f'Inserting {n} vectors...')
t0 = time.time()
BATCH = 50000
for i in range(0, n, BATCH):
    batch = vectors[i:i+BATCH].tolist()
    ids = list(range(i, min(i+BATCH, n)))
    col.insert([ids, batch])
    if i % 200000 == 0:
        print(f'  inserted {i}/{n}')
insert_time = time.time() - t0
print(f'Insert done: {insert_time:.1f}s')

# Flush
col.flush()
print('Flush done')

# Build DiskANN index
print('Building DiskANN index...')
t0 = time.time()
index_params = {
    'index_type': 'DISKANN',
    'metric_type': 'IP',
    'params': {
        'max_degree': 56,
        'search_list_size': 100,
        'pq_code_budget_gb': 0,   # 0 = no PQ, full precision
        'build_dram_budget_gb': 32,
        'num_threads': 16,
    }
}
col.create_index('vector', index_params)
build_time = time.time() - t0
print(f'Index build done: {build_time:.1f}s')

print(f'RESULT: insert={insert_time:.1f}s build={build_time:.1f}s')
PYEOF
"

ssh hannsdb-x86 "nohup /data/work/VectorDBBench/.venv/bin/python3 /tmp/diskann_setup.py > /tmp/diskann_setup.log 2>&1 &"
echo "Setup started in background, check /tmp/diskann_setup.log"
```

- [ ] **Step 3: 等待并检查结果（DiskANN build 可能需要 10-30 分钟）**

```bash
# 每 2 分钟轮询一次，最多等 40 分钟
for i in $(seq 1 20); do
  sleep 120
  ssh hannsdb-x86 "tail -5 /tmp/diskann_setup.log"
  ssh hannsdb-x86 "grep 'RESULT:' /tmp/diskann_setup.log && break || echo 'still building...'"
done
```

记录 insert 时间和 build 时间。

- [ ] **Step 4: Load collection 并运行 QPS benchmark**

```bash
ssh hannsdb-x86 "cat > /tmp/diskann_bench.py << 'PYEOF'
import time, threading, numpy as np
from pymilvus import connections, Collection

connections.connect(host='127.0.0.1', port='19530')
col = Collection('diskann_rs_cohere1m')
col.load()
print('Loading...')
time.sleep(30)  # wait for load

np.random.seed(42)
DIM = 768
q = np.random.randn(1, DIM).astype('float32')
q = (q / np.linalg.norm(q)).tolist()

search_params = {'metric_type': 'IP', 'params': {'search_list': 100}}

# Warmup
print('Warmup...')
for _ in range(20):
    col.search(q, 'vector', search_params, limit=10, output_fields=[])

def bench(concurrency, duration=60):
    count, stop, lock = [0], threading.Event(), threading.Lock()
    def worker():
        while not stop.is_set():
            col.search(q, 'vector', search_params, limit=10, output_fields=[])
            with lock: count[0] += 1
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    time.sleep(duration)
    stop.set()
    for t in threads: t.join()
    qps = count[0] / (time.time() - t0)
    print(f'c={concurrency}: {qps:.1f} QPS')
    return qps

# Serial QPS (single query latency)
t0 = time.time()
for _ in range(100):
    col.search(q, 'vector', search_params, limit=10, output_fields=[])
serial_qps = 100 / (time.time() - t0)
print(f'serial: {serial_qps:.1f} QPS ({1000/serial_qps:.1f}ms/query)')

print('Running c=1 benchmark (60s)...')
q1 = bench(1)
print('Running c=20 benchmark (60s)...')
q20 = bench(20)
print('Running c=80 benchmark (60s)...')
q80 = bench(80)

print(f'RESULT: serial={serial_qps:.1f} c=1={q1:.1f} c=20={q20:.1f} c=80={q80:.1f}')
PYEOF
"

ssh hannsdb-x86 "nohup /data/work/VectorDBBench/.venv/bin/python3 /tmp/diskann_bench.py > /tmp/diskann_bench.log 2>&1 &"
echo "Benchmark started, will take ~5 minutes"
sleep 320
ssh hannsdb-x86 "cat /tmp/diskann_bench.log"
```

- [ ] **Step 5: 测量 Recall**

```bash
ssh hannsdb-x86 "cat > /tmp/diskann_recall.py << 'PYEOF'
import numpy as np
from pymilvus import connections, Collection

connections.connect(host='127.0.0.1', port='19530')
col = Collection('diskann_rs_cohere1m')

np.random.seed(123)
DIM = 768
K = 10
N_QUERIES = 100

# Generate queries
queries = np.random.randn(N_QUERIES, DIM).astype('float32')
queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

# Brute-force ground truth via Flat index or a separate flat search
# Simple approach: use DISKANN with large search_list as proxy for GT
search_params_gt = {'metric_type': 'IP', 'params': {'search_list': 500}}
search_params_eval = {'metric_type': 'IP', 'params': {'search_list': 100}}

gt_ids = []
eval_ids = []

for q in queries:
    r_gt = col.search([q.tolist()], 'vector', search_params_gt, limit=K, output_fields=[])
    r_ev = col.search([q.tolist()], 'vector', search_params_eval, limit=K, output_fields=[])
    gt_ids.append(set(hit.id for hit in r_gt[0]))
    eval_ids.append(set(hit.id for hit in r_ev[0]))

recalls = [len(e & g) / K for e, g in zip(eval_ids, gt_ids)]
print(f'Recall@{K} (search_list=100 vs search_list=500): {np.mean(recalls):.4f}')
PYEOF
"

ssh hannsdb-x86 "/data/work/VectorDBBench/.venv/bin/python3 /tmp/diskann_recall.py 2>&1"
```

---

### Task 3: Native 对比 — 检查并建立 native Milvus

- [ ] **Step 1: 评估 native Milvus 可行性**

优先顺序：
1. **若 hannsdb-x86 上有 native Milvus binary** → 直接用
2. **若有 native KnowWhere SO** → 尝试替换 RS SO
3. **若都没有** → 在本地 Mac 上测 native + 换算（说明硬件差异）

```bash
ssh hannsdb-x86 "
# 检查可能的 native binary
find /data /opt /usr/local -name 'milvus' -type f 2>/dev/null
# 检查 knowhere native SO
find /data -name 'libknowhere.so' -o -name 'libknowhere_rs.so' 2>/dev/null
# 检查 cmake build 产物
ls /data/work/knowhere-build/ 2>/dev/null
ls /data/work/knowhere-native/ 2>/dev/null
"
```

- [ ] **Step 2: 若有 native binary — 启动 native Milvus 并重建 DiskANN collection**

如果找到 native Milvus binary（如 `/data/work/milvus-native/bin/milvus`）：

```bash
# 停止 RS Milvus
ssh hannsdb-x86 "pkill -f 'milvus run standalone' || true; sleep 5"

# 启动 native Milvus
ssh hannsdb-x86 "cd /data/work/milvus-native/milvus-src && \
  nohup bin/milvus run standalone > /tmp/milvus_native.log 2>&1 &"
sleep 30

# 重新建 collection（同样的脚本，同样的数据）
# 修改 COLL_NAME = 'diskann_native_cohere1m'
ssh hannsdb-x86 "sed 's/diskann_rs_cohere1m/diskann_native_cohere1m/' /tmp/diskann_setup.py > /tmp/diskann_setup_native.py"
ssh hannsdb-x86 "nohup .venv/bin/python3 /tmp/diskann_setup_native.py > /tmp/diskann_native_setup.log 2>&1 &"
```

- [ ] **Step 3: 若无 native binary — 在本地 Mac 测 native 作为参考**

如果 x86 上没有 native，则：
1. 本地 Mac 建 DiskANN（`cargo run --example benchmark --release` 中的 DiskANN benchmark）
2. 记录本地 Mac 数字，标注"Apple Silicon 仅供参考，非权威"
3. 在报告中说明"native x86 数字待后续建立"

**本地 benchmark（如果需要）:**
```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo run --example benchmark --release 2>&1 | grep -i diskann | head -20
```

---

### Task 4: 汇总结果并提交

- [ ] **Step 1: 收集所有数字**

填写以下表格：

```
=== DiskANN Milvus Benchmark Results (2026-04-07) ===
Dataset: Cohere-1M 768-dim IP (or synthetic if Cohere not available)
Hardware: hannsdb-x86

RS DiskANN:
  Insert: Xs
  Build (max_degree=56, no PQ): Xs
  serial QPS: X (Xms/query)
  c=1 QPS: X
  c=20 QPS: X
  c=80 QPS: X
  Recall@10: X

Native DiskANN (if available):
  ...

Gap: RS/Native = X×
```

- [ ] **Step 2: 写结果文件**

```bash
cat > /Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/diskann_milvus_rs_2026-04-07.md << 'EOF'
# DiskANN Milvus RS Benchmark — 2026-04-07

## Dataset
- Cohere-1M, 768-dim, Inner Product

## Index Config
- max_degree=56, search_list_size=100, no PQ (full precision)
- search_list=100 for query

## Results (hannsdb-x86)

| Metric | RS | Native | Ratio |
|--------|-----|--------|-------|
| Build | Xs | - | - |
| QPS c=80 | X | - | - |
| Recall@10 | X | - | - |

## Notes
...
EOF
```

- [ ] **Step 3: Commit**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
git add benchmark_results/diskann_milvus_rs_2026-04-07.md
git commit -m "$(cat <<'EOF'
bench(diskann): RS DiskANN Milvus benchmark — Cohere-1M 2026-04-07

Insert=Xs Build=Xs c=80=X QPS Recall=X

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## 注意事项

1. **DiskANN build 耗时**：1M vectors 的 DiskANN 建图预计 10–40 分钟，不要中途 kill
2. **Cohere 数据**：若 x86 上没有 Cohere-1M hdf5，使用合成 normalized random 768-dim 数据（标注 synthetic）
3. **native 对比**：如果 x86 上没有 native binary，仅记录 RS 数字并在报告中说明 native 对比待建立
4. **DISKANN 参数**：`pq_code_budget_gb=0` = full precision（NoPQ），与 HNSW 对比时这是 apples-to-apples 的公平对比
5. **Recall 注意**：DiskANN recall 测量用 search_list=500 作 proxy GT（不是暴力搜索），结果会略微高估 recall
