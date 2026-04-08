# IVF-SQ8 Milvus 调度器并发调优计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 通过调整 Milvus 调度器 `cpuRatio` 配置，突破 IVF-SQ8 的 135 QPS 天花板，目标 500+ QPS @ c=80。

**Root Cause:** 已确认 nq=1（每次 FFI call 只有 1 query）。瓶颈是 Milvus 调度器：
- `maxReadConcurrentRatio: 1` × 16 CPUs = 16 "CPU 容量"
- `cpuRatio: 10` = 每个 IVF 搜索任务占用 10 CPU 单位
- → floor(16/10) = ~1-4 个并发 IVF 搜索任务
- → QPS ceiling = 4 × 40 = 135 QPS

降低 `cpuRatio` 可直接增加并发：
- cpuRatio=1 → 16 并发 → 16 × 40 = 640 QPS（预期）
- cpuRatio=2 → 8 并发 → 320 QPS（保守估计）

**Architecture:** 修改 hannsdb-x86 上的 `milvus.yaml`，重启 Milvus，运行相同 benchmark 对比。

**Tech Stack:** YAML config, Milvus restart, pymilvus benchmark

---

## Context

### 诊断结果（2026-04-08）

```
Milvus proxy log: nq=1 per search call (confirmed)
c=1: 39 QPS → H_eff = 25ms per query
c=10+: 135 QPS ceiling → ~3-4 effective concurrent workers
c=80: 134.7 QPS (no improvement over c=10)
```

### 相关配置（milvus.yaml）

```yaml
queryNode:
  scheduler:
    maxReadConcurrentRatio: 1   # CPUNum × ratio = max concurrent capacity
    cpuRatio: 10                # estimated CPU cost per read task
```

当前有效并发 = floor(16 × 1 / 10) ≈ 1-2。观测到 3-4 是因为 Milvus 的调度有一些 buffer。

### Key Paths

```
Milvus config: hannsdb-x86:/data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml
Startup script: hannsdb-x86:/data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh
Benchmark script: /tmp/ivfsq8_concurrent_bench.py (已存在，nlist=1024, 15s per point)
Results local: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/
```

---

## Task 1: 快速测试 cpuRatio=1

**Purpose:** 用最简单的配置变更验证假设，预期 QPS 从 135 → 500+ QPS。

### Step 1: 备份并修改 milvus.yaml

```bash
ssh hannsdb-x86 'cp /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml.orig_cpuratio'
```

查找 cpuRatio 的当前值并修改为 1：
```bash
ssh hannsdb-x86 'grep -n "cpuRatio\|maxReadConcurrent" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
ssh hannsdb-x86 'sed -i "s/cpuRatio: 10/cpuRatio: 1/" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
ssh hannsdb-x86 'grep "cpuRatio\|maxReadConcurrent" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
```

预期：`cpuRatio: 1`（改变）。

### Step 2: 同时把 maxReadConcurrentRatio 从 1 → 2（默认值）

```bash
ssh hannsdb-x86 'sed -i "s/maxReadConcurrentRatio: 1/maxReadConcurrentRatio: 2/" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
ssh hannsdb-x86 'grep "cpuRatio\|maxReadConcurrent" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
```

预期：cpuRatio=1, maxReadConcurrentRatio=2 → max concurrent = floor(16×2/1) = 32 并发。

### Step 3: 停止并重启 Milvus（无 TRACE_SEARCH，正常模式）

```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone"; sleep 5; echo stopped' 2>&1 || true
ssh hannsdb-x86 'pgrep -f "milvus run standalone" | wc -l'
```

如果进程已停止（count=0），重启：
```bash
ssh hannsdb-x86 'RESET_RUNTIME_STATE=false nohup bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_cpuratio1.log 2>&1 &'
ssh hannsdb-x86 'sleep 20 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"Milvus OK\")"'
```

### Step 4: 快速诊断 benchmark（8s per point, c=1/20/80）

```bash
ssh hannsdb-x86 'timeout 120 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_diag.py 2>&1'
```

（diag.py 已存在于 /tmp/ivfsq8_diag.py，是 8s × 3 concurrency 的短版本）

**预期结果（若假设正确）**：
- c=1: ~39 QPS（不变，H 还是 25ms）
- c=20: 400-500+ QPS（16-32 worker × 25-30 QPS/worker）
- c=80: 500-600+ QPS（资源饱和）

**若无明显改善（c=20 QPS < 100）**: cpuRatio 不是瓶颈，需要查其他 config → 跳到 Task 1B。

---

## Task 1B: 若 Task 1 无效 — 检查 CGO_EXECUTOR_SLOTS 配置

**触发条件**: cpuRatio 修改后 QPS 无明显提升

可能的其他瓶颈：
- Milvus Go CGO executor 并发槽数（GoMaxProcs 或 CGO pool size）
- Milvus segcore 的 search group 大小

检查：
```bash
ssh hannsdb-x86 'grep -i "cgo\|maxproc\|goroutine\|worker\|thread" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml | head -20'
```

如果有 cgoPoolSizeRatio 或类似配置，尝试增大。

---

## Task 2: 若 Task 1 有效 — 运行完整 15s benchmark

**触发条件**: c=20 QPS 显著提升（>200 QPS）

运行和 RS 基准完全相同的 15s benchmark：

```bash
ssh hannsdb-x86 'timeout 900 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_concurrent_bench.py 2>&1 | tee /tmp/ivfsq8_cpuratio1_results.txt'
```

收集结果：
```bash
ssh hannsdb-x86 'cat /tmp/ivfsq8_cpuratio1_results.txt'
```

---

## Task 3: 恢复配置 + 记录结果 + wiki 更新

### Step 1: 恢复原始 cpuRatio（不要让调优影响后续 HNSW 测试）

```bash
ssh hannsdb-x86 'cp /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml.orig_cpuratio /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
ssh hannsdb-x86 'grep "cpuRatio\|maxReadConcurrent" /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml'
```

重启 Milvus（恢复默认配置）：
```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone"; sleep 5'
ssh hannsdb-x86 'RESET_RUNTIME_STATE=false nohup bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_restored.log 2>&1 &'
ssh hannsdb-x86 'sleep 20 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"OK\")"'
```

### Step 2: 保存结果到本地

保存到 `/Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/ivf_sq8_milvus_cpuratio_tuning_2026-04-08.md`

格式：
```markdown
# IVF-SQ8 Milvus cpuRatio 调优 — 2026-04-08

## 背景
- 默认 cpuRatio=10: c=80 QPS = 135
- 调优 cpuRatio=1, maxReadConcurrentRatio=2: c=80 QPS = ?

## 结果对比
| 配置 | c=1 | c=20 | c=80 |
...
```

### Step 3: 更新 wiki/benchmarks/authority-numbers.md

在 IVF-SQ8 Milvus 并发章节补充 cpuRatio 调优数字。

### Step 4: 更新 wiki/log.md

添加条目描述调优方法和结果。

---

## Success Criteria

1. cpuRatio 调优效果验证（提升 vs 无效）
2. 若有效：完整 18-point QPS 表收集
3. Milvus 配置恢复原始值
4. Wiki 更新完成

## Anti-patterns to Avoid

- 不要在 Milvus 生产配置中永久改变 cpuRatio（本次只是研究性测试）
- 不要跳过恢复步骤（后续 HNSW 测试需要原始配置）
- 若 cpuRatio=1 但 QPS 无明显改善：不要继续调参，而是检查 CGO 层（Task 1B）
