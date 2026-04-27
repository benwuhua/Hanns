# Hanns vs Zilliz Knowhere 当前比较总结（2026-04-24）

> **Authority surface:** HannsDB-x86 only
> **Official Knowhere source:** `https://github.com/zilliztech/knowhere` @ `868634bc5230a71a84371fea5f5b674b805643ea`
> **Status:** 当前是 **interim comparison**，不是最终 apples-to-apples verdict。

## 范围

本轮已经收集到下列 family 的 fresh authority evidence：

- 官方 Zilliz Knowhere：HNSW、IVF-SQ、IVF-PQ、DISKANN、AISAQ_P、AISAQ_S
- Hanns：DiskANN/AISAQ 线、HNSW、IVF-Flat、IVF-PQ、IVF-SQ8

仍需保持的边界：

- 官方与 Hanns 不一定来自相同 runner / 相同参数选择逻辑。
- 官方 `IVF-PQ` terminal row 需要 parser normalization，不能直接拿 terminal 输出当最终 verdict。
- Hanns `BENCH-051` 明确声明它当前是 constrained Rust AISAQ skeleton，不是 native-comparable SSD DiskANN pipeline。

## 1. HNSW

### 官方 Zilliz Knowhere

Near-0.95 recall 行（来自 `benchmark_float_qps`）:

| Variant | Recall@100 | Param | VPS@8T | Build(s) |
|---|---:|---|---:|---:|
| FP32 | 0.9502 | `ef=138` | 14546.807 | 38.275 |
| FP16 | 0.9505 | `ef=139` | 15800.933 | 38.275 |
| BF16 | 0.9504 | `ef=139` | 15672.165 | 38.142 |

### Hanns

来自 `BENCH-038` 参数扫描：

| ef_search | Build(ms) | QPS | R@10 |
|---:|---:|---:|---:|
| 64  | 1374896.88 | 1773 | 1.0000 |
| 128 | 1470908.57 | 1894 | 1.0000 |
| 256 | 1064607.59 | 1644 | 1.0000 |

附加 recall-only lane（`bench_sift1m_hnsw_quick`）:

- `ef=400`: `R@1=1.000`, `R@10=1.000`, `R@100=1.000`
- `ef=800`: `R@1=1.000`, `R@10=1.000`, `R@100=1.000`
- graph stats: `max_layer=4`, `avg_neighbors_l0=39.5`

### 当前判断

- Hanns 的 HNSW recall 证据目前非常强。
- 但官方侧是 `benchmark_float_qps` VPS runner，Hanns 侧当前主要来自参数扫描和 recall lane，**吞吐并不完全 apples-to-apples**。
- 因此本轮只能说：**Hanns HNSW quality evidence is strong; final throughput verdict still needs a stricter aligned runner.**

## 2. IVF-SQ

### 官方 Zilliz Knowhere

| Variant | Recall@100 | Param | VPS@8T |
|---|---:|---|---:|
| FP16 | 0.9519 | `nprobe=32` | 8091.813 |
| BF16 | 0.9519 | `nprobe=32` | 8179.410 |

### Hanns

| nprobe | Build(ms) | QPS | R@10 |
|---:|---:|---:|---:|
| 1   | 66800.91 | 10618 | 0.4130 |
| 5   | 67751.20 | 7761  | 0.7880 |
| 10  | 67931.14 | 5934  | 0.8720 |
| 50  | 67381.13 | 4048  | 0.9760 |
| 100 | 68103.08 | 2847  | 0.9620 |

### 当前判断

- Hanns `IVF-SQ8` 已经收到了完整 `nprobe` 扫描结果。
- Hanns 在高召回设置（`nprobe=50/100`）下 recall 很强，但吞吐低于官方 near-0.95 rows。
- 这是目前最接近可以直接读出的对照之一，但仍需要注明 runner/selection 逻辑不同。

## 3. IVF-PQ

### 官方 Zilliz Knowhere

- runner 已跑完，但 terminal row 输出不能直接拿来当 verdict。
- sweep 迹象显示 recall 大致 plateau 在 `≈0.7841`。

### Hanns

| nprobe | Build(ms) | QPS | R@10 |
|---:|---:|---:|---:|
| 1   | 93809.68 | 26951 | 0.3520 |
| 5   | 93497.81 | 7674  | 0.5370 |
| 10  | 93255.32 | 4207  | 0.5810 |
| 50  | 92713.96 | 882   | 0.5820 |
| 100 | 93312.12 | 460   | 0.5910 |

### 当前判断

- 这轮数据清楚显示：**Hanns IVF-PQ recall 仍明显偏低**。
- 官方 IVF-PQ 也没有到高召回行，但仍高于 Hanns 当前扫描结果。
- 因此这一 family 当前更接近“问题已被证据确认”，而不是接近最终性能对比结论。

## 4. DISKANN / AISAQ

### 官方 Zilliz Knowhere

| Variant | Target band | Recall@100 | Param | VPS@8T | Build(s) |
|---|---:|---:|---|---:|---:|
| DISKANN | ~0.80 lane | 0.9217 | `search_list_size=100` | 3847.820 | 232.375 |
| DISKANN | ~0.95 lane | 0.9504 | `search_list_size=108` | 3153.907 | 232.375 |
| AISAQ_P | ~0.80 lane | 0.9214 | `search_list_size=100` | 3328.575 | 214.674 |
| AISAQ_P | ~0.95 lane | 0.9502 | `search_list_size=108` | 3117.801 | 214.674 |
| AISAQ_S | ~0.80 lane | 0.9192 | `search_list_size=100` | 557.301 | 159.796 |
| AISAQ_S | ~0.95 lane | 0.9515 | `search_list_size=109` | 520.390 | 159.796 |

### Hanns

来自 `BENCH-051`:

| Config | Build(s) | QPS | Recall@10 |
|---|---:|---:|---:|
| R=32-B=4 | 22.94 | 3248 | 0.966 |
| R=48-B=8 | 38.24 | 1240 | 0.999 |
| R=64-B=16 | 56.59 | 452 | 0.999 |

### 当前判断

- Hanns 数字看起来不差，但这条 harness 自己已经明确写了：
  > constrained Rust AISAQ skeleton, not native-comparable SSD DiskANN pipeline
- 所以 **不能**据此宣称 Hanns DiskANN 领先或等价于官方 DISKANN/AISAQ。
- 这条线当前最适合作为“功能/趋势性进度证据”，不是最终排名证据。

## 当前总体结论

### 已能较清楚看到的信号

1. **Hanns HNSW recall 非常强**，但还缺严格对齐吞吐 runner 才能下最终性能 verdict。
2. **Hanns IVF-SQ8 已经有完整扫描结果**，高召回下表现较强，但吞吐较官方 near-0.95 rows 偏低。
3. **Hanns IVF-PQ 仍是明确短板**，当前 recall 远低于理想高召回目标。
4. **Hanns DiskANN/AISAQ 线不能直接当成 native-comparable SSD DiskANN 结论**。

### 仍然不能说的话

- 不能说“Hanns 全面领先官方 Knowhere”。
- 不能说“Hanns DiskANN 已经和官方 DISKANN 等价”。
- 不能把官方 IVF-PQ terminal row 直接拿来做最终结论。

## 下一步最值得做的事

1. 如果要真正收口 HNSW 性能结论：给 Hanns 补一条更贴近官方 `benchmark_float_qps` 的 HNSW throughput lane。
2. 如果要真正收口 IVF-PQ：先承认当前 sub-gate 现实，再决定是修实现还是降级结论。
3. 如果要真正收口 DiskANN/AISAQ：需要一条更接近官方 SSD / search_list_size sweep 语义的 Hanns lane。
