# Hanns vs Zilliz Knowhere progress summary (2026-04-24)

Authority: `HannsDB-x86`
Official baseline: `https://github.com/zilliztech/knowhere` @ `868634bc5230a71a84371fea5f5b674b805643ea`

## Completed official-side runs

| Family | Status | Key point |
|---|---|---|
| HNSW | done | near-0.95 recall row captured for FP32/FP16/BF16 |
| IVF-SQ | done | near-0.95 recall row captured for FP16/BF16 |
| IVF-PQ | done_but_needs_careful_interpretation | sweep rows plateau ~0.784 recall; terminal row needs parser normalization |
| DISKANN | done | near-0.95 recall row captured at search_list_size=108 |
| AISAQ_P | done | near-0.95 recall row captured at search_list_size=108 |
| AISAQ_S | done | near-0.95 recall row captured at search_list_size=109 |

## Completed Hanns-side runs

| Family | Config / Params | Build | QPS | Recall |
|---|---|---:|---:|---:|
| DiskANN/AISAQ | R=32-B=4 | 22.94s | 3248 | 0.966 |
| DiskANN/AISAQ | R=48-B=8 | 38.24s | 1240 | 0.999 |
| DiskANN/AISAQ | R=64-B=16 | 56.59s | 452 | 0.999 |
| HNSW | ef=64 | 1374896.88ms | 1773 | 1.0000 |
| HNSW | ef=128 | 1470908.57ms | 1894 | 1.0000 |
| HNSW | ef=256 | 1064607.59ms | 1644 | 1.0000 |
| IVF-Flat | nprobe=1 | 61856.59ms | 15398 | 0.3550 |
| IVF-Flat | nprobe=5 | 61074.31ms | 6574 | 0.7790 |
| IVF-Flat | nprobe=10 | 62504.92ms | 5234 | 0.9160 |
| IVF-Flat | nprobe=50 | 62048.65ms | 2620 | 0.9980 |
| IVF-Flat | nprobe=100 | 62581.26ms | 1749 | 1.0000 |
| IVF-PQ | nprobe=1 | 93809.68ms | 26951 | 0.3520 |
| IVF-PQ | nprobe=5 | 93497.81ms | 7674 | 0.5370 |
| IVF-PQ | nprobe=10 | 93255.32ms | 4207 | 0.5810 |
| IVF-PQ | nprobe=50 | 92713.96ms | 882 | 0.5820 |
| IVF-PQ | nprobe=100 | 93312.12ms | 460 | 0.5910 |
| IVF-SQ8 | nprobe=1 | 66800.91ms | 10618 | 0.4130 |
| IVF-SQ8 | nprobe=5 | 67751.20ms | 7761 | 0.7880 |
| IVF-SQ8 | nprobe=10 | 67931.14ms | 5934 | 0.8720 |
| IVF-SQ8 | nprobe=50 | 67381.13ms | 4048 | 0.9760 |
| IVF-SQ8 | nprobe=100 | 68103.08ms | 2847 | 0.9620 |

## Notes

- Hanns DiskANN harness explicitly warns that it is a constrained Rust AISAQ skeleton, not a native-comparable SSD DiskANN pipeline.
- Official IVF-PQ output needs careful normalization before entering final comparison tables.
- `hanns_hnsw_quick` also completed and confirmed `R@10=1.000` at `ef=400` and `ef=800`, but it remains a recall-only validation lane rather than the main throughput comparison surface.
