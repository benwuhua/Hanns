# Hanns vs Zilliz Knowhere DiskANN/AISAQ interim comparison (2026-04-24)

Authority: `HannsDB-x86`
Status: **partial comparison only**

## Official Zilliz Knowhere completed rows

| Variant | Target recall | Actual recall | Search param | Build(s) | VPS@8T |
|---|---:|---:|---|---:|---:|
| DISKANN | 0.80 | 0.9217 | search_list_size=100 | 232.375 | 3847.820 |
| DISKANN | 0.95 | 0.9504 | search_list_size=108 | 232.375 | 3153.907 |
| AISAQ_P | 0.80 | 0.9214 | search_list_size=100 | 214.674 | 3328.575 |
| AISAQ_P | 0.95 | 0.9502 | search_list_size=108 | 214.674 | 3117.801 |
| AISAQ_S | 0.80 | 0.9192 | search_list_size=100 | 159.796 | 557.301 |
| AISAQ_S | 0.95 | 0.9515 | search_list_size=109 | 159.796 | 520.390 |

## Hanns completed rows

| Config | Build(s) | QPS | Recall@10 |
|---|---:|---:|---:|
| R=32-B=4 | 22.94 | 3248 | 0.966 |
| R=48-B=8 | 38.24 | 1240 | 0.999 |
| R=64-B=16 | 56.59 | 452 | 0.999 |

## Important caveats

- Hanns BENCH-051 explicitly says it is a constrained Rust AISAQ skeleton, not a native-comparable SSD DiskANN pipeline.
- Official rows are search_list_size sweeps at recall targets; Hanns rows are fixed R/B configs. This is not yet a final apples-to-apples table.
- Use this artifact only as interim progress evidence while the remaining Hanns benchmarks finish.
