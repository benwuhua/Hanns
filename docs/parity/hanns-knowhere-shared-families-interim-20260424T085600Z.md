# Hanns vs Zilliz Knowhere shared families interim comparison (2026-04-24)

Authority: `HannsDB-x86`
Status: **interim / completed first-pass rows on both sides**

## HNSW

| Side | Key rows | Notes |
|---|---|---|
| Official | FP32/FP16/BF16 near-0.95 recall rows captured; 8-thread VPS ≈ 14546.8 / 15800.9 / 15672.2 | Official data comes from `benchmark_float_qps` |
| Hanns | `ef_search=64` -> `QPS=1773`, `R@10=1.0000`; `ef_search=128` -> `QPS=1894`, `R@10=1.0000`; `ef_search=256` -> `QPS=1644`, `R@10=1.0000` | Hanns rows come from BENCH-038 parameter scan, not the same runner |

## IVF-SQ

| Side | Key rows | Notes |
|---|---|---|
| Official | near-0.95 recall rows at `nprobe=32`; 8-thread VPS ≈ 8091.8 / 8179.4 | official fp16/bf16 rows |
| Hanns | `nprobe=1/5/10/50/100` -> `QPS=10618/7761/5934/4048/2847`, `R@10=0.413/0.788/0.872/0.976/0.962` | Hanns parameter scan rows |

## IVF-PQ

| Side | Key rows | Notes |
|---|---|---|
| Official | runner completed but final row is not directly trustworthy; sweep appears to plateau around recall≈0.7841 | parser normalization still required |
| Hanns | `nprobe=1/5/10/50/100` -> `QPS=26951/7674/4207/882/460`, `R@10=0.352/0.537/0.581/0.582/0.591` | weak recall across the scan |

## DiskANN / AISAQ

| Side | Key rows | Notes |
|---|---|---|
| Official | DISKANN near-0.95 at `search_list_size=108`, VPS@8T≈3153.9; AISAQ_P near-0.95 VPS@8T≈3117.8; AISAQ_S near-0.95 VPS@8T≈520.4 | same official runner family |
| Hanns | `R=32-B=4` -> `QPS=3248`, `R@10=0.966`; `R=48-B=8` -> `QPS=1240`, `R@10=0.999`; `R=64-B=16` -> `QPS=452`, `R@10=0.999` | Hanns harness explicitly says constrained AISAQ skeleton, not native-comparable SSD DiskANN |

## Important caveats

- This is still not a final apples-to-apples verdict table.
- Official and Hanns rows often come from different runners and parameter selection logic.
- Official IVF-PQ still needs normalization before any final claim.
- DiskANN/AISAQ comparison must remain labeled partial unless/until the Hanns side uses a native-comparable SSD DiskANN lane.
