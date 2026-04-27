# Hanns vs official Knowhere DiskANN/AISAQ constrained verdict (2026-04-25)

Authority: `HannsDB-x86`
Status: **non-comparable / leadership claim blocked**

## Conclusion

Hanns now has a fresh aligned `top_k=100` / `Recall@100` evidence lane for the constrained AISAQ implementation, but this is **not** a native-comparable SSD DiskANN/AISAQ verdict.  The current `PQFlashIndex::scope_audit()` reports `native_comparable=false`, so no official DiskANN/AISAQ leadership claim is allowed from these rows.

## Fresh Hanns rows

| Config | search_list_size | Recall@100 | QPS/VPS unit | Build(s) | native comparable |
|---|---:|---:|---:|---:|---|
| R48-L100-B8-EP1 | 100 | 0.9392 | 13596.515 | 160.013 | false |
| R48-L108-B8-EP1 | 108 | 0.9624 | 12907.910 | 159.442 | false |
| R48-L128-B8-EP1 | 128 | 0.9727 | 10486.586 | 163.322 | false |

The `R48-L108-B8-EP1` row is numerically strong against official near-0.95 `DISKANN` and `AISAQ_P` rows, but it remains a constrained numeric observation, not a final parity/leadership verdict.

## Official reference rows

| Variant | Goal | search_list_size | Recall@100 | VPS@8T | Build(s) |
|---|---:|---:|---:|---:|---:|
| DISKANN | 0.80 | 100 | 0.9217 | 3847.820 | 232.375 |
| DISKANN | 0.95 | 108 | 0.9504 | 3153.907 | 232.375 |
| AISAQ_P | 0.80 | 100 | 0.9214 | 3328.575 | 214.674 |
| AISAQ_P | 0.95 | 108 | 0.9502 | 3117.801 | 214.674 |
| AISAQ_S | 0.80 | 100 | 0.9192 | 557.301 | 159.796 |
| AISAQ_S | 0.95 | 109 | 0.9515 | 520.390 | 159.796 |

## Evidence

- Hanns aligned JSON: `docs/parity/raw/hanns-knowhere-diskann-aisaq-aligned-1777090211.json`
- Hanns remote log: `docs/parity/raw/remote_background_20260425T040949Z_82336.log`
- Hanns remote status: `docs/parity/raw/remote_background_20260425T040949Z_82336.status`
- Official raw logs: `docs/parity/raw/native_benchmark_20260424T032148Z_22562.log`, `docs/parity/raw/native_benchmark_20260424T035456Z_81931.log`, `docs/parity/raw/native_benchmark_20260424T071745Z_77904.log`
- Official raw statuses: `docs/parity/raw/native_benchmark_20260424T032148Z_22562.status`, `docs/parity/raw/native_benchmark_20260424T035456Z_81931.status`, `docs/parity/raw/native_benchmark_20260424T071745Z_77904.status`
- Machine: `HannsDB-x86`
- Validator: `python3 scripts/hanns_knowhere_benchmark.py validate-diskann-aisaq-verdict docs/parity/hanns-knowhere-diskann-aisaq-constrained-verdict-20260425T041817Z.json`

## Required next step before any leadership claim

Implement or expose a native-comparable SSD DiskANN/AISAQ pipeline and keep the `top_k=100`, `Recall@100`, `search_list_size` sweep and archived HannsDB-x86 evidence gates.
