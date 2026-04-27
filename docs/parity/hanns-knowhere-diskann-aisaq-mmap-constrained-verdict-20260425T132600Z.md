# Hanns vs official Knowhere DiskANN/AISAQ mmap-backed constrained verdict (2026-04-25)

Authority: `HannsDB-x86`
Status: **non-comparable / leadership claim blocked**

## Conclusion

Hanns now has fresh `top_k=100` / `Recall@100` evidence for a **mmap-backed** `PQFlashIndex::load_with_mmap` search surface. This is a stronger evidence lane than the earlier in-memory-only constrained run because `scope_audit.uses_mmap_backed_pages=true`.

It still remains **not native-comparable**: `scope_audit.native_comparable=false`, so no official DiskANN/AISAQ leadership claim is allowed from this row.

## Fresh Hanns mmap row

| Config | surface | search_list_size | Recall@100 | QPS/VPS unit | Build(s) | Persist(s) | Load(s) | mmap pages | native comparable |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| R48-L108-B8-EP1 | mmap | 108 | 0.9624 | 3287.420 | 162.430 | 0.869 | 0.000051 | True | false |

## Official near-0.95 reference rows

| Variant | search_list_size | Recall@100 | VPS@8T | Build(s) |
|---|---:|---:|---:|---:|
| DISKANN | 108 | 0.9504 | 3153.907 | 232.375 |
| AISAQ_P | 108 | 0.9502 | 3117.801 | 214.674 |
| AISAQ_S | 109 | 0.9515 | 520.390 | 159.796 |

## Numeric observation, not a leadership verdict

The mmap-backed Hanns row numerically exceeds the official near-0.95 rows on recall and search throughput, and has lower build time than official DISKANN/AISAQ_P. It does **not** beat AISAQ_S build time (162.430s vs 159.796s). Because `native_comparable=false`, all of this remains constrained evidence only.

## Evidence

- Hanns mmap artifact: `docs/parity/raw/hanns-knowhere-diskann-aisaq-aligned-1777123158.json`
- Hanns remote log/status: `docs/parity/raw/remote_background_20260425T131910Z_81536.log`, `docs/parity/raw/remote_background_20260425T131910Z_81536.status`
- Official raw logs: `docs/parity/raw/native_benchmark_20260424T032148Z_22562.log`, `docs/parity/raw/native_benchmark_20260424T035456Z_81931.log`, `docs/parity/raw/native_benchmark_20260424T071745Z_77904.log`
- Validator: `python3 scripts/hanns_knowhere_benchmark.py validate-diskann-aisaq-verdict docs/parity/hanns-knowhere-diskann-aisaq-mmap-constrained-verdict-20260425T132600Z.json`

## Required next step before any leadership claim

Close the remaining semantic gap: prove or implement native-comparable graph construction and SSD IO semantics, then re-run the same `top_k=100`, `Recall@100`, `search_list_size` evidence gates on HannsDB-x86.
