# HNSW-PQ raw-candidate/raw-refine fair-8 verdict — 2026-04-25 11:43 UTC

Status: `custom_variant_fair8_build_and_search_win_at_higher_recall`

## Scope

This is a scoped custom-variant milestone: Hanns `HnswPqIndex` currently uses a raw-vector HNSW candidate graph plus exact raw rerank. It is **not** a clean proof of same-semantics official `HNSW_PQ` family leadership and must not be used for `全面超越官方` claims.

## Fair-8 result

| Metric | Official zilliztech/knowhere `TEST_HNSW_PQ` FP32 row | Hanns raw-candidate/raw-refine |
| --- | ---: | ---: |
| Build threads | 8 | 8 |
| Query threads | 8 | 8 |
| Build time | 84.542s | 78.939s |
| Compared row | refine_k=16 | refine_k=2 |
| R@100 | 0.9576 | 0.975803 |
| QPS/VPS | 5050.505 | 9451.081 |

Deltas: R@100 +0.018203; QPS ratio 1.871x; build delta -5.603s.

## Evidence

- Hanns artifact: `docs/parity/raw/hanns-knowhere-hnsw-pq-aligned-1777117088.json`
- Hanns remote log/status: `docs/parity/raw/remote_background_20260425T113806Z_37907.log`, `docs/parity/raw/remote_background_20260425T113806Z_37907.status`
- Local source provenance: `docs/parity/raw/hanns-source-provenance-20260425T113555Z.txt`
- Official log/status: `docs/parity/raw/remote_background_20260425T061341Z_98575.log`, `docs/parity/raw/remote_background_20260425T061341Z_98575.status`
- Official source inspection: `docs/parity/raw/remote_background_20260425T091234Z_11524.log`

## Caveats

- Hanns source is an rsynced dirty local tree. The verdict records local git head and scoped file SHA-256, but this is still not an immutable clean commit.
- The custom variant remains semantically stronger than official `HNSW_PQ`; claim is custom-variant fair-8 build+search milestone only.
- Official full `TEST_HNSW_PQ` failed after FP32 emitted rows on FP16 `bad optional access`; this verdict is bounded to the FP32 rows.
