# Hanns vs Zilliz Knowhere capability probe (2026-04-23)

Authority: `HannsDB-x86`
Official Knowhere: `https://github.com/zilliztech/knowhere` @ `4489f689de32baa70dae28ee1d6784ef87c2c3c8`

| Family | Hanns | Zilliz Knowhere | Status | Notes |
|---|---|---|---|---|
| HNSW | HnswIndex / IndexType::Hnsw | HNSW / INDEX_HNSW | supported | Direct CPU HNSW family present on both sides. |
| HNSW-SQ | IndexType::HnswSq / HnswSqIndex | HNSW_SQ / INDEX_HNSW_SQ | supported | SQ/refine parameters still need benchmark matrix mapping. |
| HNSW-PQ | IndexType::HnswPq / HNSW-PQ path | HNSW_PQ / INDEX_HNSW_PQ | supported | Lossy/raw-data semantics must be reported; performance rows still require fresh benchmark. |
| DISKANN | AISAQ/PQFlash/DiskANN family | DISKANN / INDEX_DISKANN | supported | Hanns implementation is constrained; final comparison must record storage/SSD semantics before verdict. |
| DISKANN-USQ/RabitQ | AISAQ/PQFlash quantized DiskANN family | AISAQ / INDEX_AISAQ | supported | Equivalent enough for capability admission, but final benchmark must document AISAQ/USQ/RabitQ semantic mapping. |
| IVF-PQ | IvfPqIndex | IVF_PQ / INDEX_FAISS_IVFPQ | supported | Direct IVF-PQ family present on both sides. |
| IVF-SQ | IvfSq8Index | IVF_SQ8 / INDEX_FAISS_IVFSQ8 | supported | Maps first pass to SQ8 unless later matrix adds other SQ modes. |
| IVF-USQ/RabitQ | IvfUsqIndex / ExRaBitQ path | IVF_RABITQ and IVF_RABITQ_FASTSCAN | supported | USQ vs RabitQ semantics must be split or documented before numeric verdict. |

## Limitations

- This is a capability/source-symbol probe, not a performance benchmark.
- All rows remain ineligible for QPS/recall/build/latency verdicts until fresh Stage 1 HannsDB-x86 benchmark rows pass schema validation.
- DISKANN and USQ/RabitQ/AISAQ rows need semantic/parameter mapping before numeric comparison.
