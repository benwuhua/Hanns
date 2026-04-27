# Hanns vs Zilliz Knowhere IVF-SQ8 scoped verdict (HannsDB-x86)

Authority surface: `HannsDB-x86`
Official source: `https://github.com/zilliztech/knowhere` @ `868634bc5230a71a84371fea5f5b674b805643ea`
Hanns run: `/data/work/hanns-logs/remote_background_20260425T031420Z_78049.log`

| Recall goal | Candidate | R@100 | 8T QPS/VPS | Build(s) | Verdict |
|---|---|---:|---:|---:|---|
| near 0.80 | Official IVF_SQ8(BF16) nprobe=10 | 0.8009 | 28374.420 | 6.875 | baseline |
| near 0.80 | Hanns IVF-SQ8 nprobe=11 | 0.810426 | 39520.568 | 3.113 | win |
| near 0.95 | Official IVF_SQ8(BF16) nprobe=32 | 0.9519 | 13056.415 | 6.875 | baseline |
| near 0.95 | Hanns IVF-SQ8 nprobe=36 | 0.953222 | 15689.750 | 3.113 | win |

Scope: IVF-SQ8 on SIFT1M, top_k=100, nlist=1024, 8 query threads. This does not claim every index family is complete.

Archived evidence:
- Hanns aligned JSON: `docs/parity/raw/hanns-knowhere-ivfsq8-aligned-1777086937.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260425T031420Z_78049.log`, `.status`
- Official log/status: `docs/parity/raw/native_benchmark_20260425T022512Z_33583.log`, `.status`
- Machine: HannsDB-x86
