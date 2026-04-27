# Hanns IVF-PQ build-time verdict (20260424T151136Z)

Status: **win** on the normalized IVF-PQ milestone.

## Contract

- Authority surface: HannsDB-x86.
- Official Knowhere source: `https://github.com/zilliztech/knowhere` at `868634bc5230a71a84371fea5f5b674b805643ea`.
- Same `top_k=100`; official terminal zero-recall anomaly remains excluded.
- Throughput comparison uses one-vector-per-query `QPS`/`VPS` with both sides at 8 threads.
- Build comparison uses total `build_s = train_s + add_s` for Hanns and official total build seconds from the native benchmark block.
- Archived evidence is required and validated against local raw log/status copies plus the Hanns aligned JSON artifact.

## Winning row

| impl | nlist | nprobe | m | nbits | threads | Recall@100 | band | QPS/VPS | build_s | train_s | add_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hanns | 1024 | 20 | 32 | 8 | 8 | 0.905033 | 0.90 | 7520.139 | 6.428 | 1.748 | 4.681 |
| Zilliz Knowhere official FP16 | 1024 | 256 | 32 | 8 | 8 | 0.7841 | 0.78 | 1398.803 | 8.960 | n/a | n/a |

## Conclusion

Hanns now beats the normalized official IVF-PQ target on all three milestone axes:

1. Recall@100: `0.905033` vs official `0.7841`.
2. 8-thread QPS/VPS: `7520.139` vs official `1398.803`.
3. Build time: `6.428s` vs official `8.960s` (Hanns/official ratio `0.717`).

## Evidence

- Hanns remote log: `/data/work/hanns-logs/remote_background_20260424T150952Z_57912.log`
- Archived Hanns log/status: `docs/parity/raw/remote_background_20260424T150952Z_57912.log`, `docs/parity/raw/remote_background_20260424T150952Z_57912.status`
- Hanns local artifact: `docs/parity/hanns-knowhere-ivfpq-aligned-1777043481.json`
- Hanns remote artifact: `/data/work/hanns-src/docs/parity/hanns-knowhere-ivfpq-aligned-1777043481.json`
- Official remote log: `/data/work/knowhere-zilliz-official-main-logs/native_benchmark_20260424T132246Z_15129.log`
- Archived official log/status: `docs/parity/raw/native_benchmark_20260424T132246Z_15129.log`, `docs/parity/raw/native_benchmark_20260424T132246Z_15129.status`
- Machine/runtime: `HannsDB-x86`, `IVFPQ_EXPECT_THREADS=8`, `RAYON_NUM_THREADS=8`, SIFT1M full base / 10k query run.
