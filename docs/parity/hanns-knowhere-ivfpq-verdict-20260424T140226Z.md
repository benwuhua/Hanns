# Hanns vs Zilliz Knowhere IVF-PQ milestone verdict

Status: **win**
Authority surface: **HannsDB-x86**

## Normalized official target

- Source: Zilliz official Knowhere (`868634bc5230a71a84371fea5f5b674b805643ea`)
- Log: `/data/work/knowhere-zilliz-official-main-logs/native_benchmark_20260424T132246Z_15129.log`
- Target: IVF_PQ(FP16), `nlist=1024`, `m=32`, `nbits=8`, `top_k=100`, `nprobe=256`
- Metric: `Recall@100=0.7841`, band `0.78`
- Throughput table: `VPS@8T=1398.803`
- Excluded: terminal `R@=0.0000` row recorded as `terminal_zero_anomaly`.

## Hanns candidate

- Source: current rsync worktree
- Log: `/data/work/hanns-logs/remote_background_20260424T135548Z_94621.log`
- Artifact: `docs/parity/hanns-knowhere-ivfpq-aligned-1777038974.json`
- Target: IVF-PQ, `nlist=1024`, `m=32`, `nbits=8`, `top_k=100`, `nprobe=20`
- Metric: `Recall@100=0.9156`, band `0.91`
- Throughput: `QPS@8T=9246.225`

## Conclusion

Under the approved IVF-PQ normalization gate, Hanns wins the milestone: same `top_k=100`, higher Recall@100 band (`0.91` vs `0.78`), and higher one-vector-per-query throughput at 8 threads (`9246.225` vs `1398.803`).

Caveat: Hanns build time is still much slower (`332.355s` vs official `8.960s`), so this is an IVF-PQ search-quality/throughput milestone win, not a full all-metric family victory.
