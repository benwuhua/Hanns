# Performance Program

## Current Verdict

- Final criterion source: `benchmark_results/final_performance_leadership_proof.json`
- Fairness gate source: `benchmark_results/hnsw_fairness_gate.json`
- Final criterion status: `unmet`
- Program state: `blocked_on_hnsw_leadership_gap`
- Next strategic track: `hnsw-fair-lane-throughput`
- North star lane: `HNSW same-schema remote x86`
- Blocker summary: the current trusted final proof still ends with `criterion_met=false`; the latest fair-lane authority rerun has now aligned effective-`ef`, query dispatch, and datatype (BF16 on both sides), but native still remains about `2.17x` faster on this fair lane.

## Fairness Gate

- Match Rust and native on effective `ef`, not just the requested `ef_search` field.
- Match batching and thread model: Rust authority search must reflect real batch/query parallelism comparable to native `Search()`.
- Treat any compare run that violates these conditions as screen evidence only, not as leadership evidence.

## Canonical Compare Lane

- Dataset: `sift-128-euclidean.hdf5`
- Surface: existing remote x86 authority machine only
- Contract: same `top_k`, same recall gate, same effective `ef`, same datatype, same batching model, same retained logs
- Success condition for a reopen round: attributable Rust-side qps gain against the previous fair authority baseline

## Pivot Gate

- If two consecutive fair authority rounds fail to deliver attributable Rust-side qps gain, pause pure-Rust HNSW micro-optimization.
- Resume only with one of: a new architecture proposal, a narrower production-parity target, or a backend strategy change.
- Do not reopen filtered-query policy work until the unfiltered leadership lane is fair and stable.

## Immediate Next Actions

- Keep `--hnsw-adaptive-k 0 --query-dispatch-mode parallel --query-batch-size 32 --vector-datatype bfloat16` on the Rust same-schema authority lane unless a symmetric native policy changes.
- Treat `benchmark_results/rs_hnsw_sift128.full_k100.json` at `qps=4840.831171680344` as the current fair-lane Rust baseline and require attributable gains over it before reopening broader claims.
- Keep `benchmark_results/hnsw_fairness_gate.json` and the same-schema baseline artifact chain aligned with every future Rust authority rerun.
