# Performance Program

## Current Verdict

- Final criterion source: `benchmark_results/final_performance_leadership_proof.json`
- Fairness gate source: `benchmark_results/hnsw_fairness_gate.json`
- Final criterion status: `unmet`
- Program state: `blocked_on_hnsw_fairness_gate`
- Next strategic track: `hnsw-fairness-gate`
- North star lane: `HNSW same-schema remote x86`
- Blocker summary: the current trusted final proof still ends with `criterion_met=false`; the fair-ef authority rerun has closed the effective-`ef` mismatch, but the HNSW lane still lacks datatype and query-dispatch parity.

## Fairness Gate

- Match Rust and native on effective `ef`, not just the requested `ef_search` field.
- Match datatype before reading absolute qps deltas as leadership evidence.
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

- Keep `--hnsw-adaptive-k 0` on the Rust same-schema authority lane unless a symmetric native effective-`ef` policy is introduced.
- Add batch-first Rust authority search so throughput reflects real query parallelism instead of per-query serial dispatch.
- Match datatype before reading the fair-lane qps gap as leadership evidence.
- Keep `benchmark_results/hnsw_fairness_gate.json` and the same-schema baseline artifact chain aligned with every future Rust authority rerun.
