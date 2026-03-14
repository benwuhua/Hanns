# Performance Program

## Current Verdict

- Final criterion source: `benchmark_results/final_performance_leadership_proof.json`
- Fairness gate source: `benchmark_results/hnsw_fairness_gate.json`
- Final criterion status: `unmet`
- Program state: `blocked_on_hnsw_fairness_gate`
- Next strategic track: `hnsw-fairness-gate`
- North star lane: `HNSW same-schema remote x86`
- Blocker summary: the current trusted final proof still ends with `criterion_met=false`, and the last HNSW reopen rounds did not produce a durable fair-lane leadership result.

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

- Remove the dynamic effective-`ef` mismatch from the compare lane or encode the same policy on both sides.
- Add batch-first Rust authority search so throughput reflects real query parallelism instead of per-query serial dispatch.
- Keep `benchmark_results/hnsw_fairness_gate.json` aligned with the Rust baseline artifact before any new authority rerun.
- Rebuild the trusted baseline artifact chain on the fair lane before any new HNSW reopen round.
