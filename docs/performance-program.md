# Performance Program

## Current Verdict

- Final criterion source: `benchmark_results/final_performance_leadership_proof.json`
- Fairness gate source: `benchmark_results/hnsw_fairness_gate.json`
- Final criterion status: `unmet`
- Program state: `blocked_on_hnsw_leadership_gap`
- Next strategic track: `hnsw-fair-lane-throughput`
- North star lane: `HNSW same-schema remote x86`
- Blocker summary: the current trusted final proof still ends with `criterion_met=false`; the latest no-sqrt BF16 fair-lane authority rerun keeps effective-`ef`, query dispatch, and datatype aligned (BF16 on both sides) and improves Rust throughput, but native still remains about `1.24x` faster on this fair lane.

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

## Authority Stability Guardrail

- Authority A/A calibration (2026-03-15, same code same lane): `qps=8757.227` vs `qps=8800.913` (`+0.50%` drift) with recall parity.
- Treat authority noise on this lane as low; require at least about `+2%` authority uplift with recall parity before keeping new hot-path behavior changes.
- If a hypothesis shows strong local signal but fails authority pre/post, reject and roll back immediately instead of expanding local reruns.

## Immediate Next Actions

- Keep `--hnsw-adaptive-k 0 --query-dispatch-mode parallel --query-batch-size 32 --vector-datatype bfloat16` on the Rust same-schema authority lane unless a symmetric native policy changes.
- Treat authority reference artifacts in `/data/work/knowhere-rs-logs/` as the active fair-lane baseline set for fast A/B (`rs_hnsw_authority_aa_run*_opt40.json`, `rs_hnsw_opt38_*_authority_v2.json`) and require attributable gains over that envelope before reopening broader claims.
- Keep `benchmark_results/hnsw_fairness_gate.json` and the same-schema baseline artifact chain aligned with every future Rust authority rerun.
