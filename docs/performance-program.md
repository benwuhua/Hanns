# Performance Program

## Current Verdict

- Final criterion source: `benchmark_results/final_performance_leadership_proof.json`
- Fairness gate source: `benchmark_results/hnsw_fairness_gate.json`
- Final criterion status: `unmet`
- Program state: `blocked_on_hnsw_leadership_gap`
- Next strategic track: `hnsw-fair-lane-throughput`
- North star lane: `HNSW same-schema remote x86`
- Blocker summary: the current trusted final proof still ends with `criterion_met=false`; the latest authority hot-path round improves Rust equal-recall throughput on `ef=60`, but after refreshing the native BF16 anchor on the same day native still remains about `1.06x` faster on this fair lane.

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
- Use equal-recall bands for throughput claims: the 2026-03-16 authority tradeoff sweep shows Rust around `0.957` recall can run near `12.5k qps`, while the `~0.988` recall operating point is around `8.8k qps`; do not compare these cross-band points directly against native leadership claims.
- Keep the native anchor fresh on the same lane: the latest authority BF16 row (`2026-03-16`, refreshed) is `R@10=0.9500`, `qps=15344.859` (`thread_num=8`, linkfix capture), so near-equal recall still favors native; retire older `~10.5k @ 0.95` and `~12.9k @ 0.95` references from active decisions.
- Current Rust equal-recall window (authority, 2026-03-16): `ef=60` improved from `qps=13971.284` to `qps=14423.888` with recall parity (`0.9518`) in the latest opt50 round (`+3.24%` authority uplift), but remains about `-5.99%` vs the refreshed native anchor; keep targeting attributable `ef=60` gains.
- Rejected hypothesis guardrail (2026-03-16): explicit layer0 slab next-neighbor prefetch showed strong local uplift but authority `ef=60` regressed heavily (`-60.76%` with recall parity); do not reopen this direction unless a root-cause mechanism is first proven on authority.
- Local replay gate: before promoting any "local-positive" hypothesis, require controlled local ABAB replay under fixed environment; historical positive signals that fail replay (for example, `opt38` now at about `-0.86%` on local rerun) should be treated as noise/non-stationary and not escalated to authority.
- x86 prefetch ISA authority closure (2026-03-16): full-size authority compare is now complete for `t0` vs `t2` at `ef=60` with recall parity (`0.9518`); stable completed sample shows `t2` (`11600.931`) below `t0` rerun baseline (`12280.988`) by about `-5.54%`, so default remains `t0` while keeping `KNOWHERE_RS_X86_PREFETCH_HINT=t0|t1|t2|nta` as an experiment-only override.
- Additional reject guardrail (2026-03-16): a direct native-style visited-state prefetch adaptation in Rust fast path was locally negative (`-3.0%` at `ef=60`, recall parity), so keep this direction closed and prioritize ISA-hint validation (`t0` vs `t2`) over new prefetch-target experiments.
- x86 visited-state recheck closure (2026-03-16): after enforcing a clean authority window (removing concurrent foreign benchmark load), a full-size x86 A/B recheck for `vector` vs `visited` prefetch target at `t0` produced recall parity and only about `+0.355%` qps for `visited`; this is below the keep threshold (~`+2%`), so production path remains unchanged and prefetch-target reopening is closed.
- opt50 authority closure (2026-03-16): replacing safe visited marking with bounds-checked-callsite `mark_visited_unchecked` in layer-0 hot loops produced a clean-window authority uplift at `ef=60` (`13971.284 -> 14423.888`, recall parity), so this optimization is kept.
- opt51 authority closure (2026-03-16): hoisting `layer_neighbors` id->idx conversion via sliding-window traversal showed local positive signal but authority `ef=60` landed slightly negative (`14241.930 -> 14208.760`, recall parity, `-0.23%`), so this round is rejected and rolled back.
- opt52 local gate closure (2026-03-16): splitting `layer_neighbors` loop by `use_sequential_ids` (branch hoist) failed multi-sample local gate on `ef=60` (`pre=8100.626`, post median `7845.162`, recall parity, `-3.15%`), so this direction is rejected before authority promotion.
- opt53 authority closure (2026-03-16): adding dim=128 AVX-512 specialized kernels (`l2_avx512_sq_ptr_128` and `l2_batch_4_avx512_128`) delivered clean-window authority uplift at `ef=60` (`14111.214 -> 14423.421`, recall parity, `+2.21%`), so this optimization is kept.
- opt54 authority closure (2026-03-16): a follow-up dim=128 AVX-512 dual-accumulator micro-tuning regressed slightly on authority at equal recall (`14460.505 -> 14407.721`, recall parity, `-0.37%`), so this round is rejected and production stays on the opt53 specialization baseline.
- opt56 authority closure (2026-03-17): reusing thread-local `SearchScratch` for L2 unfiltered hot path survived medium-lane recheck and then passed full authority confirmation at equal recall (`1M/1000`, `ef=60`: `13406.233 -> 28574.171`, recall parity `0.9518`, `+113.14%`), so this optimization is kept and becomes the new Rust fair-lane baseline.
