#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def parse_filter(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid filter '{item}', expected key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def match_filter(row: dict[str, Any], filt: dict[str, str]) -> bool:
    for k, v in filt.items():
        if k not in row:
            return False
        if str(row[k]) != v:
            return False
    return True


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a baseline-centered DiskANN compare matrix from a grid artifact."
    )
    parser.add_argument("--grid-json", required=True, help="Grid artifact path with rows[]")
    parser.add_argument(
        "--baseline-filter",
        action="append",
        default=[],
        help="Baseline selector key=value (repeatable), e.g. intra_batch_candidates=8",
    )
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.005,
        help="Absolute recall tolerance for same-recall candidate pick",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    artifact = json.loads(Path(args.grid_json).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = artifact.get("rows", [])
    if not rows:
        raise ValueError("grid artifact has no rows")

    baseline_filter = parse_filter(args.baseline_filter)
    baseline_candidates = [r for r in rows if match_filter(r, baseline_filter)]
    if len(baseline_candidates) != 1:
        raise ValueError(f"baseline selector matched {len(baseline_candidates)} rows (expected 1)")
    baseline = baseline_candidates[0]
    bq, br = float(baseline["qps"]), float(baseline["recall_at_10"])

    matrix = []
    for r in rows:
        q, rc = float(r["qps"]), float(r["recall_at_10"])
        matrix.append(
            {
                **r,
                "delta_qps_pct": pct(q, bq),
                "delta_recall_abs": rc - br,
            }
        )

    matrix.sort(key=lambda r: (-float(r["qps"]), -float(r["recall_at_10"])))

    candidate_rows = [r for r in matrix if r.get("source") != baseline.get("source")]
    in_tol = [
        r for r in candidate_rows if abs(float(r["recall_at_10"]) - br) <= args.recall_tolerance
    ]
    if in_tol:
        same_recall_best = max(in_tol, key=lambda r: float(r["qps"]))
        same_recall_mode = "within_tolerance"
    else:
        same_recall_best = min(
            candidate_rows, key=lambda r: abs(float(r["recall_at_10"]) - br)
        )
        same_recall_mode = "nearest_recall_fallback"

    result = {
        "grid_source": args.grid_json,
        "baseline_filter": baseline_filter,
        "baseline": baseline,
        "recall_tolerance": args.recall_tolerance,
        "same_recall_pick_mode": same_recall_mode,
        "same_recall_best_qps": same_recall_best,
        "matrix": matrix,
    }

    print("## DiskANN Baseline Matrix")
    print(
        f"baseline: qps={bq:.2f}, recall={br:.4f}, filter={baseline_filter if baseline_filter else '{}'}"
    )
    print(
        f"same-recall-best({same_recall_mode}): qps={float(same_recall_best['qps']):.2f}, "
        f"recall={float(same_recall_best['recall_at_10']):.4f}, "
        f"delta_qps={float(same_recall_best['delta_qps_pct']):+.2f}%, "
        f"delta_recall={float(same_recall_best['delta_recall_abs']):+.4f}"
    )
    print("| intra | cl | qps | recall | delta_qps% | delta_recall |")
    print("|---|---|---:|---:|---:|---:|")
    for r in matrix:
        print(
            f"| {r.get('intra_batch_candidates')} | {r.get('construction_l')} | "
            f"{float(r['qps']):.2f} | {float(r['recall_at_10']):.4f} | "
            f"{float(r['delta_qps_pct']):+.2f} | {float(r['delta_recall_abs']):+.4f} |"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
