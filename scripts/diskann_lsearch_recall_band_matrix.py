#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_row(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows:
        raise ValueError(f"{path} has no rows")
    row = dict(rows[0])
    row["source"] = str(path)
    if "search_list_size" not in row:
        m = re.search(r"[._]l(\d+)[._]", path.name)
        if m:
            row["search_list_size"] = int(m.group(1))
    if "search_list_size" not in row:
        raise ValueError(f"{path} missing search_list_size")
    return row


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def pick_by_lsearch(rows: list[dict[str, Any]], lsearch: int) -> dict[str, Any]:
    exact = [r for r in rows if int(r["search_list_size"]) == lsearch]
    if exact:
        return max(exact, key=lambda r: float(r["qps"]))
    return min(rows, key=lambda r: abs(int(r["search_list_size"]) - lsearch))


def pick_in_recall_band(rows: list[dict[str, Any]], target_recall: float, tol: float) -> tuple[str, dict[str, Any]]:
    in_band = [r for r in rows if abs(float(r["recall_at_10"]) - target_recall) <= tol]
    if in_band:
        return "within_tolerance", max(in_band, key=lambda r: float(r["qps"]))
    return "nearest_recall_fallback", min(
        rows, key=lambda r: abs(float(r["recall_at_10"]) - target_recall)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DiskANN lsearch matrix with same-lsearch and fixed-recall-band compare views."
    )
    parser.add_argument("--baseline", action="append", required=True, help="Baseline JSON path (repeatable)")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate JSON path (repeatable)")
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.005,
        help="Absolute recall tolerance for recall-band selection",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    baseline_rows = [load_row(Path(p)) for p in args.baseline]
    candidate_rows = [load_row(Path(p)) for p in args.candidate]
    baseline_rows.sort(key=lambda r: int(r["search_list_size"]))

    matrix = []
    for b in baseline_rows:
        bq = float(b["qps"])
        br = float(b["recall_at_10"])
        lsearch = int(b["search_list_size"])

        same = pick_by_lsearch(candidate_rows, lsearch)
        mode, band = pick_in_recall_band(candidate_rows, br, args.recall_tolerance)

        matrix.append(
            {
                "baseline": b,
                "same_lsearch_candidate": same,
                "same_lsearch_delta_qps_pct": pct(float(same["qps"]), bq),
                "same_lsearch_delta_recall_abs": float(same["recall_at_10"]) - br,
                "recall_band_pick_mode": mode,
                "recall_band_candidate": band,
                "recall_band_delta_qps_pct": pct(float(band["qps"]), bq),
                "recall_band_delta_recall_abs": float(band["recall_at_10"]) - br,
            }
        )

    result = {
        "recall_tolerance": args.recall_tolerance,
        "matrix": matrix,
    }

    print("## DiskANN Lsearch Recall-Band Matrix")
    print("| baseline_l | baseline(qps/recall) | same_l(qps/recall,Δqps,Δrecall) | recall_band(mode,cand_l,qps/recall,Δqps,Δrecall) |")
    print("|---:|---|---|---|")
    for m in matrix:
        b = m["baseline"]
        s = m["same_lsearch_candidate"]
        r = m["recall_band_candidate"]
        print(
            f"| {int(b['search_list_size'])} | {float(b['qps']):.2f}/{float(b['recall_at_10']):.4f} | "
            f"{float(s['qps']):.2f}/{float(s['recall_at_10']):.4f},"
            f"{float(m['same_lsearch_delta_qps_pct']):+.2f}%,{float(m['same_lsearch_delta_recall_abs']):+.4f} | "
            f"{m['recall_band_pick_mode']},l{int(r['search_list_size'])},"
            f"{float(r['qps']):.2f}/{float(r['recall_at_10']):.4f},"
            f"{float(m['recall_band_delta_qps_pct']):+.2f}%,{float(m['recall_band_delta_recall_abs']):+.4f} |"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
