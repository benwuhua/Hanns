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
    row = rows[0]
    row["_source"] = str(path)
    if "search_list_size" not in row:
        m = re.search(r"[._]l(\d+)[._]", path.name)
        if m:
            row["search_list_size"] = int(m.group(1))
    return row


def pick_by_lsearch(rows: list[dict[str, Any]], lsearch: int) -> dict[str, Any]:
    exact = [r for r in rows if int(r.get("search_list_size", -1)) == lsearch]
    if exact:
        return exact[0]
    return min(rows, key=lambda r: abs(int(r.get("search_list_size", 0)) - lsearch))


def pick_by_recall(rows: list[dict[str, Any]], target_recall: float) -> dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r["recall_at_10"]) - target_recall))


def row_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row["_source"],
        "qps": float(row["qps"]),
        "recall_at_10": float(row["recall_at_10"]),
        "search_list_size": int(row.get("search_list_size", 0)),
        "construction_l": int(row.get("construction_l", 0)),
        "beamwidth": int(row.get("beamwidth", 0)),
        "intra_batch_candidates": int(row.get("intra_batch_candidates", 0)),
        "build_mode": row.get("build_mode", ""),
    }


def pct(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DiskANN benchmark profile rows under same-lsearch and near-same-recall views."
    )
    parser.add_argument("--baseline", action="append", required=True, help="Baseline report JSON path (repeatable)")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate report JSON path (repeatable)")
    parser.add_argument("--same-lsearch", type=int, required=True, help="Lsearch value for same-params compare")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    args = parser.parse_args()

    baseline_rows = [load_row(Path(p)) for p in args.baseline]
    candidate_rows = [load_row(Path(p)) for p in args.candidate]

    baseline_same = pick_by_lsearch(baseline_rows, args.same_lsearch)
    candidate_same = pick_by_lsearch(candidate_rows, args.same_lsearch)
    candidate_near_recall = pick_by_recall(candidate_rows, float(baseline_same["recall_at_10"]))

    result = {
        "same_lsearch": args.same_lsearch,
        "same_params": {
            "baseline": row_view(baseline_same),
            "candidate": row_view(candidate_same),
            "delta_qps_pct": pct(float(candidate_same["qps"]), float(baseline_same["qps"])),
            "delta_recall_abs": float(candidate_same["recall_at_10"]) - float(baseline_same["recall_at_10"]),
        },
        "near_same_recall": {
            "target_recall": float(baseline_same["recall_at_10"]),
            "baseline": row_view(baseline_same),
            "candidate": row_view(candidate_near_recall),
            "delta_qps_pct": pct(float(candidate_near_recall["qps"]), float(baseline_same["qps"])),
            "delta_recall_abs": float(candidate_near_recall["recall_at_10"])
            - float(baseline_same["recall_at_10"]),
        },
    }

    print("## DiskANN Compare")
    print(
        f"same-params@lsearch={args.same_lsearch}: "
        f"baseline qps={result['same_params']['baseline']['qps']:.2f} recall={result['same_params']['baseline']['recall_at_10']:.4f}, "
        f"candidate qps={result['same_params']['candidate']['qps']:.2f} recall={result['same_params']['candidate']['recall_at_10']:.4f}, "
        f"delta_qps={result['same_params']['delta_qps_pct']:.2f}%, "
        f"delta_recall={result['same_params']['delta_recall_abs']:+.4f}"
    )
    print(
        f"near-same-recall(target={result['near_same_recall']['target_recall']:.4f}): "
        f"candidate@lsearch={result['near_same_recall']['candidate']['search_list_size']} "
        f"qps={result['near_same_recall']['candidate']['qps']:.2f} "
        f"recall={result['near_same_recall']['candidate']['recall_at_10']:.4f}, "
        f"delta_qps={result['near_same_recall']['delta_qps_pct']:.2f}%, "
        f"delta_recall={result['near_same_recall']['delta_recall_abs']:+.4f}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
