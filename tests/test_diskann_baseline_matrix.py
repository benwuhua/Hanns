import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diskann_baseline_matrix.py"


def write_grid(path: Path) -> None:
    payload = {
        "rows": [
            {
                "source": "a",
                "qps": 100.0,
                "recall_at_10": 0.8000,
                "intra_batch_candidates": 0,
                "construction_l": 128,
            },
            {
                "source": "b",
                "qps": 105.0,
                "recall_at_10": 0.7995,
                "intra_batch_candidates": 8,
                "construction_l": 128,
            },
            {
                "source": "c",
                "qps": 120.0,
                "recall_at_10": 0.7600,
                "intra_batch_candidates": 8,
                "construction_l": 160,
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DiskAnnBaselineMatrixTests(unittest.TestCase):
    def test_pick_same_recall_within_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = root / "grid.json"
            out = root / "out.json"
            write_grid(grid)

            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--grid-json",
                    str(grid),
                    "--baseline-filter",
                    "intra_batch_candidates=0",
                    "--baseline-filter",
                    "construction_l=128",
                    "--recall-tolerance",
                    "0.002",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("same-recall-best(within_tolerance)", proc.stdout)
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["same_recall_pick_mode"], "within_tolerance")
            self.assertEqual(result["same_recall_best_qps"]["source"], "b")

    def test_pick_same_recall_fallback_when_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            grid = root / "grid.json"
            out = root / "out.json"
            write_grid(grid)

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--grid-json",
                    str(grid),
                    "--baseline-filter",
                    "intra_batch_candidates=0",
                    "--baseline-filter",
                    "construction_l=128",
                    "--recall-tolerance",
                    "0.00001",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["same_recall_pick_mode"], "nearest_recall_fallback")
            self.assertEqual(result["same_recall_best_qps"]["source"], "b")


if __name__ == "__main__":
    unittest.main()
