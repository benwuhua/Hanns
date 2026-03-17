import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diskann_lsearch_recall_band_matrix.py"


def write_report(path: Path, qps: float, recall: float, lsearch: int) -> None:
    payload = {
        "rows": [
            {
                "qps": qps,
                "recall_at_10": recall,
                "search_list_size": lsearch,
                "intra_batch_candidates": 0,
                "construction_l": 128,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DiskAnnLsearchRecallBandMatrixTests(unittest.TestCase):
    def test_matrix_same_lsearch_and_band_pick(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b120 = root / "base_l120.json"
            b128 = root / "base_l128.json"
            c120 = root / "cur_l120.json"
            c128 = root / "cur_l128.json"
            c160 = root / "cur_l160.json"
            out = root / "out.json"
            write_report(b120, 100.0, 0.7800, 120)
            write_report(b128, 90.0, 0.8200, 128)
            write_report(c120, 110.0, 0.7750, 120)
            write_report(c128, 95.0, 0.8150, 128)
            write_report(c160, 120.0, 0.7600, 160)

            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--baseline",
                    str(b120),
                    "--baseline",
                    str(b128),
                    "--candidate",
                    str(c120),
                    "--candidate",
                    str(c128),
                    "--candidate",
                    str(c160),
                    "--recall-tolerance",
                    "0.006",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("DiskANN Lsearch Recall-Band Matrix", proc.stdout)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["matrix"]), 2)
            first = data["matrix"][0]
            self.assertEqual(first["baseline"]["search_list_size"], 120)
            self.assertEqual(first["same_lsearch_candidate"]["search_list_size"], 120)
            self.assertEqual(first["recall_band_pick_mode"], "within_tolerance")

    def test_fallback_when_no_row_in_band(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = root / "base_l128.json"
            c1 = root / "cur_l120.json"
            c2 = root / "cur_l160.json"
            out = root / "out.json"
            write_report(b, 100.0, 0.9000, 128)
            write_report(c1, 90.0, 0.8000, 120)
            write_report(c2, 85.0, 0.8200, 160)

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--baseline",
                    str(b),
                    "--candidate",
                    str(c1),
                    "--candidate",
                    str(c2),
                    "--recall-tolerance",
                    "0.001",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            row = data["matrix"][0]
            self.assertEqual(row["recall_band_pick_mode"], "nearest_recall_fallback")
            self.assertEqual(row["recall_band_candidate"]["search_list_size"], 160)


if __name__ == "__main__":
    unittest.main()
