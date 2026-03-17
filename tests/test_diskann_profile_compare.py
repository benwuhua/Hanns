import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "diskann_profile_compare.py"


def write_report(path: Path, lsearch: int, qps: float, recall: float) -> None:
    payload = {
        "benchmark": "diskann_pq_ab",
        "metric": "L2",
        "rows": [
            {
                "pq_dims": 4,
                "pq_expand_pct": 125,
                "saturate_after_prune": True,
                "intra_batch_candidates": 0,
                "construction_l": 128,
                "base_size": 2000,
                "query_size": 40,
                "dim": 128,
                "top_k": 10,
                "qps": qps,
                "recall_at_10": recall,
                "search_list_size": lsearch,
                "build_mode": "load",
                "build_seconds": 0.05,
                "search_seconds": 0.01,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DiskAnnProfileCompareTests(unittest.TestCase):
    def test_same_lsearch_and_near_recall_views(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b128 = root / "b128.json"
            c128 = root / "c128.json"
            c120 = root / "c120.json"
            out = root / "out.json"
            write_report(b128, 128, 100.0, 0.8000)
            write_report(c128, 128, 95.0, 0.8020)
            write_report(c120, 120, 110.0, 0.7900)

            proc = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--baseline",
                    str(b128),
                    "--candidate",
                    str(c128),
                    "--candidate",
                    str(c120),
                    "--same-lsearch",
                    "128",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("same-params@lsearch=128", proc.stdout)

            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["same_lsearch"], 128)
            self.assertAlmostEqual(result["same_params"]["delta_qps_pct"], -5.0, places=6)
            self.assertAlmostEqual(result["same_params"]["delta_recall_abs"], 0.0020, places=6)
            self.assertEqual(result["near_same_recall"]["candidate"]["search_list_size"], 128)

    def test_lsearch_fallback_from_filename_when_field_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            b = root / "diskann_pq_ab.remote.cmp.base_l128.load.json"
            c = root / "diskann_pq_ab.remote.cmp.cur_l120.load.json"
            out = root / "out.json"
            write_report(b, 128, 100.0, 0.8000)
            write_report(c, 120, 110.0, 0.7900)

            # Remove explicit field to validate filename fallback.
            for p in [b, c]:
                payload = json.loads(p.read_text(encoding="utf-8"))
                del payload["rows"][0]["search_list_size"]
                p.write_text(json.dumps(payload), encoding="utf-8")

            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--baseline",
                    str(b),
                    "--candidate",
                    str(c),
                    "--same-lsearch",
                    "128",
                    "--output-json",
                    str(out),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(result["same_params"]["baseline"]["search_list_size"], 128)
            self.assertEqual(result["same_params"]["candidate"]["search_list_size"], 120)


if __name__ == "__main__":
    unittest.main()
