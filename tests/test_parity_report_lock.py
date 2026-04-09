import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "docs" / "parity" / "hanns-vs-native-2026-03-17.md"


class ParityReportLockTests(unittest.TestCase):
    def test_report_exists(self) -> None:
        self.assertTrue(REPORT_PATH.exists(), msg="parity report must exist")

    def test_report_locks_lane_and_date_language(self) -> None:
        content = REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("2026-03-17", content)
        self.assertIn("near-equal recall authority lane", content)
        self.assertIn("same-schema `ef=138`", content)

    def test_report_locks_core_metrics(self) -> None:
        content = REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("0.9518", content)
        self.assertIn("28479.544", content)
        self.assertIn("0.9500", content)
        self.assertIn("15918.091", content)
        self.assertIn("1.789x", content)

    def test_report_locks_caveat_and_boundaries(self) -> None:
        content = REPORT_PATH.read_text(encoding="utf-8")
        self.assertIn("lane-specific evidence", content)
        self.assertIn("in-memory simplified boundary", content)
        self.assertIn("IVF-PQ as `no-go`", content)


if __name__ == "__main__":
    unittest.main()
