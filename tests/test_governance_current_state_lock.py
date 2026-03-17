import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS = REPO_ROOT / "benchmark_results"


def load_json(name: str) -> dict:
    return json.loads((BENCHMARK_RESULTS / name).read_text(encoding="utf-8"))


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


class GovernanceCurrentStateLockTests(unittest.TestCase):
    def test_final_artifacts_are_in_accepted_state(self) -> None:
        final_proof = load_json("final_performance_leadership_proof.json")
        final_classification = load_json("final_core_path_classification.json")
        final_acceptance = load_json("final_production_acceptance.json")

        self.assertTrue(final_proof["criterion_met"])
        self.assertTrue(final_acceptance["production_accepted"])
        self.assertEqual(
            final_classification["families"][0]["family"],
            "HNSW",
        )
        self.assertEqual(
            final_classification["families"][0]["classification"],
            "leading",
        )

    def test_readme_current_status_matches_final_artifacts(self) -> None:
        content = read_text(REPO_ROOT / "README.md")
        self.assertIn("HNSW: `leading`", content)
        self.assertIn("criterion_met=true", content)
        self.assertIn("production_accepted=true", content)

    def test_task_queue_current_panel_matches_final_artifacts(self) -> None:
        content = read_text(REPO_ROOT / "TASK_QUEUE.md")
        self.assertIn("HNSW-P3-002", content)
        self.assertIn("family 级最终结论已归档为 `leading`", content)
        self.assertIn("项目最终 verdict 为 `accepted`", content)

    def test_dev_roadmap_and_gap_analysis_expose_current_state(self) -> None:
        roadmap = read_text(REPO_ROOT / "DEV_ROADMAP.md")
        gap = read_text(REPO_ROOT / "GAP_ANALYSIS.md")

        self.assertIn("criterion_met=true", roadmap)
        self.assertIn("production_accepted=true", roadmap)
        self.assertIn("HNSW=`leading`", roadmap)

        self.assertIn("当前无活跃 P3 blocker", gap)
        self.assertIn("HNSW 当前 family verdict 为 `leading`", gap)
        self.assertIn("production_accepted=true", gap)


if __name__ == "__main__":
    unittest.main()
