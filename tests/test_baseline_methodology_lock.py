import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS = REPO_ROOT / "benchmark_results"


def load_json(name: str) -> dict:
    return json.loads((BENCHMARK_RESULTS / name).read_text(encoding="utf-8"))


class BaselineMethodologyLockTests(unittest.TestCase):
    def test_hnsw_family_final_verdict_tracks_current_leadership_evidence(self) -> None:
        final_verdict = load_json("hnsw_p3_002_final_verdict.json")

        self.assertEqual(
            final_verdict["family"],
            "HNSW",
            msg="family-level verdict artifact must explicitly identify the HNSW family",
        )
        self.assertEqual(
            final_verdict["classification"],
            "leading",
            msg="HNSW should be classified as leading once near-equal-recall authority evidence shows Rust throughput leadership",
        )
        self.assertEqual(
            final_verdict["leadership_verdict"],
            "leadership_achieved_on_near_equal_recall_authority_lane",
        )
        self.assertEqual(
            final_verdict["same_schema_source"],
            "benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json",
            msg="family verdict must point at the current same-schema HDF5 evidence chain",
        )
        self.assertEqual(
            final_verdict["baseline_stop_go_source"],
            "benchmark_results/baseline_p3_001_stop_go_verdict.json",
            msg="family verdict must cross-reference the baseline stop/go artifact instead of restating stale numbers",
        )

        evidence = final_verdict["evidence"]
        self.assertAlmostEqual(evidence["rust_recall_at_10"], 0.9518, places=9)
        self.assertAlmostEqual(evidence["native_recall_at_10"], 0.95, places=9)
        self.assertAlmostEqual(evidence["rust_qps_median"], 28479.54432, places=9)
        self.assertAlmostEqual(evidence["native_qps"], 15918.091, places=9)
        self.assertGreater(evidence["rust_over_native_qps_ratio"], 1.0)
        self.assertLess(evidence["native_over_rust_qps_ratio"], 1.0)

    def test_stop_go_verdict_native_reference_tracks_same_schema_artifact(self) -> None:
        same_schema = load_json("baseline_p3_001_same_schema_hnsw_hdf5.json")
        verdict = load_json("baseline_p3_001_stop_go_verdict.json")

        native_row = same_schema["rows"]["native_full_k100"]
        native_ref = verdict["native_reference"]

        self.assertEqual(
            native_ref["params"],
            native_row["params"],
            msg="stop/go verdict must reference the current trusted native params from the same-schema artifact",
        )
        self.assertAlmostEqual(
            native_ref["recall_at_10"],
            native_row["recall_at_10"],
            places=9,
            msg="stop/go verdict must reuse the same recall@10 as the same-schema artifact",
        )
        self.assertAlmostEqual(
            native_ref["qps"],
            native_row["qps"],
            places=9,
            msg="stop/go verdict must reuse the same native qps as the same-schema artifact",
        )

    def test_methodology_gap_native_row_tracks_current_native_artifact(self) -> None:
        native_artifact = load_json("native_hnsw_sift128.remote.json")
        gap = load_json("baseline_p3_001_methodology_gap.json")

        trusted_native = next(
            row
            for row in native_artifact["rows"]
            if row["index"] == "HNSW(BF16)" and row["confidence"] == "trusted"
        )
        gap_native = gap["rows"]["native_full_dataset_row"]

        self.assertEqual(
            gap_native["params"],
            trusted_native["params"],
            msg="methodology-gap evidence must point at the current trusted native row",
        )
        self.assertAlmostEqual(
            gap_native["recall_at_10"],
            trusted_native["recall_at_10"],
            places=9,
        )
        self.assertAlmostEqual(
            gap_native["qps"],
            trusted_native["qps"],
            places=9,
        )
        self.assertEqual(
            gap_native["ground_truth_source"],
            trusted_native["ground_truth_source"],
        )


if __name__ == "__main__":
    unittest.main()
