import json
import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS = REPO_ROOT / "benchmark_results"


def load_json(name: str) -> dict:
    return json.loads((BENCHMARK_RESULTS / name).read_text(encoding="utf-8"))


def parse_param(params: str, name: str) -> int:
    match = re.search(rf"{re.escape(name)}=(\d+)", params)
    if match is None:
        raise AssertionError(f"missing `{name}` in params: {params}")
    return int(match.group(1))


class HnswFairnessGateTests(unittest.TestCase):
    def test_rs_hnsw_artifact_reports_fairness_metadata(self) -> None:
        rust_report = load_json("rs_hnsw_sift128.full_k100.json")
        rust_row = rust_report["rows"][0]
        requested_ef = parse_param(rust_row["params"], "ef")

        self.assertEqual(rust_row["requested_ef_search"], requested_ef)
        self.assertEqual(rust_row["adaptive_k"], 0.0)
        self.assertEqual(rust_row["effective_ef_search"], requested_ef)
        self.assertEqual(rust_row["vector_datatype"], "Float32")
        self.assertEqual(rust_row["query_dispatch_model"], "serial_per_query_index_search")
        self.assertEqual(rust_row["query_batch_size"], 1)

    def test_hnsw_fairness_gate_tracks_current_sources(self) -> None:
        fairness = load_json("hnsw_fairness_gate.json")

        self.assertEqual(fairness["task_id"], "HNSW-FAIRNESS-GATE")
        self.assertEqual(fairness["authority_scope"], "remote_x86_only")
        self.assertEqual(
            fairness["same_schema_source"],
            "benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json",
        )
        self.assertEqual(
            fairness["rust_source"],
            "benchmark_results/rs_hnsw_sift128.full_k100.json",
        )
        self.assertFalse(fairness["fair_for_leadership_claim"])

    def test_hnsw_fairness_gate_records_current_blockers(self) -> None:
        fairness = load_json("hnsw_fairness_gate.json")
        same_schema = load_json("baseline_p3_001_same_schema_hnsw_hdf5.json")
        rust_row = load_json("rs_hnsw_sift128.full_k100.json")["rows"][0]
        native_row = same_schema["rows"]["native_full_k100"]

        requested = fairness["checks"]["requested_ef_alignment"]
        effective = fairness["checks"]["effective_ef_alignment"]
        datatype = fairness["checks"]["datatype_alignment"]
        dispatch = fairness["checks"]["query_dispatch_alignment"]

        self.assertTrue(requested["pass"])
        self.assertEqual(requested["rust_requested_ef_search"], rust_row["requested_ef_search"])
        self.assertEqual(requested["native_compare_ef_search"], parse_param(native_row["params"], "ef"))

        self.assertTrue(effective["pass"])
        self.assertEqual(effective["rust_adaptive_k"], rust_row["adaptive_k"])
        self.assertEqual(effective["rust_effective_ef_search"], rust_row["effective_ef_search"])
        self.assertEqual(effective["native_compare_ef_search"], parse_param(native_row["params"], "ef"))

        self.assertFalse(datatype["pass"])
        self.assertEqual(datatype["rust_datatype"], rust_row["vector_datatype"])
        self.assertEqual(datatype["native_datatype"], "BF16")

        self.assertFalse(dispatch["pass"])
        self.assertEqual(dispatch["rust_query_dispatch_model"], rust_row["query_dispatch_model"])
        self.assertEqual(dispatch["rust_query_batch_size"], rust_row["query_batch_size"])


if __name__ == "__main__":
    unittest.main()
