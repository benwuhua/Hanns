import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "hanns_knowhere_benchmark.py"
spec = importlib.util.spec_from_file_location("hkb", MODULE_PATH)
hkb = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hkb)


PLAN_APPROVED_AT = "2026-04-23T02:20:00Z"
RUN_SET_ID = "hanns-knowhere-20260423T030000Z"


def timestamp(minutes: int) -> str:
    base = dt.datetime(2026, 4, 23, 2, 20, tzinfo=dt.timezone.utc)
    return (base + dt.timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def authority_manifest() -> dict:
    return {
        "runtime_proof": {
            "ssh_alias_or_host": "HannsDB-x86",
            "hostname": "hannsdb-x86",
            "uname": "Linux hannsdb-x86",
            "lscpu_hash": "abc123",
            "remote_log_root": "/data/work/hanns-logs",
            "allowed_wrapper_scripts": ["scripts/remote/test.sh"],
        }
    }


def valid_row(family: str, implementation: str, status: str = "supported") -> dict:
    repo_url = (
        "https://github.com/zilliztech/knowhere"
        if implementation == "zilliz_knowhere"
        else "https://example.invalid/hanns"
    )
    metrics = {
        "qps": 1000.0,
        "recall_at_k": 0.95,
        "build_seconds": {
            "total": 12.3,
            "train": 1.0,
            "add_or_build": 11.0,
            "persist": 0.3,
            "definition": "total=train+add_or_build+persist_when_measured",
        },
        "latency_ms": {
            "p50": 1.0,
            "p95": 2.0,
            "p99": 3.0,
            "sample_count": 100,
            "source": "synthetic_fixture",
        },
    }
    if status != "supported":
        metrics = {}
    return {
        "run_set": {
            "plan_id": hkb.PLAN_ID,
            "run_set_id": RUN_SET_ID,
            "plan_approved_at": PLAN_APPROVED_AT,
            "run_started_at": timestamp(1),
            "generated_at": timestamp(2),
            "raw_artifact_created_at": timestamp(2),
        },
        "authority": {
            "surface": "HannsDB-x86",
            "is_authority": True,
            "eligible_for_verdict": True,
            "runtime_proof": {
                "ssh_alias_or_host": "HannsDB-x86",
                "hostname": "hannsdb-x86",
                "uname": "Linux hannsdb-x86",
                "lscpu_hash": "abc123",
                "remote_log_path": "/data/work/hanns-logs/run.log",
                "wrapper_script": "scripts/remote/test.sh",
            },
        },
        "implementation": implementation,
        "source": {
            "repo_url": repo_url,
            "commit": "a" * 40,
            "branch_or_tag": "main",
            "dirty": False,
        },
        "index": {
            "user_family": family,
            "implementation_name": family.replace("-", "_"),
            "support_status": status,
        },
        "dataset": {
            "name": "sift-128-euclidean",
            "path": "/data/work/datasets/sift.hdf5",
            "checksum": "sha256:fixture",
            "metric": "L2",
            "dimension": 128,
            "base_count": 1_000_000,
            "query_count": 10_000,
        },
        "params": {
            "top_k": 10,
            "recall_at": 10,
            "recall_band": "same_parameter",
            "concurrency": 1,
            "threads": 1,
            "index_params": {},
            "search_params": {},
        },
        "metrics": metrics,
        "repeat": {"warmup": 1, "runs": 3, "run_id": "fixture"},
        "artifacts": {
            "stdout_log": "/data/work/hanns-logs/stdout.log",
            "stderr_log": "/data/work/hanns-logs/stderr.log",
            "raw_json": "/data/work/hanns-logs/raw.json",
        },
    }


def complete_rows(status: str = "supported") -> list[dict]:
    return [
        valid_row(family, implementation, status=status)
        for family in hkb.REQUIRED_FAMILIES
        for implementation in hkb.IMPLEMENTATIONS
    ]


def valid_ivfpq_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "ivfpq_milestone_verdict",
        "status": "win",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "recall_band_formula": "floor(recall * 100) / 100",
            "recall_tolerance": 0.001,
            "throughput_metric": "vps",
            "vps_qps_mapping": "one_vector_per_query",
            "thread_policy": "match_or_explicitly_normalized",
            "build_time_required": True,
        },
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": f"main@{'a' * 40}",
            "log_path": "/data/work/knowhere-zilliz-official-main-logs/ivfpq.log",
            "normalization_source": "sweep_row",
            "excluded_rows": ["terminal_zero_anomaly"],
            "top_k": 100,
            "recall_at_100": 0.7841,
            "recall_band_floor": 0.78,
            "nlist": 1024,
            "nprobe": 64,
            "m": 16,
            "nbits": 8,
            "threads": 8,
            "build_s": 12.0,
            "qps_or_vps": 800.0,
        },
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/ivfpq",
            "log_path": "/data/work/hanns-logs/ivfpq.log",
            "top_k": 100,
            "recall_at_100": 0.785,
            "recall_band_floor": 0.78,
            "nlist": 1024,
            "nprobe": 100,
            "m": 16,
            "nbits": 8,
            "threads": 8,
            "build_s": 10.0,
            "qps_or_vps": 900.0,
            "refine": {
                "enabled": True,
                "candidate_pool": "min(scanned, top_k * 4)",
                "exact_refine_multiplier": 4,
            },
        },
        "verdict_checks": {
            "official_normalized": True,
            "same_top_k": True,
            "same_or_higher_recall_band": True,
            "exact_recall_within_tolerance": True,
            "throughput_units_comparable": True,
            "thread_policy_satisfied": True,
            "hanns_qps_vps_gt_official": True,
            "hanns_build_s_lt_official": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_hnsw_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "hnsw_milestone_verdict",
        "status": "win",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "recall_tolerance": 0.0,
            "throughput_metric": "vps",
            "vps_qps_mapping": "one_vector_per_query",
            "thread_policy": "match_or_explicitly_normalized",
            "build_time_required": True,
        },
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": f"main@{'a' * 40}",
            "log_path": "/data/work/knowhere-zilliz-official-main-logs/hnsw.log",
            "variant": "FP16",
            "recall_goal": "near_0.80",
            "top_k": 100,
            "recall_at_100": 0.9178,
            "m": 16,
            "ef_construction": 100,
            "ef": 100,
            "threads": 8,
            "build_threads": 8,
            "build_s": 29.339,
            "qps_or_vps": 30013.983,
        },
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/hnsw",
            "log_path": "/data/work/hanns-logs/hnsw.log",
            "top_k": 100,
            "recall_at_100": 0.9198,
            "m": 6,
            "ef_construction": 34,
            "ef": 850,
            "threads": 8,
            "build_threads": 8,
            "build_s": 23.878,
            "train_s": 0.0,
            "add_s": 23.878,
            "qps_or_vps": 30972.456,
        },
        "verdict_checks": {
            "same_top_k": True,
            "same_or_higher_recall": True,
            "throughput_units_comparable": True,
            "thread_policy_satisfied": True,
            "hanns_qps_vps_gt_official": True,
            "hanns_build_s_lt_official": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_ivfsq8_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "ivfsq8_milestone_verdict",
        "status": "win",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "recall_tolerance": 0.0,
            "throughput_metric": "vps",
            "vps_qps_mapping": "one_vector_per_query",
            "thread_policy": "match_or_explicitly_normalized",
            "build_time_required": True,
        },
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": "main",
            "log_path": "/data/work/knowhere-zilliz-official-main-logs/ivfsq8.log",
            "variant": "BF16",
            "recall_goal": "near_0.95",
            "top_k": 100,
            "recall_at_100": 0.9519,
            "nlist": 1024,
            "nprobe": 32,
            "threads": 8,
            "build_s": 6.875,
            "qps_or_vps": 13056.415,
        },
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/ivfsq8",
            "log_path": "/data/work/hanns-logs/ivfsq8.log",
            "top_k": 100,
            "recall_at_100": 0.9532,
            "nlist": 1024,
            "nprobe": 36,
            "threads": 8,
            "build_s": 3.177,
            "train_s": 1.610,
            "add_s": 1.567,
            "qps_or_vps": 16046.2,
        },
        "verdict_checks": {
            "same_top_k": True,
            "same_or_higher_recall": True,
            "throughput_units_comparable": True,
            "thread_policy_satisfied": True,
            "hanns_qps_vps_gt_official": True,
            "hanns_build_s_lt_official": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_diskann_aisaq_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "diskann_aisaq_constrained_verdict",
        "status": "non_comparable",
        "authority_surface": "HannsDB-x86",
        "family": "DiskANN/AISAQ",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "leadership_claim_allowed": False,
        "native_comparable": False,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "throughput_metric": "vps",
            "vps_qps_mapping": "one_vector_per_query",
            "archived_evidence_required": False,
        },
        "official_targets": [
            {
                "repo_url": "https://github.com/zilliztech/knowhere",
                "variant": "AISAQ_P",
                "log_path": "/data/work/knowhere-zilliz-official-main-logs/aisaq_p.log",
                "top_k": 100,
                "recall_at_100": 0.9502,
                "search_list_size": 108,
                "threads": 8,
                "build_s": 214.674,
                "qps_or_vps": 3117.801,
            }
        ],
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/diskann-aisaq",
            "log_path": "/data/work/hanns-logs/aisaq.log",
            "config_name": "R48-L108-B8-EP1",
            "top_k": 100,
            "recall_at_100": 0.955,
            "max_degree": 48,
            "search_list_size": 108,
            "beamwidth": 8,
            "num_entry_points": 1,
            "threads": 8,
            "build_s": 38.0,
            "qps_or_vps": 3000.0,
            "native_comparable": False,
            "comparability_reason": "constrained Rust AISAQ skeleton",
        },
        "verdict_checks": {
            "same_top_k": True,
            "aligned_metric_available": True,
            "native_comparable": False,
            "leadership_claim_blocked": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_hnsw_pq_blocker(**overrides) -> dict:
    verdict = {
        "artifact_type": "hnsw_pq_blocker_verdict",
        "status": "blocked_no_comparable_hanns_refine",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "evidence_required": False,
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": f"main@{'a' * 40}",
            "runner": "benchmark_float",
            "variant": "FP32",
            "full_test_status": "failed_after_fp32_rows",
            "top_k": 100,
            "m": 16,
            "ef_construction": 200,
            "ef": 128,
            "refine_k": 16,
            "nq": 10000,
            "elapsed_s": 1.980,
            "recall_at_100": 0.9576,
            "build_s": 84.542,
        },
        "hanns_capability": {
            "implementation": "HnswPqIndex",
            "has_raw_data": False,
            "get_vector_by_ids": "unsupported",
            "comparability_reason": "official HNSW_PQ FP32 rows use FLAT refine; Hanns HnswPqIndex stores lossy PQ codes and cannot provide FLAT refine raw vectors",
        },
        "verdict_checks": {
            "official_fp32_rows_available": True,
            "official_full_test_failed_after_fp32": True,
            "official_qps_runner_missing": True,
            "hanns_raw_refine_unavailable": True,
            "leadership_claim_blocked": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_hnsw_sq_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "hnsw_sq_milestone_verdict",
        "status": "win",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "recall_tolerance": 0.001,
            "throughput_metric": "qps",
            "vps_qps_mapping": "one_vector_per_query",
            "thread_policy": "match_or_explicitly_normalized",
            "archived_evidence_required": False,
        },
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": f"main@{'a' * 40}",
            "log_path": "/data/work/knowhere-zilliz-official-main-logs/hnsw-sq.log",
            "runner": "benchmark_float",
            "variant": "FP32",
            "full_test_status": "failed_after_fp32_rows",
            "top_k": 100,
            "m": 16,
            "ef_construction": 200,
            "ef": 128,
            "refine_k": 1,
            "threads": 8,
            "build_threads": 8,
            "nq": 10000,
            "elapsed_s": 0.323,
            "recall_at_100": 0.9531,
            "build_s": 66.281,
            "qps_or_vps": 30959.752321981426,
        },
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/hnsw-sq",
            "log_path": "/data/work/hanns-logs/hnsw-sq.log",
            "top_k": 100,
            "recall_at_100": 0.9595,
            "m": 16,
            "ef_construction": 128,
            "ef": 256,
            "sq_mode": "SQ8Refine",
            "threads": 8,
            "build_threads": 8,
            "build_s": 59.486,
            "train_s": 0.0,
            "add_s": 59.486,
            "qps_or_vps": 38561.913,
        },
        "verdict_checks": {
            "same_top_k": True,
            "same_or_higher_recall_with_tolerance": True,
            "throughput_units_comparable": True,
            "thread_policy_satisfied": True,
            "hanns_qps_gt_official": True,
            "hanns_build_s_lt_official": True,
            "official_fp32_row_bound": True,
            "official_full_test_failed_after_fp32": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


def valid_ivfusq_verdict(**overrides) -> dict:
    verdict = {
        "artifact_type": "ivfusq_search_milestone_verdict",
        "status": "search_win",
        "authority_surface": "HannsDB-x86",
        "run_set_id": RUN_SET_ID,
        "plan_approved_at": PLAN_APPROVED_AT,
        "comparison_contract": {
            "metric_of_record": "recall_at_100",
            "top_k": 100,
            "recall_tolerance": 0.001,
            "throughput_metric": "vps",
            "vps_qps_mapping": "one_vector_per_query",
            "archived_evidence_required": False,
        },
        "official_target": {
            "repo_url": "https://github.com/zilliztech/knowhere",
            "commit": "a" * 40,
            "ref": "main",
            "log_path": "/data/work/knowhere-zilliz-official-main-logs/ivfrabitq.log",
            "runner": "benchmark_float",
            "variant": "FP32",
            "top_k": 100,
            "nlist": 1024,
            "nprobe": 64,
            "nq": 10000,
            "elapsed_s": 1.294,
            "recall_at_100": 0.6146,
            "qps_or_vps": 7727.975270479134,
        },
        "hanns_candidate": {
            "commit": "b" * 40,
            "ref": "feature/ivfusq",
            "log_path": "/data/work/hanns-logs/ivfusq.log",
            "top_k": 100,
            "recall_at_100": 0.6144,
            "nlist": 1024,
            "nprobe": 4,
            "bits_per_dim": 8,
            "threads": 8,
            "build_s": 85.762,
            "train_s": 77.841,
            "add_s": 7.922,
            "qps_or_vps": 19661.756,
        },
        "verdict_checks": {
            "same_top_k": True,
            "same_or_higher_recall_with_tolerance": True,
            "throughput_units_comparable": True,
            "hanns_qps_gt_official": True,
            "official_runner_non_qps": True,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(verdict.get(key), dict):
            verdict[key].update(value)
        else:
            verdict[key] = value
    return verdict


class HannsKnowhereBenchmarkTests(unittest.TestCase):
    def test_matrix_contains_all_required_user_families(self) -> None:
        matrix = hkb.default_matrix()
        hkb.validate_matrix(matrix)
        self.assertEqual(set(matrix["required_families"]), set(hkb.REQUIRED_FAMILIES))

    def test_rejects_non_zilliz_knowhere_rows(self) -> None:
        rows = complete_rows()
        rows[1]["source"]["repo_url"] = "https://github.com/benwuhua/knowhere"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("zilliztech/knowhere", "\n".join(ctx.exception.errors))

    def test_rejects_local_rows_from_final_aggregation(self) -> None:
        rows = complete_rows()
        rows[0]["authority"] = {
            "surface": "local",
            "is_authority": False,
            "eligible_for_verdict": False,
            "purpose": "read_only_inspection_or_non_authority_smoke",
        }
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("final row must use HannsDB-x86", "\n".join(ctx.exception.errors))

    def test_rejects_hannsdb_string_without_runtime_proof(self) -> None:
        rows = complete_rows()
        del rows[0]["authority"]["runtime_proof"]
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("runtime_proof", "\n".join(ctx.exception.errors))

    def test_rejects_spoofed_runtime_proof_not_matching_manifest(self) -> None:
        rows = complete_rows()
        rows[0]["authority"]["runtime_proof"]["hostname"] = "fake-local-host"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("does not match manifest", "\n".join(ctx.exception.errors))

    def test_rejects_empty_authority_manifest(self) -> None:
        rows = complete_rows()
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest={},
            )
        self.assertIn("authority manifest missing hostname", "\n".join(ctx.exception.errors))

    def test_rejects_manifest_missing_allowed_wrappers(self) -> None:
        rows = complete_rows()
        manifest = authority_manifest()
        manifest["runtime_proof"].pop("allowed_wrapper_scripts")
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=manifest,
            )
        self.assertIn("allowed_wrapper_scripts", "\n".join(ctx.exception.errors))

    def test_rejects_disallowed_wrapper_script(self) -> None:
        rows = complete_rows()
        rows[0]["authority"]["runtime_proof"]["wrapper_script"] = "scripts/remote/evil.sh"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("wrapper_script is not allowed", "\n".join(ctx.exception.errors))

    def test_rejects_sibling_prefix_log_root_spoofing(self) -> None:
        rows = complete_rows()
        rows[0]["authority"]["runtime_proof"][
            "remote_log_path"
        ] = "/data/work/hanns-logs-evil/run.log"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("outside manifest log root", "\n".join(ctx.exception.errors))

    def test_rejects_missing_row_plan_approved_at(self) -> None:
        rows = complete_rows()
        rows[0]["run_set"].pop("plan_approved_at")
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("plan_approved_at", "\n".join(ctx.exception.errors))

    def test_rejects_rows_from_old_run_set(self) -> None:
        rows = complete_rows()
        rows[0]["run_set"]["run_set_id"] = "old-run-set"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("current run set", "\n".join(ctx.exception.errors))

    def test_rejects_rows_older_than_plan_approval(self) -> None:
        rows = complete_rows()
        rows[0]["run_set"]["generated_at"] = "2026-04-23T02:00:00Z"
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("predates plan approval", "\n".join(ctx.exception.errors))

    def test_supported_row_requires_qps_recall_build_latency(self) -> None:
        rows = complete_rows()
        rows[0]["metrics"]["latency_ms"].pop("p99")
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_rows(
                rows,
                plan_approved_at=PLAN_APPROVED_AT,
                run_set_id=RUN_SET_ID,
                authority_manifest=authority_manifest(),
            )
        self.assertIn("latency_ms.p99", "\n".join(ctx.exception.errors))

    def test_capability_report_has_status_for_each_family_and_implementation(self) -> None:
        rows = complete_rows(status="non_comparable")
        hkb.validate_rows(
            rows,
            plan_approved_at=PLAN_APPROVED_AT,
            run_set_id=RUN_SET_ID,
            authority_manifest=authority_manifest(),
        )
        rendered = hkb.render_markdown(
            rows,
            plan_approved_at=PLAN_APPROVED_AT,
            run_set_id=RUN_SET_ID,
            authority_manifest=authority_manifest(),
        )
        for family in hkb.REQUIRED_FAMILIES:
            self.assertIn(family, rendered)

    def test_unsupported_rows_are_not_dropped(self) -> None:
        rows = complete_rows(status="unsupported")
        rendered = hkb.render_markdown(
            rows,
            plan_approved_at=PLAN_APPROVED_AT,
            run_set_id=RUN_SET_ID,
            authority_manifest=authority_manifest(),
        )
        self.assertEqual(rendered.count("unsupported"), len(hkb.REQUIRED_FAMILIES) * 2)

    def test_check_knowhere_source_requires_zilliz_origin_and_clean_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/benwuhua/knowhere.git"],
                cwd=repo,
                check=True,
            )
            (repo / "README.md").write_text("fixture\n", encoding="utf-8")
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.inspect_knowhere_repo(repo, "main")
            self.assertIn("zilliztech/knowhere", "\n".join(ctx.exception.errors))

    def test_check_knowhere_source_requires_expected_ref_to_match_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.invalid"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "remote", "add", "origin", "https://github.com/zilliztech/knowhere.git"],
                cwd=repo,
                check=True,
            )
            (repo / "README.md").write_text("fixture\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.inspect_knowhere_repo(repo, "missing-ref")
            self.assertIn("expected Knowhere ref", "\n".join(ctx.exception.errors))

    def test_cli_render_report_requires_manifest_and_fresh_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            rows_path = tmp_path / "rows.json"
            manifest_path = tmp_path / "manifest.json"
            report_path = tmp_path / "report.md"
            rows_path.write_text(json.dumps({"rows": complete_rows(status="unsupported")}), encoding="utf-8")
            manifest_path.write_text(json.dumps(authority_manifest()), encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(MODULE_PATH),
                    "render-report",
                    "--rows",
                    str(rows_path),
                    "--output",
                    str(report_path),
                    "--plan-approved-at",
                    PLAN_APPROVED_AT,
                    "--run-set-id",
                    RUN_SET_ID,
                    "--authority-manifest",
                    str(manifest_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("DISKANN-USQ/RabitQ", report_path.read_text(encoding="utf-8"))

    def test_cli_write_and_validate_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "matrix.json"
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "write-matrix", "--output", str(out)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            validate = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-matrix", str(out)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validate.returncode, 0, msg=validate.stderr)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["authority_surface"], "HannsDB-x86")

    def test_ivfpq_recall_band_uses_floor_percent_bucket(self) -> None:
        self.assertEqual(hkb.ivfpq_recall_band_floor(0.7841), 0.78)
        self.assertEqual(hkb.ivfpq_recall_band_floor(0.9502), 0.95)
        self.assertEqual(hkb.ivfpq_recall_band_floor(1.0), 1.0)

    def test_hnsw_verdict_win_requires_all_checks_true(self) -> None:
        hkb.validate_hnsw_verdict(valid_hnsw_verdict())
        verdict = valid_hnsw_verdict(
            verdict_checks={"hanns_qps_vps_gt_official": False}
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_hnsw_verdict(verdict)
        self.assertIn("status=win requires all verdict checks", "\n".join(ctx.exception.errors))

    def test_hnsw_verdict_marks_unit_or_thread_mismatch_non_comparable(self) -> None:
        verdict = valid_hnsw_verdict(
            status="non_comparable",
            comparison_contract={"vps_qps_mapping": "missing"},
            hanns_candidate={"threads": 4},
            verdict_checks={
                "throughput_units_comparable": False,
                "thread_policy_satisfied": False,
            },
        )
        hkb.validate_hnsw_verdict(verdict)
        bad = valid_hnsw_verdict(
            comparison_contract={"vps_qps_mapping": "missing"},
            verdict_checks={"throughput_units_comparable": False},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_hnsw_verdict(bad)
        self.assertIn("non_comparable", "\n".join(ctx.exception.errors))

    def test_hnsw_archived_evidence_binds_logs_and_artifact_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns.json"
            artifact.write_text(
                json.dumps(
                    {
                        "authority_surface": "HannsDB-x86",
                        "rows": [
                            {
                                "m": 6,
                                "ef_construction": 34,
                                "ef": 850,
                                "top_k": 100,
                                "threads": 8,
                                "build_threads": 8,
                                "recall_at_100": 0.9198,
                                "qps": 30972.456,
                                "build_s": 23.878,
                                "train_s": 0.0,
                                "add_s": 23.878,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "HNSW_M=6 ef_construction=34 HNSW_BUILD_THREADS=8\n"
                "HNSW aligned ef=850: build=23.878s train=0.000s "
                "add=23.878s qps=30972.456 R@10=0.9469 R@100=0.9198\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            official_log = root / "official.log"
            official_log.write_text(
                "Build index HNSW time: 29.339s\n"
                "sift-128-euclidean | HNSW(FP16) | M=16 | "
                "efConstruction=100, ef=100, k=100, R@=0.9178\n"
                "thread_num =  8, elapse = 0.333s, VPS = 30013.983\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_status = root / "official.status"
            official_status.write_text("status=ok\n", encoding="utf-8")

            verdict = valid_hnsw_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                },
            )
            hkb.validate_hnsw_verdict(verdict)

            mismatched = valid_hnsw_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={"qps_or_vps": 30973.0},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_hnsw_verdict(mismatched)
            self.assertIn("artifact qps", "\n".join(ctx.exception.errors))

    def test_cli_validate_hnsw_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(json.dumps(valid_hnsw_verdict()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-hnsw-verdict", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_ivfsq8_verdict_win_requires_all_checks_true(self) -> None:
        hkb.validate_ivfsq8_verdict(valid_ivfsq8_verdict())
        verdict = valid_ivfsq8_verdict(
            verdict_checks={"hanns_qps_vps_gt_official": False}
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfsq8_verdict(verdict)
        self.assertIn("status=win requires all verdict checks", "\n".join(ctx.exception.errors))

    def test_ivfsq8_archived_evidence_binds_logs_and_artifact_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns.json"
            artifact.write_text(
                json.dumps(
                    {
                        "authority_surface": "HannsDB-x86",
                        "rows": [
                            {
                                "nprobe": 36,
                                "top_k": 100,
                                "threads": 8,
                                "recall_at_100": 0.9532,
                                "qps": 16046.2,
                                "build_s": 3.177,
                                "train_s": 1.61,
                                "add_s": 1.567,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "IVFSQ8_EXPECT_THREADS=8\n"
                "IVF-SQ8 aligned nprobe=36: build=3.177s train=1.610s "
                "add=1.567s qps=16046.200 R@10=0.9622 R@100=0.9532\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            official_log = root / "official.log"
            official_log.write_text(
                "Build index IVF_SQ8 time: 6.875s\n"
                "sift-128-euclidean | IVF_SQ8(BF16) | "
                "nlist=1024, nprobe=  32, k=100, R@=0.9519\n"
                "thread_num =  8, elapse = 0.766s, VPS = 13056.415\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_status = root / "official.status"
            official_status.write_text("status=ok\n", encoding="utf-8")

            verdict = valid_ivfsq8_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                },
            )
            hkb.validate_ivfsq8_verdict(verdict)

            mismatched = valid_ivfsq8_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={"qps_or_vps": 16047.0},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_ivfsq8_verdict(mismatched)
            self.assertIn("artifact qps", "\n".join(ctx.exception.errors))

    def test_cli_validate_ivfsq8_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(json.dumps(valid_ivfsq8_verdict()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-ivfsq8-verdict", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_diskann_aisaq_verdict_blocks_leadership_claims(self) -> None:
        hkb.validate_diskann_aisaq_verdict(valid_diskann_aisaq_verdict())
        verdict = valid_diskann_aisaq_verdict(
            leadership_claim_allowed=True,
            verdict_checks={"leadership_claim_blocked": False},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_diskann_aisaq_verdict(verdict)
        self.assertIn("leadership_claim_allowed must be false", "\n".join(ctx.exception.errors))

    def test_diskann_aisaq_archived_evidence_binds_hanns_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns-aisaq.json"
            artifact.write_text(
                json.dumps(
                    {
                        "artifact_type": "diskann_aisaq_aligned_hanns_rows",
                        "authority_surface": "HannsDB-x86",
                        "native_comparable": False,
                        "leadership_claim_allowed": False,
                        "rows": [
                            {
                                "config_name": "R48-L108-B8-EP1",
                                "search_surface": "mmap",
                                "top_k": 100,
                                "threads": 8,
                                "recall_at_100": 0.955,
                                "qps": 3000.0,
                                "build_s": 38.0,
                                "persist_s": 0.5,
                                "load_s": 0.01,
                                "search_list_size": 108,
                                "native_comparable": False,
                                "scope_audit": {
                                    "uses_mmap_backed_pages": True,
                                    "native_comparable": False,
                                },
                            }
                        ],
                        "search_surface": "mmap",
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "AISAQ_EXPECT_THREADS=8\n"
                "AISAQ aligned config=R48-L108-B8-EP1 surface=mmap search_list_size=108 "
                "build=38.000s qps=3000.000 R@10=0.9700 R@100=0.9550 "
                "native_comparable=false\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_log = root / "official-aisaq-p.log"
            official_log.write_text(
                "Build index: done (214674.027647 ms)\n"
                "sift-128-euclidean | AISAQ(FP32) | "
                "search_list_size=108, k=100, R@=0.9502\n"
                "  thread_num =  8, elapse =  3.207s, VPS = 3117.801\n",
                encoding="utf-8",
            )
            official_status = root / "official-aisaq-p.status"
            official_status.write_text("status=ok\n", encoding="utf-8")

            verdict = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={
                    "search_surface": "mmap",
                    "persist_s": 0.5,
                    "load_s": 0.01,
                    "uses_mmap_backed_pages": True,
                },
                official_targets=[
                    {
                        **valid_diskann_aisaq_verdict()["official_targets"][0],
                        "archived_log": str(official_log),
                        "archived_status": str(official_status),
                    }
                ],
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_logs": [str(official_log)],
                    "archived_official_statuses": [str(official_status)],
                },
            )
            hkb.validate_diskann_aisaq_verdict(verdict)

            page_artifact = root / "hanns-page-cache.json"
            page_artifact.write_text(
                json.dumps(
                    {
                        "artifact_type": "diskann_aisaq_aligned_hanns_rows",
                        "authority_surface": "HannsDB-x86",
                        "native_comparable": False,
                        "leadership_claim_allowed": False,
                        "rows": [
                            {
                                "config_name": "R48-L108-B8-EP1",
                                "search_surface": "page_cache",
                                "disk_pq_dims": 8,
                                "pq_candidate_expand_pct": 400,
                                "rerank_expand_pct": 400,
                                "top_k": 100,
                                "threads": 8,
                                "recall_at_100": 0.955,
                                "qps": 3000.0,
                                "build_s": 38.0,
                                "persist_s": 0.5,
                                "load_s": 0.01,
                                "search_list_size": 108,
                                "native_comparable": False,
                                "scope_audit": {
                                    "uses_mmap_backed_pages": False,
                                    "has_page_cache": True,
                                    "native_comparable": False,
                                },
                            }
                        ],
                        "search_surface": "page_cache",
                        "disk_pq_dims": 8,
                        "pq_candidate_expand_pct": 400,
                        "rerank_expand_pct": 400,
                    }
                ),
                encoding="utf-8",
            )
            page_log = root / "hanns-page-cache.log"
            page_log.write_text(
                "AISAQ_EXPECT_THREADS=8\n"
                "AISAQ aligned config=R48-L108-B8-EP1 surface=page_cache disk_pq_dims=8 "
                "pq_candidate_expand_pct=400 rerank_expand_pct=400 "
                "search_list_size=108 build=38.000s qps=3000.000 R@10=0.9700 "
                "R@100=0.9550 native_comparable=false\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            page_verdict = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={
                    "search_surface": "page_cache",
                    "disk_pq_dims": 8,
                    "pq_candidate_expand_pct": 400,
                    "rerank_expand_pct": 400,
                    "persist_s": 0.5,
                    "load_s": 0.01,
                    "has_page_cache": True,
                },
                official_targets=[
                    {
                        **valid_diskann_aisaq_verdict()["official_targets"][0],
                        "archived_log": str(official_log),
                        "archived_status": str(official_status),
                    }
                ],
                evidence={
                    "hanns_aligned_artifact": str(page_artifact),
                    "archived_hanns_log": str(page_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_logs": [str(official_log)],
                    "archived_official_statuses": [str(official_status)],
                },
            )
            hkb.validate_diskann_aisaq_verdict(page_verdict)

            bad_page_artifact = root / "hanns-page-cache-bad.json"
            bad_payload = json.loads(page_artifact.read_text(encoding="utf-8"))
            bad_payload["rows"][0]["scope_audit"]["has_page_cache"] = False
            bad_page_artifact.write_text(json.dumps(bad_payload), encoding="utf-8")
            bad_page_verdict = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={
                    "search_surface": "page_cache",
                    "disk_pq_dims": 8,
                    "persist_s": 0.5,
                    "load_s": 0.01,
                    "has_page_cache": True,
                },
                evidence={
                    **page_verdict["evidence"],
                    "hanns_aligned_artifact": str(bad_page_artifact),
                },
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(bad_page_verdict)
            self.assertIn("has_page_cache", "\n".join(ctx.exception.errors))

            mismatched = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={"qps_or_vps": 3001.0},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(mismatched)
            self.assertIn("artifact qps", "\n".join(ctx.exception.errors))

            mismatched_surface = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={
                    "search_surface": "page_cache",
                    "persist_s": 0.5,
                    "load_s": 0.01,
                    "uses_mmap_backed_pages": True,
                },
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(mismatched_surface)
            self.assertIn("search_surface", "\n".join(ctx.exception.errors))

            missing_mmap_proof = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={
                    "search_surface": "mmap",
                    "persist_s": 0.5,
                    "load_s": 0.01,
                    "uses_mmap_backed_pages": False,
                },
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(missing_mmap_proof)
            self.assertIn("uses_mmap_backed_pages", "\n".join(ctx.exception.errors))

            mismatched_official = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                official_targets=[
                    {
                        **valid_diskann_aisaq_verdict()["official_targets"][0],
                        "archived_log": str(official_log),
                        "archived_status": str(official_status),
                        "qps_or_vps": 3118.0,
                    }
                ],
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(mismatched_official)
            self.assertIn("thread/VPS tokens", "\n".join(ctx.exception.errors))

            missing_official_status = valid_diskann_aisaq_verdict(
                comparison_contract={"archived_evidence_required": True},
                official_targets=[
                    {
                        **valid_diskann_aisaq_verdict()["official_targets"][0],
                        "archived_log": str(official_log),
                    }
                ],
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_logs": [str(official_log)],
                },
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_diskann_aisaq_verdict(missing_official_status)
            self.assertIn(
                "evidence.archived_official_statuses", "\n".join(ctx.exception.errors)
            )

    def test_cli_validate_diskann_aisaq_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(
                json.dumps(valid_diskann_aisaq_verdict()), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    "python3",
                    str(MODULE_PATH),
                    "validate-diskann-aisaq-verdict",
                    str(verdict_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_hnsw_pq_blocker_requires_raw_refine_block(self) -> None:
        hkb.validate_hnsw_pq_blocker(valid_hnsw_pq_blocker())
        verdict = valid_hnsw_pq_blocker(hanns_capability={"has_raw_data": True})
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_hnsw_pq_blocker(verdict)
        self.assertIn("has_raw_data must be false", "\n".join(ctx.exception.errors))

    def test_hnsw_pq_blocker_archived_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            official_log = root / "official.log"
            official_log.write_text(
                "Note: Google Test filter = Benchmark_float.TEST_HNSW_PQ\n"
                "Build index HNSW_PQ time: 84.542s \n"
                "sift-128-euclidean | HNSW_PQ(FP32) | hnsw_M=16, efc=200, ef=128\n"
                "  refine_k =  16, nq = 10000, k =  100, elapse =  1.980s, R@ = 0.9576\n"
                "C++ exception with description \"bad optional access\" thrown in the test body.\n"
                "[  FAILED  ] Benchmark_float.TEST_HNSW_PQ\n",
                encoding="utf-8",
            )
            official_status = root / "official.status"
            official_status.write_text("status=failed\n", encoding="utf-8")
            qps_list = root / "qps-list.log"
            qps_list.write_text(
                f"commit={'a' * 40}\nBenchmark_float_qps.\n  TEST_HNSW\n",
                encoding="utf-8",
            )
            hanns_log = root / "hanns-capability.log"
            hanns_log.write_text(
                "test faiss::hnsw_pq::tests::test_hnsw_pq_has_raw_data_is_false ... ok\n"
                "test faiss::hnsw_pq::tests::test_hnsw_pq_get_vector_by_ids_returns_stable_unsupported ... ok\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns-capability.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            source_log = root / "source.log"
            source_log.write_text(
                "provenance_kind=official_knowhere_source\n"
                "origin_url=https://github.com/zilliztech/knowhere.git\n"
                f"head_commit={'a' * 40}\n"
                "official_source_ok=true\n",
                encoding="utf-8",
            )
            source_status = root / "source.status"
            source_status.write_text("status=ok\n", encoding="utf-8")
            verdict = valid_hnsw_pq_blocker(
                evidence_required=True,
                evidence={
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                    "archived_qps_gtest_list": str(qps_list),
                    "archived_hanns_capability_log": str(hanns_log),
                    "archived_hanns_capability_status": str(hanns_status),
                    "official_source_inspection": str(source_log),
                    "official_source_status": str(source_status),
                },
            )
            hkb.validate_hnsw_pq_blocker(verdict)

            exposed_qps = root / "qps-exposes-pq.log"
            exposed_qps.write_text(
                qps_list.read_text() + "  TEST_HNSW_PQ\n", encoding="utf-8"
            )
            bad = valid_hnsw_pq_blocker(
                evidence_required=True,
                evidence={**verdict["evidence"], "archived_qps_gtest_list": str(exposed_qps)},
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_hnsw_pq_blocker(bad)
            self.assertIn("must not expose HNSW_PQ", "\n".join(ctx.exception.errors))

    def test_cli_validate_hnsw_pq_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "blocker.json"
            verdict_path.write_text(json.dumps(valid_hnsw_pq_blocker()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-hnsw-pq-blocker", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_hnsw_sq_verdict_requires_official_partial_row_checks(self) -> None:
        hkb.validate_hnsw_sq_verdict(valid_hnsw_sq_verdict())
        verdict = valid_hnsw_sq_verdict(
            official_target={"full_test_status": "ok"},
            verdict_checks={"official_fp32_row_bound": True},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_hnsw_sq_verdict(verdict)
        self.assertIn("official_fp32_row_bound", "\n".join(ctx.exception.errors))

    def test_hnsw_sq_archived_evidence_binds_rows_and_failed_official_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns-hnsw-sq.json"
            artifact.write_text(
                json.dumps(
                    {
                        "artifact_type": "hnsw_sq_aligned_hanns_rows",
                        "authority_surface": "HannsDB-x86",
                        "rows": [
                            {
                                "m": 16,
                                "ef_construction": 128,
                                "ef": 256,
                                "top_k": 100,
                                "threads": 8,
                                "build_threads": 8,
                                "sq_mode": "SQ8Refine",
                                "recall_at_100": 0.9595,
                                "qps": 38561.913,
                                "build_s": 59.486,
                                "train_s": 0.0,
                                "add_s": 59.486,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "HNSWSQ_EXPECT_THREADS=8 HNSWSQ_M=16 HNSWSQ_EF_CONSTRUCTION=128\n"
                "HNSW-SQ aligned mode=SQ8Refine ef=256: build=59.486s train=0.000s "
                "add=59.486s qps=38561.913 R@10=0.9774 R@100=0.9595\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            official_log = root / "official.log"
            official_log.write_text(
                "Note: Google Test filter = Benchmark_float.TEST_HNSW_SQ\n"
                "Build index HNSW_SQ time: 66.281s \n"
                "sift-128-euclidean | HNSW_SQ(FP32) | hnsw_M=16, efc=200, ef=128\n"
                "  refine_k =   1, nq = 10000, k =  100, elapse =  0.323s, R@ = 0.9531\n"
                "C++ exception with description \"bad optional access\" thrown in the test body.\n"
                "[  FAILED  ] Benchmark_float.TEST_HNSW_SQ\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_status = root / "official.status"
            official_status.write_text("status=failed\n", encoding="utf-8")
            source_log = root / "source.log"
            source_log.write_text(
                f"provenance_kind=official_knowhere_source\n"
                f"origin_url=https://github.com/zilliztech/knowhere.git\n"
                f"head_commit={'a' * 40}\n"
                f"official_source_ok=true\n",
                encoding="utf-8",
            )
            source_status = root / "source.status"
            source_status.write_text("status=ok\n", encoding="utf-8")
            verdict = valid_hnsw_sq_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                    "official_source_inspection": str(source_log),
                    "official_source_status": str(source_status),
                },
            )
            hkb.validate_hnsw_sq_verdict(verdict)

            bad = valid_hnsw_sq_verdict(
                comparison_contract={"archived_evidence_required": True},
                hanns_candidate={"ef": 384},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_hnsw_sq_verdict(bad)
            self.assertIn("no HNSW-SQ row matching", "\n".join(ctx.exception.errors))

            missing_source = valid_hnsw_sq_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={
                    **verdict["evidence"],
                    "official_source_inspection": str(root / "missing-source.log"),
                },
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_hnsw_sq_verdict(missing_source)
            self.assertIn("official_source_inspection", "\n".join(ctx.exception.errors))

            bad_ref = valid_hnsw_sq_verdict(official_target={"ref": "main"})
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_hnsw_sq_verdict(bad_ref)
            self.assertIn("must pin official_target.commit", "\n".join(ctx.exception.errors))

    def test_cli_validate_hnsw_sq_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(json.dumps(valid_hnsw_sq_verdict()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-hnsw-sq-verdict", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_ivfusq_search_verdict_requires_checks_for_win(self) -> None:
        hkb.validate_ivfusq_verdict(valid_ivfusq_verdict())
        verdict = valid_ivfusq_verdict(verdict_checks={"hanns_qps_gt_official": False})
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfusq_verdict(verdict)
        self.assertIn(
            "status=search_win requires all verdict checks", "\n".join(ctx.exception.errors)
        )

    def test_ivfusq_archived_evidence_binds_hanns_and_official_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns-ivfusq.json"
            artifact.write_text(
                json.dumps(
                    {
                        "artifact_type": "ivfusq_aligned_hanns_rows",
                        "authority_surface": "HannsDB-x86",
                        "rows": [
                            {
                                "bits_per_dim": 8,
                                "nprobe": 4,
                                "top_k": 100,
                                "threads": 8,
                                "recall_at_100": 0.6144,
                                "qps": 19661.756,
                                "build_s": 85.762,
                                "train_s": 77.841,
                                "add_s": 7.922,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "IVFUSQ_EXPECT_THREADS=8\n"
                "IVF-USQ aligned bits=8 nprobe=4: build=85.762s train=77.841s "
                "add=7.922s qps=19661.756 R@10=0.7051 R@100=0.6144\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            official_log = root / "official.log"
            official_log.write_text(
                "Note: Google Test filter = Benchmark_float.TEST_IVF_RABITQ\n"
                "[142.052 s] sift-128-euclidean | IVF_RABITQ(FP32) | nlist=1024\n"
                "  nprobe =   64, nq = 10000, k =  100, elapse =  1.294s, R@ = 0.6146\n"
                "[  PASSED  ] 1 test.\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_status = root / "official.status"
            official_status.write_text("status=ok\n", encoding="utf-8")
            qps_list = root / "qps-list.log"
            qps_list.write_text(
                f"commit={'a' * 40}\n"
                "Benchmark_float_qps.\n"
                "  TEST_IDMAP\n"
                "  TEST_IVF_SQ8\n"
                "  TEST_IVF_PQ\n",
                encoding="utf-8",
            )

            verdict = valid_ivfusq_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                    "archived_qps_gtest_list": str(qps_list),
                },
            )
            hkb.validate_ivfusq_verdict(verdict)

            mismatched = valid_ivfusq_verdict(
                comparison_contract={"archived_evidence_required": True},
                official_target={"elapsed_s": 1.290},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_ivfusq_verdict(mismatched)
            self.assertIn("archived official log missing token", "\n".join(ctx.exception.errors))

            exposed_qps = root / "qps-list-exposes-rabitq.log"
            exposed_qps.write_text(
                qps_list.read_text() + "  TEST_IVF_RABITQ\n",
                encoding="utf-8",
            )
            bad_qps = valid_ivfusq_verdict(
                comparison_contract={"archived_evidence_required": True},
                evidence={**verdict["evidence"], "archived_qps_gtest_list": str(exposed_qps)},
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_ivfusq_verdict(bad_qps)
            self.assertIn("must not expose IVF_RABITQ", "\n".join(ctx.exception.errors))

    def test_cli_validate_ivfusq_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(json.dumps(valid_ivfusq_verdict()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-ivfusq-verdict", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_ivfpq_verdict_win_requires_all_checks_true(self) -> None:
        hkb.validate_ivfpq_verdict(valid_ivfpq_verdict())
        verdict = valid_ivfpq_verdict(
            verdict_checks={"hanns_qps_vps_gt_official": False}
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        self.assertIn("status=win requires all verdict checks", "\n".join(ctx.exception.errors))

    def test_ivfpq_verdict_blocks_without_official_normalization(self) -> None:
        verdict = valid_ivfpq_verdict(
            status="blocked_official_normalization",
            official_target={
                "normalization_source": "none",
                "excluded_rows": ["terminal_zero_anomaly"],
            },
            verdict_checks={"official_normalized": False},
        )
        hkb.validate_ivfpq_verdict(verdict)
        bad = valid_ivfpq_verdict(
            status="win",
            official_target={"normalization_source": "none"},
            verdict_checks={"official_normalized": False},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(bad)
        self.assertIn("blocked_official_normalization", "\n".join(ctx.exception.errors))

    def test_ivfpq_verdict_marks_unit_or_thread_mismatch_non_comparable(self) -> None:
        verdict = valid_ivfpq_verdict(
            status="non_comparable",
            comparison_contract={"vps_qps_mapping": "missing"},
            official_target={"threads": 8},
            hanns_candidate={"threads": 1},
            verdict_checks={
                "throughput_units_comparable": False,
                "thread_policy_satisfied": False,
            },
        )
        hkb.validate_ivfpq_verdict(verdict)
        bad = valid_ivfpq_verdict(
            comparison_contract={"vps_qps_mapping": "missing"},
            verdict_checks={"throughput_units_comparable": False},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(bad)
        self.assertIn("non_comparable", "\n".join(ctx.exception.errors))

    def test_ivfpq_verdict_records_excluded_terminal_anomaly(self) -> None:
        verdict = valid_ivfpq_verdict(official_target={"excluded_rows": []})
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        self.assertIn("terminal_zero_anomaly", "\n".join(ctx.exception.errors))

    def test_ivfpq_hanns_row_requires_same_top_k_and_metric_of_record(self) -> None:
        verdict = valid_ivfpq_verdict(hanns_candidate={"top_k": 10})
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        self.assertIn("same top_k", "\n".join(ctx.exception.errors))

    def test_ivfpq_hanns_row_allows_only_noise_tolerance_not_lower_band(self) -> None:
        verdict = valid_ivfpq_verdict(
            hanns_candidate={
                "recall_at_100": 0.779,
                "recall_band_floor": 0.77,
            }
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        errors = "\n".join(ctx.exception.errors)
        self.assertIn("recall", errors)
        self.assertIn("band", errors)


    def test_ivfpq_build_time_required_matches_build_values(self) -> None:
        hkb.validate_ivfpq_verdict(valid_ivfpq_verdict())
        verdict = valid_ivfpq_verdict(
            hanns_candidate={"build_s": 13.0},
            verdict_checks={"hanns_build_s_lt_official": False},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        self.assertIn("build_s below official", "\n".join(ctx.exception.errors))

        mismatched = valid_ivfpq_verdict(
            hanns_candidate={"build_s": 13.0},
            verdict_checks={"hanns_build_s_lt_official": True},
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(mismatched)
        self.assertIn("does not match build_s values", "\n".join(ctx.exception.errors))


    def test_ivfpq_archived_evidence_binds_logs_and_artifact_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            artifact = root / "hanns.json"
            artifact.write_text(
                json.dumps(
                    {
                        "authority_surface": "HannsDB-x86",
                        "rows": [
                            {
                                "nprobe": 100,
                                "top_k": 100,
                                "threads": 8,
                                "recall_at_100": 0.785,
                                "qps": 900.0,
                                "build_s": 10.0,
                                "train_s": 1.0,
                                "add_s": 9.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            hanns_log = root / "hanns.log"
            hanns_log.write_text(
                "IVFPQ_EXPECT_THREADS=8\n"
                "IVF-PQ aligned nprobe=100: build=10.000s train=1.000s "
                "add=9.000s qps=900.000 R@10=0.9000 R@100=0.7850\n"
                "test result: ok\n",
                encoding="utf-8",
            )
            official_log = root / "official.log"
            official_log.write_text(
                "sift-128-euclidean_IVF_PQ_1024_32_fp16.index\n"
                "Build index IVF_PQ time: 12.000s\n"
                "nprobe=  64, k=100, R@=0.7841\n"
                "thread_num =  8, elapse = 1.0s, VPS = 800.000\n",
                encoding="utf-8",
            )
            hanns_status = root / "hanns.status"
            hanns_status.write_text("status=ok\n", encoding="utf-8")
            official_status = root / "official.status"
            official_status.write_text("status=ok\n", encoding="utf-8")

            verdict = valid_ivfpq_verdict(
                comparison_contract={"archived_evidence_required": True},
                official_target={"m": 32},
                hanns_candidate={"m": 32, "train_s": 1.0, "add_s": 9.0},
                evidence={
                    "hanns_aligned_artifact": str(artifact),
                    "archived_hanns_log": str(hanns_log),
                    "archived_hanns_status": str(hanns_status),
                    "archived_official_log": str(official_log),
                    "archived_official_status": str(official_status),
                },
            )
            hkb.validate_ivfpq_verdict(verdict)

            mismatched = valid_ivfpq_verdict(
                comparison_contract={"archived_evidence_required": True},
                official_target={"m": 32},
                hanns_candidate={"m": 32, "qps_or_vps": 901.0, "train_s": 1.0, "add_s": 9.0},
                evidence=verdict["evidence"],
            )
            with self.assertRaises(hkb.ValidationError) as ctx:
                hkb.validate_ivfpq_verdict(mismatched)
            self.assertIn("artifact qps", "\n".join(ctx.exception.errors))

    def test_ivfpq_win_requires_refine_metadata(self) -> None:
        verdict = valid_ivfpq_verdict(
            hanns_candidate={"refine": {"enabled": False, "candidate_pool": ""}}
        )
        with self.assertRaises(hkb.ValidationError) as ctx:
            hkb.validate_ivfpq_verdict(verdict)
        errors = "\n".join(ctx.exception.errors)
        self.assertIn("refine.enabled", errors)
        self.assertIn("exact_refine_multiplier", errors)
        self.assertIn("candidate_pool", errors)

    def test_cli_validate_ivfpq_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict_path = pathlib.Path(tmp) / "verdict.json"
            verdict_path.write_text(json.dumps(valid_ivfpq_verdict()), encoding="utf-8")
            result = subprocess.run(
                ["python3", str(MODULE_PATH), "validate-ivfpq-verdict", str(verdict_path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
