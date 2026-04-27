#!/usr/bin/env python3
"""Schema, matrix, and report gates for Hanns vs official Knowhere benchmarks.

This module intentionally does *not* run performance benchmarks.  It provides
the fail-closed validation layer required before fresh HannsDB-x86 benchmark
rows can be admitted into a final comparison report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import posixpath
import re
import subprocess
import sys
from typing import Any


PLAN_ID = "hanns-knowhere-benchmark-v1"
AUTHORITY_SURFACE = "HannsDB-x86"
OFFICIAL_KNOWHERE_URL = "https://github.com/zilliztech/knowhere"
IMPLEMENTATIONS = ("hanns", "zilliz_knowhere")
SUPPORT_STATUSES = ("supported", "unsupported", "non_comparable", "failed")
RECALL_BANDS = ("same_parameter", "near_equal_recall")
IVFPQ_VERDICT_STATUSES = (
    "win",
    "not_yet_win",
    "blocked_official_normalization",
    "non_comparable",
    "failed",
)
IVFPQ_REQUIRED_CHECKS = (
    "official_normalized",
    "same_top_k",
    "same_or_higher_recall_band",
    "exact_recall_within_tolerance",
    "throughput_units_comparable",
    "thread_policy_satisfied",
    "hanns_qps_vps_gt_official",
)
HNSW_VERDICT_STATUSES = ("win", "not_yet_win", "non_comparable", "failed")
HNSW_REQUIRED_CHECKS = (
    "same_top_k",
    "same_or_higher_recall",
    "throughput_units_comparable",
    "thread_policy_satisfied",
    "hanns_qps_vps_gt_official",
    "hanns_build_s_lt_official",
)

HNSW_SQ_VERDICT_STATUSES = ("win", "not_yet_win", "non_comparable", "failed")
HNSW_SQ_REQUIRED_CHECKS = (
    "same_top_k",
    "same_or_higher_recall_with_tolerance",
    "throughput_units_comparable",
    "thread_policy_satisfied",
    "hanns_qps_gt_official",
    "hanns_build_s_lt_official",
    "official_fp32_row_bound",
    "official_full_test_failed_after_fp32",
)
HNSW_PQ_BLOCKER_STATUSES = ("blocked_no_comparable_hanns_refine", "failed")
HNSW_PQ_BLOCKER_REQUIRED_CHECKS = (
    "official_fp32_rows_available",
    "official_full_test_failed_after_fp32",
    "official_qps_runner_missing",
    "hanns_raw_refine_unavailable",
    "leadership_claim_blocked",
)
IVFSQ8_VERDICT_STATUSES = ("win", "not_yet_win", "non_comparable", "failed")
IVFSQ8_REQUIRED_CHECKS = (
    "same_top_k",
    "same_or_higher_recall",
    "throughput_units_comparable",
    "thread_policy_satisfied",
    "hanns_qps_vps_gt_official",
    "hanns_build_s_lt_official",
)
DISKANN_AISAQ_VERDICT_STATUSES = (
    "non_comparable",
    "numeric_evidence_non_comparable",
    "failed",
)
DISKANN_AISAQ_REQUIRED_CHECKS = (
    "same_top_k",
    "aligned_metric_available",
    "native_comparable",
    "leadership_claim_blocked",
)
IVFUSQ_VERDICT_STATUSES = ("search_win", "not_yet_win", "non_comparable", "failed")
IVFUSQ_REQUIRED_CHECKS = (
    "same_top_k",
    "same_or_higher_recall_with_tolerance",
    "throughput_units_comparable",
    "hanns_qps_gt_official",
    "official_runner_non_qps",
)
REQUIRED_FAMILIES = (
    "HNSW",
    "HNSW-SQ",
    "HNSW-PQ",
    "DISKANN",
    "DISKANN-USQ/RabitQ",
    "IVF-PQ",
    "IVF-SQ",
    "IVF-USQ/RabitQ",
)


class ValidationError(Exception):
    """Validation failed with one or more human-readable errors."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def load_json(path: str | pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def write_json(path: str | pathlib.Path, payload: Any) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_time(value: str, *, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty timestamp string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include timezone")
    return parsed.astimezone(dt.timezone.utc)


def canonicalize_repo_url(url: str) -> str:
    """Return a stable URL identity for supported GitHub URL forms."""

    if not isinstance(url, str):
        return ""
    value = url.strip()
    if value.endswith(".git"):
        value = value[:-4]

    ssh_match = re.fullmatch(r"git@github\.com:([^/]+)/(.+)", value)
    if ssh_match:
        owner, repo = ssh_match.groups()
        return f"https://github.com/{owner}/{repo}".lower()

    ssh_url_match = re.fullmatch(r"ssh://git@github\.com/([^/]+)/(.+)", value)
    if ssh_url_match:
        owner, repo = ssh_url_match.groups()
        return f"https://github.com/{owner}/{repo}".lower()

    https_match = re.fullmatch(r"https://github\.com/([^/]+)/(.+)", value)
    if https_match:
        owner, repo = https_match.groups()
        return f"https://github.com/{owner}/{repo}".lower()

    return value.lower()


def is_official_knowhere_url(url: str) -> bool:
    return canonicalize_repo_url(url) == OFFICIAL_KNOWHERE_URL


def default_matrix() -> dict[str, Any]:
    families = [
        {
            "user_family": family,
            "matrix_status": "to_probe",
            "hanns": {
                "implementation": "hanns",
                "implementation_name": None,
                "mapping_notes": "to be discovered on HannsDB-x86",
            },
            "zilliz_knowhere": {
                "implementation": "zilliz_knowhere",
                "implementation_name": None,
                "mapping_notes": "to be discovered from zilliztech/knowhere on HannsDB-x86",
            },
        }
        for family in REQUIRED_FAMILIES
    ]
    return {
        "plan_id": PLAN_ID,
        "authority_surface": AUTHORITY_SURFACE,
        "official_knowhere_url": OFFICIAL_KNOWHERE_URL,
        "required_families": list(REQUIRED_FAMILIES),
        "implementations": list(IMPLEMENTATIONS),
        "families": families,
        "metrics": ["qps", "recall_at_k", "build_seconds", "latency_ms"],
        "stages": [
            {
                "stage": 0,
                "name": "capability_smoke",
                "verdict_eligible": False,
            },
            {
                "stage": 1,
                "name": "sift1m_final_matrix",
                "dataset": "sift-128-euclidean",
                "verdict_eligible": True,
            },
        ],
    }


def collect_matrix_errors(matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if matrix.get("plan_id") != PLAN_ID:
        errors.append(f"matrix.plan_id must be {PLAN_ID!r}")
    if matrix.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"matrix.authority_surface must be {AUTHORITY_SURFACE!r}")
    if not is_official_knowhere_url(matrix.get("official_knowhere_url", "")):
        errors.append("matrix.official_knowhere_url must be zilliztech/knowhere")

    required = set(REQUIRED_FAMILIES)
    declared = set(matrix.get("required_families", []))
    if declared != required:
        errors.append(
            "matrix.required_families must exactly match "
            f"{sorted(required)}; got {sorted(declared)}"
        )

    families = matrix.get("families", [])
    if not isinstance(families, list):
        errors.append("matrix.families must be a list")
        families = []
    seen = {entry.get("user_family") for entry in families if isinstance(entry, dict)}
    if seen != required:
        errors.append(
            f"matrix.families must contain all required families; got {sorted(seen)}"
        )
    for entry in families:
        if not isinstance(entry, dict):
            errors.append("matrix.families entries must be objects")
            continue
        for impl in IMPLEMENTATIONS:
            if impl not in entry:
                errors.append(f"{entry.get('user_family')}: missing {impl} mapping")
    return errors


def validate_matrix(matrix: dict[str, Any]) -> None:
    errors = collect_matrix_errors(matrix)
    if errors:
        raise ValidationError(errors)


def validate_capability_rows(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    required_pairs = {
        (family, impl) for family in REQUIRED_FAMILIES for impl in IMPLEMENTATIONS
    }
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        family = row.get("index", {}).get("user_family")
        impl = row.get("implementation")
        status = row.get("index", {}).get("support_status")
        if family not in REQUIRED_FAMILIES:
            errors.append(f"row[{index}]: unexpected family {family!r}")
        if impl not in IMPLEMENTATIONS:
            errors.append(f"row[{index}]: unexpected implementation {impl!r}")
        if status not in SUPPORT_STATUSES:
            errors.append(
                f"row[{index}]: support_status must be one of {SUPPORT_STATUSES}"
            )
        if family in REQUIRED_FAMILIES and impl in IMPLEMENTATIONS:
            seen_pairs.add((family, impl))
    missing = required_pairs - seen_pairs
    if missing:
        errors.append(f"missing capability rows: {sorted(missing)}")
    if errors:
        raise ValidationError(errors)


def _require_path(
    row: dict[str, Any], path: tuple[str, ...], errors: list[str], row_label: str
) -> Any:
    current: Any = row
    for part in path:
        if not isinstance(current, dict) or part not in current:
            errors.append(f"{row_label}: missing {'.'.join(path)}")
            return None
        current = current[part]
    return current


def _manifest_errors(
    proof: dict[str, Any], manifest: dict[str, Any] | None, row_label: str
) -> list[str]:
    if manifest is None:
        return [f"{row_label}: final validation requires authority manifest"]
    expected = manifest.get("runtime_proof", manifest)
    errors: list[str] = []
    required_fields = (
        "ssh_alias_or_host",
        "hostname",
        "uname",
        "lscpu_hash",
        "remote_log_root",
    )
    for field in required_fields:
        if not expected.get(field):
            errors.append(f"{row_label}: authority manifest missing {field}")
    allowed_wrappers = expected.get("allowed_wrapper_scripts")
    if not isinstance(allowed_wrappers, list) or not allowed_wrappers:
        errors.append(f"{row_label}: authority manifest missing allowed_wrapper_scripts")
    for field in ("ssh_alias_or_host", "hostname", "uname", "lscpu_hash"):
        if expected.get(field) and proof.get(field) != expected.get(field):
            errors.append(f"{row_label}: runtime_proof.{field} does not match manifest")
    log_root = expected.get("remote_log_root")
    if log_root:
        root_norm = posixpath.normpath(str(log_root))
        path_norm = posixpath.normpath(str(proof.get("remote_log_path", "")))
        try:
            if posixpath.commonpath([root_norm, path_norm]) != root_norm:
                errors.append(
                    f"{row_label}: runtime_proof.remote_log_path is outside manifest log root"
                )
        except ValueError:
            errors.append(
                f"{row_label}: runtime_proof.remote_log_path is outside manifest log root"
            )
    if allowed_wrappers and proof.get("wrapper_script") not in allowed_wrappers:
        errors.append(f"{row_label}: runtime_proof.wrapper_script is not allowed")
    return errors


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def ivfpq_recall_band_floor(recall: float) -> float:
    """Return the normative IVF-PQ recall-band floor.

    The approved plan defines the band as floor(recall * 100) / 100.  Rounding
    the result to two decimals avoids binary-float display artifacts while
    preserving the bucket semantics.
    """

    if not _is_number(recall) or recall < 0.0 or recall > 1.0:
        raise ValueError("recall must be a number in [0, 1]")
    return round(int((float(recall) * 100.0) + 1e-9) / 100.0, 2)


def _require_number(
    obj: dict[str, Any], field: str, errors: list[str], label: str
) -> float | None:
    value = obj.get(field)
    if not _is_number(value):
        errors.append(f"{label}: {field} must be numeric")
        return None
    return float(value)


def _require_positive_int(
    obj: dict[str, Any], field: str, errors: list[str], label: str
) -> int | None:
    value = obj.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{label}: {field} must be positive integer")
        return None
    return value



def _read_existing_text_path(
    value: Any, errors: list[str], label: str
) -> tuple[pathlib.Path | None, str | None]:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} is required")
        return None, None
    path = pathlib.Path(value)
    if not path.exists() or not path.is_file():
        errors.append(f"{label} must point to an existing local file: {value}")
        return path, None
    try:
        return path, path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - filesystem edge case
        errors.append(f"{label} is not readable: {exc}")
        return path, None


def _numbers_close(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    return _is_number(left) and _is_number(right) and abs(float(left) - float(right)) <= tolerance


def _validate_ivfpq_archived_evidence(
    verdict: dict[str, Any],
    official: dict[str, Any],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("nprobe") == hanns.get("nprobe")
                        and row.get("top_k") == hanns.get("top_k")
                        and row.get("threads") == hanns.get("threads")
                    ):
                        matched = row
                        break
                if matched is None:
                    suffix = " and search_surface" if "search_surface" in hanns else ""
                    errors.append(
                        "hanns aligned artifact has no row matching verdict "
                        f"candidate{suffix}"
                    )
                else:
                    comparisons = (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("train_s", "train_s"),
                        ("add_s", "add_s"),
                    )
                    for row_field, verdict_field in comparisons:
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")
    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=ok" not in official_status:
        errors.append("archived official status must contain status=ok")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"nprobe={hanns.get('nprobe')}",
            f"R@100={float(hanns.get('recall_at_100')):.4f}" if _is_number(hanns.get("recall_at_100")) else None,
            f"qps={float(hanns.get('qps_or_vps')):.3f}" if _is_number(hanns.get("qps_or_vps")) else None,
            f"build={float(hanns.get('build_s')):.3f}" if _is_number(hanns.get("build_s")) else None,
            "IVFPQ_EXPECT_THREADS=8" if hanns.get("threads") == 8 else None,
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        expected_tokens = [
            "sift-128-euclidean_IVF_PQ_1024_32_fp16.index",
            f"Build index IVF_PQ time: {float(official.get('build_s')):.3f}s"
            if _is_number(official.get("build_s"))
            else None,
            f"nprobe= {int(official.get('nprobe')):3d}, k={official.get('top_k')}, R@={float(official.get('recall_at_100')):.4f}"
            if isinstance(official.get("nprobe"), int)
            and isinstance(official.get("top_k"), int)
            and _is_number(official.get("recall_at_100"))
            else None,
            f"thread_num =  {int(official.get('threads'))}"
            if isinstance(official.get("threads"), int)
            else None,
            f"VPS = {float(official.get('qps_or_vps')):.3f}"
            if _is_number(official.get("qps_or_vps"))
            else None,
        ]
        for token in filter(None, expected_tokens):
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")


def _validate_hnsw_archived_evidence(
    verdict: dict[str, Any],
    official: dict[str, Any],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("m") == hanns.get("m")
                        and row.get("ef_construction") == hanns.get("ef_construction")
                        and row.get("ef") == hanns.get("ef")
                        and row.get("top_k") == hanns.get("top_k")
                        and row.get("threads") == hanns.get("threads")
                    ):
                        matched = row
                        break
                if matched is None:
                    suffix = " and search_surface" if "search_surface" in hanns else ""
                    errors.append(
                        "hanns aligned artifact has no row matching verdict "
                        f"candidate{suffix}"
                    )
                else:
                    comparisons = (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("train_s", "train_s"),
                        ("add_s", "add_s"),
                        ("build_threads", "build_threads"),
                    )
                    for row_field, verdict_field in comparisons:
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")
    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=ok" not in official_status:
        errors.append("archived official status must contain status=ok")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"HNSW_M={hanns.get('m')}",
            f"ef_construction={hanns.get('ef_construction')}",
            f"HNSW_BUILD_THREADS={hanns.get('build_threads')}",
            f"HNSW aligned ef={hanns.get('ef')}:",
            f"R@100={float(hanns.get('recall_at_100')):.4f}"
            if _is_number(hanns.get("recall_at_100"))
            else None,
            f"qps={float(hanns.get('qps_or_vps')):.3f}"
            if _is_number(hanns.get("qps_or_vps"))
            else None,
            f"build={float(hanns.get('build_s')):.3f}s"
            if _is_number(hanns.get("build_s"))
            else None,
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        expected_tokens = [
            f"HNSW({official.get('variant')})" if official.get("variant") else None,
            f"Build index HNSW time: {float(official.get('build_s')):.3f}s"
            if _is_number(official.get("build_s"))
            else None,
            f"ef={int(official.get('ef'))}, k={official.get('top_k')}, R@={float(official.get('recall_at_100')):.4f}"
            if isinstance(official.get("ef"), int)
            and isinstance(official.get("top_k"), int)
            and _is_number(official.get("recall_at_100"))
            else None,
            f"thread_num =  {int(official.get('threads'))}"
            if isinstance(official.get("threads"), int)
            else None,
            f"VPS = {float(official.get('qps_or_vps')):.3f}"
            if _is_number(official.get("qps_or_vps"))
            else None,
        ]
        for token in filter(None, expected_tokens):
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")


def _validate_ivfsq8_archived_evidence(
    verdict: dict[str, Any],
    official: dict[str, Any],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("nprobe") == hanns.get("nprobe")
                        and row.get("top_k") == hanns.get("top_k")
                        and row.get("threads") == hanns.get("threads")
                    ):
                        matched = row
                        break
                if matched is None:
                    errors.append("hanns aligned artifact has no row matching verdict candidate")
                else:
                    comparisons = (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("train_s", "train_s"),
                        ("add_s", "add_s"),
                    )
                    for row_field, verdict_field in comparisons:
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")
    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=ok" not in official_status:
        errors.append("archived official status must contain status=ok")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"IVFSQ8_EXPECT_THREADS={hanns.get('threads')}",
            f"IVF-SQ8 aligned nprobe={hanns.get('nprobe')}:",
            f"R@100={float(hanns.get('recall_at_100')):.4f}"
            if _is_number(hanns.get("recall_at_100"))
            else None,
            f"qps={float(hanns.get('qps_or_vps')):.3f}"
            if _is_number(hanns.get("qps_or_vps"))
            else None,
            f"build={float(hanns.get('build_s')):.3f}s"
            if _is_number(hanns.get("build_s"))
            else None,
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        expected_tokens = [
            f"IVF_SQ8({official.get('variant')})" if official.get("variant") else None,
            f"Build index IVF_SQ8 time: {float(official.get('build_s')):.3f}s"
            if _is_number(official.get("build_s"))
            else None,
            f"nprobe= {int(official.get('nprobe')):3d}, k={official.get('top_k')}, R@={float(official.get('recall_at_100')):.4f}"
            if isinstance(official.get("nprobe"), int)
            and isinstance(official.get("top_k"), int)
            and _is_number(official.get("recall_at_100"))
            else None,
            f"thread_num =  {int(official.get('threads'))}"
            if isinstance(official.get("threads"), int)
            else None,
            f"VPS = {float(official.get('qps_or_vps')):.3f}"
            if _is_number(official.get("qps_or_vps"))
            else None,
        ]
        for token in filter(None, expected_tokens):
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")


def collect_ivfsq8_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate an IVF-SQ8 milestone verdict object against archived authority evidence."""

    errors: list[str] = []
    if verdict.get("artifact_type") != "ivfsq8_milestone_verdict":
        errors.append("artifact_type must be 'ivfsq8_milestone_verdict'")

    status = verdict.get("status")
    if status not in IVFSQ8_VERDICT_STATUSES:
        errors.append(f"status must be one of {IVFSQ8_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "ivfsq8") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100 for this official lane")
    tolerance = _require_number(contract, "recall_tolerance", errors, "comparison_contract")
    if tolerance is not None and tolerance < 0:
        errors.append("comparison_contract.recall_tolerance must be non-negative")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")

    official = _require_path(verdict, ("official_target",), errors, "ivfsq8") or {}
    hanns = _require_path(verdict, ("hanns_candidate",), errors, "ivfsq8") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "ivfsq8") or {}

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    for label, obj in (("official_target", official), ("hanns_candidate", hanns)):
        for field in ("commit", "ref", "log_path"):
            if not obj.get(field):
                errors.append(f"{label}.{field} is required")
        for field in ("top_k", "nlist", "nprobe", "threads"):
            _require_positive_int(obj, field, errors, label)
        for field in ("recall_at_100", "build_s", "qps_or_vps"):
            _require_number(obj, field, errors, label)
    for field in ("train_s", "add_s"):
        _require_number(hanns, field, errors, "hanns_candidate")
    if not official.get("variant"):
        errors.append("official_target.variant is required")
    if not official.get("recall_goal"):
        errors.append("official_target.recall_goal is required")

    for check in IVFSQ8_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    same_top_k = official.get("top_k") == hanns.get("top_k") == contract.get("top_k")
    if checks.get("same_top_k") != same_top_k:
        errors.append("verdict_checks.same_top_k does not match top_k fields")

    if _is_number(official.get("recall_at_100")) and _is_number(hanns.get("recall_at_100")):
        recall_ok = float(hanns["recall_at_100"]) + float(tolerance or 0.0) >= float(
            official["recall_at_100"]
        )
        if checks.get("same_or_higher_recall") != recall_ok:
            errors.append("verdict_checks.same_or_higher_recall does not match recall values")

    units_comparable = contract.get("vps_qps_mapping") == "one_vector_per_query"
    if checks.get("throughput_units_comparable") != units_comparable:
        errors.append(
            "verdict_checks.throughput_units_comparable does not match vps_qps_mapping"
        )
    threads_match = (
        contract.get("thread_policy") == "match_or_explicitly_normalized"
        and isinstance(official.get("threads"), int)
        and official.get("threads") == hanns.get("threads")
    )
    if checks.get("thread_policy_satisfied") != threads_match:
        errors.append("verdict_checks.thread_policy_satisfied does not match thread fields")

    if _is_number(official.get("qps_or_vps")) and _is_number(hanns.get("qps_or_vps")):
        tput_ok = float(hanns["qps_or_vps"]) > float(official["qps_or_vps"])
        if checks.get("hanns_qps_vps_gt_official") != tput_ok:
            errors.append(
                "verdict_checks.hanns_qps_vps_gt_official does not match throughput values"
            )
    if _is_number(official.get("build_s")) and _is_number(hanns.get("build_s")):
        build_ok = float(hanns["build_s"]) < float(official["build_s"])
        if checks.get("hanns_build_s_lt_official") != build_ok:
            errors.append("verdict_checks.hanns_build_s_lt_official does not match build_s values")

    if contract.get("archived_evidence_required") is True:
        _validate_ivfsq8_archived_evidence(verdict, official, hanns, errors)

    all_checks_true = all(checks.get(check) is True for check in IVFSQ8_REQUIRED_CHECKS)
    if status == "win" and not all_checks_true:
        errors.append("status=win requires all verdict checks to be true")
    if status != "non_comparable" and (
        checks.get("throughput_units_comparable") is False
        or checks.get("thread_policy_satisfied") is False
    ):
        errors.append("unit or thread mismatch requires status=non_comparable")

    return errors


def validate_ivfsq8_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_ivfsq8_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)


def _diskann_aisaq_log_name(variant: Any) -> str | None:
    if variant == "DISKANN":
        return "DISKANN"
    if variant in ("AISAQ_P", "AISAQ_S"):
        return "AISAQ"
    return None


def _validate_diskann_aisaq_official_log(
    target: dict[str, Any],
    log_text: str,
    errors: list[str],
    label: str,
) -> None:
    build_match = re.search(r"Build index: done \(([0-9.]+) ms\)", log_text)
    if build_match and _is_number(target.get("build_s")):
        actual_build_s = float(build_match.group(1)) / 1000.0
        if abs(actual_build_s - float(target["build_s"])) > 0.001:
            errors.append(f"{label}: build_s does not match archived official log")
    elif _is_number(target.get("build_s")):
        errors.append(f"{label}: archived official log missing build_s token")

    log_name = _diskann_aisaq_log_name(target.get("variant"))
    if log_name is None:
        errors.append(f"{label}: unsupported variant for official log binding")
        return
    if not isinstance(target.get("search_list_size"), int):
        errors.append(f"{label}: search_list_size must be integer for log binding")
        return
    if not isinstance(target.get("top_k"), int):
        errors.append(f"{label}: top_k must be integer for log binding")
        return
    if not _is_number(target.get("recall_at_100")):
        errors.append(f"{label}: recall_at_100 must be numeric for log binding")
        return

    row_token = (
        f"sift-128-euclidean | {log_name}(FP32) | "
        f"search_list_size={target['search_list_size']}, "
        f"k={target['top_k']}, R@={float(target['recall_at_100']):.4f}"
    )
    row_pos = log_text.find(row_token)
    if row_pos < 0:
        errors.append(f"{label}: archived official log missing row token {row_token!r}")
        return

    row_window = log_text[row_pos : row_pos + 2000]
    thread_token = (
        f"thread_num =  {int(target['threads'])}"
        if isinstance(target.get("threads"), int)
        else None
    )
    if thread_token is None:
        errors.append(f"{label}: threads must be integer for log binding")
    if not _is_number(target.get("qps_or_vps")):
        errors.append(f"{label}: qps_or_vps must be numeric for log binding")
        return
    vps_token = f"VPS = {float(target['qps_or_vps']):.3f}"
    if thread_token is not None:
        matched_line = any(
            thread_token in line and vps_token in line for line in row_window.splitlines()
        )
        if not matched_line:
            errors.append(
                f"{label}: archived official log missing coupled thread/VPS tokens "
                f"{thread_token!r} and {vps_token!r}"
            )


def _official_artifact_path(
    target: dict[str, Any],
    field: str,
    evidence_values: list[Any],
) -> Any:
    value = target.get(field)
    if value:
        return value
    remote_basename = pathlib.PurePosixPath(str(target.get("log_path", ""))).name
    for candidate in evidence_values:
        if not isinstance(candidate, str):
            continue
        candidate_name = pathlib.Path(candidate).name
        if field.endswith("status") and candidate_name == remote_basename.replace(
            ".log", ".status"
        ):
            return candidate
        if not field.endswith("status") and candidate_name == remote_basename:
            return candidate
    return ""


def _validate_diskann_aisaq_official_evidence(
    official_targets: list[dict[str, Any]],
    evidence: dict[str, Any],
    errors: list[str],
) -> None:
    official_logs = evidence.get("archived_official_logs")
    if not isinstance(official_logs, list) or not official_logs:
        errors.append("evidence.archived_official_logs must be a non-empty list")
        official_logs = []
    official_statuses = evidence.get("archived_official_statuses")
    if not isinstance(official_statuses, list) or not official_statuses:
        errors.append("evidence.archived_official_statuses must be a non-empty list")
        official_statuses = []

    for index, target in enumerate(official_targets):
        label = f"official_targets[{index}]"
        log_value = _official_artifact_path(target, "archived_log", official_logs)
        _, log_text = _read_existing_text_path(log_value, errors, f"{label}.archived_log")
        if log_text is not None:
            _validate_diskann_aisaq_official_log(target, log_text, errors, label)

        status_value = _official_artifact_path(
            target, "archived_status", official_statuses
        )
        _, status_text = _read_existing_text_path(
            status_value, errors, f"{label}.archived_status"
        )
        if status_text is not None and "status=ok" not in status_text:
            errors.append(f"{label}: archived official status must contain status=ok")


def _validate_diskann_aisaq_archived_evidence(
    verdict: dict[str, Any],
    official_targets: list[dict[str, Any]],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("artifact_type") != "diskann_aisaq_aligned_hanns_rows":
                errors.append(
                    "hanns aligned artifact artifact_type must be "
                    "'diskann_aisaq_aligned_hanns_rows'"
                )
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            if artifact.get("native_comparable") is not False:
                errors.append("hanns aligned artifact must be native_comparable=false")
            if artifact.get("leadership_claim_allowed") is not False:
                errors.append("hanns aligned artifact must block leadership claims")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if row.get("config_name") != hanns.get("config_name"):
                        continue
                    if row.get("top_k") != hanns.get("top_k"):
                        continue
                    if row.get("threads") != hanns.get("threads"):
                        continue
                    if (
                        "search_surface" in hanns
                        and row.get("search_surface") != hanns.get("search_surface")
                    ):
                        continue
                    if (
                        "disk_pq_dims" in hanns
                        and row.get("disk_pq_dims") != hanns.get("disk_pq_dims")
                    ):
                        continue
                    if (
                        "pq_candidate_expand_pct" in hanns
                        and row.get("pq_candidate_expand_pct")
                        != hanns.get("pq_candidate_expand_pct")
                    ):
                        continue
                    if (
                        "rerank_expand_pct" in hanns
                        and row.get("rerank_expand_pct") != hanns.get("rerank_expand_pct")
                    ):
                        continue
                    matched = row
                    break
                if matched is None:
                    suffix = ""
                    for field in (
                        "search_surface",
                        "disk_pq_dims",
                        "pq_candidate_expand_pct",
                        "rerank_expand_pct",
                    ):
                        if field in hanns:
                            suffix += f" and {field}"
                    errors.append(
                        "hanns aligned artifact has no row matching verdict "
                        f"candidate{suffix}"
                    )
                else:
                    if matched.get("native_comparable") is not False:
                        errors.append("matched Hanns row must be native_comparable=false")
                    if "search_surface" in hanns:
                        expected_surface = hanns.get("search_surface")
                        if matched.get("search_surface") != expected_surface:
                            errors.append(
                                "hanns aligned artifact search_surface does not match "
                                "hanns_candidate.search_surface"
                            )
                        if artifact.get("search_surface") != expected_surface:
                            errors.append(
                                "hanns aligned artifact top-level search_surface does not match "
                                "hanns_candidate.search_surface"
                            )
                    if "disk_pq_dims" in hanns:
                        if matched.get("disk_pq_dims") != hanns.get("disk_pq_dims"):
                            errors.append(
                                "hanns aligned artifact disk_pq_dims does not match "
                                "hanns_candidate.disk_pq_dims"
                            )
                        if artifact.get("disk_pq_dims") != hanns.get("disk_pq_dims"):
                            errors.append(
                                "hanns aligned artifact top-level disk_pq_dims does not match "
                                "hanns_candidate.disk_pq_dims"
                            )
                    for field in ("pq_candidate_expand_pct", "rerank_expand_pct"):
                        if field in hanns:
                            if matched.get(field) != hanns.get(field):
                                errors.append(
                                    f"hanns aligned artifact {field} does not match "
                                    f"hanns_candidate.{field}"
                                )
                            if artifact.get(field) != hanns.get(field):
                                errors.append(
                                    f"hanns aligned artifact top-level {field} does not match "
                                    f"hanns_candidate.{field}"
                                )
                    if hanns.get("uses_mmap_backed_pages") is True:
                        scope_audit = matched.get("scope_audit")
                        if not isinstance(scope_audit, dict):
                            errors.append("matched Hanns row scope_audit must be an object")
                        elif scope_audit.get("uses_mmap_backed_pages") is not True:
                            errors.append(
                                "matched Hanns row must prove uses_mmap_backed_pages=true"
                            )
                    if hanns.get("has_page_cache") is True:
                        scope_audit = matched.get("scope_audit")
                        if not isinstance(scope_audit, dict):
                            errors.append("matched Hanns row scope_audit must be an object")
                        elif scope_audit.get("has_page_cache") is not True:
                            errors.append(
                                "matched Hanns row must prove has_page_cache=true"
                            )
                    for row_field, verdict_field in (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("search_list_size", "search_list_size"),
                        ("persist_s", "persist_s"),
                        ("load_s", "load_s"),
                    ):
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"AISAQ_EXPECT_THREADS={hanns.get('threads')}",
            f"search_list_size={hanns.get('search_list_size')}",
            f"R@100={float(hanns.get('recall_at_100')):.4f}"
            if _is_number(hanns.get("recall_at_100"))
            else None,
            "native_comparable=false",
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")
        config_token = f"config={hanns.get('config_name')}"
        legacy_config_token = f"AISAQ aligned config={hanns.get('config_name')}"
        if config_token not in hanns_log and legacy_config_token not in hanns_log:
            errors.append(
                f"archived Hanns log missing AISAQ config token {config_token!r}"
            )
        if "search_surface" in hanns:
            surface_token = f"surface={hanns.get('search_surface')}"
            if surface_token not in hanns_log:
                errors.append(f"archived Hanns log missing token {surface_token!r}")
        if "disk_pq_dims" in hanns:
            disk_pq_token = f"disk_pq_dims={hanns.get('disk_pq_dims')}"
            if disk_pq_token not in hanns_log:
                errors.append(f"archived Hanns log missing token {disk_pq_token!r}")
        for field in ("pq_candidate_expand_pct", "rerank_expand_pct"):
            if field in hanns:
                token = f"{field}={hanns.get(field)}"
                if token not in hanns_log:
                    errors.append(f"archived Hanns log missing token {token!r}")

    _validate_diskann_aisaq_official_evidence(official_targets, evidence, errors)


def collect_diskann_aisaq_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate a constrained DiskANN/AISAQ verdict.

    This validator is intentionally fail-closed: current Hanns AISAQ evidence can
    be archived and compared numerically, but it must remain non-comparable until
    the implementation exposes a native-comparable SSD DiskANN/AISAQ pipeline.
    """

    errors: list[str] = []
    if verdict.get("artifact_type") != "diskann_aisaq_constrained_verdict":
        errors.append("artifact_type must be 'diskann_aisaq_constrained_verdict'")

    status = verdict.get("status")
    if status not in DISKANN_AISAQ_VERDICT_STATUSES:
        errors.append(f"status must be one of {DISKANN_AISAQ_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if verdict.get("family") != "DiskANN/AISAQ":
        errors.append("family must be 'DiskANN/AISAQ'")
    if verdict.get("leadership_claim_allowed") is not False:
        errors.append("leadership_claim_allowed must be false")
    if verdict.get("native_comparable") is not False:
        errors.append("native_comparable must be false for this constrained verdict")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "diskann") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")

    official_targets = verdict.get("official_targets")
    if not isinstance(official_targets, list) or not official_targets:
        errors.append("official_targets must be a non-empty list")
        official_targets = []
    for index, target in enumerate(official_targets):
        if not isinstance(target, dict):
            errors.append(f"official_targets[{index}] must be an object")
            continue
        if not is_official_knowhere_url(target.get("repo_url", "")):
            errors.append(f"official_targets[{index}].repo_url must be zilliztech/knowhere")
        for field in ("variant", "log_path", "top_k"):
            if not target.get(field):
                errors.append(f"official_targets[{index}].{field} is required")
        for field in ("top_k", "search_list_size", "threads"):
            _require_positive_int(target, field, errors, f"official_targets[{index}]")
        for field in ("recall_at_100", "build_s", "qps_or_vps"):
            _require_number(target, field, errors, f"official_targets[{index}]")

    hanns = _require_path(verdict, ("hanns_candidate",), errors, "diskann") or {}
    for field in (
        "commit",
        "ref",
        "log_path",
        "config_name",
        "comparability_reason",
    ):
        if not hanns.get(field):
            errors.append(f"hanns_candidate.{field} is required")
    for field in (
        "top_k",
        "max_degree",
        "search_list_size",
        "beamwidth",
        "num_entry_points",
        "threads",
    ):
        _require_positive_int(hanns, field, errors, "hanns_candidate")
    for field in ("recall_at_100", "build_s", "qps_or_vps"):
        _require_number(hanns, field, errors, "hanns_candidate")
    if hanns.get("native_comparable") is not False:
        errors.append("hanns_candidate.native_comparable must be false")
    if hanns.get("search_surface") is not None:
        if hanns.get("search_surface") not in ("memory", "mmap", "page_cache"):
            errors.append("hanns_candidate.search_surface must be memory, mmap, or page_cache")
        if hanns.get("search_surface") == "mmap" and hanns.get("uses_mmap_backed_pages") is not True:
            errors.append(
                "hanns_candidate.search_surface=mmap requires uses_mmap_backed_pages=true"
            )
        if hanns.get("search_surface") == "page_cache":
            if hanns.get("has_page_cache") is not True:
                errors.append(
                    "hanns_candidate.search_surface=page_cache requires has_page_cache=true"
                )
            _require_positive_int(hanns, "disk_pq_dims", errors, "hanns_candidate")
            for field in ("pq_candidate_expand_pct", "rerank_expand_pct"):
                if field in hanns:
                    _require_positive_int(hanns, field, errors, "hanns_candidate")

    checks = _require_path(verdict, ("verdict_checks",), errors, "diskann") or {}
    for check in DISKANN_AISAQ_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    same_top_k = hanns.get("top_k") == contract.get("top_k") == 100
    if checks.get("same_top_k") != same_top_k:
        errors.append("verdict_checks.same_top_k does not match top_k fields")
    aligned_metric = _is_number(hanns.get("recall_at_100"))
    if checks.get("aligned_metric_available") != aligned_metric:
        errors.append(
            "verdict_checks.aligned_metric_available does not match hanns recall"
        )
    if checks.get("native_comparable") is not False:
        errors.append("verdict_checks.native_comparable must be false")
    if checks.get("leadership_claim_blocked") is not True:
        errors.append("verdict_checks.leadership_claim_blocked must be true")
    if status != "non_comparable" and checks.get("native_comparable") is False:
        errors.append("native_comparable=false requires status=non_comparable")

    if contract.get("archived_evidence_required") is True:
        _validate_diskann_aisaq_archived_evidence(
            verdict, official_targets, hanns, errors
        )

    return errors


def validate_diskann_aisaq_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_diskann_aisaq_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)


def _validate_ivfusq_archived_evidence(
    verdict: dict[str, Any],
    official: dict[str, Any],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("artifact_type") != "ivfusq_aligned_hanns_rows":
                errors.append("hanns aligned artifact artifact_type must be ivfusq_aligned_hanns_rows")
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("bits_per_dim") == hanns.get("bits_per_dim")
                        and row.get("nprobe") == hanns.get("nprobe")
                        and row.get("top_k") == hanns.get("top_k")
                        and row.get("threads") == hanns.get("threads")
                    ):
                        matched = row
                        break
                if matched is None:
                    errors.append("hanns aligned artifact has no row matching verdict candidate")
                else:
                    for row_field, verdict_field in (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("train_s", "train_s"),
                        ("add_s", "add_s"),
                    ):
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")
    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=ok" not in official_status:
        errors.append("archived official status must contain status=ok")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"IVFUSQ_EXPECT_THREADS={hanns.get('threads')}",
            f"IVF-USQ aligned bits={hanns.get('bits_per_dim')} nprobe={hanns.get('nprobe')}:",
            f"R@100={float(hanns.get('recall_at_100')):.4f}"
            if _is_number(hanns.get("recall_at_100"))
            else None,
            f"qps={float(hanns.get('qps_or_vps')):.3f}"
            if _is_number(hanns.get("qps_or_vps"))
            else None,
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        row_token = (
            f"nprobe = {int(official.get('nprobe')):4d}, nq = {int(official.get('nq'))}, "
            f"k = {int(official.get('top_k')):4d}, elapse = {float(official.get('elapsed_s')):6.3f}s, "
            f"R@ = {float(official.get('recall_at_100')):.4f}"
        )
        expected_tokens = [
            "Benchmark_float.TEST_IVF_RABITQ",
            f"IVF_RABITQ({official.get('variant')})",
            f"nlist={official.get('nlist')}",
            row_token,
            "PASSED",
        ]
        for token in expected_tokens:
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")

    _, qps_list = _read_existing_text_path(
        evidence.get("archived_qps_gtest_list"), errors, "evidence.archived_qps_gtest_list"
    )
    if qps_list is not None:
        if f"commit={official.get('commit')}" not in qps_list:
            errors.append("archived qps gtest list commit does not match official_target.commit")
        if "Benchmark_float_qps." not in qps_list:
            errors.append("archived qps gtest list must list Benchmark_float_qps tests")
        if "TEST_IVF_RABITQ" in qps_list or "IVF_RABITQ" in qps_list:
            errors.append("archived qps gtest list must not expose IVF_RABITQ")


def collect_ivfusq_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate a scoped IVF-USQ/RabitQ search-throughput verdict."""

    errors: list[str] = []
    if verdict.get("artifact_type") != "ivfusq_search_milestone_verdict":
        errors.append("artifact_type must be 'ivfusq_search_milestone_verdict'")

    status = verdict.get("status")
    if status not in IVFUSQ_VERDICT_STATUSES:
        errors.append(f"status must be one of {IVFUSQ_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "ivfusq") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100")
    tolerance = _require_number(contract, "recall_tolerance", errors, "comparison_contract")
    if tolerance is not None and tolerance < 0:
        errors.append("comparison_contract.recall_tolerance must be non-negative")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")

    official = _require_path(verdict, ("official_target",), errors, "ivfusq") or {}
    hanns = _require_path(verdict, ("hanns_candidate",), errors, "ivfusq") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "ivfusq") or {}

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    for label, obj in (("official_target", official), ("hanns_candidate", hanns)):
        for field in ("commit", "ref", "log_path"):
            if not obj.get(field):
                errors.append(f"{label}.{field} is required")
        for field in ("top_k", "nlist", "nprobe"):
            _require_positive_int(obj, field, errors, label)
        for field in ("recall_at_100", "qps_or_vps"):
            _require_number(obj, field, errors, label)
    for field in ("variant", "runner", "nq", "elapsed_s"):
        if not official.get(field):
            errors.append(f"official_target.{field} is required")
    _require_positive_int(official, "nq", errors, "official_target")
    _require_number(official, "elapsed_s", errors, "official_target")
    for field in ("bits_per_dim", "threads"):
        _require_positive_int(hanns, field, errors, "hanns_candidate")
    for field in ("build_s", "train_s", "add_s"):
        _require_number(hanns, field, errors, "hanns_candidate")

    for check in IVFUSQ_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    same_top_k = official.get("top_k") == hanns.get("top_k") == contract.get("top_k")
    if checks.get("same_top_k") != same_top_k:
        errors.append("verdict_checks.same_top_k does not match top_k fields")
    if _is_number(official.get("recall_at_100")) and _is_number(hanns.get("recall_at_100")):
        recall_ok = float(hanns["recall_at_100"]) + float(tolerance or 0.0) >= float(
            official["recall_at_100"]
        )
        if checks.get("same_or_higher_recall_with_tolerance") != recall_ok:
            errors.append("verdict_checks.same_or_higher_recall_with_tolerance mismatch")
    units_comparable = contract.get("vps_qps_mapping") == "one_vector_per_query"
    if checks.get("throughput_units_comparable") != units_comparable:
        errors.append("verdict_checks.throughput_units_comparable does not match mapping")
    if _is_number(official.get("qps_or_vps")) and _is_number(hanns.get("qps_or_vps")):
        tput_ok = float(hanns["qps_or_vps"]) > float(official["qps_or_vps"])
        if checks.get("hanns_qps_gt_official") != tput_ok:
            errors.append("verdict_checks.hanns_qps_gt_official does not match throughput")
    official_non_qps = official.get("runner") == "benchmark_float"
    if checks.get("official_runner_non_qps") != official_non_qps:
        errors.append("verdict_checks.official_runner_non_qps does not match runner")

    all_checks_true = all(checks.get(check) is True for check in IVFUSQ_REQUIRED_CHECKS)
    if status == "search_win" and not all_checks_true:
        errors.append("status=search_win requires all verdict checks to be true")
    if contract.get("archived_evidence_required") is True:
        _validate_ivfusq_archived_evidence(verdict, official, hanns, errors)

    return errors


def validate_ivfusq_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_ivfusq_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)


def _validate_hnsw_pq_blocker_archived_evidence(
    verdict: dict[str, Any], official: dict[str, Any], errors: list[str]
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=failed" not in official_status:
        errors.append("archived official status must contain status=failed")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        expected_tokens = [
            "Benchmark_float.TEST_HNSW_PQ",
            "HNSW_PQ(FP32)",
            f"Build index HNSW_PQ time: {float(official.get('build_s')):.3f}s"
            if _is_number(official.get("build_s"))
            else None,
            f"refine_k = {int(official.get('refine_k')):3d}, nq = {int(official.get('nq'))}, "
            f"k = {int(official.get('top_k')):4d}, elapse = {float(official.get('elapsed_s')):6.3f}s, "
            f"R@ = {float(official.get('recall_at_100')):.4f}",
            'C++ exception with description "bad optional access" thrown in the test body.',
            "[  FAILED  ] Benchmark_float.TEST_HNSW_PQ",
        ]
        for token in filter(None, expected_tokens):
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")

    _, qps_list = _read_existing_text_path(
        evidence.get("archived_qps_gtest_list"), errors, "evidence.archived_qps_gtest_list"
    )
    if qps_list is not None:
        if f"commit={official.get('commit')}" not in qps_list:
            errors.append("archived qps gtest list commit does not match official_target.commit")
        if "Benchmark_float_qps." not in qps_list:
            errors.append("archived qps gtest list must list Benchmark_float_qps tests")
        if "TEST_HNSW_PQ" in qps_list or "HNSW_PQ" in qps_list:
            errors.append("archived qps gtest list must not expose HNSW_PQ")

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_capability_status"),
        errors,
        "evidence.archived_hanns_capability_status",
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns capability status must contain status=ok")
    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_capability_log"),
        errors,
        "evidence.archived_hanns_capability_log",
    )
    if hanns_log is not None:
        expected_tokens = [
            "test_hnsw_pq_has_raw_data_is_false ... ok",
            "test_hnsw_pq_get_vector_by_ids_returns_stable_unsupported ... ok",
        ]
        for token in expected_tokens:
            if token not in hanns_log:
                errors.append(f"archived Hanns capability log missing token {token!r}")

    _, source_status = _read_existing_text_path(
        evidence.get("official_source_status"), errors, "evidence.official_source_status"
    )
    if source_status is not None and "status=ok" not in source_status:
        errors.append("official source status must contain status=ok")
    _, source_log = _read_existing_text_path(
        evidence.get("official_source_inspection"), errors, "evidence.official_source_inspection"
    )
    if source_log is not None:
        expected_tokens = [
            "provenance_kind=official_knowhere_source",
            "origin_url=https://github.com/zilliztech/knowhere.git",
            f"head_commit={official.get('commit')}",
            "official_source_ok=true",
        ]
        for token in expected_tokens:
            if token not in source_log:
                errors.append(f"official source inspection missing token {token!r}")


def collect_hnsw_pq_blocker_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate a fail-closed HNSW-PQ comparability blocker verdict."""

    errors: list[str] = []
    if verdict.get("artifact_type") != "hnsw_pq_blocker_verdict":
        errors.append("artifact_type must be 'hnsw_pq_blocker_verdict'")
    status = verdict.get("status")
    if status not in HNSW_PQ_BLOCKER_STATUSES:
        errors.append(f"status must be one of {HNSW_PQ_BLOCKER_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    official = _require_path(verdict, ("official_target",), errors, "hnsw_pq") or {}
    hanns = _require_path(verdict, ("hanns_capability",), errors, "hnsw_pq") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "hnsw_pq") or {}

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    if official.get("commit") and str(official.get("commit")) not in str(official.get("ref", "")):
        errors.append("official_target.ref must pin official_target.commit")
    for field in ("commit", "ref", "runner", "variant", "full_test_status"):
        if not official.get(field):
            errors.append(f"official_target.{field} is required")
    for field in ("top_k", "m", "ef_construction", "ef", "refine_k", "nq"):
        _require_positive_int(official, field, errors, "official_target")
    for field in ("elapsed_s", "recall_at_100", "build_s"):
        _require_number(official, field, errors, "official_target")

    if hanns.get("implementation") != "HnswPqIndex":
        errors.append("hanns_capability.implementation must be HnswPqIndex")
    if hanns.get("has_raw_data") is not False:
        errors.append("hanns_capability.has_raw_data must be false")
    if hanns.get("get_vector_by_ids") != "unsupported":
        errors.append("hanns_capability.get_vector_by_ids must be unsupported")
    reason = str(hanns.get("comparability_reason", ""))
    if "FLAT refine" not in reason:
        errors.append("hanns_capability.comparability_reason must mention FLAT refine")

    for check in HNSW_PQ_BLOCKER_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    expected = {
        "official_fp32_rows_available": official.get("variant") == "FP32",
        "official_full_test_failed_after_fp32": official.get("full_test_status")
        == "failed_after_fp32_rows",
        "hanns_raw_refine_unavailable": hanns.get("has_raw_data") is False
        and hanns.get("get_vector_by_ids") == "unsupported",
        "leadership_claim_blocked": status == "blocked_no_comparable_hanns_refine",
    }
    for check, value in expected.items():
        if checks.get(check) != value:
            errors.append(f"verdict_checks.{check} does not match verdict fields")
    if checks.get("official_qps_runner_missing") is not True:
        errors.append("verdict_checks.official_qps_runner_missing must be true")

    if verdict.get("evidence_required") is True:
        _validate_hnsw_pq_blocker_archived_evidence(verdict, official, errors)

    all_checks_true = all(checks.get(check) is True for check in HNSW_PQ_BLOCKER_REQUIRED_CHECKS)
    if status == "blocked_no_comparable_hanns_refine" and not all_checks_true:
        errors.append("blocked HNSW-PQ verdict requires all blocker checks to be true")
    return errors


def validate_hnsw_pq_blocker(verdict: dict[str, Any]) -> None:
    errors = collect_hnsw_pq_blocker_errors(verdict)
    if errors:
        raise ValidationError(errors)


def _validate_hnsw_sq_archived_evidence(
    verdict: dict[str, Any],
    official: dict[str, Any],
    hanns: dict[str, Any],
    errors: list[str],
) -> None:
    evidence = verdict.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object when archived evidence is required")
        return

    artifact_path, _ = _read_existing_text_path(
        evidence.get("hanns_aligned_artifact"), errors, "evidence.hanns_aligned_artifact"
    )
    if artifact_path is not None and artifact_path.exists():
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            errors.append(f"evidence.hanns_aligned_artifact is not valid JSON: {exc}")
            artifact = None
        if isinstance(artifact, dict):
            if artifact.get("artifact_type") != "hnsw_sq_aligned_hanns_rows":
                errors.append(
                    "hanns aligned artifact artifact_type must be hnsw_sq_aligned_hanns_rows"
                )
            if artifact.get("authority_surface") != AUTHORITY_SURFACE:
                errors.append("hanns aligned artifact authority_surface mismatch")
            rows = artifact.get("rows")
            if not isinstance(rows, list):
                errors.append("hanns aligned artifact rows must be a list")
            else:
                matched = None
                for row in rows:
                    if (
                        isinstance(row, dict)
                        and row.get("m") == hanns.get("m")
                        and row.get("ef_construction") == hanns.get("ef_construction")
                        and row.get("ef") == hanns.get("ef")
                        and row.get("top_k") == hanns.get("top_k")
                        and row.get("threads") == hanns.get("threads")
                        and row.get("sq_mode") == hanns.get("sq_mode")
                    ):
                        matched = row
                        break
                if matched is None:
                    errors.append(
                        "hanns aligned artifact has no HNSW-SQ row matching verdict candidate"
                    )
                else:
                    for row_field, verdict_field in (
                        ("recall_at_100", "recall_at_100"),
                        ("qps", "qps_or_vps"),
                        ("build_s", "build_s"),
                        ("train_s", "train_s"),
                        ("add_s", "add_s"),
                        ("build_threads", "build_threads"),
                    ):
                        if verdict_field in hanns and not _numbers_close(
                            matched.get(row_field), hanns.get(verdict_field), 1e-6
                        ):
                            errors.append(
                                "hanns aligned artifact "
                                f"{row_field} does not match hanns_candidate.{verdict_field}"
                            )

    _, hanns_status = _read_existing_text_path(
        evidence.get("archived_hanns_status"), errors, "evidence.archived_hanns_status"
    )
    if hanns_status is not None and "status=ok" not in hanns_status:
        errors.append("archived Hanns status must contain status=ok")
    _, official_status = _read_existing_text_path(
        evidence.get("archived_official_status"), errors, "evidence.archived_official_status"
    )
    if official_status is not None and "status=failed" not in official_status:
        errors.append(
            "archived official status must contain status=failed for current HNSW-SQ blocker"
        )

    _, source_status = _read_existing_text_path(
        evidence.get("official_source_status"), errors, "evidence.official_source_status"
    )
    if source_status is not None and "status=ok" not in source_status:
        errors.append("official source status must contain status=ok")
    _, source_log = _read_existing_text_path(
        evidence.get("official_source_inspection"), errors, "evidence.official_source_inspection"
    )
    if source_log is not None:
        expected_tokens = [
            "provenance_kind=official_knowhere_source",
            "origin_url=https://github.com/zilliztech/knowhere.git",
            f"head_commit={official.get('commit')}",
            "official_source_ok=true",
        ]
        for token in expected_tokens:
            if token not in source_log:
                errors.append(f"official source inspection missing token {token!r}")

    _, hanns_log = _read_existing_text_path(
        evidence.get("archived_hanns_log"), errors, "evidence.archived_hanns_log"
    )
    if hanns_log is not None:
        expected_tokens = [
            f"HNSWSQ_EXPECT_THREADS={hanns.get('threads')}",
            f"HNSWSQ_M={hanns.get('m')}",
            f"HNSWSQ_EF_CONSTRUCTION={hanns.get('ef_construction')}",
            f"HNSW-SQ aligned mode={hanns.get('sq_mode')} ef={hanns.get('ef')}:",
            f"R@100={float(hanns.get('recall_at_100')):.4f}"
            if _is_number(hanns.get("recall_at_100"))
            else None,
            f"qps={float(hanns.get('qps_or_vps')):.3f}"
            if _is_number(hanns.get("qps_or_vps"))
            else None,
            f"build={float(hanns.get('build_s')):.3f}s"
            if _is_number(hanns.get("build_s"))
            else None,
            "test result: ok",
        ]
        for token in filter(None, expected_tokens):
            if token not in hanns_log:
                errors.append(f"archived Hanns log missing token {token!r}")

    _, official_log = _read_existing_text_path(
        evidence.get("archived_official_log"), errors, "evidence.archived_official_log"
    )
    if official_log is not None:
        row_token = (
            f"refine_k = {int(official.get('refine_k')):3d}, nq = {int(official.get('nq'))}, "
            f"k = {int(official.get('top_k')):4d}, elapse = {float(official.get('elapsed_s')):6.3f}s, "
            f"R@ = {float(official.get('recall_at_100')):.4f}"
        )
        expected_tokens = [
            "Benchmark_float.TEST_HNSW_SQ",
            f"HNSW_SQ({official.get('variant')})",
            f"Build index HNSW_SQ time: {float(official.get('build_s')):.3f}s"
            if _is_number(official.get("build_s"))
            else None,
            row_token,
            'C++ exception with description "bad optional access" thrown in the test body.',
            "[  FAILED  ] Benchmark_float.TEST_HNSW_SQ",
        ]
        for token in filter(None, expected_tokens):
            if token not in official_log:
                errors.append(f"archived official log missing token {token!r}")


def collect_hnsw_sq_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate a scoped HNSW-SQ verdict against official FP32 partial-row evidence."""

    errors: list[str] = []
    if verdict.get("artifact_type") != "hnsw_sq_milestone_verdict":
        errors.append("artifact_type must be 'hnsw_sq_milestone_verdict'")

    status = verdict.get("status")
    if status not in HNSW_SQ_VERDICT_STATUSES:
        errors.append(f"status must be one of {HNSW_SQ_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "hnsw_sq") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100")
    tolerance = _require_number(contract, "recall_tolerance", errors, "comparison_contract")
    if tolerance is not None and tolerance < 0:
        errors.append("comparison_contract.recall_tolerance must be non-negative")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")

    official = _require_path(verdict, ("official_target",), errors, "hnsw_sq") or {}
    hanns = _require_path(verdict, ("hanns_candidate",), errors, "hnsw_sq") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "hnsw_sq") or {}

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    if official.get("commit") and str(official.get("commit")) not in str(official.get("ref", "")):
        errors.append("official_target.ref must pin official_target.commit")
    for label, obj in (("official_target", official), ("hanns_candidate", hanns)):
        for field in ("commit", "ref", "log_path"):
            if not obj.get(field):
                errors.append(f"{label}.{field} is required")
        for field in ("top_k", "m", "ef_construction", "ef", "threads", "build_threads"):
            _require_positive_int(obj, field, errors, label)
        for field in ("recall_at_100", "build_s", "qps_or_vps"):
            _require_number(obj, field, errors, label)
    for field in ("runner", "variant", "nq", "elapsed_s", "refine_k"):
        if not official.get(field):
            errors.append(f"official_target.{field} is required")
    _require_positive_int(official, "nq", errors, "official_target")
    _require_positive_int(official, "refine_k", errors, "official_target")
    _require_number(official, "elapsed_s", errors, "official_target")
    for field in ("train_s", "add_s"):
        _require_number(hanns, field, errors, "hanns_candidate")
    if hanns.get("sq_mode") not in ("SQ8", "SQ8Refine"):
        errors.append("hanns_candidate.sq_mode must be SQ8 or SQ8Refine")

    for check in HNSW_SQ_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    same_top_k = official.get("top_k") == hanns.get("top_k") == contract.get("top_k")
    if checks.get("same_top_k") != same_top_k:
        errors.append("verdict_checks.same_top_k does not match top_k fields")
    if _is_number(official.get("recall_at_100")) and _is_number(hanns.get("recall_at_100")):
        recall_ok = float(hanns["recall_at_100"]) + float(tolerance or 0.0) >= float(
            official["recall_at_100"]
        )
        if checks.get("same_or_higher_recall_with_tolerance") != recall_ok:
            errors.append("verdict_checks.same_or_higher_recall_with_tolerance mismatch")
    units_comparable = contract.get("vps_qps_mapping") == "one_vector_per_query"
    if checks.get("throughput_units_comparable") != units_comparable:
        errors.append("verdict_checks.throughput_units_comparable does not match mapping")
    threads_match = (
        contract.get("thread_policy") == "match_or_explicitly_normalized"
        and official.get("threads") == hanns.get("threads")
        and official.get("build_threads") == hanns.get("build_threads")
    )
    if checks.get("thread_policy_satisfied") != threads_match:
        errors.append("verdict_checks.thread_policy_satisfied does not match thread fields")
    if _is_number(official.get("qps_or_vps")) and _is_number(hanns.get("qps_or_vps")):
        tput_ok = float(hanns["qps_or_vps"]) > float(official["qps_or_vps"])
        if checks.get("hanns_qps_gt_official") != tput_ok:
            errors.append("verdict_checks.hanns_qps_gt_official does not match throughput")
    if _is_number(official.get("build_s")) and _is_number(hanns.get("build_s")):
        build_ok = float(hanns["build_s"]) < float(official["build_s"])
        if checks.get("hanns_build_s_lt_official") != build_ok:
            errors.append("verdict_checks.hanns_build_s_lt_official does not match build_s values")

    official_bound = (
        official.get("runner") == "benchmark_float"
        and official.get("variant") == "FP32"
        and official.get("full_test_status") == "failed_after_fp32_rows"
    )
    if checks.get("official_fp32_row_bound") != official_bound:
        errors.append("verdict_checks.official_fp32_row_bound does not match official_target")
    if checks.get("official_full_test_failed_after_fp32") != official_bound:
        errors.append(
            "verdict_checks.official_full_test_failed_after_fp32 does not match official_target"
        )

    if contract.get("archived_evidence_required") is True:
        _validate_hnsw_sq_archived_evidence(verdict, official, hanns, errors)

    all_checks_true = all(checks.get(check) is True for check in HNSW_SQ_REQUIRED_CHECKS)
    if status == "win" and not all_checks_true:
        errors.append("status=win requires all verdict checks to be true")
    return errors


def validate_hnsw_sq_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_hnsw_sq_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)


def collect_hnsw_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate an HNSW milestone verdict object against archived authority evidence."""

    errors: list[str] = []
    if verdict.get("artifact_type") != "hnsw_milestone_verdict":
        errors.append("artifact_type must be 'hnsw_milestone_verdict'")

    status = verdict.get("status")
    if status not in HNSW_VERDICT_STATUSES:
        errors.append(f"status must be one of {HNSW_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "hnsw") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100 for this official lane")
    tolerance = _require_number(contract, "recall_tolerance", errors, "comparison_contract")
    if tolerance is not None and tolerance < 0:
        errors.append("comparison_contract.recall_tolerance must be non-negative")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")

    official = _require_path(verdict, ("official_target",), errors, "hnsw") or {}
    hanns = _require_path(verdict, ("hanns_candidate",), errors, "hnsw") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "hnsw") or {}

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    for label, obj in (("official_target", official), ("hanns_candidate", hanns)):
        for field in ("commit", "ref", "log_path"):
            if not obj.get(field):
                errors.append(f"{label}.{field} is required")
        for field in ("top_k", "m", "ef_construction", "ef", "threads", "build_threads"):
            _require_positive_int(obj, field, errors, label)
        for field in ("recall_at_100", "build_s", "qps_or_vps"):
            _require_number(obj, field, errors, label)
    for field in ("train_s", "add_s"):
        _require_number(hanns, field, errors, "hanns_candidate")
    if not official.get("variant"):
        errors.append("official_target.variant is required")
    if not official.get("recall_goal"):
        errors.append("official_target.recall_goal is required")

    for check in HNSW_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    same_top_k = official.get("top_k") == hanns.get("top_k") == contract.get("top_k")
    if checks.get("same_top_k") != same_top_k:
        errors.append("verdict_checks.same_top_k does not match top_k fields")

    if _is_number(official.get("recall_at_100")) and _is_number(hanns.get("recall_at_100")):
        recall_ok = float(hanns["recall_at_100"]) + float(tolerance or 0.0) >= float(
            official["recall_at_100"]
        )
        if checks.get("same_or_higher_recall") != recall_ok:
            errors.append("verdict_checks.same_or_higher_recall does not match recall values")

    units_comparable = contract.get("vps_qps_mapping") == "one_vector_per_query"
    if checks.get("throughput_units_comparable") != units_comparable:
        errors.append(
            "verdict_checks.throughput_units_comparable does not match vps_qps_mapping"
        )
    thread_policy = contract.get("thread_policy")
    threads_match = (
        thread_policy == "match_or_explicitly_normalized"
        and isinstance(official.get("threads"), int)
        and official.get("threads") == hanns.get("threads")
        and isinstance(official.get("build_threads"), int)
        and official.get("build_threads") == hanns.get("build_threads")
    )
    if checks.get("thread_policy_satisfied") != threads_match:
        errors.append(
            "verdict_checks.thread_policy_satisfied does not match thread fields"
        )

    if _is_number(official.get("qps_or_vps")) and _is_number(hanns.get("qps_or_vps")):
        tput_ok = float(hanns["qps_or_vps"]) > float(official["qps_or_vps"])
        if checks.get("hanns_qps_vps_gt_official") != tput_ok:
            errors.append(
                "verdict_checks.hanns_qps_vps_gt_official does not match throughput values"
            )
    if _is_number(official.get("build_s")) and _is_number(hanns.get("build_s")):
        build_ok = float(hanns["build_s"]) < float(official["build_s"])
        if checks.get("hanns_build_s_lt_official") != build_ok:
            errors.append(
                "verdict_checks.hanns_build_s_lt_official does not match build_s values"
            )

    if contract.get("archived_evidence_required") is True:
        _validate_hnsw_archived_evidence(verdict, official, hanns, errors)

    all_checks_true = all(checks.get(check) is True for check in HNSW_REQUIRED_CHECKS)
    if status == "win" and not all_checks_true:
        errors.append("status=win requires all verdict checks to be true")
    if status != "non_comparable" and (
        checks.get("throughput_units_comparable") is False
        or checks.get("thread_policy_satisfied") is False
    ):
        errors.append("unit or thread mismatch requires status=non_comparable")

    return errors


def validate_hnsw_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_hnsw_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)

def _validate_ivfpq_refine_metadata(
    refine: Any, errors: list[str], status: str | None
) -> None:
    if not isinstance(refine, dict):
        errors.append("hanns_candidate.refine must be an object")
        return

    if status == "win" and refine.get("enabled") is not True:
        errors.append("status=win requires hanns_candidate.refine.enabled=true")

    multiplier = refine.get("exact_refine_multiplier")
    if not isinstance(multiplier, int) or isinstance(multiplier, bool) or multiplier <= 0:
        errors.append(
            "hanns_candidate.refine.exact_refine_multiplier must be positive integer"
        )

    if not refine.get("candidate_pool"):
        errors.append("hanns_candidate.refine.candidate_pool is required")


def collect_ivfpq_verdict_errors(verdict: dict[str, Any]) -> list[str]:
    """Validate the approved IVF-PQ milestone verdict object.

    This is intentionally a verdict-object validator, not a benchmark parser.
    Parsers may produce this object only after normalizing raw official/Hanns
    logs; this function enforces the fail-closed comparison contract.
    """

    errors: list[str] = []
    if verdict.get("artifact_type") != "ivfpq_milestone_verdict":
        errors.append("artifact_type must be 'ivfpq_milestone_verdict'")

    status = verdict.get("status")
    if status not in IVFPQ_VERDICT_STATUSES:
        errors.append(f"status must be one of {IVFPQ_VERDICT_STATUSES}")
    if verdict.get("authority_surface") != AUTHORITY_SURFACE:
        errors.append(f"authority_surface must be {AUTHORITY_SURFACE!r}")
    if not verdict.get("run_set_id"):
        errors.append("run_set_id is required")
    if not verdict.get("plan_approved_at"):
        errors.append("plan_approved_at is required")

    contract = _require_path(verdict, ("comparison_contract",), errors, "ivfpq") or {}
    if contract.get("metric_of_record") != "recall_at_100":
        errors.append("comparison_contract.metric_of_record must be recall_at_100")
    if contract.get("top_k") != 100:
        errors.append("comparison_contract.top_k must be 100 for this official lane")
    if contract.get("recall_band_formula") != "floor(recall * 100) / 100":
        errors.append("comparison_contract.recall_band_formula is invalid")
    tolerance = _require_number(contract, "recall_tolerance", errors, "comparison_contract")
    if tolerance is not None and tolerance < 0:
        errors.append("comparison_contract.recall_tolerance must be non-negative")
    if contract.get("throughput_metric") not in ("qps", "vps"):
        errors.append("comparison_contract.throughput_metric must be qps or vps")
    build_time_required = contract.get("build_time_required") is True

    official = _require_path(verdict, ("official_target",), errors, "ivfpq") or {}
    hanns = _require_path(verdict, ("hanns_candidate",), errors, "ivfpq") or {}
    checks = _require_path(verdict, ("verdict_checks",), errors, "ivfpq") or {}
    _validate_ivfpq_refine_metadata(hanns.get("refine"), errors, status)

    if not is_official_knowhere_url(official.get("repo_url", "")):
        errors.append("official_target.repo_url must be zilliztech/knowhere")
    for label, obj in (("official_target", official), ("hanns_candidate", hanns)):
        for field in ("commit", "ref", "log_path"):
            if not obj.get(field):
                errors.append(f"{label}.{field} is required")
        for field in ("top_k", "nlist", "nprobe", "m", "nbits", "threads"):
            _require_positive_int(obj, field, errors, label)
        for field in ("recall_at_100", "recall_band_floor", "build_s", "qps_or_vps"):
            _require_number(obj, field, errors, label)

    excluded = official.get("excluded_rows")
    if not isinstance(excluded, list):
        errors.append("official_target.excluded_rows must be a list")
    elif "terminal_zero_anomaly" not in excluded:
        errors.append("official_target.excluded_rows must record terminal_zero_anomaly")

    normalization_source = official.get("normalization_source")
    if normalization_source != "sweep_row":
        if status != "blocked_official_normalization":
            errors.append(
                "missing trustworthy official normalization requires "
                "status=blocked_official_normalization"
            )
    elif checks.get("official_normalized") is not True:
        errors.append("verdict_checks.official_normalized must be true for sweep_row")

    for check in IVFPQ_REQUIRED_CHECKS:
        if check not in checks:
            errors.append(f"verdict_checks.{check} is required")
        elif not isinstance(checks[check], bool):
            errors.append(f"verdict_checks.{check} must be boolean")

    official_top_k = official.get("top_k")
    hanns_top_k = hanns.get("top_k")
    if official_top_k is not None and hanns_top_k is not None:
        same_top_k = official_top_k == hanns_top_k == contract.get("top_k")
        if not same_top_k:
            errors.append("hanns_candidate must use the same top_k as official_target")
        if checks.get("same_top_k") != same_top_k:
            errors.append("verdict_checks.same_top_k does not match top_k fields")

    official_recall = official.get("recall_at_100")
    hanns_recall = hanns.get("recall_at_100")
    if _is_number(official_recall) and _is_number(hanns_recall):
        recall_ok = float(hanns_recall) + float(tolerance or 0.0) >= float(
            official_recall
        )
        if checks.get("exact_recall_within_tolerance") != recall_ok:
            errors.append(
                "verdict_checks.exact_recall_within_tolerance does not match recall values"
            )
        if not recall_ok and status == "win":
            errors.append("status=win requires Hanns recall within tolerance")

    official_band = official.get("recall_band_floor")
    hanns_band = hanns.get("recall_band_floor")
    if _is_number(official_recall) and _is_number(official_band):
        expected = ivfpq_recall_band_floor(float(official_recall))
        if abs(float(official_band) - expected) > 1e-6:
            errors.append("official_target.recall_band_floor does not match recall")
    if _is_number(hanns_recall) and _is_number(hanns_band):
        expected = ivfpq_recall_band_floor(float(hanns_recall))
        if abs(float(hanns_band) - expected) > 1e-6:
            errors.append("hanns_candidate.recall_band_floor does not match recall")
    if _is_number(official_band) and _is_number(hanns_band):
        band_ok = float(hanns_band) >= float(official_band)
        if checks.get("same_or_higher_recall_band") != band_ok:
            errors.append(
                "verdict_checks.same_or_higher_recall_band does not match recall bands"
            )
        if not band_ok and status == "win":
            errors.append("status=win requires same-or-higher recall band")

    mapping = contract.get("vps_qps_mapping")
    units_comparable = mapping == "one_vector_per_query"
    if checks.get("throughput_units_comparable") != units_comparable:
        errors.append(
            "verdict_checks.throughput_units_comparable does not match vps_qps_mapping"
        )
    thread_policy = contract.get("thread_policy")
    official_threads = official.get("threads")
    hanns_threads = hanns.get("threads")
    threads_match = (
        thread_policy == "match_or_explicitly_normalized"
        and isinstance(official_threads, int)
        and official_threads == hanns_threads
    )
    if checks.get("thread_policy_satisfied") != threads_match:
        errors.append(
            "verdict_checks.thread_policy_satisfied does not match thread fields"
        )

    official_tput = official.get("qps_or_vps")
    hanns_tput = hanns.get("qps_or_vps")
    if _is_number(official_tput) and _is_number(hanns_tput):
        tput_ok = float(hanns_tput) > float(official_tput)
        if checks.get("hanns_qps_vps_gt_official") != tput_ok:
            errors.append(
                "verdict_checks.hanns_qps_vps_gt_official does not match throughput values"
            )

    build_check = checks.get("hanns_build_s_lt_official")
    if build_time_required and build_check is None:
        errors.append(
            "comparison_contract.build_time_required requires "
            "verdict_checks.hanns_build_s_lt_official"
        )
    if build_check is not None and not isinstance(build_check, bool):
        errors.append("verdict_checks.hanns_build_s_lt_official must be boolean")
    official_build = official.get("build_s")
    hanns_build = hanns.get("build_s")
    if _is_number(official_build) and _is_number(hanns_build):
        build_ok = float(hanns_build) < float(official_build)
        if build_check is not None and build_check != build_ok:
            errors.append(
                "verdict_checks.hanns_build_s_lt_official does not match build_s values"
            )
        if build_time_required and not build_ok and status == "win":
            errors.append("status=win requires Hanns build_s below official_target build_s")

    all_checks_true = all(checks.get(check) is True for check in IVFPQ_REQUIRED_CHECKS)
    if build_time_required:
        all_checks_true = all_checks_true and build_check is True

    if contract.get("archived_evidence_required") is True:
        _validate_ivfpq_archived_evidence(verdict, official, hanns, errors)

    if status == "win" and not all_checks_true:
        errors.append("status=win requires all verdict checks to be true")
    if status != "non_comparable" and (
        checks.get("throughput_units_comparable") is False
        or checks.get("thread_policy_satisfied") is False
    ):
        errors.append("unit or thread mismatch requires status=non_comparable")
    if status == "blocked_official_normalization" and checks.get(
        "official_normalized"
    ):
        errors.append(
            "status=blocked_official_normalization requires official_normalized=false"
        )

    return errors


def validate_ivfpq_verdict(verdict: dict[str, Any]) -> None:
    errors = collect_ivfpq_verdict_errors(verdict)
    if errors:
        raise ValidationError(errors)


def collect_row_errors(
    row: dict[str, Any],
    *,
    row_index: int,
    plan_approved_at: str,
    run_set_id: str,
    final: bool,
    authority_manifest: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    label = f"row[{row_index}]"

    run_set = _require_path(row, ("run_set",), errors, label) or {}
    if run_set.get("plan_id") != PLAN_ID:
        errors.append(f"{label}: run_set.plan_id must be {PLAN_ID!r}")
    if run_set.get("run_set_id") != run_set_id:
        errors.append(f"{label}: run_set.run_set_id must be current run set")
    try:
        approved = parse_time(plan_approved_at, field="plan_approved_at")
    except Exception as exc:  # pragma: no cover - caller error path
        errors.append(f"invalid plan_approved_at: {exc}")
        approved = None
    if run_set.get("plan_approved_at") != plan_approved_at:
        errors.append(f"{label}: run_set.plan_approved_at must match plan approval")
    for field in ("run_started_at", "generated_at", "raw_artifact_created_at"):
        value = run_set.get(field)
        try:
            parsed = parse_time(value, field=f"run_set.{field}")
            if approved and parsed < approved:
                errors.append(f"{label}: run_set.{field} predates plan approval")
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    authority = _require_path(row, ("authority",), errors, label) or {}
    surface = authority.get("surface")
    eligible = authority.get("eligible_for_verdict")
    is_authority = authority.get("is_authority")
    if final:
        if surface != AUTHORITY_SURFACE:
            errors.append(f"{label}: final row must use {AUTHORITY_SURFACE}")
        if is_authority is not True or eligible is not True:
            errors.append(f"{label}: final row must be authority/verdict eligible")
        proof = authority.get("runtime_proof")
        if not isinstance(proof, dict):
            errors.append(f"{label}: missing authority.runtime_proof")
        else:
            for field in (
                "ssh_alias_or_host",
                "hostname",
                "uname",
                "lscpu_hash",
                "remote_log_path",
                "wrapper_script",
            ):
                if not proof.get(field):
                    errors.append(f"{label}: missing runtime_proof.{field}")
            errors.extend(_manifest_errors(proof, authority_manifest, label))
    elif surface == "local" and eligible is not False:
        errors.append(f"{label}: local row must not be verdict eligible")

    impl = row.get("implementation")
    if impl not in IMPLEMENTATIONS:
        errors.append(f"{label}: invalid implementation {impl!r}")
    source = _require_path(row, ("source",), errors, label) or {}
    if impl == "zilliz_knowhere" and not is_official_knowhere_url(
        source.get("repo_url", "")
    ):
        errors.append(f"{label}: official Knowhere source must be zilliztech/knowhere")
    for field in ("repo_url", "commit", "dirty"):
        if field not in source:
            errors.append(f"{label}: missing source.{field}")
    if source.get("dirty") is not False:
        errors.append(f"{label}: source.dirty must be false")

    index = _require_path(row, ("index",), errors, label) or {}
    family = index.get("user_family")
    status = index.get("support_status")
    if family not in REQUIRED_FAMILIES:
        errors.append(f"{label}: invalid index.user_family {family!r}")
    if status not in SUPPORT_STATUSES:
        errors.append(f"{label}: invalid index.support_status {status!r}")

    dataset = _require_path(row, ("dataset",), errors, label) or {}
    for field in (
        "name",
        "path",
        "checksum",
        "metric",
        "dimension",
        "base_count",
        "query_count",
    ):
        if dataset.get(field) in (None, ""):
            errors.append(f"{label}: missing dataset.{field}")

    params = _require_path(row, ("params",), errors, label) or {}
    for field in ("top_k", "recall_at", "concurrency", "threads"):
        if not isinstance(params.get(field), int) or params.get(field) <= 0:
            errors.append(f"{label}: params.{field} must be positive integer")
    if params.get("recall_band") not in RECALL_BANDS:
        errors.append(f"{label}: params.recall_band must be one of {RECALL_BANDS}")
    for field in ("index_params", "search_params"):
        if not isinstance(params.get(field), dict):
            errors.append(f"{label}: params.{field} must be object")

    if status == "supported":
        metrics = _require_path(row, ("metrics",), errors, label) or {}
        if not _is_number(metrics.get("qps")):
            errors.append(f"{label}: supported row requires numeric metrics.qps")
        if not _is_number(metrics.get("recall_at_k")):
            errors.append(
                f"{label}: supported row requires numeric metrics.recall_at_k"
            )
        build = metrics.get("build_seconds")
        if not isinstance(build, dict):
            errors.append(f"{label}: metrics.build_seconds must be object")
        else:
            if not _is_number(build.get("total")) and not build.get(
                "unavailable_reason"
            ):
                errors.append(
                    f"{label}: build_seconds requires total or unavailable_reason"
                )
            if not build.get("definition"):
                errors.append(f"{label}: build_seconds.definition is required")
        latency = metrics.get("latency_ms")
        if not isinstance(latency, dict):
            errors.append(f"{label}: metrics.latency_ms must be object")
        else:
            for field in ("p50", "p95", "p99"):
                if not _is_number(latency.get(field)):
                    errors.append(f"{label}: latency_ms.{field} must be numeric")
            if not isinstance(latency.get("sample_count"), int) or latency.get(
                "sample_count"
            ) <= 0:
                errors.append(f"{label}: latency_ms.sample_count must be positive")
            if not latency.get("source"):
                errors.append(f"{label}: latency_ms.source is required")

    repeat = _require_path(row, ("repeat",), errors, label) or {}
    for field in ("warmup", "runs"):
        if not isinstance(repeat.get(field), int) or repeat.get(field) < 0:
            errors.append(f"{label}: repeat.{field} must be non-negative integer")
    if not repeat.get("run_id"):
        errors.append(f"{label}: repeat.run_id is required")

    artifacts = _require_path(row, ("artifacts",), errors, label) or {}
    if final:
        for field in ("stdout_log", "raw_json"):
            if not artifacts.get(field):
                errors.append(f"{label}: artifacts.{field} is required")

    return errors


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    plan_approved_at: str,
    run_set_id: str,
    final: bool = True,
    authority_manifest: dict[str, Any] | None = None,
) -> None:
    errors: list[str] = []
    for index, row in enumerate(rows):
        errors.extend(
            collect_row_errors(
                row,
                row_index=index,
                plan_approved_at=plan_approved_at,
                run_set_id=run_set_id,
                final=final,
                authority_manifest=authority_manifest,
            )
        )
    if final:
        try:
            validate_capability_rows(rows)
        except ValidationError as exc:
            errors.extend(exc.errors)
    if errors:
        raise ValidationError(errors)


def git_output(repo_dir: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def inspect_knowhere_repo(repo_dir: pathlib.Path, expected_ref: str) -> dict[str, Any]:
    if not expected_ref:
        raise ValidationError(["expected Knowhere ref/commit is required"])
    errors: list[str] = []
    if not (repo_dir / ".git").exists():
        errors.append(f"{repo_dir} is not a git repository")
    try:
        origin = git_output(repo_dir, "remote", "get-url", "origin")
        if not is_official_knowhere_url(origin):
            errors.append(f"origin is not zilliztech/knowhere: {origin}")
    except subprocess.CalledProcessError as exc:
        errors.append(f"failed to read origin: {exc.stderr.strip()}")
        origin = ""
    try:
        status = git_output(repo_dir, "status", "--porcelain")
        if status:
            errors.append("Knowhere worktree is dirty")
    except subprocess.CalledProcessError as exc:
        errors.append(f"failed to inspect dirty state: {exc.stderr.strip()}")
    try:
        commit = git_output(repo_dir, "rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        errors.append(f"failed to read commit: {exc.stderr.strip()}")
        commit = ""
    try:
        expected_commit = git_output(repo_dir, "rev-parse", f"{expected_ref}^{{commit}}")
        if commit and expected_commit != commit:
            errors.append(
                f"HEAD {commit} does not match expected Knowhere ref "
                f"{expected_ref} ({expected_commit})"
            )
    except subprocess.CalledProcessError as exc:
        errors.append(
            f"expected Knowhere ref {expected_ref!r} is not available locally; "
            "fetch/reset with scripts/remote/official_knowhere_provenance.sh "
            f"first: {exc.stderr.strip()}"
        )
    if errors:
        raise ValidationError(errors)
    return {
        "repo_url": canonicalize_repo_url(origin),
        "commit": commit,
        "ref": expected_ref,
        "dirty": False,
    }


def render_markdown(
    rows: list[dict[str, Any]],
    *,
    plan_approved_at: str,
    run_set_id: str,
    authority_manifest: dict[str, Any],
) -> str:
    validate_rows(
        rows,
        plan_approved_at=plan_approved_at,
        run_set_id=run_set_id,
        final=True,
        authority_manifest=authority_manifest,
    )
    validate_capability_rows(rows)
    lines = [
        "# Hanns vs Zilliz Knowhere Benchmark Report",
        "",
        "> Headline tables may include only fresh HannsDB-x86 authority rows.",
        "",
        "## Capability status",
        "",
        "| Family | Hanns | Zilliz Knowhere |",
        "|---|---|---|",
    ]
    by_pair = {
        (row["index"]["user_family"], row["implementation"]): row["index"][
            "support_status"
        ]
        for row in rows
    }
    for family in REQUIRED_FAMILIES:
        lines.append(
            f"| {family} | {by_pair[(family, 'hanns')]} | "
            f"{by_pair[(family, 'zilliz_knowhere')]} |"
        )
    supported = [
        row
        for row in rows
        if row.get("index", {}).get("support_status") == "supported"
        and row.get("authority", {}).get("eligible_for_verdict") is True
    ]
    lines.extend(
        [
            "",
            "## Supported authority rows",
            "",
            "| Family | Implementation | QPS | Recall | Build(s) | p50 | p95 | p99 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in supported:
        metrics = row["metrics"]
        latency = metrics["latency_ms"]
        build = metrics["build_seconds"]
        lines.append(
            f"| {row['index']['user_family']} | {row['implementation']} | "
            f"{metrics['qps']} | {metrics['recall_at_k']} | "
            f"{build.get('total', 'n/a')} | {latency['p50']} | "
            f"{latency['p95']} | {latency['p99']} |"
        )
    return "\n".join(lines) + "\n"


def lscpu_hash(lscpu_text: str) -> str:
    return hashlib.sha256(lscpu_text.encode("utf-8")).hexdigest()


def command_write_matrix(args: argparse.Namespace) -> int:
    write_json(args.output, default_matrix())
    return 0


def command_validate_matrix(args: argparse.Namespace) -> int:
    validate_matrix(load_json(args.matrix))
    return 0


def command_validate_rows(args: argparse.Namespace) -> int:
    payload = load_json(args.rows)
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    if not isinstance(rows, list):
        raise ValidationError(["rows input must be a list or object with rows list"])
    authority_manifest = load_json(args.authority_manifest) if args.authority_manifest else None
    validate_rows(
        rows,
        plan_approved_at=args.plan_approved_at,
        run_set_id=args.run_set_id,
        final=not args.non_final,
        authority_manifest=authority_manifest,
    )
    return 0


def command_validate_ivfpq_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["IVF-PQ verdict input must be an object"])
    validate_ivfpq_verdict(payload)
    return 0


def command_validate_hnsw_pq_blocker(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["HNSW-PQ blocker input must be an object"])
    validate_hnsw_pq_blocker(payload)
    return 0


def command_validate_hnsw_sq_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["HNSW-SQ verdict input must be an object"])
    validate_hnsw_sq_verdict(payload)
    return 0


def command_validate_hnsw_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["HNSW verdict input must be an object"])
    validate_hnsw_verdict(payload)
    return 0


def command_validate_ivfsq8_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["IVF-SQ8 verdict input must be an object"])
    validate_ivfsq8_verdict(payload)
    return 0


def command_validate_diskann_aisaq_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["DiskANN/AISAQ verdict input must be an object"])
    validate_diskann_aisaq_verdict(payload)
    return 0


def command_validate_ivfusq_verdict(args: argparse.Namespace) -> int:
    payload = load_json(args.verdict)
    if not isinstance(payload, dict):
        raise ValidationError(["IVF-USQ verdict input must be an object"])
    validate_ivfusq_verdict(payload)
    return 0


def command_render_report(args: argparse.Namespace) -> int:
    rows = load_json(args.rows)
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    authority_manifest = load_json(args.authority_manifest)
    output = render_markdown(
        rows,
        plan_approved_at=args.plan_approved_at,
        run_set_id=args.run_set_id,
        authority_manifest=authority_manifest,
    )
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(output, encoding="utf-8")
    return 0


def command_check_knowhere_source(args: argparse.Namespace) -> int:
    payload = inspect_knowhere_repo(pathlib.Path(args.repo_dir), args.expected_ref)
    print(json.dumps(payload, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_matrix = subparsers.add_parser("write-matrix")
    write_matrix.add_argument("--output", required=True)
    write_matrix.set_defaults(func=command_write_matrix)

    validate_matrix_cmd = subparsers.add_parser("validate-matrix")
    validate_matrix_cmd.add_argument("matrix")
    validate_matrix_cmd.set_defaults(func=command_validate_matrix)

    validate_rows_cmd = subparsers.add_parser("validate-rows")
    validate_rows_cmd.add_argument("rows")
    validate_rows_cmd.add_argument("--plan-approved-at", required=True)
    validate_rows_cmd.add_argument("--run-set-id", required=True)
    validate_rows_cmd.add_argument("--authority-manifest")
    validate_rows_cmd.add_argument("--non-final", action="store_true")
    validate_rows_cmd.set_defaults(func=command_validate_rows)

    validate_ivfpq_cmd = subparsers.add_parser("validate-ivfpq-verdict")
    validate_ivfpq_cmd.add_argument("verdict")
    validate_ivfpq_cmd.set_defaults(func=command_validate_ivfpq_verdict)

    validate_hnsw_cmd = subparsers.add_parser("validate-hnsw-verdict")
    validate_hnsw_cmd.add_argument("verdict")
    validate_hnsw_cmd.set_defaults(func=command_validate_hnsw_verdict)


    validate_hnsw_pq_blocker_cmd = subparsers.add_parser("validate-hnsw-pq-blocker")
    validate_hnsw_pq_blocker_cmd.add_argument("verdict")
    validate_hnsw_pq_blocker_cmd.set_defaults(func=command_validate_hnsw_pq_blocker)
    validate_hnsw_sq_cmd = subparsers.add_parser("validate-hnsw-sq-verdict")
    validate_hnsw_sq_cmd.add_argument("verdict")
    validate_hnsw_sq_cmd.set_defaults(func=command_validate_hnsw_sq_verdict)

    validate_ivfsq8_cmd = subparsers.add_parser("validate-ivfsq8-verdict")
    validate_ivfsq8_cmd.add_argument("verdict")
    validate_ivfsq8_cmd.set_defaults(func=command_validate_ivfsq8_verdict)

    validate_diskann_aisaq_cmd = subparsers.add_parser(
        "validate-diskann-aisaq-verdict"
    )
    validate_diskann_aisaq_cmd.add_argument("verdict")
    validate_diskann_aisaq_cmd.set_defaults(func=command_validate_diskann_aisaq_verdict)

    validate_ivfusq_cmd = subparsers.add_parser("validate-ivfusq-verdict")
    validate_ivfusq_cmd.add_argument("verdict")
    validate_ivfusq_cmd.set_defaults(func=command_validate_ivfusq_verdict)

    render_report = subparsers.add_parser("render-report")
    render_report.add_argument("--rows", required=True)
    render_report.add_argument("--output", required=True)
    render_report.add_argument("--plan-approved-at", required=True)
    render_report.add_argument("--run-set-id", required=True)
    render_report.add_argument("--authority-manifest", required=True)
    render_report.set_defaults(func=command_render_report)

    check_source = subparsers.add_parser("check-knowhere-source")
    check_source.add_argument("--repo-dir", required=True)
    check_source.add_argument("--expected-ref", required=True)
    check_source.set_defaults(func=command_check_knowhere_source)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValidationError as exc:
        for error in exc.errors:
            print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
