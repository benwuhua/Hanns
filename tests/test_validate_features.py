from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_features.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_features", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator_module()


def base_features() -> list[dict[str, object]]:
    return [
        {
            "id": "feature-alpha",
            "category": "baseline",
            "title": "Feature Alpha",
            "description": "Already completed baseline feature.",
            "priority": "high",
            "status": "passing",
            "verification_steps": [
                "python3 -m unittest tests/test_alpha.py",
            ],
            "dependencies": [],
        },
        {
            "id": "feature-beta",
            "category": "baseline",
            "title": "Feature Beta",
            "description": "Current active feature.",
            "priority": "high",
            "status": "failing",
            "verification_steps": [
                "cargo test --test test_beta -- --nocapture",
            ],
            "dependencies": [
                "feature-alpha",
            ],
        },
    ]


def build_task_progress(
    *,
    current_focus: str = "feature-beta",
    next_feature: str = "feature-beta",
    strategic_state: str | None = None,
    passing: int = 1,
    total: int = 2,
    percent: int = 50,
    latest_verification: str = "`python3 scripts/validate_features.py feature-list.json` -> `VALID`",
) -> str:
    strategic_state_line = (
        f"- Strategic state: `{strategic_state}`\n" if strategic_state is not None else ""
    )
    return textwrap.dedent(
        f"""\
        # Task Progress

        ## Current State

        - Current focus: `{current_focus}`
        - Next feature: `{next_feature}` (selected for execution)
        {strategic_state_line}\
        - Last updated: 2026-03-12
        - Progress: {passing}/{total} features passing ({percent}%)

        ## Session Log

        ### Session 9 - 2026-03-12
        - Focus: `feature-alpha`
        - Verification:
          - {latest_verification}
        - Result:
          - `feature-alpha` is now `passing`
        """
    )


def build_release_notes() -> str:
    return textwrap.dedent(
        """\
        # Release Notes

        ## [Unreleased]

        ### Fixed
        - `feature-alpha` now passes with durable-state updates.
        """
    )


def build_final_performance_proof(*, criterion_met: bool = False) -> dict[str, object]:
    return {
        "task_id": "FINAL-PERFORMANCE-LEADERSHIP-PROOF",
        "criterion_met": criterion_met,
    }


def build_performance_program(
    *,
    program_state: str = "blocked_on_hnsw_fairness_gate",
    next_track: str = "hnsw-fairness-gate",
    final_status: str = "unmet",
    final_source: str = "benchmark_results/final_performance_leadership_proof.json",
) -> str:
    return textwrap.dedent(
        f"""\
        # Performance Program

        ## Current Verdict

        - Final criterion source: `{final_source}`
        - Final criterion status: `{final_status}`
        - Program state: `{program_state}`
        - Next strategic track: `{next_track}`

        ## Fairness Gate

        - Match effective `ef` before reading qps deltas.

        ## Canonical Compare Lane

        - Use the remote x86 same-schema lane only.

        ## Pivot Gate

        - Pause pure-Rust HNSW reopen work after repeated non-attributable fair-lane rounds.
        """
    )


class ValidateFeaturesWorkflowTests(unittest.TestCase):
    def write_repo(
        self,
        root: pathlib.Path,
        *,
        features: list[dict[str, object]] | None = None,
        task_progress: str | None = None,
        release_notes: str | None = None,
        final_performance_proof: dict[str, object] | None = None,
        performance_program: str | None = None,
        extra_files: dict[str, str] | None = None,
    ) -> pathlib.Path:
        feature_list_path = root / "feature-list.json"
        feature_list_path.write_text(
            json.dumps({"features": features or base_features()}, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "task-progress.md").write_text(
            task_progress or build_task_progress(),
            encoding="utf-8",
        )
        (root / "RELEASE_NOTES.md").write_text(
            release_notes or build_release_notes(),
            encoding="utf-8",
        )
        if final_performance_proof is not None:
            proof_path = root / "benchmark_results" / "final_performance_leadership_proof.json"
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            proof_path.write_text(
                json.dumps(final_performance_proof, indent=2) + "\n",
                encoding="utf-8",
            )
        if performance_program is not None:
            program_path = root / "docs" / "performance-program.md"
            program_path.parent.mkdir(parents=True, exist_ok=True)
            program_path.write_text(performance_program, encoding="utf-8")

        for relative_path, contents in (extra_files or {}).items():
            file_path = root / relative_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents, encoding="utf-8")

        return feature_list_path

    def test_validate_accepts_consistent_workflow_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            feature_list_path = self.write_repo(root)

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertEqual(errors, [])

    def test_validate_rejects_progress_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            feature_list_path = self.write_repo(
                root,
                task_progress=build_task_progress(passing=0, total=2, percent=0),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "task-progress.md progress summary says 0/2 features passing but feature-list.json has 1/2",
                errors,
            )

    def test_validate_accepts_none_focus_when_all_features_are_passing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            features = base_features()
            features[1]["status"] = "passing"
            feature_list_path = self.write_repo(
                root,
                features=features,
                task_progress=build_task_progress(
                    current_focus="none",
                    next_feature="none",
                    passing=2,
                    total=2,
                    percent=100,
                ),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertEqual(errors, [])

    def test_validate_rejects_none_focus_when_features_remain_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            feature_list_path = self.write_repo(
                root,
                task_progress=build_task_progress(
                    current_focus="none",
                    next_feature="none",
                ),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "task-progress.md current focus may be `none` only when all features are passing",
                errors,
            )
            self.assertIn(
                "task-progress.md next feature may be `none` only when all features are passing",
                errors,
            )

    def test_validate_rejects_latest_session_pending_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            feature_list_path = self.write_repo(
                root,
                task_progress=build_task_progress(
                    latest_verification="`python3 scripts/validate_features.py feature-list.json` -> pending rerun after durable-state update"
                ),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "Latest session still contains unresolved placeholder: pending rerun after durable-state update",
                errors,
            )

    def test_validate_rejects_bad_integration_test_filter_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            features = base_features()
            features[1]["verification_steps"] = [
                "cargo test --tests -q test_beta",
            ]
            feature_list_path = self.write_repo(root, features=features)

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "Feature id=feature-beta: verification step 'cargo test --tests -q test_beta' uses `--tests` with a bare filter; use `--test <binary>` for integration test binaries",
                errors,
            )

    def test_validate_rejects_temp_artifact_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            feature_list_path = self.write_repo(
                root,
                extra_files={"src/faiss/hnsw.rs.new": "// orphan temp artifact\n"},
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "Repository contains temporary/orphan artifact: src/faiss/hnsw.rs.new",
                errors,
            )

    def test_validate_rejects_unmet_final_proof_without_strategic_program(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            features = base_features()
            features[1]["status"] = "passing"
            feature_list_path = self.write_repo(
                root,
                features=features,
                task_progress=build_task_progress(
                    current_focus="none",
                    next_feature="none",
                    passing=2,
                    total=2,
                    percent=100,
                ),
                final_performance_proof=build_final_performance_proof(),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "docs/performance-program.md is required when all tracked features are passing but the final performance leadership criterion remains unmet",
                errors,
            )
            self.assertIn(
                "task-progress.md missing `- Strategic state: ` line required when all tracked features are passing but the final performance leadership criterion remains unmet",
                errors,
            )

    def test_validate_accepts_unmet_final_proof_with_strategic_program(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            features = base_features()
            features[1]["status"] = "passing"
            feature_list_path = self.write_repo(
                root,
                features=features,
                task_progress=build_task_progress(
                    current_focus="none",
                    next_feature="none",
                    strategic_state="blocked_on_hnsw_fairness_gate",
                    passing=2,
                    total=2,
                    percent=100,
                ),
                final_performance_proof=build_final_performance_proof(),
                performance_program=build_performance_program(),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertEqual(errors, [])

    def test_validate_rejects_strategic_state_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            features = base_features()
            features[1]["status"] = "passing"
            feature_list_path = self.write_repo(
                root,
                features=features,
                task_progress=build_task_progress(
                    current_focus="none",
                    next_feature="none",
                    strategic_state="stale_state",
                    passing=2,
                    total=2,
                    percent=100,
                ),
                final_performance_proof=build_final_performance_proof(),
                performance_program=build_performance_program(),
            )

            errors = VALIDATOR.validate(str(feature_list_path))

            self.assertIn(
                "task-progress.md strategic state `stale_state` does not match docs/performance-program.md program state `blocked_on_hnsw_fairness_gate`",
                errors,
            )


if __name__ == "__main__":
    unittest.main()
