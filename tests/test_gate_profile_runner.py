from __future__ import annotations

import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "gate_profile_runner.sh"


def run_print_checks(profile: str) -> list[str]:
    proc = subprocess.run(
        [str(RUNNER), "--profile", profile, "--print-checks"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


class GateProfileRunnerTests(unittest.TestCase):
    def test_default_regression_checks(self) -> None:
        lines = run_print_checks("default_regression")
        self.assertEqual(lines[0], "profile=default_regression")
        self.assertIn("cargo test -q", lines)

    def test_full_regression_includes_governance_locks(self) -> None:
        lines = run_print_checks("full_regression")
        self.assertEqual(lines[0], "profile=full_regression")
        self.assertIn("cargo test --lib -q", lines)
        self.assertIn("cargo test --tests -q", lines)
        self.assertIn("cargo test --doc -q", lines)
        self.assertIn("python3 -m unittest tests/test_validate_features.py", lines)
        self.assertIn(
            "python3 -m unittest tests/test_baseline_methodology_lock.py",
            lines,
        )
        self.assertIn("python3 -m unittest tests/test_parity_report_lock.py", lines)
        self.assertIn(
            "python3 -m unittest tests/test_governance_current_state_lock.py",
            lines,
        )

    def test_long_regression_checks(self) -> None:
        lines = run_print_checks("long_regression")
        self.assertEqual(lines[0], "profile=long_regression")
        self.assertIn("cargo test --tests --features long-tests -q", lines)
        self.assertIn("cargo test --test opt_p2_stable_regression_matrix -q", lines)
        self.assertIn("cargo test --test bench_recall_gated_baseline -q", lines)


if __name__ == "__main__":
    unittest.main()
