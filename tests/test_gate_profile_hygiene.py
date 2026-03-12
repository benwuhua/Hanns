import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OPT013_TEST = REPO_ROOT / "tests" / "opt013_test.rs"
class GateProfileHygieneTests(unittest.TestCase):
    def test_opt013_fast_build_benchmark_is_excluded_from_default_gate(self) -> None:
        contents = OPT013_TEST.read_text(encoding="utf-8")

        self.assertIn(
            "#[ignore]\nfn test_opt013_ivf_flat_fast_build()",
            contents,
            msg=(
                "opt013 long-test must use a plain #[ignore] attribute so the "
                "remote rust test harness reliably excludes it from default gates"
            ),
        )

    def test_pq_perf_long_tests_are_excluded_from_default_gate(self) -> None:
        contents = (REPO_ROOT / "tests" / "pq_perf_test.rs").read_text(encoding="utf-8")

        for test_name in (
            "test_opq_recall_improvement",
            "test_residual_pq_recall_improvement",
            "test_ivf_opq_index",
        ):
            self.assertIn(
                f"#[ignore]\nfn {test_name}()",
                contents,
                msg=(
                    f"{test_name} carries hard recall/perf assertions and must use "
                    "a plain #[ignore] attribute for default regression gates"
                ),
            )

    def test_adaptive_ef_long_tests_are_excluded_from_default_gate(self) -> None:
        contents = (REPO_ROOT / "tests" / "test_adaptive_ef.rs").read_text(encoding="utf-8")

        for test_name in (
            "test_adaptive_ef_100k",
            "test_adaptive_ef_different_top_k",
            "test_adaptive_ef_full",
        ):
            self.assertIn(
                f"#[ignore]\nfn {test_name}()",
                contents,
                msg=(
                    f"{test_name} is a long-running adaptive-ef benchmark flow and must "
                    "use a plain #[ignore] attribute for default regression gates"
                ),
            )


if __name__ == "__main__":
    unittest.main()
