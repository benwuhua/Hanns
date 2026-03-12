import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_RS = REPO_ROOT / "build.rs"
CARGO_TOML = REPO_ROOT / "Cargo.toml"
INIT_SH = REPO_ROOT / "init.sh"
REMOTE_TEST_SH = REPO_ROOT / "scripts" / "remote" / "test.sh"
REMOTE_ENV_SH = REPO_ROOT / "scripts" / "remote" / "remote_env.sh"


class ProdGateHygieneTests(unittest.TestCase):
    def test_faiss_cxx_build_path_declares_cxx_build_dependency(self) -> None:
        build_rs = BUILD_RS.read_text(encoding="utf-8")
        cargo_toml = CARGO_TOML.read_text(encoding="utf-8")

        self.assertIn(
            '#[cfg(feature = "faiss-cxx")]',
            build_rs,
            msg="build.rs must keep the faiss-cxx bridge behind an explicit feature gate",
        )
        self.assertIn(
            "cxx_build::bridge",
            build_rs,
            msg="build.rs hygiene test only makes sense while the build script still uses cxx_build",
        )
        self.assertRegex(
            cargo_toml,
            re.compile(r"(?ms)^\[build-dependencies\]\n.*^cxx-build\s*="),
            msg=(
                "Cargo.toml must declare cxx-build in [build-dependencies] so "
                "clippy --all-features can compile build.rs when faiss-cxx is enabled"
            ),
        )

    def test_remote_bootstrap_and_test_wrapper_ensure_rustfmt_and_clippy(self) -> None:
        init_sh = INIT_SH.read_text(encoding="utf-8")
        remote_test_sh = REMOTE_TEST_SH.read_text(encoding="utf-8")
        remote_env_sh = REMOTE_ENV_SH.read_text(encoding="utf-8")

        self.assertIn(
            "ensure_remote_rust_components()",
            remote_env_sh,
            msg="remote_env.sh must expose a reusable rust component bootstrap helper",
        )
        self.assertIn(
            "ensure_remote_rust_components rustfmt clippy",
            init_sh,
            msg="init.sh must provision rustfmt and clippy on the remote authority toolchain",
        )
        self.assertIn(
            "ensure_remote_rust_components rustfmt clippy",
            remote_test_sh,
            msg="remote test wrapper must self-bootstrap rustfmt and clippy before running gate commands",
        )


if __name__ == "__main__":
    unittest.main()
