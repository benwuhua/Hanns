import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
INIT_SH = REPO_ROOT / "init.sh"


class InitShInjectionTests(unittest.TestCase):
    def test_init_sh_supports_injected_sync_and_probe_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            env_file = tmp_path / "remote.env"
            sync_marker = tmp_path / "sync.marker"
            probe_marker = tmp_path / "probe.marker"
            sync_stub = tmp_path / "sync_stub.sh"
            probe_stub = tmp_path / "probe_stub.sh"

            env_file.write_text(
                textwrap.dedent(
                    """
                    KNOWHERE_RS_REMOTE_HOST=dummy-host
                    KNOWHERE_RS_REMOTE_USER=dummy-user
                    KNOWHERE_RS_REMOTE_PORT=22
                    KNOWHERE_RS_REMOTE_REPO_DIR=/tmp/knowhere-rs-src
                    KNOWHERE_RS_REMOTE_TARGET_DIR=/tmp/knowhere-rs-target
                    KNOWHERE_RS_REMOTE_LOG_DIR=/tmp/knowhere-rs-logs
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            sync_stub.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'sync_stub\\n' > \"$1\"\n",
                encoding="utf-8",
            )
            probe_stub.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf 'probe_stub\\n' > \"$1\"\nprintf 'probe_override_ok\\n'\n",
                encoding="utf-8",
            )
            sync_stub.chmod(0o755)
            probe_stub.chmod(0o755)

            env = os.environ.copy()
            env["KNOWHERE_RS_REMOTE_ENV"] = str(env_file)
            env["KNOWHERE_RS_INIT_SYNC_CMD"] = f"bash {sync_stub} {sync_marker}"
            env["KNOWHERE_RS_INIT_PROBE_CMD"] = f"bash {probe_stub} {probe_marker}"

            result = subprocess.run(
                ["bash", str(INIT_SH)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
            )
            self.assertIn("=== knowhere-rs remote bootstrap ===", result.stdout)
            self.assertIn("probe_override_ok", result.stdout)
            self.assertTrue(sync_marker.exists(), "sync override was not executed")
            self.assertTrue(probe_marker.exists(), "probe override was not executed")


if __name__ == "__main__":
    unittest.main()
