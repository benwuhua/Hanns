import os
import pathlib
import subprocess
import tempfile
import textwrap
import time
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "remote" / "remote_background.sh"


class RemoteBackgroundTests(unittest.TestCase):
    def test_start_wait_tail_with_fake_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            fake_ssh = fake_bin / "ssh"

            fake_ssh.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail

                    args=("$@")
                    target_index=-1
                    i=0
                    while [[ ${i} -lt ${#args[@]} ]]; do
                        case "${args[$i]}" in
                            -o|-p|-i)
                                ((i += 2))
                                ;;
                            -*)
                                ((i += 1))
                                ;;
                            *)
                                target_index=${i}
                                break
                                ;;
                        esac
                    done

                    if [[ ${target_index} -lt 0 ]]; then
                        echo "fake ssh: missing target" >&2
                        exit 2
                    fi

                    cmd_parts=("${args[@]:$((target_index + 1))}")
                    remote_cmd="${cmd_parts[*]}"
                    stdin_copy="$(mktemp)"
                    cat >"${stdin_copy}"
                    bash -lc "${remote_cmd}" <"${stdin_copy}"
                    """
                ),
                encoding="utf-8",
            )
            fake_ssh.chmod(0o755)

            remote_root = tmp_path / "remote"
            repo_dir = remote_root / "repo"
            log_dir = remote_root / "logs"
            repo_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["HANNS_REMOTE_HOST"] = "dummy-host"
            env["HANNS_REMOTE_USER"] = "dummy-user"
            env["HANNS_REMOTE_PORT"] = "22"
            env["HANNS_SSH_IDENTITY_FILE"] = ""
            env["HANNS_REMOTE_REPO_DIR"] = str(repo_dir)
            env["HANNS_REMOTE_LOG_DIR"] = str(log_dir)

            start = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "start",
                    "--repo-dir",
                    str(repo_dir),
                    "--log-dir",
                    str(log_dir),
                    "--command",
                    "echo background_ok && sleep 1 && echo done",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(start.returncode, 0, msg=start.stderr)

            metadata = {}
            for line in start.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key] = value

            self.assertIn("status_file", metadata)
            self.assertIn("log", metadata)

            time.sleep(2)
            wait = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "wait",
                    "--status-file",
                    metadata["status_file"],
                    "--poll-interval",
                    "1",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(wait.returncode, 0, msg=wait.stderr)
            self.assertIn("status=ok", wait.stdout)

            tail = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "tail",
                    "--log",
                    metadata["log"],
                    "--lines",
                    "20",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tail.returncode, 0, msg=tail.stderr)
            self.assertIn("background_ok", tail.stdout)
            self.assertIn("done", tail.stdout)


if __name__ == "__main__":
    unittest.main()
