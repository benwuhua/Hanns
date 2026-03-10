import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class RemoteCommonRunRemoteScriptTests(unittest.TestCase):
    def test_run_remote_script_preserves_empty_and_spaced_args(self) -> None:
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

            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"

            command = textwrap.dedent(
                """\
                source scripts/remote/common.sh
                REMOTE_HOST=dummy-host
                REMOTE_USER=dummy-user
                REMOTE_PORT=22
                SSH_IDENTITY_FILE=
                run_remote_script "alpha" "" "two words" "cargo test --lib -q" <<'EOF'
                printf 'arg1=%s\\n' "$1"
                printf 'arg2=%s\\n' "$2"
                printf 'arg3=%s\\n' "$3"
                printf 'arg4=%s\\n' "$4"
                printf 'argc=%s\\n' "$#"
                EOF
                """
            )

            result = subprocess.run(
                ["bash", "-c", command],
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
            self.assertEqual(
                result.stdout,
                "arg1=alpha\narg2=\narg3=two words\narg4=cargo test --lib -q\nargc=4\n",
            )

    def test_remote_env_helper_expands_home_based_cargo_env_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            home_dir = tmp_path / "remote-home"
            cargo_dir = home_dir / ".cargo"
            cargo_dir.mkdir(parents=True)
            cargo_env = cargo_dir / "env"
            cargo_env.write_text(
                "export TEST_REMOTE_CARGO_ENV=loaded\n",
                encoding="utf-8",
            )

            command = textwrap.dedent(
                """\
                source scripts/remote/remote_env.sh
                load_remote_cargo_env '$HOME/.cargo/env' ''
                printf 'dollar_home=%s\\n' "${TEST_REMOTE_CARGO_ENV:-missing}"
                unset TEST_REMOTE_CARGO_ENV
                load_remote_cargo_env '~/.cargo/env' ''
                printf 'tilde_home=%s\\n' "${TEST_REMOTE_CARGO_ENV:-missing}"
                """
            )

            env = os.environ.copy()
            env["HOME"] = str(home_dir)

            result = subprocess.run(
                ["bash", "-c", command],
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
            self.assertEqual(
                result.stdout,
                "dollar_home=loaded\ntilde_home=loaded\n",
            )


if __name__ == "__main__":
    unittest.main()
