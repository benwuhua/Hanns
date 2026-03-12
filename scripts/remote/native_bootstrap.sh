#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_LOG_DIR

run_remote_script "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_LOG_DIR}" <<'EOF'
set -euo pipefail

src="$1"
log_dir="$2"
base="${src}/build/Release"
venv_root="${HOME}/.local/share/knowhere-native-bootstrap"
venv_dir="${venv_root}/conan-venv"
conan_bin="${venv_dir}/bin/conan"
bootstrap_log="${log_dir}/native-bootstrap.log"
profile_log="${log_dir}/native-conan-profile.log"
install_log="${log_dir}/native-conan-install.log"

mkdir -p "${log_dir}" "${base}" "${venv_root}"

if [[ ! -x "${conan_bin}" ]]; then
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/pip" install --upgrade pip > "${log_dir}/native-bootstrap-pip-upgrade.log" 2>&1
  "${venv_dir}/bin/pip" install 'conan<2' > "${bootstrap_log}" 2>&1
fi

export PATH="${venv_dir}/bin:${PATH}"
if ! conan --version | grep -q 'Conan version 1\.'; then
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/pip" install --upgrade pip > "${log_dir}/native-bootstrap-pip-upgrade.log" 2>&1
  "${venv_dir}/bin/pip" install 'conan<2' > "${bootstrap_log}" 2>&1
fi

if [[ ! -f "${HOME}/.conan2/profiles/default" ]]; then
  conan profile detect --force > "${profile_log}" 2>&1
else
  conan profile path default > "${profile_log}" 2>&1 || true
fi

cd "${base}"
conan install ../.. \
  --output-folder . \
  --build=missing \
  --update \
  -s build_type=Release \
  -s compiler.cppstd=17 \
  -s compiler.libcxx=libstdc++11 \
  -o with_benchmark=True \
  -o with_ut=False \
  -o with_diskann=False \
  -o with_profiler=False \
  -o with_coverage=False \
  -o with_faiss_tests=False \
  > "${install_log}" 2>&1

toolchain_path="$(find "${base}" -path '*/generators/conan_toolchain.cmake' -print -quit)"
if [[ -z "${toolchain_path}" ]]; then
  toolchain_path="${base}/generators/conan_toolchain.cmake"
fi

printf 'conan=%s\n' "$(conan --version)"
printf 'conan_bin=%s\n' "${conan_bin}"
printf 'profile_log=%s\n' "${profile_log}"
printf 'install_log=%s\n' "${install_log}"
printf 'toolchain=%s\n' "${toolchain_path}"
EOF
