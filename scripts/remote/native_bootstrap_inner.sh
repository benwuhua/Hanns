#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: native_bootstrap_inner.sh <native-src> <native-log-dir> [with-diskann] [conan-home] [tmp-dir]" >&2
    exit 1
fi

src="$1"
log_dir="$2"
with_diskann="${3:-True}"
conan_home="${4:-/data/work/knowhere-native-conan2}"
tmp_dir="${5:-/data/work/knowhere-native-tmp}"
base="${src}/build/Release"
venv_root="${HOME}/.local/share/knowhere-native-bootstrap"
venv_dir="${venv_root}/conan-venv"
conan_bin="${venv_dir}/bin/conan"
conan_remote_url="https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2"
bootstrap_log="${log_dir}/native-bootstrap.log"
profile_log="${log_dir}/native-conan-profile.log"
remote_log="${log_dir}/native-conan-remote.log"
install_log="${log_dir}/native-conan-install.log"

mkdir -p "${log_dir}" "${base}" "${venv_root}" "${conan_home}" "${tmp_dir}"
export CONAN_HOME="${conan_home}"
export TMPDIR="${tmp_dir}"

if [[ ! -x "${conan_bin}" ]]; then
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/pip" install --upgrade pip > "${log_dir}/native-bootstrap-pip-upgrade.log" 2>&1
  "${venv_dir}/bin/pip" install 'conan>=2,<3' > "${bootstrap_log}" 2>&1
fi

export PATH="${venv_dir}/bin:${PATH}"
if ! conan --version | grep -q 'Conan version 2\.'; then
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/pip" install --upgrade pip > "${log_dir}/native-bootstrap-pip-upgrade.log" 2>&1
  "${venv_dir}/bin/pip" install 'conan>=2,<3' > "${bootstrap_log}" 2>&1
  export PATH="${venv_dir}/bin:${PATH}"
fi

if [[ ! -f "${CONAN_HOME}/profiles/default" ]]; then
  conan profile detect --force > "${profile_log}" 2>&1
else
  conan profile show -pr default > "${profile_log}" 2>&1 || true
fi

conan remote add default-conan-local2 "${conan_remote_url}" > "${remote_log}" 2>&1 || true

cd "${base}"
conan install ../.. \
  --output-folder . \
  --build=missing \
  --update \
  -s build_type=Release \
  -s compiler.cppstd=20 \
  -s:b compiler.cppstd=20 \
  -s compiler.libcxx=libstdc++11 \
  -s:b compiler.libcxx=libstdc++11 \
  -o '&:with_benchmark=True' \
  -o '&:with_ut=False' \
  -o '&:with_diskann='"${with_diskann}" \
  -o '&:with_profiler=False' \
  -o '&:with_coverage=False' \
  -o '&:with_faiss_tests=False' \
  > "${install_log}" 2>&1

toolchain_path="$(find "${base}" -path '*/generators/conan_toolchain.cmake' -print -quit)"
if [[ -z "${toolchain_path}" ]]; then
  toolchain_path="${base}/generators/conan_toolchain.cmake"
fi

printf 'conan=%s\n' "$(conan --version)"
printf 'conan_bin=%s\n' "${conan_bin}"
printf 'profile_log=%s\n' "${profile_log}"
printf 'remote_log=%s\n' "${remote_log}"
printf 'install_log=%s\n' "${install_log}"
printf 'toolchain=%s\n' "${toolchain_path}"
