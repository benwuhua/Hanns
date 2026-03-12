#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_BUILD_DIR REMOTE_NATIVE_LOG_DIR

run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_BUILD_DIR}" "${REMOTE_NATIVE_LOG_DIR}" "${REMOTE_NATIVE_REPO_URL}" "${REMOTE_NATIVE_DEFAULT_BRANCH}" <<'EOF'
set -euo pipefail

repo_root="$1"
src="$2"
build="$3"
log_dir="$4"
repo_url="$5"
repo_branch="$6"
base="${src}/build/Release"
gtest_build=/tmp/gtest-build
ngtest_install=/tmp/gtest-install
gtest_src=/tmp/googletest-src
gtest_config="${ngtest_install}/lib/cmake/GTest/GTestConfig.cmake"
gtest_lib_dir="${ngtest_install}/lib"
bootstrap_venv="${HOME}/.local/share/knowhere-native-bootstrap/conan-venv"
if [[ -x "${bootstrap_venv}/bin/conan" ]]; then
  export PATH="${bootstrap_venv}/bin:${PATH}"
fi
conan_toolchain="$(find "${base}" -path '*/generators/conan_toolchain.cmake' -print -quit)"
if [[ -z "${conan_toolchain}" ]]; then
  conan_toolchain="${base}/generators/conan_toolchain.cmake"
fi
preflight_log="${log_dir}/preflight.log"

mkdir -p "${build}" "${log_dir}"

if [[ ! -d "${src}/.git" ]]; then
  if [[ -z "${repo_url}" ]]; then
    echo "missing remote native repo url for isolated native benchmark source" >&2
    exit 1
  fi
  rm -rf "${src}"
  mkdir -p "$(dirname "${src}")"
  git clone --depth=1 --branch "${repo_branch}" "${repo_url}" "${src}" \
    > "${log_dir}/native-clone.log" 2>&1
fi

if [[ ! -f "${gtest_config}" ]]; then
  rm -rf "${gtest_build}" "${ngtest_install}"
  if [[ -d /usr/src/googletest ]]; then
    gtest_source=/usr/src/googletest
  else
    rm -rf "${gtest_src}"
    git clone --depth=1 --branch v1.14.0 https://github.com/google/googletest.git "${gtest_src}" \
      > "${log_dir}/gtest-clone.log" 2>&1
    gtest_source="${gtest_src}"
  fi
  cmake -S "${gtest_source}" -B "${gtest_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ngtest_install}" \
    > "${log_dir}/gtest-configure.log" 2>&1
  cmake --build "${gtest_build}" -j4 \
    > "${log_dir}/gtest-build.log" 2>&1
  cmake --install "${gtest_build}" \
    > "${log_dir}/gtest-install.log" 2>&1
fi

{
  echo "[preflight] src=${src}"
  echo "[preflight] build=${build}"
  echo "[preflight] conan_toolchain=${conan_toolchain}"
  echo "[preflight] gtest_config=${gtest_config}"
} > "${preflight_log}"

if ! command -v conan >/dev/null 2>&1; then
  echo "missing conan on remote PATH; native benchmark build requires Conan-generated toolchain at ${conan_toolchain}" | tee -a "${preflight_log}" >&2
  printf 'preflight_log=%s\n' "${preflight_log}"
  exit 127
fi

if [[ ! -f "${conan_toolchain}" ]]; then
  echo "missing Conan toolchain: ${conan_toolchain}; run the native repo's Conan install/bootstrap before probing benchmark_float_qps" | tee -a "${preflight_log}" >&2
  printf 'preflight_log=%s\n' "${preflight_log}"
  exit 127
fi

conan_generators_dir="$(dirname "${conan_toolchain}")"

cmake -S "${src}" -B "${build}" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${conan_toolchain}" \
  -Dfolly_DIR="${conan_generators_dir}" \
  -Dfmt_DIR="${conan_generators_dir}" \
  -Dglog_DIR="${conan_generators_dir}" \
  -Dprometheus-cpp_DIR="${conan_generators_dir}" \
  -DxxHash_DIR="${conan_generators_dir}" \
  -Dsimde_DIR="${conan_generators_dir}" \
  -Dopentelemetry-cpp_DIR="${conan_generators_dir}" \
  -Dopentelemetry-proto_DIR="${conan_generators_dir}" \
  -DCMAKE_PREFIX_PATH="${ngtest_install}" \
  -DGTest_DIR="${ngtest_install}/lib/cmake/GTest" \
  -DGTEST_LIBRARY="${gtest_lib_dir}/libgtest.a" \
  -DGTEST_MAIN_LIBRARY="${gtest_lib_dir}/libgtest_main.a" \
  -DGTEST_INCLUDE_DIR="${ngtest_install}/include" \
  -DWITH_BENCHMARK=ON \
  -DWITH_UT=OFF \
  -DWITH_DISKANN=OFF \
  > "${log_dir}/configure.log" 2>&1

set +e
cmake --build "${build}" --target benchmark_float_qps -j4 \
  > "${log_dir}/build.log" 2>&1
build_rc=$?
set -e

printf 'build_dir=%s\n' "${build}"
printf 'configure_log=%s\n' "${log_dir}/configure.log"
printf 'build_log=%s\n' "${log_dir}/build.log"
printf 'native_repo_dir=%s\n' "${src}"
printf 'gtest_config=%s\n' "${gtest_config}"
printf 'build_exit_code=%s\n' "${build_rc}"

if [[ ${build_rc} -eq 0 ]]; then
  benchmark_bin="${build}/benchmark/benchmark_float_qps"
  runtime_libs="$({
    printf '%s\n' "${build}"
    printf '%s\n' "${build}/milvus-common-build"
    readelf -d "${benchmark_bin}" 2>/dev/null \
      | sed -n 's/.*\[\(.*\)\].*/\1/p' \
      | tr ':' '\n'
  } | awk 'NF' | awk '!seen[$0]++' | paste -sd ':' -)"
  if [[ -z "${runtime_libs}" ]]; then
    echo 'failed to resolve runtime library directories for benchmark_float_qps' >&2
    exit 127
  fi
  list_log="${log_dir}/gtest_list.log"
  export LD_LIBRARY_PATH="${runtime_libs}:${LD_LIBRARY_PATH:-}"
  set +e
  "${benchmark_bin}" --gtest_list_tests > "${list_log}" 2>&1
  list_rc=$?
  set -e
  if [[ ${list_rc} -ne 0 ]] && grep -q 'Metric name grpc.xds_client.resource_updates_valid has already been registered.' "${list_log}"; then
    patched_src="/data/work/knowhere-native-linkfix-src"
    patched_build="/data/work/knowhere-native-linkfix-build"
    linkfix_meta="$(bash "${repo_root}/scripts/remote/native_linkfix_remote.sh" \
      "${repo_root}" "${src}" "${patched_src}" "${patched_build}" "${log_dir}" "benchmark_float_qps")"
    patched_repo_dir="$(awk -F= '/^patched_repo_dir=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_build_dir="$(awk -F= '/^patched_build_dir=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_bin="$(awk -F= '/^patched_bin=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_link_txt="${patched_build_dir}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
    patched_link_rpath="$(python3 - "${patched_link_txt}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'-Wl,-rpath,([^ ]+)', text)
print(m.group(1) if m else '')
PY
)"
    export LD_LIBRARY_PATH="${patched_link_rpath}:${patched_build_dir}:${patched_build_dir}/milvus-common-build:${LD_LIBRARY_PATH:-}"
    fallback_log="${log_dir}/gtest_list_linkfix.log"
    (cd "${patched_repo_dir}" && "${patched_bin}" --gtest_list_tests > "${fallback_log}" 2>&1)
    printf 'fallback=linkfix_worktree\n'
    printf 'patched_repo_dir=%s\n' "${patched_repo_dir}"
    printf 'patched_build_dir=%s\n' "${patched_build_dir}"
    printf 'patched_bin=%s\n' "${patched_bin}"
    printf 'gtest_list_log=%s\n' "${fallback_log}"
  elif [[ ${list_rc} -eq 0 ]]; then
    printf 'gtest_list_log=%s\n' "${list_log}"
  else
    exit "${list_rc}"
  fi
  printf 'runtime_libs=%s\n' "${runtime_libs}"
fi
EOF
