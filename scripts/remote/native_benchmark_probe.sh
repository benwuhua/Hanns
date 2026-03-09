#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER

run_remote_script <<'EOF'
set -euo pipefail

src=/data/work/knowhere-src
base=/data/work/knowhere-src/build/Release
build=/data/work/knowhere-build-benchmark
log_dir="${build}"
gtest_build=/tmp/gtest-build
ngtest_install=/tmp/gtest-install

rm -rf "${build}"
mkdir -p "${build}"

if [[ ! -f "${ngtest_install}/lib/x86_64-linux-gnu/cmake/GTest/GTestConfig.cmake" ]]; then
  rm -rf "${gtest_build}" "${ngtest_install}"
  cmake -S /usr/src/googletest -B "${gtest_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ngtest_install}" \
    > "${log_dir}/gtest-configure.log" 2>&1
  cmake --build "${gtest_build}" -j4 \
    > "${log_dir}/gtest-build.log" 2>&1
  cmake --install "${gtest_build}" \
    > "${log_dir}/gtest-install.log" 2>&1
fi

cmake -S "${src}" -B "${build}" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${base}/generators/conan_toolchain.cmake" \
  -Dfolly_DIR="${base}/generators" \
  -Dfmt_DIR="${base}/generators" \
  -Dglog_DIR="${base}/generators" \
  -Dprometheus-cpp_DIR="${base}/generators" \
  -DxxHash_DIR="${base}/generators" \
  -Dsimde_DIR="${base}/generators" \
  -Dopentelemetry-cpp_DIR="${base}/generators" \
  -Dopentelemetry-proto_DIR="${base}/generators" \
  -DCMAKE_PREFIX_PATH="${ngtest_install}" \
  -DGTest_DIR="${ngtest_install}/lib/x86_64-linux-gnu/cmake/GTest" \
  -DGTEST_LIBRARY="${ngtest_install}/lib/x86_64-linux-gnu/libgtest.so" \
  -DGTEST_MAIN_LIBRARY="${ngtest_install}/lib/x86_64-linux-gnu/libgtest_main.so" \
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
printf 'gtest_config=%s\n' "${ngtest_install}/lib/x86_64-linux-gnu/cmake/GTest/GTestConfig.cmake"
printf 'build_exit_code=%s\n' "${build_rc}"

if [[ ${build_rc} -eq 0 ]]; then
  "${build}/benchmark/benchmark_float_qps" --gtest_list_tests \
    > "${log_dir}/gtest_list.log" 2>&1
  printf 'gtest_list_log=%s\n' "${log_dir}/gtest_list.log"
fi
EOF
