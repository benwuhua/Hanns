#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_BUILD_DIR REMOTE_NATIVE_LOG_DIR

run_remote_script "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_BUILD_DIR}" "${REMOTE_NATIVE_LOG_DIR}" <<'EOF'
set -euo pipefail

src="$1"
build="$2"
log_dir="$3"
mkdir -p "${log_dir}"
cd "${src}"

backup="CMakeLists.txt.bak.builder-usage-probe"
cp CMakeLists.txt "${backup}"
restore() {
  if [[ -f "${backup}" ]]; then
    mv "${backup}" CMakeLists.txt
  fi
}
trap restore EXIT

python3 - <<'PY'
from pathlib import Path
p = Path('CMakeLists.txt')
text = p.read_text()
text = text.replace(
    'set(KNOWHERE_LINKER_LIBS "")\nset(CARDINAL_LIBS "")',
    'set(KNOWHERE_PUBLIC_LINKER_LIBS "")\nset(KNOWHERE_PRIVATE_LINKER_LIBS "")\nset(CARDINAL_LIBS "")',
    1,
)
repls = {
    'list(APPEND KNOWHERE_LINKER_LIBS Boost::boost)': 'list(APPEND KNOWHERE_PUBLIC_LINKER_LIBS Boost::boost)',
    'list(APPEND KNOWHERE_LINKER_LIBS nlohmann_json::nlohmann_json)': 'list(APPEND KNOWHERE_PUBLIC_LINKER_LIBS nlohmann_json::nlohmann_json)',
    'list(APPEND KNOWHERE_LINKER_LIBS glog::glog)': 'list(APPEND KNOWHERE_PUBLIC_LINKER_LIBS glog::glog)',
    'list(APPEND KNOWHERE_LINKER_LIBS simde::simde)': 'list(APPEND KNOWHERE_PUBLIC_LINKER_LIBS simde::simde)',
    'list(APPEND KNOWHERE_LINKER_LIBS faiss)': 'list(APPEND KNOWHERE_PRIVATE_LINKER_LIBS faiss)',
    'list(APPEND KNOWHERE_LINKER_LIBS prometheus-cpp::core prometheus-cpp::push)': 'list(APPEND KNOWHERE_PRIVATE_LINKER_LIBS prometheus-cpp::core prometheus-cpp::push)',
    'list(APPEND KNOWHERE_LINKER_LIBS fmt::fmt)': 'list(APPEND KNOWHERE_PRIVATE_LINKER_LIBS fmt::fmt)',
    'list(APPEND KNOWHERE_LINKER_LIBS Folly::folly)': 'list(APPEND KNOWHERE_PRIVATE_LINKER_LIBS Folly::folly)',
    'list(APPEND KNOWHERE_LINKER_LIBS milvus-common)': 'list(APPEND KNOWHERE_PRIVATE_LINKER_LIBS milvus-common)',
}
for old, new in repls.items():
    if old not in text:
        raise SystemExit(f'missing replacement: {old}')
    text = text.replace(old, new, 1)
text = text.replace(
    'add_library(knowhere SHARED ${KNOWHERE_SRCS})\nadd_dependencies(knowhere ${KNOWHERE_LINKER_LIBS})',
    'add_library(knowhere SHARED ${KNOWHERE_SRCS})\nadd_dependencies(knowhere ${KNOWHERE_PUBLIC_LINKER_LIBS} ${KNOWHERE_PRIVATE_LINKER_LIBS})',
    1,
)
text = text.replace(
    '    APPEND\n    KNOWHERE_LINKER_LIBS\n    raft::raft\n    cuvs::cuvs\n    CUDA::cublas\n    CUDA::cusparse\n    CUDA::cusolver)',
    '    APPEND\n    KNOWHERE_PRIVATE_LINKER_LIBS\n    raft::raft\n    cuvs::cuvs\n    CUDA::cublas\n    CUDA::cusparse\n    CUDA::cusolver)',
    1,
)
text = text.replace(
    'target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})',
    'target_link_libraries(knowhere\n  PUBLIC ${KNOWHERE_PUBLIC_LINKER_LIBS}\n  PRIVATE ${KNOWHERE_PRIVATE_LINKER_LIBS})',
    1,
)
p.write_text(text)
PY

conan_toolchain="$(find "${src}/build/Release" -path '*/generators/conan_toolchain.cmake' -print -quit)"
conan_generators_dir="$(dirname "${conan_toolchain}")"
configure_log="${log_dir}/usage_requirements_probe.configure.log"
build_log="${log_dir}/usage_requirements_probe.build.log"
gtest_log="${log_dir}/usage_requirements_probe.gtest_list.log"

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
  -DCMAKE_PREFIX_PATH=/tmp/gtest-install \
  -DGTest_DIR=/tmp/gtest-install/lib/cmake/GTest \
  -DGTEST_LIBRARY=/tmp/gtest-install/lib/libgtest.a \
  -DGTEST_MAIN_LIBRARY=/tmp/gtest-install/lib/libgtest_main.a \
  -DGTEST_INCLUDE_DIR=/tmp/gtest-install/include \
  -DWITH_BENCHMARK=ON \
  -DWITH_UT=OFF \
  -DWITH_DISKANN=OFF \
  > "${configure_log}" 2>&1

set +e
cmake --build "${build}" --target benchmark_float_qps -j4 > "${build_log}" 2>&1
build_rc=$?
set -e

printf 'configure_log=%s\n' "${configure_log}"
printf 'build_log=%s\n' "${build_log}"
printf 'build_exit_code=%s\n' "${build_rc}"

if [[ ${build_rc} -ne 0 ]]; then
  tail -40 "${build_log}"
  exit 0
fi

link_txt="${build}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
printf 'link_txt=%s\n' "${link_txt}"
export LD_LIBRARY_PATH="${build}:${build}/milvus-common-build:${LD_LIBRARY_PATH:-}"
set +e
"${build}/benchmark/benchmark_float_qps" --gtest_list_tests > "${gtest_log}" 2>&1
gtest_rc=$?
set -e
printf 'gtest_log=%s\n' "${gtest_log}"
printf 'gtest_exit_code=%s\n' "${gtest_rc}"
EOF
