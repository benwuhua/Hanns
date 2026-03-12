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

cmake_backup="CMakeLists.txt.bak.builder-public-header-probe"
index_node_backup="include/knowhere/index/index_node.h.bak.builder-public-header-probe"
sparse_index_node_backup="src/index/sparse/sparse_index_node.cc.bak.builder-public-header-probe"
cp CMakeLists.txt "${cmake_backup}"
cp include/knowhere/index/index_node.h "${index_node_backup}"
cp src/index/sparse/sparse_index_node.cc "${sparse_index_node_backup}"
restore() {
  if [[ -f "${cmake_backup}" ]]; then
    mv "${cmake_backup}" CMakeLists.txt
  fi
  if [[ -f "${index_node_backup}" ]]; then
    mv "${index_node_backup}" include/knowhere/index/index_node.h
  fi
  if [[ -f "${sparse_index_node_backup}" ]]; then
    mv "${sparse_index_node_backup}" src/index/sparse/sparse_index_node.cc
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

p = Path('include/knowhere/index/index_node.h')
text = p.read_text()
old = '''#if defined(NOT_COMPILE_FOR_SWIG)
#include "common/OpContext.h"
#else
namespace milvus {
struct OpContext;
}  // namespace milvus
#endif
'''
new = '''namespace milvus {
struct OpContext;
}  // namespace milvus
'''
if old not in text:
    raise SystemExit('missing OpContext include block in index_node.h')
text = text.replace(old, new, 1)
old = '''#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif
'''
new = '''#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
// builder probe: keep heavy runtime/task headers out of benchmark-facing public surface
#endif
'''
if old not in text:
    raise SystemExit('missing heavy public include block in index_node.h')
text = text.replace(old, new, 1)

replacements = [
    (
        '''        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                update_next_func();
            }));
            WaitAllSuccess(futs);
#else
            update_next_func();
#endif
        } else {
            update_next_func();
        }
''',
        '''        if (use_knowhere_search_pool_) {
            update_next_func();
        } else {
            update_next_func();
        }
'''
    ),
    (
        '''        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                sort_next();
            }));
            WaitAllSuccess(futs);
#else
            sort_next();
#endif
        } else {
            sort_next();
        }
''',
        '''        if (use_knowhere_search_pool_) {
            sort_next();
        } else {
            sort_next();
        }
'''
    ),
    (
        '''        if (use_knowhere_search_pool_) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&]() {
                ThreadPool::ScopedSearchOmpSetter setter(1);
                results_ = compute_dist_func_();
            }));
            WaitAllSuccess(futs);
#else
            results_ = compute_dist_func_();
#endif
        } else {
            results_ = compute_dist_func_();
        }
''',
        '''        if (use_knowhere_search_pool_) {
            results_ = compute_dist_func_();
        } else {
            results_ = compute_dist_func_();
        }
'''
    ),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit('missing inline runtime block in index_node.h for builder probe')
    text = text.replace(old, new, 1)
p.write_text(text)

p = Path('src/index/sparse/sparse_index_node.cc')
text = p.read_text()
include = '#include "knowhere/comp/task.h"\n'
if include not in text:
    marker = '#include "knowhere/index/index_node.h"\n'
    if marker not in text:
        raise SystemExit('missing sparse_index_node.cc include anchor')
    text = text.replace(marker, marker + include, 1)
p.write_text(text)
PY

conan_toolchain="$(find "${src}/build/Release" -path '*/generators/conan_toolchain.cmake' -print -quit)"
conan_generators_dir="$(dirname "${conan_toolchain}")"
configure_log="${log_dir}/public_header_probe.configure.log"
build_log="${log_dir}/public_header_probe.build.log"
gtest_log="${log_dir}/public_header_probe.gtest_list.log"

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
  tail -60 "${build_log}"
  exit 0
fi

link_txt="${build}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
printf 'link_txt=%s\n' "${link_txt}"
link_rpath="$(python3 - "${link_txt}" <<'PY2'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'-Wl,-rpath,([^ ]+)', text)
print(m.group(1) if m else '')
PY2
)"
export LD_LIBRARY_PATH="${link_rpath}:${build}:${build}/milvus-common-build:${LD_LIBRARY_PATH:-}"
set +e
"${build}/benchmark/benchmark_float_qps" --gtest_list_tests > "${gtest_log}" 2>&1
gtest_rc=$?
set -e
printf 'gtest_log=%s\n' "${gtest_log}"
printf 'gtest_exit_code=%s\n' "${gtest_rc}"
if [[ ${gtest_rc} -ne 0 ]]; then
  tail -60 "${gtest_log}"
fi
EOF
