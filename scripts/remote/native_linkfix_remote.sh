#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "usage: native_linkfix_remote.sh <rs-repo-dir> <native-src> <patched-src> <patched-build> <log-dir> <target>" >&2
    exit 1
fi

rs_repo_dir="$1"
native_src="$2"
patched_src="$3"
patched_build="$4"
log_dir="$5"
target_name="$6"

mkdir -p "${log_dir}"

worktree_log="${log_dir}/linkfix-worktree.log"
patch_log="${log_dir}/linkfix-patch.log"
configure_log="${log_dir}/linkfix-configure.log"
build_log="${log_dir}/linkfix-build.log"

git -C "${native_src}" worktree remove --force "${patched_src}" >/dev/null 2>&1 || true
rm -rf "${patched_src}" "${patched_build}"

git -C "${native_src}" worktree add --detach "${patched_src}" HEAD >"${worktree_log}" 2>&1
python3 "${rs_repo_dir}/scripts/remote/native_linkfix_patch.py" "${patched_src}" >"${patch_log}" 2>&1

for artifact in \
    sift-128-euclidean.hdf5 \
    sift-128-euclidean_HNSW_16_100_fp32.index \
    sift-128-euclidean_HNSW_16_100_fp16.index \
    sift-128-euclidean_HNSW_16_100_bf16.index
do
    if [[ -f "${native_src}/${artifact}" && ! -e "${patched_src}/${artifact}" ]]; then
        ln -s "${native_src}/${artifact}" "${patched_src}/${artifact}"
    fi
done

base="${native_src}/build/Release"
conan_toolchain="$(find "${base}" -path '*/generators/conan_toolchain.cmake' -print -quit)"
if [[ -z "${conan_toolchain}" ]]; then
    conan_toolchain="${base}/generators/conan_toolchain.cmake"
fi
gen_dir="$(dirname "${conan_toolchain}")"

cmake -S "${patched_src}" -B "${patched_build}" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${conan_toolchain}" \
  -Dfolly_DIR="${gen_dir}" \
  -Dfmt_DIR="${gen_dir}" \
  -Dglog_DIR="${gen_dir}" \
  -Dprometheus-cpp_DIR="${gen_dir}" \
  -DxxHash_DIR="${gen_dir}" \
  -Dsimde_DIR="${gen_dir}" \
  -Dopentelemetry-cpp_DIR="${gen_dir}" \
  -Dopentelemetry-proto_DIR="${gen_dir}" \
  -DCMAKE_PREFIX_PATH=/tmp/gtest-install \
  -DGTest_DIR=/tmp/gtest-install/lib/cmake/GTest \
  -DGTEST_LIBRARY=/tmp/gtest-install/lib/libgtest.a \
  -DGTEST_MAIN_LIBRARY=/tmp/gtest-install/lib/libgtest_main.a \
  -DGTEST_INCLUDE_DIR=/tmp/gtest-install/include \
  -DWITH_BENCHMARK=ON \
  -DWITH_UT=OFF \
  -DWITH_DISKANN=OFF \
  >"${configure_log}" 2>&1

cmake --build "${patched_build}" --target "${target_name}" -j4 >"${build_log}" 2>&1

printf 'patched_repo_dir=%s\n' "${patched_src}"
printf 'patched_build_dir=%s\n' "${patched_build}"
printf 'patched_bin=%s/%s/%s\n' "${patched_build}" "benchmark" "${target_name}"
printf 'worktree_log=%s\n' "${worktree_log}"
printf 'patch_log=%s\n' "${patch_log}"
printf 'configure_log=%s\n' "${configure_log}"
printf 'build_log=%s\n' "${build_log}"
