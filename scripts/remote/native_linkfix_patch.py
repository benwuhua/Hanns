#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys


ROOT_OLD = "target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})"
ROOT_NEW = "target_link_libraries(knowhere PRIVATE ${KNOWHERE_LINKER_LIBS})"

BENCH_DEPEND_OLD = """set(depend_libs
        knowhere
        ${HDF5_LIBRARIES}
        )"""

BENCH_DEPEND_NEW = """set(depend_libs
        knowhere
        nlohmann_json::nlohmann_json
        glog::glog
        fmt::fmt
        xxHash::xxhash
        Folly::folly
        simde::simde
        Boost::boost
        ${HDF5_LIBRARIES}
        )"""

BENCH_LINK_NEEDLE = """    target_link_libraries(${target} ${depend_libs} ${unittest_libs})

    # this is needed for clang in Debug compilation mode
"""

BENCH_LINK_REPLACEMENT = """    target_link_libraries(${target} ${depend_libs} ${unittest_libs})
    target_include_directories(${target} PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${glog_INCLUDE_DIRS}
        ${xxHash_INCLUDE_DIRS}
        ${folly_INCLUDE_DIRS}
        ${opentelemetry-cpp_INCLUDE_DIRS}
        ${prometheus-cpp_INCLUDE_DIRS})

    # this is needed for clang in Debug compilation mode
"""


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"{label}: expected pattern not found")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: native_linkfix_patch.py <native-repo-dir>")

    repo = pathlib.Path(sys.argv[1]).resolve()
    root_cmake = repo / "CMakeLists.txt"
    bench_cmake = repo / "benchmark" / "CMakeLists.txt"

    root_text = root_cmake.read_text(encoding="utf-8")
    root_text = replace_once(root_text, ROOT_OLD, ROOT_NEW, label="root CMakeLists.txt")
    root_cmake.write_text(root_text, encoding="utf-8")

    bench_text = bench_cmake.read_text(encoding="utf-8")
    bench_text = replace_once(
        bench_text,
        BENCH_DEPEND_OLD,
        BENCH_DEPEND_NEW,
        label="benchmark/CMakeLists.txt depend_libs",
    )
    bench_text = replace_once(
        bench_text,
        BENCH_LINK_NEEDLE,
        BENCH_LINK_REPLACEMENT,
        label="benchmark/CMakeLists.txt benchmark_test macro",
    )
    bench_cmake.write_text(bench_text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
