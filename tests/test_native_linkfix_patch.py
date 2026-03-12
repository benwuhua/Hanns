import pathlib
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / "scripts" / "remote" / "native_linkfix_patch.py"


class NativeLinkfixPatchTests(unittest.TestCase):
    def test_patch_rewrites_known_linkage_patterns_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            (repo / "benchmark").mkdir()

            root_cmake = repo / "CMakeLists.txt"
            bench_cmake = repo / "benchmark" / "CMakeLists.txt"

            root_cmake.write_text(
                textwrap.dedent(
                    """\
                    add_library(knowhere SHARED ${KNOWHERE_SRCS})
                    target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})
                    """
                ),
                encoding="utf-8",
            )
            bench_cmake.write_text(
                textwrap.dedent(
                    """\
                    set(depend_libs
                            knowhere
                            ${HDF5_LIBRARIES}
                            )

                    macro(benchmark_test target file)
                        set(FILE_SRCS ${file})
                        add_executable(${target} ${FILE_SRCS})
                        target_link_libraries(${target} ${depend_libs} ${unittest_libs})

                        # this is needed for clang in Debug compilation mode
                        if(NOT APPLE)
                            target_link_libraries(${target} atomic)
                        endif()
                    endmacro()
                    """
                ),
                encoding="utf-8",
            )

            first = subprocess.run(
                ["python3", str(PATCH_SCRIPT), str(repo)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr)

            second = subprocess.run(
                ["python3", str(PATCH_SCRIPT), str(repo)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.returncode, 0, msg=second.stderr)

            root_text = root_cmake.read_text(encoding="utf-8")
            bench_text = bench_cmake.read_text(encoding="utf-8")

            self.assertIn(
                "target_link_libraries(knowhere PRIVATE ${KNOWHERE_LINKER_LIBS})",
                root_text,
            )
            self.assertNotIn(
                "target_link_libraries(knowhere PUBLIC ${KNOWHERE_LINKER_LIBS})",
                root_text,
            )
            self.assertEqual(root_text.count("PRIVATE ${KNOWHERE_LINKER_LIBS}"), 1)

            self.assertIn("nlohmann_json::nlohmann_json", bench_text)
            self.assertIn("glog::glog", bench_text)
            self.assertIn("Folly::folly", bench_text)
            self.assertIn("Boost::boost", bench_text)
            self.assertIn("target_include_directories(${target} PRIVATE", bench_text)
            self.assertIn("${prometheus-cpp_INCLUDE_DIRS}", bench_text)
            self.assertIn("${opentelemetry-cpp_INCLUDE_DIRS}", bench_text)
            self.assertEqual(bench_text.count("target_include_directories(${target} PRIVATE"), 1)


if __name__ == "__main__":
    unittest.main()
