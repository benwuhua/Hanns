import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SnapshotHeaderTests(unittest.TestCase):
    def test_snapshot_header_compiles_for_c_hosts(self) -> None:
        source = textwrap.dedent(
            """
            #include <stdint.h>
            #include <stddef.h>
            #include "knowhere_snapshot.h"

            static int len_cb(void* ctx, const char* name, uint64_t* out_len) {
                (void)ctx;
                (void)name;
                *out_len = 0;
                return 0;
            }

            static int read_cb(
                void* ctx,
                const char* name,
                uint64_t offset,
                size_t len,
                uint8_t* out
            ) {
                (void)ctx;
                (void)name;
                (void)offset;
                (void)len;
                (void)out;
                return 0;
            }

            int main(void) {
                CSnapshotArtifactCallbacks callbacks = {
                    .context = 0,
                    .section_len = len_cb,
                    .read_range = read_cb,
                };
                (void)callbacks;
                (void)CSnapshotLoadMode_OwnedMemory;
                (void)knowhere_snapshot_manifest_plan;
                (void)knowhere_free_cstring;
                (void)knowhere_load_snapshot_from_callbacks;
                (void)knowhere_snapshot_runtime_search;
                (void)knowhere_snapshot_runtime_search_with_params;
                CSnapshotSearchParams params = {
                    .top_k = 3,
                    .nprobe = 16,
                };
                (void)params.top_k;
                (void)params.nprobe;
                (void)knowhere_snapshot_runtime_search_with_search_params;
                CSearchResult result = {
                    .ids = 0,
                    .distances = 0,
                    .num_results = 0,
                    .elapsed_ms = 0.0f,
                };
                (void)result.ids;
                (void)result.distances;
                (void)result.num_results;
                (void)result.elapsed_ms;
                (void)knowhere_free_result;
                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "snapshot_header_smoke.c"
            obj = Path(tmp) / "snapshot_header_smoke.o"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-I",
                    str(REPO_ROOT / "include"),
                    "-c",
                    str(src),
                    "-o",
                    str(obj),
                ],
                check=True,
            )

    def test_snapshot_header_compiles_for_cpp_hosts(self) -> None:
        source = textwrap.dedent(
            """
            #include "knowhere_snapshot.h"

            int main() {
                CSnapshotArtifactCallbacks callbacks{};
                callbacks.context = nullptr;
                callbacks.section_len = nullptr;
                callbacks.read_range = nullptr;
                (void)callbacks;
                (void)CSnapshotLoadMode_OwnedMemory;
                (void)knowhere_load_snapshot_from_callbacks;
                return 0;
            }
            """
        )

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "snapshot_header_smoke.cc"
            obj = Path(tmp) / "snapshot_header_smoke.o"
            src.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "c++",
                    "-std=c++17",
                    "-I",
                    str(REPO_ROOT / "include"),
                    "-c",
                    str(src),
                    "-o",
                    str(obj),
                ],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
