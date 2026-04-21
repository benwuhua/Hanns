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


if __name__ == "__main__":
    unittest.main()
