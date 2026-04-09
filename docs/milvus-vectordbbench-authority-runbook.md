# Milvus + VectorDBBench Authority Runbook

This file records the canonical single-host authority workflow for Milvus
replacement checks on `hannsdb-x86`.

## Authority Host

- Host: `hannsdb-x86`
- Milvus integration repo: `/data/work/milvus-rs-integ/milvus-src`
- hanns integration checkout: `/data/work/milvus-rs-integ/hanns`
- Rust release target dir: `/data/work/milvus-rs-integ/hanns-target`
- VectorDBBench repo: `/data/work/VectorDBBench`
- Milvus runtime root: `/data/work/milvus-rs-integ/milvus-var`

## Canonical Startup Rule

Do not hand-roll:

- `nohup ... milvus run standalone`
- ad-hoc `LD_LIBRARY_PATH`
- ad-hoc external etcd startup

Use the checked-in remote wrapper:

- `/data/work/milvus-rs-integ/milvus-src/scripts/hanns-shim/start_standalone_remote.sh`

That wrapper sources:

- `/data/work/milvus-rs-integ/milvus-src/scripts/hanns-shim/remote_env.sh`

It is the canonical entrypoint because it already carries:

- embed etcd: `ETCD_USE_EMBED=true`
- remote toolchain paths
- library paths
- runtime directories under `/data/work/milvus-rs-integ/milvus-var`
- health wait on `http://127.0.0.1:9091/healthz`

## Canonical rs Library Rebuild

On `hannsdb-x86`:

```bash
cd /data/work/milvus-rs-integ/hanns
source "$HOME/.cargo/env" >/dev/null 2>&1 || true
CARGO_TARGET_DIR=/data/work/milvus-rs-integ/hanns-target cargo build --release --lib
```

Expected artifact:

- `/data/work/milvus-rs-integ/hanns-target/release/libhanns.so`

## Canonical Standalone Restart

On `hannsdb-x86`:

```bash
cd /data/work/milvus-rs-integ/milvus-src
scripts/hanns-shim/start_standalone_remote.sh
```

Expected success output:

- `PID=...`
- `LOG=/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log`
- `HEALTHY`

Primary Milvus log:

- `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log`

## Canonical VectorDBBench Entry Points

VectorDBBench venv:

- `/data/work/VectorDBBench/.venv`

Native vs rs Cohere 1M HNSW:

- native: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py`
- rs: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`

Other existing quick lanes:

- `/data/work/VectorDBBench/run_milvus_hnsw_500k_native.py`
- `/data/work/VectorDBBench/run_milvus_hnsw_500k_rs.py`
- `/data/work/VectorDBBench/run_milvus_hnsw_50k_native.py`
- `/data/work/VectorDBBench/run_milvus_hnsw_50k.py`

## Canonical Result Locations

VectorDBBench logs:

- `/data/work/VectorDBBench/logs/`

Milvus result JSONs:

- `/data/work/VectorDBBench/vectordb_bench/results/Milvus/`

Known reference results:

- native Cohere 1M:
  - `/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_20260330_milvus-native-knowhere-hnsw-cohere1m-20260330_milvus.json`
- rs Cohere 1M:
  - `/data/work/VectorDBBench/vectordb_bench/results/Milvus/result_20260401_milvus-hanns-hnsw-cohere1m-20260330_milvus.json`

## Recommended Session Order

1. `bash init.sh`
2. Sync or edit the Milvus integration checkout:
   - `/data/work/milvus-rs-integ/hanns`
3. Rebuild the Rust library:
   - `CARGO_TARGET_DIR=/data/work/milvus-rs-integ/hanns-target cargo build --release --lib`
4. Restart standalone only through:
   - `/data/work/milvus-rs-integ/milvus-src/scripts/hanns-shim/start_standalone_remote.sh`
5. Run the required native or rs lane by the checked-in VectorDBBench script.
6. Read authority evidence only from:
   - `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log`
   - `/data/work/VectorDBBench/logs/*.log`
   - `/data/work/VectorDBBench/vectordb_bench/results/Milvus/*.json`

## Known Failure Modes

If someone bypasses the wrapper and runs `milvus run standalone` directly, the
usual failure modes are:

- missing runtime libs such as `libdouble-conversion.so.3`
- missing `folly_exception_tracer_base` and related deps
- missing embed etcd, causing:
  - `dial tcp 127.0.0.1:2379: connect: connection refused`
- runtime state accidentally pointing at the wrong directories

If any of those show up, stop and go back to the wrapper instead of rebuilding
the startup flow by hand.
