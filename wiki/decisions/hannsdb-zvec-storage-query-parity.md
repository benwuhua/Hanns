# HannsDB storage/query parity slice：先保证状态诚实，再追更大 parity

**结论**：在 HannsDB 对标 zvec 的当前 storage/query 收尾阶段，最重要的不是继续铺新 surface，而是把 `index_completeness` / `ann_ready` 这类状态声明做成 **optimize → reopen → subsequent write** 全链路一致、可验证、可回归的事实。为此，优先把 `db.rs` 中的存储职责拆清，再用 core + daemon 测试把状态边界钉死。

---

## 背景

HannsDB 当前这一轮对 zvec 的差距，已经不再主要是“有没有某个 API 名字”，而更像是 **storage/runtime 组织成熟度** 的差距。

zvec 在 `VersionManager`、segment metadata、recovery 组织上更系统；HannsDB 这轮 dirty slice 则集中在：

- `crates/hannsdb-core/src/db.rs`
- `crates/hannsdb-core/src/document.rs`
- `crates/hannsdb-core/src/query/*`
- `crates/hannsdb-core/src/storage/*`
- `crates/hannsdb-core/tests/{collection_api.rs, wal_recovery.rs}`
- `crates/hannsdb-daemon/tests/http_smoke.rs`

核心问题不是“怎么把页面看起来更像 zvec”，而是：

1. optimize 后声称 ANN ready / index complete，reopen 后还是否成立？
2. subsequent write 发生后，这个状态会不会被正确清空？
3. daemon 层对外报告的状态，是否和 core 内部状态同源一致？

---

## 做了什么

### 1. 把 `db.rs` 中的存储职责拆成显式模块

新增：

- `crates/hannsdb-core/src/storage/paths.rs`
- `crates/hannsdb-core/src/storage/primary_keys.rs`
- `crates/hannsdb-core/src/storage/recovery.rs`
- `crates/hannsdb-core/src/storage/segment_io.rs`
- `crates/hannsdb-core/src/storage/wal.rs`

同时把一些共享结构拆出：

- `crates/hannsdb-core/src/db_types.rs`
- `crates/hannsdb-core/src/query/hits.rs`

效果是：路径、主键、WAL、replay、segment IO、hit 排序/投影，不再混在单个超大文件里。

### 2. 把 schema/document/index descriptor 校验前移并显式化

在 `crates/hannsdb-core/src/document.rs` 中补上：

- document batch 校验
- vector field 维度校验
- index descriptor 参数校验
- HNSW_HVQ / IVF_USQ 等 runtime honesty 约束

这避免了“接口能写进去、运行时才 silently 偏掉”的假 parity。

### 3. 用回归测试锁死 ANN 状态诚实性

新增/强化的验证重点：

- `collection_api_get_collection_info_marks_index_complete_after_optimize`
- `collection_api_get_collection_info_clears_index_complete_after_subsequent_write`
- `wal_truncate_optimize_preserves_ann_completeness_after_reopen`
- `collection_info_route_preserves_index_complete_after_rebuild_router`
- `collection_info_route_clears_index_complete_after_subsequent_write`
- `admin_segments_route_preserves_ann_ready_after_rebuild_router`
- `admin_segments_route_reports_ann_ready_after_optimize_then_false_after_write`

这些测试把“优化完成”“重启后仍然成立”“后续写入后必须失效”三个阶段串起来，防止只验证其中一个瞬时状态。

---

## 为什么有效

### 第一性原理

一个存储/query 系统对外暴露的状态，如果不能跨 `persist / reopen / mutate` 保持一致，那这个状态就不是能力，只是瞬时现象。

因此，这一轮真正缩小的不是“API 名字差距”，而是：

- **状态所有权更清晰**：路径 / PK / replay / segment IO 各归其位
- **状态边界更清晰**：什么时候 ANN complete，什么时候必须清空
- **验证链更完整**：core 与 daemon 都证明同一件事

### 对 zvec 对比的意义

这并不代表 HannsDB 已经达到 zvec 的整体 runtime 成熟度；zvec 的 versioned segment/runtime 体系仍然更完整。

但它把一个容易失真的 parity 区域变成了：

- 可解释
- 可回归
- 可继续迭代

换句话说，先把“说出来的话”变成真，再去追下一层能力深度。

---

## 教训

- **不要把 engine 内部能力直接当成 product parity。** 只有接通 public/runtime/verification，才算真正能力。
- **凡是状态位，就必须测三段：** `set`（optimize/build 后）→ `persist/reopen` → `invalidate on mutate`。
- **大文件拆分要按状态边界拆，不要按“看起来差不多”拆。** `paths`、`primary_keys`、`recovery`、`segment_io`、`wal` 的边界天然比“再塞回 db.rs”更稳定。
- **daemon 状态测试不能省。** 如果 HTTP 层报告与 core 不一致，用户看到的就是错误系统。

---

## 验证

本轮收尾时，HannsDB 本地验证为：

- `cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture`
- `cargo test -p hannsdb-daemon --test http_smoke -- --nocapture`
- `bash scripts/run_zvec_parity_smoke.sh`

其中 parity smoke 完成：Rust parity suites、lifecycle、compaction、collection_api、wal_recovery、segment_storage、daemon http smoke、Python parity（168 passed / 4 skipped）。

---

## 相关页面

- [[machines/knowhere-x86-hk-proxy]] — HannsDB / zvec 对比使用的 x86 机器
- [[concepts/vdbbench]] — VectorDBBench 背景与流程
- [[decisions/materialize-storage]] — 另一个“状态声明必须与真实 runtime 对齐”的案例
