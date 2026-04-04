# ISSUE-AISAQ-SAVELOAD-COMPAT
**日期**: 2026-03-24 | **严重程度**: P0
## 问题
AISAQ 当前 save/load 使用自定义 `bincode metadata + 单 data 文件 + 自定义尾部 payload` 方案，不兼容 native DiskANN/Knowhere 的多文件索引格式与 `PQFlashIndex::load` 约定。

## Native 对比（引用文件+行号）
- AISAQ 自定义 save 格式：`src/faiss/diskann_aisaq.rs:1209-1265`
- AISAQ 自定义 load 解析：`src/faiss/diskann_aisaq.rs:1268-1365`
- Knowhere native 依赖 DiskANN 标准文件族：`/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc:193-223`
- Knowhere native 构建走 `build_disk_index`：`/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc:306-310`
- Knowhere native 加载走 `PQFlashIndex::load`：`/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc:382-385`

## 影响
- Rust AISAQ 产物无法直接被 native knowhere/diskann 工具链读取，反向亦然。
- 跨语言部署与回滚路径复杂，增加线上迁移风险和运维成本。
- benchmark 对比时很难做到“同一索引文件、不同实现”公平验证。

## 建议方向
- 定义“native 兼容序列化模式”（输出标准 DiskANN 文件族），保留当前格式为私有模式。
- 增加双向兼容测试：AISAQ 生成 -> native 读取；native 生成 -> AISAQ 读取。
- 若短期无法完全兼容，至少提供离线转换器并固定版本协议。
