# Prefetch 导致 L1 Cache Thrash

**结论**：在 HNSW beam search 的内层循环中加 prefetch 指令，导致 QPS 从 285 降至 141（−51%）。根因是 prefetch 的 working set 远超 L1 cache 容量，把热数据全部驱逐。

---

## 背景

R2（`6ef694e`）混杂了多个变更，其中包括对每个候选节点的邻居做 prefetch。R3（`0294852`）去掉 TRACE_SLAB，修改 prefetch offset，结果 QPS 从 260 降到 141。

---

## 根因分析

### Prefetch 的 working set 计算

```
64 邻居/候选 × 4 条 prefetch cache line × 128 个候选
= 32,768 次 L1 cache fill per search
```

### L1 cache 容量

```
L1 data cache = 32KB = 512 cache lines（64B/line）
```

### 结论

32,768 >> 512：每次 beam search 的 prefetch 请求量是 L1 容量的 **64×**，完全无法缓存。

实际效果：
- 每个候选处理完后，L1 已被 prefetch 数据淹没
- **scratch buffer（~17KB）被驱逐出 L1**
- 后续所有 heap/visited_list 操作变成 L2 miss
- 净效果：prefetch 帮倒忙，延迟增加而非减少

---

## 数字

| 版本 | QPS | 说明 |
|------|-----|------|
| R1（无 prefetch）| 286 | baseline |
| R2（有 prefetch，混杂）| 260 | 其他 fix 部分抵消 |
| R3（有 prefetch，去其他噪声）| **141** | prefetch 损失完全暴露 |
| R4（去掉 prefetch）| **349** | FFI fix 收益 +22% 显现 |

---

## 可复用规律

1. **Prefetch 只在 working set < L1 时有效**。先算：prefetch 总量 vs L1 大小。
2. **Beam search 的候选集太大**：L=100-200 候选 × 每候选多条 prefetch = 必然 thrash。
3. **Prefetch 驱逐热数据的代价高于 prefetch 收益**：scratch buffer 等循环热数据要保在 L1。
4. **混杂变更掩盖了 regression**：R2 多个变更叠加，直到 R3 剥离才看清 prefetch 的影响。→ 每次只改一个变量。

---

## 相关页面

- [[concepts/hnsw]] — HNSW beam search 结构
- [[decisions/optimization-log]] — R1-R4 完整时间线
