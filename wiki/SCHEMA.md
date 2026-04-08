# Wiki Schema — knowhere-rs

> LLM 维护的项目知识库。人类负责 ingest 触发；LLM 负责写作、更新、交叉链接。

---

## 目录结构

```
wiki/
  SCHEMA.md              # 本文件：页面约定 + 工作流规则
  index.md               # 全局目录（每次 ingest 更新）
  log.md                 # append-only 时间线
  concepts/              # 算法/架构/系统概念
  benchmarks/            # 权威数字 + 历史轮次
  decisions/             # 优化决策 + 经验教训
  machines/              # 机器环境 + 工具链
```

---

## 页面类型约定

### concept 页 (`concepts/*.md`)

```markdown
# [概念名]

**一句话定义**：...

## 在 knowhere-rs 中的实现
...

## 关键参数
| 参数 | 含义 | 默认值 |
...

## 已知坑 / 教训
- ...

## 相关页面
- [[concepts/xxx]] — 原因
- [[decisions/xxx]] — 相关决策
```

### benchmark 页 (`benchmarks/*.md`)

```markdown
# [Benchmark 名]

**数据集**：... **机器**：... **日期**：...

## 权威数字
| 指标 | 值 | 对比 native |
...

## 历史轮次
| Round | Commit | 关键变更 | 指标变化 |
...

## 方法论说明
...

```

### decision 页 (`decisions/*.md`)

```markdown
# [决策/经验名]

**结论**：一句话

**背景**：为什么要做这个决策

## 做了什么
...

## 为什么有效 / 为什么失败
第一性原理分析...

## 教训
- 可复用的规律
- 反例 / 陷阱

## 相关页面
```

### machine 页 (`machines/*.md`)

```markdown
# [机器名]

**角色**：...

## 关键路径
| 用途 | 路径 |
...

## 常用命令
...

## 注意事项
...
```

---

## 工作流

### Ingest（新 benchmark / 新优化轮次）

触发词：「把这次结果 ingest 进 wiki」或「更新 wiki」

LLM 操作：
1. 读原始数据（benchmark log、commit message、对话中的数字）
2. 更新 `log.md`（append 一条记录）
3. 更新对应的 benchmarks/ 页（添加新轮次行）
4. 若有新的经验教训，更新或新建 decisions/ 页
5. 更新 `index.md`（若有新页面）

### Query（问题 → wiki）

触发词：任何「knowhere-rs 的 X 是什么」「为什么 Y」类问题

LLM 操作：
1. 读 index.md 确定相关页面
2. 读相关页面合成答案
3. 若答案值得保留，写入新的 decisions/ 页或补充现有页面

### Lint（健康检查）

触发词：「lint wiki」

LLM 检查：
- 数字不一致（同一指标在不同页面有矛盾）
- 孤立页面（没有被任何其他页面链接）
- 过时声明（标有日期的数字，较新轮次已覆盖）
- 缺少交叉链接（一个页面提到了另一个概念但没有 [[链接]]）

---

## 交叉链接约定

- 内部链接用 `[[path/page]]` 格式（Obsidian wiki-link）
- 链接到 benchmark_results/ 原始文件用 `[[../benchmark_results/xxx]]`
- 链接到源代码用普通文本 `src/faiss/diskann_aisaq.rs:2502`（不用 wiki-link）

---

## 版本化

wiki/ 与代码在同一 git repo。每次 ingest 时：
- benchmark 数字更新 → 与对应 benchmark commit 一起打包
- 纯 wiki 更新 → `docs(wiki): ...` commit 前缀
