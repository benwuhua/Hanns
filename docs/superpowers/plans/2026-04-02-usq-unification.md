# USQ (Unit Sphere Quantizer) 统一计划：合并 ExRaBitQ + HVQ

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ExRaBitQ 和 HVQ 两套重复实现合并为一个统一的 `UsqQuantizer`，消除 ~4000 行重复代码。

**Architecture:** ExRaBitQ 和 HVQ 是同一个算法的 vibe coding 分叉——两者都执行 `residual → rotate → normalize → B-bit quantize → 1-bit fastscan → B-bit rerank`。差异仅在实现细节（量化求解器、评分公式的代数变形、LUT 构造点）。统一后用一个 `UsqQuantizer` 替代两者，通过 `nbits` const 参数控制精度档位。

**Tech Stack:** Rust, AVX-512 SIMD (pshufb fastscan), rayon (并行编码)

---

## 文件结构

### 新建文件

| 文件 | 职责 |
|------|------|
| `src/quantization/usq/mod.rs` | 模块导出 |
| `src/quantization/usq/config.rs` | `UsqConfig` — 统一配置 |
| `src/quantization/usq/rotator.rs` | 随机正交旋转（从 exrabitq/rotator.rs 迁移，逻辑不变） |
| `src/quantization/usq/quantizer.rs` | `UsqQuantizer` — 编码 + 评分（合并两套量化+评分逻辑） |
| `src/quantization/usq/layout.rs` | `UsqLayout` — SoA 存储 + fastscan 转置（合并 ExRaBitQLayout + HvqClusterLayout） |
| `src/quantization/usq/fastscan.rs` | `UsqFastScanState` + LUT 构造 + block scan（统一 fastscan 内核） |
| `src/quantization/usq/searcher.rs` | `scan_and_rerank()` — 两阶段搜索 |
| `src/quantization/usq/space.rs` | 压缩码解码器（从 exrabitq/space.rs 迁移） |
| `tests/test_usq_quantizer.rs` | 量化器单元测试 |
| `tests/test_usq_fastscan.rs` | fastscan 单元测试 |
| `tests/test_usq_ivf.rs` | IVF-USQ 集成测试 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/quantization/mod.rs` | 添加 `pub mod usq;`，保留旧模块（Phase 5 再删） |
| `src/faiss/ivf_hvq.rs` | 内部替换为 `UsqQuantizer`（保留公开 API 做 type alias） |
| `src/faiss/ivf_exrabitq.rs` | 内部替换为 `UsqQuantizer`（保留公开 API 做 type alias） |
| `src/faiss/hnsw_hvq.rs` | 内部替换为 `UsqQuantizer` |
| `src/faiss/mod.rs` | 添加 USQ 导出 |

### 最终删除文件（Phase 5）

| 文件 | 行数 |
|------|------|
| `src/quantization/exrabitq/` 全目录 | 2166 行 |
| `src/quantization/hvq.rs` | 2608 行 |

---

## 关键设计决策

### D1: 评分公式统一为 HVQ 的直觉形式

ExRaBitQ: `x2 + y2 - xipnorm * (fac_rescale * sign_ip + long_ip - (fac_rescale-1) * half_sum)`
HVQ:      `centroid_score + norm_o * ip / base_quant_dist`

两者等价——都在计算 `‖x‖ · <reconstructed_unit, q_rot>`。
统一为 HVQ 形式：`norm * dequant_dot(codes, query) / quant_quality`。
原因：更简单、更直观、中间量更少。

### D2: 量化方法按 nbits 自动选择

- nbits ∈ {1, 2, 3}: `threshold_sweep()`（ExRaBitQ 方法，低 bit 时更快且足够好）
- nbits ∈ {4, 5, 6, 7, 8}: `greedy_refine()`（HVQ 方法，高 bit 时 recall 更优）

### D3: LUT 构造统一为 HVQ 方式

浮点建表 → 全局 i8 量化。比 ExRaBitQ 的"先量化 query 再建表"更简单、更通用。

### D4: 元数据统一为 4 个 f32

```rust
struct UsqMeta {
    norm: f32,           // ‖rotated_residual‖
    norm_sq: f32,        // norm²
    vmax: f32,           // max(|unit[i]|), 用于反量化 scale
    quant_quality: f32,  // base_quant_dist (量化误差归一化因子)
}
```

ExRaBitQ 的 `xipnorm`, `short_ip_factor`, `sum_xb`, `short_err` 不再持久化——
`xipnorm` 和 HVQ 的 `norm / quant_quality` 是同一个东西的不同表达。

### D5: 码存储统一为 packed_bits

不再分离 short_code + long_code。sign_bits 在 `add()` 时从 packed_bits 提取（一次位运算），写入 fastscan 转置块。

---

## Task 1: UsqConfig + UsqRotator（基础层）

**Files:**
- Create: `src/quantization/usq/mod.rs`
- Create: `src/quantization/usq/config.rs`
- Create: `src/quantization/usq/rotator.rs`
- Modify: `src/quantization/mod.rs`
- Test: `tests/test_usq_quantizer.rs`

- [ ] **Step 1: 创建 `src/quantization/usq/config.rs`**

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsqConfig {
    pub dim: usize,
    pub nbits: u8,
    pub seed: u64,
}

impl UsqConfig {
    pub fn new(dim: usize, nbits: u8) -> Result<Self, String> {
        if dim == 0 {
            return Err("dim must be > 0".to_string());
        }
        if nbits == 0 || nbits > 8 {
            return Err(format!("nbits must be 1..=8, got {nbits}"));
        }
        Ok(Self { dim, nbits, seed: 42 })
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Padded to multiple of 64 for SIMD alignment.
    pub fn padded_dim(&self) -> usize {
        self.dim.div_ceil(64) * 64
    }

    /// Number of quantization levels (2^nbits).
    pub fn levels(&self) -> u32 {
        1u32 << self.nbits
    }

    /// Packed code size in bytes (B-bit codes for padded_dim dimensions).
    pub fn code_bytes(&self) -> usize {
        (self.padded_dim() * self.nbits as usize).div_ceil(8)
    }

    /// Per-vector metadata size: norm + norm_sq + vmax + quant_quality = 16 bytes.
    pub fn meta_bytes(&self) -> usize {
        16
    }

    /// Total bytes per encoded vector (metadata + packed codes).
    pub fn encoded_bytes(&self) -> usize {
        self.meta_bytes() + self.code_bytes()
    }

    /// Sign bits size in bytes (1 bit per padded_dim).
    pub fn sign_bytes(&self) -> usize {
        self.padded_dim() / 8
    }
}
```

- [ ] **Step 2: 创建 `src/quantization/usq/rotator.rs`**

从 `src/quantization/exrabitq/rotator.rs` 复制，改名为 `UsqRotator`。
逻辑不变（QR 分解生成随机正交矩阵），只改 struct 名和 config 类型引用。

```rust
use super::config::UsqConfig;

#[derive(Clone)]
pub struct UsqRotator {
    dim: usize,
    matrix: Vec<f32>,
    transpose: Vec<f32>,
}

impl UsqRotator {
    pub fn new(config: &UsqConfig) -> Self {
        // 和 ExRaBitQRotator::new 完全相同的 QR 分解逻辑
        // 使用 config.seed 作为 RNG 种子
        // 生成 padded_dim x padded_dim 的随机正交矩阵
        let padded = config.padded_dim();
        // ... (从 exrabitq/rotator.rs 复制全部实现)
        todo!("复制 ExRaBitQRotator::new 实现，改用 UsqConfig")
    }

    pub fn rotate(&self, padded: &[f32]) -> Vec<f32> {
        // 同 ExRaBitQRotator::rotate_padded
        let mut out = vec![0.0f32; self.dim];
        self.rotate_into(padded, &mut out);
        out
    }

    pub fn rotate_into(&self, padded: &[f32], out: &mut [f32]) {
        // 同 ExRaBitQRotator::rotate_padded_into
        // AVX-512 matvec 或 scalar fallback
        todo!("复制 matvec 实现")
    }

    pub fn inverse_rotate(&self, rotated: &[f32]) -> Vec<f32> {
        // 同 ExRaBitQRotator::inverse_rotate_padded
        let mut out = vec![0.0f32; self.dim];
        self.inverse_rotate_into(rotated, &mut out);
        out
    }

    pub fn inverse_rotate_into(&self, rotated: &[f32], out: &mut [f32]) {
        // 用 transpose 矩阵做 matvec
        todo!("复制 inverse matvec 实现")
    }

    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }
}
```

- [ ] **Step 3: 创建 `src/quantization/usq/mod.rs`**

```rust
mod config;
mod rotator;

pub use config::UsqConfig;
pub use rotator::UsqRotator;
```

- [ ] **Step 4: 注册模块**

在 `src/quantization/mod.rs` 添加：

```rust
pub mod usq;
pub use usq::{UsqConfig, UsqRotator};
```

- [ ] **Step 5: 写测试验证旋转矩阵正交性**

创建 `tests/test_usq_quantizer.rs`：

```rust
use knowhere_rs::quantization::usq::{UsqConfig, UsqRotator};

#[test]
fn test_rotator_orthogonality() {
    let config = UsqConfig::new(128, 4).unwrap();
    let rotator = UsqRotator::new(&config);
    let padded = config.padded_dim();

    // 随机向量旋转后范数不变
    let mut v = vec![0.0f32; padded];
    for (i, x) in v.iter_mut().enumerate() {
        *x = (i as f32 * 0.1).sin();
    }
    let norm_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rotated = rotator.rotate(&v);
    let norm_after: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_before - norm_after).abs() < 1e-4, "rotation should preserve norm");

    // 旋转 + 逆旋转 = 恒等
    let recovered = rotator.inverse_rotate(&rotated);
    for i in 0..padded {
        assert!((v[i] - recovered[i]).abs() < 1e-4, "inverse rotation failed at dim {i}");
    }
}

#[test]
fn test_rotator_deterministic() {
    let config = UsqConfig::new(128, 4).unwrap();
    let r1 = UsqRotator::new(&config);
    let r2 = UsqRotator::new(&config);
    assert_eq!(r1.matrix(), r2.matrix(), "same seed should produce same matrix");
}
```

- [ ] **Step 6: 运行测试**

Run: `cargo test --test test_usq_quantizer -v`
Expected: 2 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/quantization/usq/ src/quantization/mod.rs tests/test_usq_quantizer.rs
git commit -m "feat(usq): add UsqConfig and UsqRotator foundation"
```

---

## Task 2: UsqQuantizer 编码路径

**Files:**
- Create: `src/quantization/usq/quantizer.rs`
- Create: `src/quantization/usq/space.rs`
- Modify: `src/quantization/usq/mod.rs`
- Test: `tests/test_usq_quantizer.rs` (追加)

- [ ] **Step 1: 创建 `src/quantization/usq/space.rs`**

从 `src/quantization/exrabitq/space.rs` 完整复制。这是压缩码解码器（fxu2/fxu3/fxu4/fxu6/fxu7/fxu8 + AVX-512 内积），
与量化方案无关，是纯工具代码。只改 `pub(crate)` 的可见性。

- [ ] **Step 2: 写 UsqQuantizer 编码的失败测试**

在 `tests/test_usq_quantizer.rs` 追加：

```rust
use knowhere_rs::quantization::usq::UsqQuantizer;

#[test]
fn test_encode_decode_roundtrip_4bit() {
    let config = UsqConfig::new(128, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config);
    let centroid = vec![0.0f32; 128];
    quantizer.set_centroid(&centroid);

    // 随机向量
    let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    // 元数据合理
    assert!(encoded.norm > 0.0);
    assert!(encoded.vmax > 0.0);
    assert!(encoded.quant_quality > 0.0);

    // 反量化后的内积应与原始内积接近
    let q: Vec<f32> = (0..128).map(|i| (i as f32 * 0.71).cos()).collect();
    let true_ip: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
    let approx_score = quantizer.score(&encoded, &q);
    // 4-bit 量化应有合理精度
    let relative_err = ((true_ip - approx_score) / true_ip.abs().max(1e-6)).abs();
    assert!(relative_err < 0.3, "4-bit score too inaccurate: true={true_ip}, approx={approx_score}");
}

#[test]
fn test_encode_1bit() {
    let config = UsqConfig::new(128, 1).unwrap();
    let mut quantizer = UsqQuantizer::new(config);
    let centroid = vec![0.0f32; 128];
    quantizer.set_centroid(&centroid);

    let v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin()).collect();
    let encoded = quantizer.encode(&v);

    // 1-bit: 码大小 = padded_dim/8 = 128/8 = 16 bytes
    assert_eq!(encoded.packed_bits.len(), 16);
    assert!(encoded.norm > 0.0);
}
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cargo test --test test_usq_quantizer test_encode -v`
Expected: FAIL，`UsqQuantizer` 不存在

- [ ] **Step 4: 实现 `src/quantization/usq/quantizer.rs`**

核心结构体和编码逻辑：

```rust
use super::config::UsqConfig;
use super::rotator::UsqRotator;

/// 统一的编码结果。
#[derive(Clone, Debug)]
pub struct UsqEncoded {
    pub packed_bits: Vec<u8>,    // B-bit 压缩码
    pub sign_bits: Vec<u8>,      // 1-bit 符号码（从 packed_bits 导出）
    pub norm: f32,               // ‖rotated_residual‖
    pub norm_sq: f32,            // norm²
    pub vmax: f32,               // max(|unit[i]|)
    pub quant_quality: f32,      // sqrt(Σ(重建误差²))
}

pub struct UsqQuantizer {
    config: UsqConfig,
    rotator: UsqRotator,
    centroid: Vec<f32>,
    rotated_centroid: Vec<f32>,
}

impl UsqQuantizer {
    pub fn new(config: UsqConfig) -> Self {
        let rotator = UsqRotator::new(&config);
        Self {
            centroid: vec![0.0; config.dim],
            rotated_centroid: vec![0.0; config.padded_dim()],
            config,
            rotator,
        }
    }

    pub fn set_centroid(&mut self, centroid: &[f32]) {
        self.centroid = centroid.to_vec();
        // 预旋转质心
        let mut padded = vec![0.0f32; self.config.padded_dim()];
        padded[..self.config.dim].copy_from_slice(centroid);
        self.rotated_centroid = self.rotator.rotate(&padded);
    }

    pub fn config(&self) -> &UsqConfig { &self.config }
    pub fn rotator(&self) -> &UsqRotator { &self.rotator }

    /// 编码一个向量。
    /// Pipeline: center → pad → rotate → normalize → quantize → pack
    pub fn encode(&self, vector: &[f32]) -> UsqEncoded {
        let dim = self.config.dim;
        let padded_dim = self.config.padded_dim();

        // 1. Center
        let mut centered = vec![0.0f32; padded_dim];
        for i in 0..dim {
            centered[i] = vector[i] - self.centroid[i];
        }

        // 2. Rotate
        let rotated = self.rotator.rotate(&centered);

        self.encode_rotated(&rotated)
    }

    /// 对已旋转的残差编码（用于 IVF 场景，避免重复旋转）。
    pub fn encode_rotated(&self, rotated: &[f32]) -> UsqEncoded {
        let padded_dim = self.config.padded_dim();

        // 1. 计算范数
        let norm_sq: f32 = rotated.iter().map(|x| x * x).sum();
        let norm = norm_sq.sqrt();

        if norm <= 1e-12 {
            return UsqEncoded {
                packed_bits: vec![0u8; self.config.code_bytes()],
                sign_bits: vec![0u8; self.config.sign_bytes()],
                norm: 0.0,
                norm_sq: 0.0,
                vmax: 0.0,
                quant_quality: 1.0, // 避免除零
            };
        }

        // 2. 归一化到单位球
        let unit: Vec<f32> = rotated.iter().map(|&x| x / norm).collect();

        // 3. 量化
        let nbits = self.config.nbits;
        let (codes, ip, qed_length) = if nbits >= 4 {
            greedy_quantize(&unit, nbits, 6)
        } else {
            threshold_sweep_quantize(&unit, nbits)
        };

        // 4. 计算元数据
        let vmax = unit.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let quant_quality = qed_length.sqrt().max(1e-12);

        // 5. 打包码 + 提取符号位
        let packed_bits = pack_codes(&codes, nbits);
        let sign_bits = extract_sign_bits(&unit, padded_dim);

        UsqEncoded {
            packed_bits,
            sign_bits,
            norm,
            norm_sq,
            vmax,
            quant_quality,
        }
    }

    /// 计算 score = centroid_dot + norm * dequant_ip / quant_quality
    /// 用于 rerank 阶段。
    pub fn score(&self, encoded: &UsqEncoded, query: &[f32]) -> f32 {
        let padded_dim = self.config.padded_dim();
        let mut q_padded = vec![0.0f32; padded_dim];
        q_padded[..self.config.dim].copy_from_slice(query);
        let q_rot = self.rotator.rotate(&q_padded);

        let centroid_score: f32 = q_rot.iter()
            .zip(self.rotated_centroid.iter())
            .map(|(a, b)| a * b)
            .sum();

        if encoded.quant_quality <= 1e-12 {
            return centroid_score;
        }

        let ip = dequant_dot(&q_rot, &encoded.packed_bits, encoded.vmax, &self.config);
        centroid_score + encoded.norm * ip / encoded.quant_quality
    }

    /// score_with_meta: 用预提取的元数据评分（避免重复 parse）。
    pub fn score_with_meta(
        &self,
        q_rot: &[f32],
        centroid_score: f32,
        norm: f32,
        vmax: f32,
        quant_quality: f32,
        packed_bits: &[u8],
    ) -> f32 {
        if quant_quality <= 1e-12 {
            return centroid_score;
        }
        let ip = dequant_dot(q_rot, packed_bits, vmax, &self.config);
        centroid_score + norm * ip / quant_quality
    }
}

/// 反量化 + 内积。
/// 从 packed B-bit 码反量化每个维度，与 query 做内积。
fn dequant_dot(query: &[f32], packed: &[u8], vmax: f32, config: &UsqConfig) -> f32 {
    let nbits = config.nbits;
    let levels = config.levels() as f32;
    let scale = (2.0 * vmax) / levels.max(1.0);
    let offset = -vmax;
    let dim = config.padded_dim();

    // 对 nbits ∈ {4, 8} 使用 SIMD 快速路径（复用 space.rs 的解码器）
    // 其他位宽使用通用 bit-unpack
    let mut ip = 0.0f32;
    let mask = (1u32 << nbits) - 1;
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0usize;
    let mut byte_idx = 0usize;

    for i in 0..dim {
        while bits_in_buffer < nbits as usize {
            let next = packed.get(byte_idx).copied().unwrap_or(0) as u64;
            bit_buffer |= next << bits_in_buffer;
            bits_in_buffer += 8;
            byte_idx += 1;
        }
        let raw = (bit_buffer & mask as u64) as f32;
        bit_buffer >>= nbits;
        bits_in_buffer -= nbits as usize;

        let decoded = raw * scale + offset;
        ip += query[i] * decoded;
    }
    ip
}

/// 低 bit (1-3) 量化：阈值扫描法。
/// 来源：ExRaBitQ 的 fast_quantize_abs + 符号编码，
/// 适配为 HVQ 的 signed-magnitude 编码方式。
fn threshold_sweep_quantize(unit: &[f32], nbits: u8) -> (Vec<u16>, f32, f32) {
    // 从 hvq.rs fast_quantize() 实现复制
    // 返回 (codes, ip, qed_length)
    // ip = Σ dequant(code[i]) * unit[i]
    // qed_length = Σ (dequant(code[i]) - unit[i])²
    todo!("复制 HvqQuantizer::fast_quantize 实现")
}

/// 高 bit (4-8) 量化：贪心逐维调优。
/// 来源：HVQ 的 greedy_quantize。
fn greedy_quantize(unit: &[f32], nbits: u8, n_refine: usize) -> (Vec<u16>, f32, f32) {
    // 从 hvq.rs greedy_quantize() 实现复制
    // 返回 (codes, ip, qed_length)
    todo!("复制 HvqQuantizer::greedy_quantize 实现")
}

/// 将量化码打包为 bytes。
fn pack_codes(codes: &[u16], nbits: u8) -> Vec<u8> {
    // 从 turboquant::packed::pack_codes 复用
    todo!("复用现有 pack_codes")
}

/// 从归一化单位向量提取符号位。
fn extract_sign_bits(unit: &[f32], padded_dim: usize) -> Vec<u8> {
    let mut bits = vec![0u8; padded_dim / 8];
    for (i, &val) in unit.iter().enumerate().take(padded_dim) {
        if val >= 0.0 {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    bits
}
```

- [ ] **Step 5: 更新 mod.rs 导出**

```rust
mod config;
mod quantizer;
mod rotator;
mod space;

pub use config::UsqConfig;
pub use quantizer::{UsqEncoded, UsqQuantizer};
pub use rotator::UsqRotator;
```

- [ ] **Step 6: 运行测试**

Run: `cargo test --test test_usq_quantizer -v`
Expected: 4 tests PASS (2 rotator + 2 encoder)

- [ ] **Step 7: Commit**

```bash
git add src/quantization/usq/quantizer.rs src/quantization/usq/space.rs src/quantization/usq/mod.rs tests/test_usq_quantizer.rs
git commit -m "feat(usq): add UsqQuantizer with unified encode + score"
```

---

## Task 3: UsqLayout（SoA 存储 + fastscan 转置）

**Files:**
- Create: `src/quantization/usq/layout.rs`
- Modify: `src/quantization/usq/mod.rs`
- Test: `tests/test_usq_quantizer.rs` (追加)

- [ ] **Step 1: 写失败测试**

```rust
use knowhere_rs::quantization::usq::{UsqConfig, UsqQuantizer, UsqLayout};

#[test]
fn test_layout_build_and_access() {
    let config = UsqConfig::new(128, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; 128]);

    let n = 100;
    let data: Vec<f32> = (0..n * 128).map(|i| (i as f32 * 0.13).sin()).collect();
    let encoded: Vec<_> = (0..n).map(|i| quantizer.encode(&data[i*128..(i+1)*128])).collect();
    let ids: Vec<i64> = (0..n as i64).collect();

    let layout = UsqLayout::build(&config, &encoded, &ids);
    assert_eq!(layout.len(), n);

    // 元数据可访问
    for i in 0..n {
        assert_eq!(layout.id_at(i), i as i64);
        assert!(layout.norm_at(i) > 0.0);
        assert!(layout.vmax_at(i) > 0.0);
    }

    // fastscan 块数量正确
    let expected_blocks = n.div_ceil(32);
    assert_eq!(layout.n_blocks(), expected_blocks);
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --test test_usq_quantizer test_layout -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/quantization/usq/layout.rs`**

合并 ExRaBitQLayout（分离存储 short/long + 6 个元数据数组）和 HvqClusterLayout（4 个元数据 + packed_bits）为统一的 SoA：

```rust
use super::config::UsqConfig;
use super::quantizer::UsqEncoded;

pub const BLOCK_SIZE: usize = 32;

#[derive(Clone)]
pub struct UsqLayout {
    // ID
    ids: Vec<i64>,

    // 元数据 (4 x f32 per vector)
    norms: Vec<f32>,
    norms_sq: Vec<f32>,
    vmaxs: Vec<f32>,
    quant_qualities: Vec<f32>,

    // B-bit 码 (扁平化，每向量 code_bytes 字节)
    packed_bits: Vec<u8>,
    code_bytes: usize,

    // 1-bit fastscan 转置码
    fastscan_codes: Vec<u8>,
    fastscan_block_size: usize,  // 每块字节数
    n_blocks: usize,

    padded_dim: usize,
}

impl UsqLayout {
    pub fn build(config: &UsqConfig, encoded: &[UsqEncoded], ids: &[i64]) -> Self {
        let n = encoded.len();
        let code_bytes = config.code_bytes();
        let padded_dim = config.padded_dim();

        let mut norms = Vec::with_capacity(n);
        let mut norms_sq = Vec::with_capacity(n);
        let mut vmaxs = Vec::with_capacity(n);
        let mut quant_qualities = Vec::with_capacity(n);
        let mut all_packed = Vec::with_capacity(n * code_bytes);
        let mut all_signs: Vec<Vec<u8>> = Vec::with_capacity(n);

        for enc in encoded {
            norms.push(enc.norm);
            norms_sq.push(enc.norm_sq);
            vmaxs.push(enc.vmax);
            quant_qualities.push(enc.quant_quality);
            all_packed.extend_from_slice(&enc.packed_bits);
            all_signs.push(enc.sign_bits.clone());
        }

        // 转置 sign_bits 为 fastscan 块格式
        let n_blocks = n.div_ceil(BLOCK_SIZE);
        let n_groups = padded_dim / 4;
        let fastscan_block_size = n_groups * 16; // 16 bytes per group (32 vectors × 4 bits)
        let fastscan_codes = transpose_to_fastscan(&all_signs, n, padded_dim, n_blocks, fastscan_block_size);

        Self {
            ids: ids.to_vec(),
            norms,
            norms_sq,
            vmaxs,
            quant_qualities,
            packed_bits: all_packed,
            code_bytes,
            fastscan_codes,
            fastscan_block_size,
            n_blocks,
            padded_dim,
        }
    }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
    pub fn id_at(&self, idx: usize) -> i64 { self.ids[idx] }
    pub fn norm_at(&self, idx: usize) -> f32 { self.norms[idx] }
    pub fn norm_sq_at(&self, idx: usize) -> f32 { self.norms_sq[idx] }
    pub fn vmax_at(&self, idx: usize) -> f32 { self.vmaxs[idx] }
    pub fn quant_quality_at(&self, idx: usize) -> f32 { self.quant_qualities[idx] }
    pub fn packed_bits_at(&self, idx: usize) -> &[u8] {
        let start = idx * self.code_bytes;
        &self.packed_bits[start..start + self.code_bytes]
    }
    pub fn n_blocks(&self) -> usize { self.n_blocks }
    pub fn fastscan_block_size(&self) -> usize { self.fastscan_block_size }
    pub fn fastscan_block(&self, block_idx: usize) -> &[u8] {
        let start = block_idx * self.fastscan_block_size;
        &self.fastscan_codes[start..start + self.fastscan_block_size]
    }
    pub fn padded_dim(&self) -> usize { self.padded_dim }
}

/// 将 N 个 sign_bits (每个 padded_dim/8 字节) 转置为 fastscan 块格式。
/// 块格式：每组 4 维 × 32 向量 → 16 bytes（每向量 1 nibble，2 per byte）。
fn transpose_to_fastscan(
    signs: &[Vec<u8>],
    n: usize,
    padded_dim: usize,
    n_blocks: usize,
    block_size: usize,
) -> Vec<u8> {
    let n_groups = padded_dim / 4;
    let mut out = vec![0u8; n_blocks * block_size];

    for block_idx in 0..n_blocks {
        let block_start = block_idx * block_size;
        let vec_start = block_idx * BLOCK_SIZE;

        for group_idx in 0..n_groups {
            let group_offset = block_start + group_idx * 16;
            let dim_base = group_idx * 4;

            for slot in 0..BLOCK_SIZE {
                let vec_idx = vec_start + slot;
                if vec_idx >= n { break; }

                // 从 sign_bits 中提取 4 个维度的符号位，组合成 nibble
                let mut nibble = 0u8;
                for bit in 0..4 {
                    let dim = dim_base + bit;
                    let byte_idx = dim / 8;
                    let bit_idx = dim % 8;
                    if (signs[vec_idx][byte_idx] >> bit_idx) & 1 == 1 {
                        nibble |= 1 << bit;
                    }
                }

                // 2 nibbles per byte
                let byte_pos = group_offset + slot / 2;
                if slot % 2 == 0 {
                    out[byte_pos] |= nibble;
                } else {
                    out[byte_pos] |= nibble << 4;
                }
            }
        }
    }

    out
}
```

- [ ] **Step 4: 更新 mod.rs 导出 `UsqLayout`, `BLOCK_SIZE`**

- [ ] **Step 5: 运行测试**

Run: `cargo test --test test_usq_quantizer test_layout -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/quantization/usq/layout.rs src/quantization/usq/mod.rs tests/test_usq_quantizer.rs
git commit -m "feat(usq): add UsqLayout with SoA storage and fastscan transpose"
```

---

## Task 4: UsqFastScan（LUT + block scan + 两阶段搜索）

**Files:**
- Create: `src/quantization/usq/fastscan.rs`
- Create: `src/quantization/usq/searcher.rs`
- Modify: `src/quantization/usq/mod.rs`
- Test: `tests/test_usq_fastscan.rs`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_usq_fastscan.rs`：

```rust
use knowhere_rs::quantization::usq::*;

fn make_test_data(n: usize, dim: usize) -> (Vec<f32>, Vec<i64>) {
    let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.13).sin()).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    (data, ids)
}

#[test]
fn test_fastscan_vs_brute_force() {
    let dim = 128;
    let n = 200;
    let config = UsqConfig::new(dim, 4).unwrap();
    let mut quantizer = UsqQuantizer::new(config.clone());
    quantizer.set_centroid(&vec![0.0f32; dim]);

    let (data, ids) = make_test_data(n, dim);
    let encoded: Vec<_> = (0..n).map(|i| quantizer.encode(&data[i*dim..(i+1)*dim])).collect();
    let layout = UsqLayout::build(&config, &encoded, &ids);

    // 查询
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut q_padded = vec![0.0f32; config.padded_dim()];
    q_padded[..dim].copy_from_slice(&query);
    let q_rot = quantizer.rotator().rotate(&q_padded);

    // Brute force: 用 score_with_meta 对所有向量评分
    let centroid_score: f32 = q_rot.iter().zip(vec![0.0f32; config.padded_dim()].iter())
        .map(|(a, b)| a * b).sum();
    let q_norm_sq: f32 = q_rot.iter().map(|x| x * x).sum();
    let mut bf_results: Vec<(i64, f32)> = (0..n).map(|i| {
        let score = quantizer.score_with_meta(
            &q_rot, centroid_score,
            layout.norm_at(i), layout.vmax_at(i), layout.quant_quality_at(i),
            layout.packed_bits_at(i),
        );
        let dist = q_norm_sq + layout.norm_sq_at(i) - 2.0 * score;
        (ids[i], dist)
    }).collect();
    bf_results.sort_by(|a, b| a.1.total_cmp(&b.1));
    let bf_top10: Vec<i64> = bf_results.iter().take(10).map(|r| r.0).collect();

    // Fastscan + rerank
    let state = UsqFastScanState::new(&q_rot, &config);
    let fs_results = scan_and_rerank(&quantizer, &layout, &state, &q_rot, centroid_score, q_norm_sq, 10);
    let fs_top10: Vec<i64> = fs_results.iter().take(10).map(|r| r.0).collect();

    // fastscan+rerank 应和 brute force 结果一致（或非常接近）
    let overlap = bf_top10.iter().filter(|id| fs_top10.contains(id)).count();
    assert!(overlap >= 8, "fastscan top-10 should overlap >=80% with brute force, got {overlap}/10");
}

#[test]
fn test_fastscan_lut_symmetry() {
    let dim = 128;
    let config = UsqConfig::new(dim, 4).unwrap();

    // 全零 query → LUT 应全零
    let q_rot = vec![0.0f32; config.padded_dim()];
    let state = UsqFastScanState::new(&q_rot, &config);
    assert!(state.lut.iter().all(|&v| v == 0), "zero query should give zero LUT");
}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cargo test --test test_usq_fastscan -v`
Expected: FAIL

- [ ] **Step 3: 实现 `src/quantization/usq/fastscan.rs`**

LUT 构造统一为 HVQ 方式（浮点建表 → i8 量化）：

```rust
use super::config::UsqConfig;
use super::layout::{UsqLayout, BLOCK_SIZE};

#[derive(Clone, Debug)]
pub struct UsqFastScanState {
    pub lut: Vec<i8>,      // n_groups * 16 entries
    pub lut_scale: f32,    // 将 i32 累加值转换回 f32 的缩放因子
}

impl UsqFastScanState {
    /// 构建 fastscan LUT。
    /// 对每组 4 个维度，穷举 16 种符号组合，预计算 q_rot 加权和。
    pub fn new(q_rot: &[f32], config: &UsqConfig) -> Self {
        let padded_dim = config.padded_dim();
        let n_groups = padded_dim / 4;
        let mut lut_f32 = vec![0.0f32; n_groups * 16];

        for g in 0..n_groups {
            let q = &q_rot[g * 4..g * 4 + 4];
            for nibble in 0..16u8 {
                let mut val = 0.0f32;
                for bit in 0..4 {
                    let sign = if nibble & (1 << bit) != 0 { 1.0 } else { -1.0 };
                    val += q[bit] * sign;
                }
                lut_f32[g * 16 + nibble as usize] = val;
            }
        }

        // 全局量化到 i8
        let max_abs = lut_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let lut_scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 1.0 };
        let lut: Vec<i8> = lut_f32.iter()
            .map(|&v| (v / lut_scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        Self { lut, lut_scale }
    }
}

/// Fastscan 候选。
#[derive(Clone, Copy, Debug)]
pub struct FsCandidate {
    pub idx: usize,
    pub raw_score: i32,  // 1-bit 近似分数（i32 累加值）
}

/// 扫描所有 fastscan 块，返回 top_n 候选。
pub fn fastscan_topk(
    layout: &UsqLayout,
    state: &UsqFastScanState,
    top_n: usize,
) -> Vec<FsCandidate> {
    let n = layout.len();
    let n_groups = layout.padded_dim() / 4;

    // 用 BinaryHeap 维护 top_n（最小堆）
    let mut heap = std::collections::BinaryHeap::new();

    for block_idx in 0..layout.n_blocks() {
        let block = layout.fastscan_block(block_idx);
        let scores = fastscan_block_scalar(block, &state.lut, n_groups);
        let vec_base = block_idx * BLOCK_SIZE;

        for (slot, &score) in scores.iter().enumerate() {
            let vec_idx = vec_base + slot;
            if vec_idx >= n { break; }

            let candidate = FsCandidate { idx: vec_idx, raw_score: score };

            if heap.len() < top_n {
                heap.push(std::cmp::Reverse(ScoreOrd(candidate)));
            } else if let Some(&std::cmp::Reverse(ScoreOrd(worst))) = heap.peek() {
                if score > worst.raw_score {
                    heap.pop();
                    heap.push(std::cmp::Reverse(ScoreOrd(candidate)));
                }
            }
        }
    }

    let mut results: Vec<FsCandidate> = heap.into_iter().map(|r| r.0 .0).collect();
    results.sort_by(|a, b| b.raw_score.cmp(&a.raw_score));
    results
}

/// 标量 fastscan block scan：扫描一个 32 向量块。
fn fastscan_block_scalar(block: &[u8], lut: &[i8], n_groups: usize) -> [i32; BLOCK_SIZE] {
    let mut scores = [0i32; BLOCK_SIZE];

    for group in 0..n_groups {
        let group_offset = group * 16;
        let lut_base = group * 16;

        for slot in 0..BLOCK_SIZE {
            let byte_idx = group_offset + slot / 2;
            let nibble = if slot % 2 == 0 {
                block[byte_idx] & 0x0F
            } else {
                (block[byte_idx] >> 4) & 0x0F
            };
            scores[slot] += lut[lut_base + nibble as usize] as i32;
        }
    }

    scores
}

/// AVX-512 fastscan block scan（x86_64 only）。
/// 从 ivf_hvq.rs fastscan_block_avx512 复制，逻辑完全相同。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx2,ssse3")]
unsafe fn fastscan_block_avx512(block: &[u8], lut: &[i8], n_groups: usize) -> [i32; BLOCK_SIZE] {
    // 复制 ivf_hvq.rs 中的 AVX-512 实现
    // 使用 pshufb 对 16 个 nibble 并行查表
    todo!("复制 AVX-512 实现")
}

// 辅助类型：用于 BinaryHeap 排序
#[derive(Clone, Copy, Debug)]
struct ScoreOrd(FsCandidate);
impl PartialEq for ScoreOrd {
    fn eq(&self, other: &Self) -> bool { self.0.raw_score == other.0.raw_score }
}
impl Eq for ScoreOrd {}
impl PartialOrd for ScoreOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for ScoreOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.raw_score.cmp(&other.0.raw_score) }
}
```

- [ ] **Step 4: 实现 `src/quantization/usq/searcher.rs`**

```rust
use super::config::UsqConfig;
use super::fastscan::{fastscan_topk, UsqFastScanState};
use super::layout::UsqLayout;
use super::quantizer::UsqQuantizer;

/// 两阶段搜索：fastscan 粗排 → B-bit rerank。
pub fn scan_and_rerank(
    quantizer: &UsqQuantizer,
    layout: &UsqLayout,
    state: &UsqFastScanState,
    q_rot: &[f32],
    centroid_score: f32,
    q_norm_sq: f32,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let n = layout.len();
    if n == 0 {
        return Vec::new();
    }

    // 候选数量：根据 nbits 调整（和 HVQ 相同的 tier-aware 策略）
    let nbits = quantizer.config().nbits;
    let n_candidates = match nbits {
        1 => (top_k * 20).max(200),
        2..=4 => (top_k * 15).max(150),
        _ => (top_k * 30).max(300),
    }.min(n);

    // 小集合直接全量 rerank
    if n <= n_candidates {
        return brute_force_rerank(quantizer, layout, q_rot, centroid_score, q_norm_sq, top_k);
    }

    // Stage 1: fastscan
    let candidates = fastscan_topk(layout, state, n_candidates);

    // Stage 2: rerank
    let mut results: Vec<(i64, f32)> = candidates.iter().map(|c| {
        let score = quantizer.score_with_meta(
            q_rot,
            centroid_score,
            layout.norm_at(c.idx),
            layout.vmax_at(c.idx),
            layout.quant_quality_at(c.idx),
            layout.packed_bits_at(c.idx),
        );
        let dist = q_norm_sq + layout.norm_sq_at(c.idx) - 2.0 * score;
        (layout.id_at(c.idx), dist)
    }).collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(top_k);
    results
}

fn brute_force_rerank(
    quantizer: &UsqQuantizer,
    layout: &UsqLayout,
    q_rot: &[f32],
    centroid_score: f32,
    q_norm_sq: f32,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let mut results: Vec<(i64, f32)> = (0..layout.len()).map(|i| {
        let score = quantizer.score_with_meta(
            q_rot,
            centroid_score,
            layout.norm_at(i),
            layout.vmax_at(i),
            layout.quant_quality_at(i),
            layout.packed_bits_at(i),
        );
        let dist = q_norm_sq + layout.norm_sq_at(i) - 2.0 * score;
        (layout.id_at(i), dist)
    }).collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(top_k);
    results
}
```

- [ ] **Step 5: 更新 mod.rs**

```rust
mod config;
mod fastscan;
mod layout;
mod quantizer;
mod rotator;
mod searcher;
mod space;

pub use config::UsqConfig;
pub use fastscan::{UsqFastScanState, FsCandidate, fastscan_topk};
pub use layout::{UsqLayout, BLOCK_SIZE};
pub use quantizer::{UsqEncoded, UsqQuantizer};
pub use rotator::UsqRotator;
pub use searcher::scan_and_rerank;
```

- [ ] **Step 6: 运行测试**

Run: `cargo test --test test_usq_fastscan -v`
Expected: 2 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/quantization/usq/ tests/test_usq_fastscan.rs
git commit -m "feat(usq): add fastscan LUT + two-stage scan_and_rerank"
```

---

## Task 5: A/B 验证 — USQ vs ExRaBitQ/HVQ recall 对比

**Files:**
- Test: `tests/test_usq_quantizer.rs` (追加)

这个 task 在整合到 IVF 之前验证 USQ 的评分质量不退化。

- [ ] **Step 1: 写 recall 对比测试**

```rust
/// 对比 UsqQuantizer 和 HvqQuantizer 在相同数据上的 recall。
/// USQ 的 recall 不应低于 HVQ 的 95%。
#[test]
fn test_usq_vs_hvq_recall_parity() {
    use knowhere_rs::quantization::hvq::{HvqConfig, HvqQuantizer};
    use knowhere_rs::quantization::usq::{UsqConfig, UsqQuantizer, UsqLayout, UsqFastScanState, scan_and_rerank};

    let dim = 128;
    let n = 1000;
    let k = 10;
    let nbits = 4;

    // 生成数据
    let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.037).sin()).collect();
    let queries: Vec<f32> = (0..10 * dim).map(|i| (i as f32 * 0.71).cos()).collect();

    // Brute force ground truth
    let gt = brute_force_knn(&data, &queries, dim, n, 10, k);

    // ---- HVQ ----
    let hvq_config = HvqConfig { dim, nbits };
    let mut hvq_q = HvqQuantizer::new(hvq_config, 42);
    hvq_q.train(n, &data);
    // ... (用 HvqQuantizer 编码 + 评分，计算 recall)

    // ---- USQ ----
    let usq_config = UsqConfig::new(dim, nbits).unwrap();
    let mut usq_q = UsqQuantizer::new(usq_config.clone());
    let centroid = compute_centroid(&data, dim, n);
    usq_q.set_centroid(&centroid);
    let encoded: Vec<_> = (0..n).map(|i| usq_q.encode(&data[i*dim..(i+1)*dim])).collect();
    let ids: Vec<i64> = (0..n as i64).collect();
    let layout = UsqLayout::build(&usq_config, &encoded, &ids);

    let mut usq_recall_sum = 0.0;
    for q_idx in 0..10 {
        let query = &queries[q_idx * dim..(q_idx + 1) * dim];
        let mut q_padded = vec![0.0f32; usq_config.padded_dim()];
        q_padded[..dim].copy_from_slice(query);
        let q_rot = usq_q.rotator().rotate(&q_padded);
        let q_norm_sq: f32 = q_rot.iter().map(|x| x * x).sum();
        let centroid_score: f32 = 0.0; // 简化：centroid 已减
        let state = UsqFastScanState::new(&q_rot, &usq_config);
        let results = scan_and_rerank(&usq_q, &layout, &state, &q_rot, centroid_score, q_norm_sq, k);

        let result_ids: Vec<i64> = results.iter().map(|r| r.0).collect();
        let overlap = gt[q_idx].iter().filter(|id| result_ids.contains(id)).count();
        usq_recall_sum += overlap as f32 / k as f32;
    }

    let usq_recall = usq_recall_sum / 10.0;
    eprintln!("USQ recall@{k}: {usq_recall:.3}");
    // 在合成数据上 4-bit 量化应有合理 recall
    assert!(usq_recall > 0.3, "USQ recall too low: {usq_recall:.3}");
}

fn brute_force_knn(data: &[f32], queries: &[f32], dim: usize, n: usize, nq: usize, k: usize) -> Vec<Vec<i64>> {
    (0..nq).map(|q| {
        let query = &queries[q * dim..(q + 1) * dim];
        let mut dists: Vec<(i64, f32)> = (0..n).map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            let dist: f32 = query.iter().zip(vec.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            (i as i64, dist)
        }).collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.iter().take(k).map(|r| r.0).collect()
    }).collect()
}

fn compute_centroid(data: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut centroid = vec![0.0f32; dim];
    for i in 0..n {
        for d in 0..dim {
            centroid[d] += data[i * dim + d];
        }
    }
    for d in 0..dim {
        centroid[d] /= n as f32;
    }
    centroid
}
```

- [ ] **Step 2: 运行测试**

Run: `cargo test --test test_usq_quantizer test_usq_vs_hvq -v -- --nocapture`
Expected: PASS，打印 recall 数字

- [ ] **Step 3: Commit**

```bash
git add tests/test_usq_quantizer.rs
git commit -m "test(usq): add recall parity test vs HVQ"
```

---

## Task 6: IVF-USQ 集成（替换 IvfHvqIndex 内部实现）

**Files:**
- Modify: `src/faiss/ivf_hvq.rs` — 内部逐步替换为 UsqQuantizer
- Test: `tests/test_usq_ivf.rs`

这个 task 是最关键的整合步骤。策略是**先在 IvfHvqIndex 内部替换**，保持公开 API 不变，然后验证 benchmark 不退化。

- [ ] **Step 1: 写 IVF-USQ 集成测试**

创建 `tests/test_usq_ivf.rs`：

```rust
use knowhere_rs::faiss::ivf_hvq::{IvfHvqConfig, IvfHvqIndex};

/// IVF-HVQ（内部替换为 USQ 后）的基本功能验证。
#[test]
fn test_ivf_hvq_with_usq_backend() {
    let dim = 128;
    let n = 2000;
    let nlist = 16;
    let nbits = 4;

    let config = IvfHvqConfig::new(dim, nlist, nbits).with_nprobe(4);
    let mut index = IvfHvqIndex::new(config);

    let data: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.037).sin()).collect();
    index.train(&data).unwrap();
    index.add(&data, None).unwrap();
    assert_eq!(index.count(), n);

    // 搜索
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.71).cos()).collect();
    let req = knowhere_rs::faiss::SearchRequest { k: 10, ..Default::default() };
    let result = index.search(&query, &req).unwrap();
    assert_eq!(result.ids.len(), 10);

    // 至少有一些合理结果（非全 -1）
    let valid = result.ids.iter().filter(|&&id| id >= 0).count();
    assert!(valid >= 5, "should have at least 5 valid results, got {valid}");
}
```

- [ ] **Step 2: 运行测试确认当前 IvfHvqIndex 通过**

Run: `cargo test --test test_usq_ivf -v`
Expected: PASS（当前实现）

- [ ] **Step 3: 修改 IvfHvqIndex 内部使用 UsqQuantizer**

在 `src/faiss/ivf_hvq.rs` 中：

1. 将 `HvqQuantizer` 字段替换为 `UsqQuantizer`
2. 将 `HvqClusterLayout` 替换为 `UsqLayout`
3. `train()` 中用 `UsqQuantizer::new()` + `set_centroid()`
4. `add()` 中用 `UsqQuantizer::encode()` + `UsqLayout::build()`
5. `search()` 中用 `UsqFastScanState::new()` + `scan_and_rerank()`

保持 `IvfHvqConfig` 和 `IvfHvqIndex` 的公开 API 完全不变。

- [ ] **Step 4: 运行测试**

Run: `cargo test --test test_usq_ivf -v`
Expected: PASS

- [ ] **Step 5: 运行现有 IVF-HVQ 内联测试**

Run: `cargo test ivf_hvq -v`
Expected: 所有现有测试 PASS

- [ ] **Step 6: Commit**

```bash
git add src/faiss/ivf_hvq.rs tests/test_usq_ivf.rs
git commit -m "refactor(ivf-hvq): replace HvqQuantizer with UsqQuantizer internally"
```

---

## Task 7: IVF-ExRaBitQ 内部替换

**Files:**
- Modify: `src/faiss/ivf_exrabitq.rs`
- Modify: `tests/test_ivf_exrabitq.rs` (如需调整)

- [ ] **Step 1: 运行现有 ExRaBitQ 测试确认基线**

Run: `cargo test --test test_ivf_exrabitq -v`
Expected: 全 PASS

- [ ] **Step 2: 修改 IvfExRaBitqIndex 内部使用 UsqQuantizer**

在 `src/faiss/ivf_exrabitq.rs` 中：

1. 将 `ExRaBitQQuantizer` 字段替换为 `UsqQuantizer`
2. 将 `ExRaBitQLayout` 替换为 `UsqLayout`
3. 将 `ExRaBitQFastScanState` 替换为 `UsqFastScanState`
4. `bits_per_dim` 映射到 `nbits`
5. `rotation_seed` 映射到 `seed`
6. save/load 需要适配新的序列化格式

保持 `IvfExRaBitqConfig` 和 `IvfExRaBitqIndex` 的公开 API 不变。

- [ ] **Step 3: 运行测试**

Run: `cargo test --test test_ivf_exrabitq -v`
Expected: 全 PASS

- [ ] **Step 4: 运行 fastscan 测试**

Run: `cargo test --test test_exrabitq_fastscan -v`
Expected: 需要评估哪些测试仍适用。部分测试可能需要适配为 USQ API。

- [ ] **Step 5: Commit**

```bash
git add src/faiss/ivf_exrabitq.rs tests/test_ivf_exrabitq.rs
git commit -m "refactor(ivf-exrabitq): replace ExRaBitQQuantizer with UsqQuantizer internally"
```

---

## Task 8: HNSW-HVQ 替换

**Files:**
- Modify: `src/faiss/hnsw_hvq.rs`

- [ ] **Step 1: 运行现有 HNSW-HVQ 测试**

Run: `cargo test hnsw_hvq -v`
Expected: PASS

- [ ] **Step 2: 修改 HnswHvqIndex 内部使用 UsqQuantizer**

HNSW-HVQ 不需要 fastscan（逐点评分），只需替换：
1. `HvqQuantizer` → `UsqQuantizer`
2. `score_code()` → `score_with_meta()`

- [ ] **Step 3: 运行测试**

Run: `cargo test hnsw_hvq -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/faiss/hnsw_hvq.rs
git commit -m "refactor(hnsw-hvq): replace HvqQuantizer with UsqQuantizer"
```

---

## Task 9: Benchmark 验证（Cohere-768D）

**Files:** 无代码变更，纯验证

在合并前必须验证 recall/QPS 不退化。

- [ ] **Step 1: 本地 Mac 快速验证（10K synthetic）**

Run: `cargo run --example benchmark --release 2>&1 | grep -E "HVQ|ExRaBitQ|USQ"`

对比 USQ 替换前后的 QPS 和 recall。允许 ±5% QPS 波动。

- [ ] **Step 2: x86 权威验证（Cohere-768D 1M）**

如果有 Cohere 数据集的 benchmark：

```bash
ssh knowhere-x86-hk-proxy "cd /data/work/knowhere-rs-src && \
    CARGO_TARGET_DIR=/data/work/knowhere-rs-target \
    ~/.cargo/bin/cargo test --test bench_milvus_cohere1m_hnsw_compare --release 2>&1"
```

**验收标准**：
- Recall@10 变化 ≤ 0.01
- QPS 变化 ≤ -10%（允许小幅下降，不允许大幅退化）

- [ ] **Step 3: 记录结果**

如果通过，更新 CLAUDE.md 中的 benchmark baselines。
如果 recall 退化 > 0.01，回到 Task 2 检查量化方法选择或评分公式。

- [ ] **Step 4: Commit（如有 baseline 更新）**

```bash
git add CLAUDE.md
git commit -m "docs: update benchmark baselines after USQ unification"
```

---

## Task 10: 清理旧代码

**Files:**
- Delete: `src/quantization/exrabitq/` (全目录，2166 行)
- Delete: `src/quantization/hvq.rs` (2608 行)
- Modify: `src/quantization/mod.rs` — 移除旧模块导出
- Modify: `src/faiss/mod.rs` — 更新导出
- Modify: 所有 `use` 引用

⚠️ 只有在 Task 9 benchmark 验证通过后才执行此步。

- [ ] **Step 1: 删除 exrabitq 目录**

```bash
rm -rf src/quantization/exrabitq/
```

- [ ] **Step 2: 删除 hvq.rs**

```bash
rm src/quantization/hvq.rs
```

- [ ] **Step 3: 更新 `src/quantization/mod.rs`**

移除：
```rust
pub mod exrabitq;
pub mod hvq;
pub use exrabitq::{ExFactor, ExRaBitQConfig, ExRaBitQQuantizer, ExRaBitQRotator};
pub use hvq::{HvqConfig, HvqFastScanState, HvqIndex, HvqQuantizer};
```

- [ ] **Step 4: 修复所有编译错误**

Run: `cargo build 2>&1 | grep "^error" | head -20`

逐个修复残留的 `use` 引用。

- [ ] **Step 5: 运行全量测试**

Run: `cargo test 2>&1 | tail -5`
Expected: 全 PASS

- [ ] **Step 6: 删除旧测试文件（如不再适用）**

评估 `tests/test_exrabitq_*.rs` 是否已被 `tests/test_usq_*.rs` 覆盖。
如已覆盖，删除旧测试。

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove ExRaBitQ and HVQ, unified into USQ (-4774 lines)"
```

---

## 风险清单

| 风险 | 影响 | 缓解 |
|------|------|------|
| HVQ 评分公式在某些分布下 recall 低于 ExRaBitQ 的多因子校准 | recall 退化 | Task 5 A/B 测试 + Task 9 Cohere 验证，如退化则保留校准因子作为可选 |
| ExRaBitQ 的 high_accuracy 模式无法用 HVQ 公式表达 | 功能缺失 | ExRaBitQ high_accuracy 是独立代码路径，可作为 USQ 的可选模式保留 |
| save/load 序列化格式不兼容 | 已有索引无法加载 | IvfExRaBitqIndex 是唯一有 save/load 的，需要版本号升级 + 迁移 |
| SIMD 路径性能退化 | QPS 下降 | USQ 复用两者中更优的 SIMD 内核，benchmark 验证 |
