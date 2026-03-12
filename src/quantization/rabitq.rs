//! RaBitQ (Randomized Bit Quantization)
//!
//! 高效的二进制量化方法，支持 32x 压缩
//! 参考：https://arxiv.org/abs/2405.12497
//!
//! 核心思想（与 C++ Faiss 对齐）:
//! 1. 计算向量与质心的残差
//! 2. 直接对残差进行二值化 (residual[i] > 0 ? 1 : 0)
//! 3. 存储校正因子用于无偏距离估计

/// 查询量化结果（qb=8 模式）
///
/// 存储量化后的查询向量和预计算的系数
/// 参考 C++ Faiss RaBitDistanceComputerQ::set_query
#[derive(Clone, Debug)]
pub struct QueryQuantization {
    /// 量化后的查询向量（8-bit）
    pub quantized_query: Vec<u8>,
    /// 量化值之和
    pub sum_qq: usize,
    /// 系数 c1 = 2 * delta / sqrt(d)
    pub c1: f32,
    /// 系数 c2 = 2 * v_min / sqrt(d)
    pub c2: f32,
    /// 系数 c34 = (delta * sum_qq + d * v_min) / sqrt(d)
    pub c34: f32,
    /// ||qr - c||^2 查询残差范数的平方
    pub qr_to_c_l2sqr: f32,
}

/// RaBitQ 编码器
///
/// 将高维向量量化为二进制码，支持基于质心的残差量化
///
/// 量化流程（与 C++ Faiss 对齐）:
/// 1. 计算残差：residual = vector - centroid
/// 2. 直接二值化：code[i] = residual[i] > 0 ? 1 : 0
/// 3. 存储校正因子：
///    - centroid_dist = ||residual||
///    - ip = <sign(residual), residual> / ||residual|| / sqrt(d)
pub struct RaBitQEncoder {
    dim: usize,
    /// 是否已训练
    trained: bool,
    /// 查询量化位数（0 = 不量化，8 = 8-bit 量化）
    pub qb: u8,
}

impl RaBitQEncoder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            trained: false,
            qb: 8, // 默认使用 qb=8 模式
        }
    }

    pub fn with_qb(mut self, qb: u8) -> Self {
        self.qb = qb;
        self
    }

    /// 训练码书 - RaBitQ 无需训练
    ///
    /// C++ Faiss 版本：train() does nothing
    pub fn train(&mut self, _data: &[f32]) {
        self.trained = true;
    }

    /// 编码到二进制 (带校正因子)
    ///
    /// 返回：(binary_code, centroid_dist, inner_product)
    /// - centroid_dist: ||vector|| (向量 L2 范数)
    /// - inner_product: <sign(vector), vector> / ||vector|| / sqrt(d)
    pub fn encode(&self, vector: &[f32]) -> (Vec<u8>, f32, f32) {
        if !self.trained {
            return (vec![0u8; self.dim.div_ceil(8)], 0.0, 0.0);
        }

        // 计算向量范数
        let norm_l2sqr: f32 = vector.iter().map(|&x| x * x).sum();
        let norm_l2 = norm_l2sqr.sqrt().max(1e-10);

        // 直接二值化（不旋转）
        let total_bytes = self.dim.div_ceil(8);
        let mut codes = vec![0u8; total_bytes];
        let sqrt_dim = (self.dim as f32).sqrt();

        for (i, &value) in vector.iter().enumerate().take(self.dim) {
            if value > 0.0 {
                codes[i / 8] |= 1 << (i % 8);
            }
        }

        // 计算校正因子（与 C++ Faiss 一致）
        // dp_o_o = sum(sign(or_minus_c) * or_minus_c) / ||or_minus_c|| / sqrt(d)
        let mut dp_o_o = 0.0f32;
        for &value in vector.iter().take(self.dim) {
            let sign_val = if value > 0.0 { 1.0 } else { -1.0 };
            dp_o_o += sign_val * value;
        }
        dp_o_o /= norm_l2 * sqrt_dim;

        (codes, norm_l2, dp_o_o)
    }

    /// 编码带质心的向量（用于 IVF 残差量化）
    ///
    /// 返回：(binary_code, centroid_dist, inner_product, sum_xb)
    /// - centroid_dist: ||vector - centroid|| (残差的 L2 范数)
    /// - inner_product: dp_multiplier 校正因子
    /// - sum_xb: 二进制位之和（C++ FactorsData.sum_xb）
    pub fn encode_with_centroid(
        &self,
        vector: &[f32],
        centroid: &[f32],
    ) -> (Vec<u8>, f32, f32, f32) {
        if !self.trained {
            return (vec![0u8; self.dim.div_ceil(8)], 0.0, 0.0, 0.0);
        }

        // 计算残差
        let residual: Vec<f32> = (0..self.dim).map(|i| vector[i] - centroid[i]).collect();

        // 计算残差范数 (centroid_dist)
        let norm_l2sqr: f32 = residual.iter().map(|&x| x * x).sum();
        let norm_l2 = norm_l2sqr.sqrt().max(1e-10);

        // 直接二值化（不旋转，与 C++ Faiss 一致）
        let total_bytes = self.dim.div_ceil(8);
        let mut codes = vec![0u8; total_bytes];
        let sqrt_dim = (self.dim as f32).sqrt();

        // 统计二进制位之和（C++ FactorsData.sum_xb）
        let mut sum_xb = 0.0f32;
        for i in 0..self.dim {
            if residual[i] > 0.0 {
                codes[i / 8] |= 1 << (i % 8);
                sum_xb += 1.0;
            }
        }

        // 计算校正因子（与 C++ Faiss 一致）
        // dp_o_o = sum(sign(or_minus_c) * or_minus_c) / ||or_minus_c|| / sqrt(d)
        let mut dp_o_o = 0.0f32;
        for &value in residual.iter().take(self.dim) {
            let sign_val = if value > 0.0 { 1.0 } else { -1.0 };
            dp_o_o += sign_val * value;
        }
        dp_o_o /= norm_l2 * sqrt_dim;

        (codes, norm_l2, dp_o_o, sum_xb)
    }

    /// 解码 (重建量化向量)
    ///
    /// 返回单位球面上的量化向量（无旋转）
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let sqrt_dim = (self.dim as f32).sqrt();
        (0..self.dim)
            .map(|i| {
                let bit = (codes[i / 8] >> (i % 8)) & 1;
                if bit == 0 {
                    -1.0 / sqrt_dim
                } else {
                    1.0 / sqrt_dim
                }
            })
            .collect()
    }

    /// 构建查询的距离表
    ///
    /// 返回：(query_normalized, query_norm)
    pub fn build_distance_table(&self, query: &[f32]) -> (Vec<f32>, f32) {
        if !self.trained {
            return (vec![0.0f32; self.dim], 0.0);
        }

        // 计算范数
        let norm: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm = norm.max(1e-10);

        // 归一化
        let normalized: Vec<f32> = query.iter().map(|&x| x / norm).collect();

        (normalized, norm)
    }

    /// 使用质心构建查询的距离表
    ///
    /// 返回：(query_residual_normalized, centroid, query_norm, residual_norm)
    pub fn build_distance_table_with_centroid(
        &self,
        query: &[f32],
        centroid: &[f32],
    ) -> (Vec<f32>, Vec<f32>, f32, f32) {
        if !self.trained {
            return (vec![0.0f32; self.dim], vec![], 0.0, 0.0);
        }

        // 计算残差
        let residual: Vec<f32> = query
            .iter()
            .zip(centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        // 残差范数
        let residual_norm: f32 = residual.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let residual_norm = residual_norm.max(1e-10);

        // 归一化残差
        let normalized: Vec<f32> = residual.iter().map(|&x| x / residual_norm).collect();

        (normalized, centroid.to_vec(), residual_norm, 0.0)
    }

    /// 使用距离表计算与编码向量的距离 (qb=0 模式，不量化查询)
    ///
    /// 参考 C++ Faiss RaBitDistanceComputerNotQ::distance_to_code
    ///
    /// C++ 公式：
    /// 1. dot_qo = fvec_masked_sum(rotated_q, binary_data, d)
    ///    即 sum(rotated_q[i] * bit[i])，其中 bit[i] 是数据库向量的第 i 位
    /// 2. c1 = 2 / sqrt(d), c2 = 0, c34 = sum(rotated_q) / sqrt(d)
    /// 3. final_dot = c1 * dot_qo + c2 * sum_xb - c34
    /// 4. pre_dist = or_c_l2sqr + qr_to_c_l2sqr - 2 * dp_multiplier * final_dot
    ///
    /// 注意：query_normalized 实际传入的是归一化残差 (residual/||residual||)
    /// query_norm 是残差的范数 ||residual||
    /// 所以 raw_residual[i] = query_normalized[i] * query_norm
    pub fn compute_distance(
        &self,
        query_normalized: &[f32],
        query_norm: f32,
        data_code: &[u8],
        data_centroid_dist: f32,
        data_ip: f32, // dp_oO = sum(|r|)/||r||/sqrt(d)
        _data_sum_xb: f32,
    ) -> f32 {
        if !self.trained {
            return f32::MAX;
        }

        let sqrt_dim = (self.dim as f32).sqrt();
        let inv_d_sqrt = 1.0 / sqrt_dim;

        // 还原原始残差: rotated_q[i] = query_normalized[i] * query_norm
        // C++ dot_qo = fvec_masked_sum(rotated_q, binary_data, d)
        //            = sum(rotated_q[i] where bit[i]=1)
        let mut dot_qo = 0.0f32;
        let mut sum_q = 0.0f32;
        for i in 0..self.dim {
            let raw_q = query_normalized[i] * query_norm;
            let bit = ((data_code[i / 8] >> (i % 8)) & 1) as f32;
            dot_qo += raw_q * bit;
            sum_q += raw_q;
        }

        // c1 = 2 / sqrt(d), c2 = 0, c34 = sum_q / sqrt(d)
        let c1 = 2.0 * inv_d_sqrt;
        let c34 = sum_q * inv_d_sqrt;
        let final_dot = c1 * dot_qo - c34;

        // C++ dp_multiplier = inv_dp_oO * ||r|| = ||r|| / dp_oO
        let dp_multiplier = if data_ip.abs() > f32::EPSILON {
            data_centroid_dist / data_ip
        } else {
            data_centroid_dist
        };

        let or_c_l2sqr = data_centroid_dist * data_centroid_dist;
        let qr_to_c_l2sqr = query_norm * query_norm;

        let pre_dist = or_c_l2sqr + qr_to_c_l2sqr - 2.0 * dp_multiplier * final_dot;

        pre_dist.max(0.0).sqrt()
    }

    /// 构建 qb=8 查询量化距离表（带质心）
    ///
    /// 参考 C++ Faiss RaBitDistanceComputerQ::set_query (line 392-476)
    ///
    /// 流程：
    /// 1. 计算查询残差 rotated_q = query - centroid
    /// 2. 找 v_min/v_max，计算 delta = (v_max - v_min) / 255
    /// 3. 量化查询到 8-bit: v_qq = round((v_q - v_min) / delta)
    /// 4. 计算系数 c1, c2, c34
    pub fn build_distance_table_q8_with_centroid(
        &self,
        query: &[f32],
        centroid: &[f32],
    ) -> QueryQuantization {
        let d = self.dim;
        let sqrt_d = (d as f32).sqrt();
        let inv_d = 1.0 / sqrt_d;

        // 计算残差
        let mut residual = vec![0.0f32; d];
        let mut qr_to_c_l2sqr = 0.0f32;
        for i in 0..d {
            residual[i] = query[i] - centroid[i];
            qr_to_c_l2sqr += residual[i] * residual[i];
        }

        // 找 v_min / v_max
        let mut v_min = f32::MAX;
        let mut v_max = f32::MIN;
        for &value in residual.iter().take(d) {
            v_min = v_min.min(value);
            v_max = v_max.max(value);
        }

        // 量化参数
        let pow_2_qb = 255.0f32; // 2^8 - 1
        let delta = (v_max - v_min) / pow_2_qb;
        let inv_delta = if delta > 1e-30 { 1.0 / delta } else { 0.0 };

        // 量化查询到 8-bit
        let mut quantized_query = vec![0u8; d];
        let mut sum_qq: usize = 0;
        for i in 0..d {
            let v_qq = ((residual[i] - v_min) * inv_delta).round() as i32;
            let clamped = v_qq.clamp(0, 255) as u8;
            quantized_query[i] = clamped;
            sum_qq += clamped as usize;
        }

        // 计算系数（与 C++ Faiss 一致）
        // c1 = 2 * delta / sqrt(d)
        let c1 = 2.0 * delta * inv_d;
        // c2 = 2 * v_min / sqrt(d)
        let c2 = 2.0 * v_min * inv_d;
        // c34 = (delta * sum_qq + d * v_min) / sqrt(d)
        let c34 = inv_d * (delta * sum_qq as f32 + d as f32 * v_min);

        QueryQuantization {
            quantized_query,
            sum_qq,
            c1,
            c2,
            c34,
            qr_to_c_l2sqr,
        }
    }

    /// 使用 qb=8 量化计算距离
    ///
    /// 参考 C++ Faiss RaBitDistanceComputerQ::distance_to_code (line 325-389)
    ///
    /// 公式：
    /// 1. dot_qo = sum(binary_bit[i] * quantized_query[i])
    /// 2. final_dot = c1 * dot_qo + c2 * sum_xb - c34
    /// 3. pre_dist = or_c_l2sqr + qr_to_c_l2sqr - 2 * dp_multiplier * final_dot
    ///
    /// 注意：C++ dp_multiplier = inv_dp_oO * ||r|| = ||r|| / dp_oO
    /// 其中 dp_oO = sum(|r|) / ||r|| / sqrt(d)（我们存储为 data_ip）
    pub fn compute_distance_q8(
        &self,
        qq: &QueryQuantization,
        data_code: &[u8],
        data_centroid_dist: f32,
        data_ip: f32, // dp_oO
        data_sum_xb: f32,
    ) -> f32 {
        if !self.trained {
            return f32::MAX;
        }

        // 计算 dot_qo = sum(binary_bit[i] * quantized_query[i])
        // 其中 binary_bit[i] 是数据库向量的第 i 位（0 或 1）
        let mut dot_qo = 0.0f32;
        for i in 0..self.dim {
            let bit = ((data_code[i / 8] >> (i % 8)) & 1) as f32;
            dot_qo += bit * qq.quantized_query[i] as f32;
        }

        // final_dot = c1 * dot_qo + c2 * sum_xb - c34
        let final_dot = qq.c1 * dot_qo + qq.c2 * data_sum_xb - qq.c34;

        // C++ dp_multiplier = inv_dp_oO * sqrt(norm_L2sqr)
        //                   = (1/dp_oO) * ||r||
        //                   = data_centroid_dist / data_ip
        let dp_multiplier = if data_ip.abs() > f32::EPSILON {
            data_centroid_dist / data_ip
        } else {
            data_centroid_dist
        };

        // pre_dist = or_c_l2sqr + qr_to_c_l2sqr - 2 * dp_multiplier * final_dot
        let or_c_l2sqr = data_centroid_dist * data_centroid_dist;

        let pre_dist = or_c_l2sqr + qq.qr_to_c_l2sqr - 2.0 * dp_multiplier * final_dot;

        pre_dist.max(0.0)
    }

    /// 使用汉明距离近似
    pub fn compute_distance_hamming(&self, code1: &[u8], code2: &[u8]) -> f32 {
        let mut hamming = 0;
        for (b1, b2) in code1.iter().zip(code2.iter()) {
            hamming += (b1 ^ b2).count_ones() as usize;
        }

        // 转换为 L2 距离近似
        // 对于单位向量：L2^2 = 2 - 2*cos(theta) ≈ 2 * (hamming / dim)
        hamming as f32 * 2.0 / (self.dim as f32)
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn code_size(&self) -> usize {
        self.dim.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_encoder_train() {
        let mut encoder = RaBitQEncoder::new(64);

        // RaBitQ 无需训练，直接标记为已训练
        let data: Vec<f32> = (0..6400).map(|i| (i as f32) * 0.01).collect();
        encoder.train(&data);

        assert!(encoder.is_trained());
    }

    #[test]
    fn test_rabitq_encode_decode() {
        let mut encoder = RaBitQEncoder::new(32);

        // 训练
        let data: Vec<f32> = (0..3200).map(|i| (i as f32) * 0.01).collect();
        encoder.train(&data);

        // 测试编码
        let vector: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (code, centroid_dist, _ip) = encoder.encode(&vector);

        assert_eq!(code.len(), 32_usize.div_ceil(8));
        assert!(centroid_dist > 0.0);

        // 测试解码
        let decoded = encoder.decode(&code);
        assert_eq!(decoded.len(), 32);

        // 解码向量应该在单位球面上
        let decoded_norm: f32 = decoded.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((decoded_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rabitq_encode_with_centroid() {
        let mut encoder = RaBitQEncoder::new(16);

        // 训练
        let data: Vec<f32> = (0..1600).map(|i| (i as f32) * 0.01).collect();
        encoder.train(&data);

        // 测试带质心编码
        let vector: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let centroid: Vec<f32> = vec![0.5; 16];
        let (code, centroid_dist, ip, _sum_xb) = encoder.encode_with_centroid(&vector, &centroid);

        assert_eq!(code.len(), 16_usize.div_ceil(8));
        assert!(centroid_dist > 0.0);

        // 验证校正因子在合理范围内
        // ip = <sign(residual), residual> / ||residual|| / sqrt(d)
        // 应该在 [0, 1] 范围内
        assert!((0.0..=2.0).contains(&ip));
    }

    #[test]
    fn test_rabitq_distance_computation() {
        let mut encoder = RaBitQEncoder::new(32);
        encoder.qb = 0; // 使用 qb=0 模式测试
        encoder.train(&[]);

        // 编码两个向量
        let v1: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..32).map(|i| (i + 16) as f32 * 0.1).collect();

        let (_code1, _dist1, _ip1) = encoder.encode(&v1);
        let (code2, dist2, ip2) = encoder.encode(&v2);

        // 构建查询距离表
        let (query_normalized, query_norm) = encoder.build_distance_table(&v1);

        // 计算距离 (qb=0)
        let distance =
            encoder.compute_distance(&query_normalized, query_norm, &code2, dist2, ip2, 0.0);

        // 距离应该为非负
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_rabitq_q8_distance_computation() {
        let mut encoder = RaBitQEncoder::new(128);
        encoder.qb = 8;
        encoder.train(&[]);

        // 生成一些向量和质心
        let centroid: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let v2: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 + 5.0).collect();

        // 编码数据库向量
        let (code1, dist1, ip1, sum_xb1) = encoder.encode_with_centroid(&v1, &centroid);
        let (code2, dist2, ip2, sum_xb2) = encoder.encode_with_centroid(&v2, &centroid);

        // 用 v1 作为查询，构建 q8 距离表
        let qq = encoder.build_distance_table_q8_with_centroid(&v1, &centroid);

        // v1 对自己应该距离很近
        let d1 = encoder.compute_distance_q8(&qq, &code1, dist1, ip1, sum_xb1);
        // v1 对 v2 应该距离更远
        let d2 = encoder.compute_distance_q8(&qq, &code2, dist2, ip2, sum_xb2);

        assert!(d1 >= 0.0, "d1 should be non-negative, got {}", d1);
        assert!(d2 >= 0.0, "d2 should be non-negative, got {}", d2);
        // v1 距自己应该比距 v2 近
        assert!(d1 < d2, "d1={} should be < d2={}", d1, d2);
    }

    #[test]
    fn test_rabitq_q8_recall() {
        // 测试 qb=8 在小数据集上的召回率
        let dim = 128;
        let n = 1000;
        let nq = 10;
        let k = 10;

        let mut rng = 42u64; // 简单伪随机 (LCG)
        let mut next_f32 = || -> f32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            (rng >> 33) as f32 / (1u64 << 30) as f32 - 1.0
        };

        // 生成数据
        let base: Vec<f32> = (0..n * dim).map(|_| next_f32()).collect();
        let queries: Vec<f32> = (0..nq * dim).map(|_| next_f32()).collect();

        // 计算 ground truth (brute-force L2)
        let mut gt = vec![vec![0usize; k]; nq];
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut dists: Vec<(usize, f32)> = (0..n)
                .map(|j| {
                    let v = &base[j * dim..(j + 1) * dim];
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for ki in 0..k {
                gt[qi][ki] = dists[ki].0;
            }
        }

        // 使用单个质心（简化测试，相当于 nlist=1）
        let centroid = vec![0.0f32; dim];
        let mut encoder = RaBitQEncoder::new(dim);
        encoder.qb = 8;
        encoder.train(&[]);

        // 编码所有数据库向量
        let encoded: Vec<_> = (0..n)
            .map(|j| {
                let v = &base[j * dim..(j + 1) * dim];
                encoder.encode_with_centroid(v, &centroid)
            })
            .collect();

        // 搜索并计算召回率
        let mut total_recall = 0.0;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let qq = encoder.build_distance_table_q8_with_centroid(q, &centroid);

            let mut results: Vec<(usize, f32)> = encoded
                .iter()
                .enumerate()
                .map(|(j, (code, cd, ip, sxb))| {
                    let d = encoder.compute_distance_q8(&qq, code, *cd, *ip, *sxb);
                    (j, d)
                })
                .collect();
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let topk: Vec<usize> = results.iter().take(k).map(|(j, _)| *j).collect();

            let hits = topk.iter().filter(|j| gt[qi].contains(j)).count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / nq as f64;
        println!(
            "qb=8 recall@{}: {:.3} (single centroid, n={})",
            k, avg_recall, n
        );
        assert!(
            avg_recall > 0.5,
            "qb=8 recall@{} should be > 0.5, got {:.3}",
            k,
            avg_recall
        );
    }
}
