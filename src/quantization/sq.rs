//! SQ (Scalar Quantization) 量化器
//! 
//! 将浮点数映射到低精度表示 (通常是 8-bit)
//! 参考: https://faiss.ai/cpp/html/ScalarQuantizer_8bit_avx512_8h.html

/// SQ 量化类型
#[derive(Debug, Clone, Copy)]
pub enum QuantizerType {
    /// 8-bit 均匀量化
    Uniform,
    /// 8-bit 非均匀 (Learnable)
    Learned,
    /// 4-bit 量化
    Quant4,
}

/// SQ 量化器配置
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    pub dim: usize,
    pub bit: usize,           // 量化位数 (4 或 8)
    pub quantizer_type: QuantizerType,
    
    // 量化参数
    pub min_val: f32,        // 最小值
    pub max_val: f32,        // 最大值
    pub scale: f32,          // 缩放因子
    pub offset: f32,         // 偏移
}

impl ScalarQuantizer {
    /// 创建新的量化器
    pub fn new(dim: usize, bit: usize) -> Self {
        Self {
            dim,
            bit: bit.min(8),
            quantizer_type: QuantizerType::Uniform,
            min_val: f32::MAX,
            max_val: f32::MIN,
            scale: 0.0,
            offset: 0.0,
        }
    }
    
    /// 训练量化器 (确定 min/max/scale)
    pub fn train(&mut self, data: &[f32]) {
        if data.is_empty() { return; }
        
        // 计算 min/max
        self.min_val = data.iter().cloned().fold(f32::MAX, f32::min);
        self.max_val = data.iter().cloned().fold(f32::MIN, f32::max);
        
        // 计算 scale 和 offset
        let range = self.max_val - self.min_val;
        let levels = (1 << self.bit) as f32;
        
        self.scale = if range > 0.0 {
            (levels - 1.0) / range
        } else {
            1.0
        };
        self.offset = self.min_val;
    }
    
    /// 量化：将 f32 转换为 u8/u4
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let levels = (1 << self.bit) as f32;
        vector.iter()
            .map(|&v| {
                let scaled = (v - self.offset) * self.scale;
                let quantized = scaled.clamp(0.0, levels - 1.0).round() as u8;
                quantized
            })
            .collect()
    }
    
    /// 解码：将 u8/u4 转换回 f32
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let levels = (1 << self.bit) as f32;
        codes.iter()
            .map(|&c| {
                let v = c as f32 / self.scale + self.offset;
                v.clamp(self.min_val, self.max_val)
            })
            .collect()
    }
    
    /// 计算量化误差
    pub fn compute_error(&self, original: &[f32], reconstructed: &[f32]) -> f32 {
        original.iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// 批量编码
    pub fn encode_batch(&self, vectors: &[f32]) -> Vec<Vec<u8>> {
        let n = vectors.len() / self.dim;
        (0..n)
            .map(|i| {
                let v = &vectors[i * self.dim..(i + 1) * self.dim];
                self.encode(v)
            })
            .collect()
    }
    
    /// 批量解码
    pub fn decode_batch(&self, codes: &[Vec<u8>]) -> Vec<f32> {
        let mut result = Vec::with_capacity(codes.len() * self.dim);
        for code in codes {
            result.extend_from_slice(&self.decode(code));
        }
        result
    }
}

/// SQ8 量化器 (8-bit, 简化别名)
pub type Sq8Quantizer = ScalarQuantizer;

/// SQ4 量化器 (4-bit)
pub struct Sq4Quantizer {
    sq8: ScalarQuantizer,
}

impl Sq4Quantizer {
    pub fn new(dim: usize) -> Self {
        Self {
            sq8: ScalarQuantizer::new(dim, 4),
        }
    }
    
    pub fn train(&mut self, data: &[f32]) {
        self.sq8.train(data);
    }
    
    /// 编码到 4-bit (每 byte 存2个值)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let codes8 = self.sq8.encode(vector);
        let mut result = Vec::with_capacity(codes8.len() / 2);
        
        for chunk in codes8.chunks(2) {
            let low = chunk.get(0).unwrap_or(&0);
            let high = chunk.get(1).unwrap_or(&0);
            let byte = ((high & 0x0F) << 4) | (low & 0x0F);
            result.push(byte);
        }
        
        result
    }
    
    /// 从 4-bit 解码
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut codes8 = Vec::with_capacity(codes.len() * 2);
        
        for &byte in codes {
            codes8.push((byte & 0x0F) as u8);
            codes8.push(((byte >> 4) & 0x0F) as u8);
        }
        
        self.sq8.decode(&codes8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sq8_new() {
        let sq = ScalarQuantizer::new(128, 8);
        assert_eq!(sq.bit, 8);
    }
    
    #[test]
    fn test_sq8_train() {
        let mut sq = ScalarQuantizer::new(4, 8);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        
        sq.train(&data);
        
        assert_eq!(sq.min_val, 0.0);
        assert_eq!(sq.max_val, 3.0);
    }
    
    #[test]
    fn test_sq8_encode_decode() {
        let mut sq = ScalarQuantizer::new(4, 8);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        sq.train(&data);
        
        let codes = sq.encode(&data);
        assert_eq!(codes.len(), 4);
        
        let decoded = sq.decode(&codes);
        assert_eq!(decoded.len(), 4);
        
        // 检查误差
        let error = sq.compute_error(&data, &decoded);
        assert!(error < 0.1, "Error {} too large", error);
    }
    
    #[test]
    fn test_sq4() {
        let mut sq4 = Sq4Quantizer::new(4);
        let data = vec![0.0, 1.0, 2.0, 3.0];
        sq4.train(&data);
        
        let codes = sq4.encode(&data);
        assert_eq!(codes.len(), 2);
        
        let decoded = sq4.decode(&codes);
        assert_eq!(decoded.len(), 4);
    }
}
