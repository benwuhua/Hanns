//! Metrics 距离度量枚举与计算
//! 
//! 对齐 C++ 的 MetricType，实现 L2, IP, COSINE 等距离计算
//! 
//! SIMD 优化：当启用 "simd" feature 时，使用 SIMD 加速计算

/// 距离度量类型
/// 
/// 与 Milvus/KnowHere 对齐
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum MetricType {
    /// L2 欧氏距离
    #[default]
    L2 = 0,
    /// 内积 (Inner Product)
    IP = 1,
    /// 余弦相似度 (需要归一化向量)
    COSINE = 2,
    /// Jaccard 距离（用于二进制向量）
    Jaccard = 3,
    /// Hamming 距离（用于二进制向量）
    Hamming = 4,
}

impl MetricType {
    /// 从字符串转换
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "l2" | "l2_distance" | "euclidean" => Some(MetricType::L2),
            "ip" | "inner_product" | "dot" => Some(MetricType::IP),
            "cosine" | "cos" | "cosine_similarity" => Some(MetricType::COSINE),
            "jaccard" => Some(MetricType::Jaccard),
            "hamming" => Some(MetricType::Hamming),
            _ => None,
        }
    }
}

/// 距离计算 trait
pub trait Distance {
    /// 计算两个向量的距离
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// 批量计算距离矩阵
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32>;
}

/// L2 距离计算（使用 SIMD 加速）
pub struct L2Distance;

impl L2Distance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for L2Distance {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for L2Distance {
    #[inline]
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        
        #[cfg(feature = "simd")]
        {
            // Use SIMD-optimized L2 distance
            crate::simd::l2_distance(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0f32;
            for i in 0..a.len() {
                let diff = a[i] - b[i];
                sum += diff * diff;
            }
            sum.sqrt()
        }
    }
    
    #[inline]
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        #[cfg(feature = "simd")]
        {
            crate::simd::l2_batch(a, b, dim)
        }
        #[cfg(not(feature = "simd"))]
        {
            let na = a.len() / dim;
            let nb = b.len() / dim;
            
            let mut result = Vec::with_capacity(na * nb);
            
            for i in 0..na {
                let a_start = i * dim;
                let a_vec = &a[a_start..a_start + dim];
                
                for j in 0..nb {
                    let b_start = j * dim;
                    let b_vec = &b[b_start..b_start + dim];
                    result.push(self.compute(a_vec, b_vec));
                }
            }
            
            result
        }
    }
}

/// 内积距离计算（使用 SIMD 加速）
pub struct InnerProductDistance;

impl InnerProductDistance {
    pub fn new() -> Self {
        Self
    }
}

impl Default for InnerProductDistance {
    fn default() -> Self {
        Self::new()
    }
}

impl Distance for InnerProductDistance {
    #[inline]
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        
        #[cfg(feature = "simd")]
        {
            // Use SIMD-optimized inner product
            crate::simd::inner_product(a, b)
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut sum = 0.0f32;
            for i in 0..a.len() {
                sum += a[i] * b[i];
            }
            sum
        }
    }
    
    #[inline]
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        #[cfg(feature = "simd")]
        {
            crate::simd::ip_batch(a, b, dim)
        }
        #[cfg(not(feature = "simd"))]
        {
            let na = a.len() / dim;
            let nb = b.len() / dim;
            
            let mut result = Vec::with_capacity(na * nb);
            
            for i in 0..na {
                let a_start = i * dim;
                let a_vec = &a[a_start..a_start + dim];
                
                for j in 0..nb {
                    let b_start = j * dim;
                    let b_vec = &b[b_start..b_start + dim];
                    result.push(self.compute(a_vec, b_vec));
                }
            }
            
            result
        }
    }
}

/// 余弦相似度计算
pub struct CosineDistance;

impl CosineDistance {
    #[inline]
    fn norm(v: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for &x in v {
            sum += x * x;
        }
        sum.sqrt()
    }
}

impl Distance for CosineDistance {
    #[inline]
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        
        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let norm_a = Self::norm(a);
        let norm_b = Self::norm(b);
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        // 返回 1 - similarity 作为"距离"
        1.0 - (dot / (norm_a * norm_b))
    }
    
    #[inline]
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        let na = a.len() / dim;
        let nb = b.len() / dim;
        
        let mut result = Vec::with_capacity(na * nb);
        
        for i in 0..na {
            let a_start = i * dim;
            let a_vec = &a[a_start..a_start + dim];
            
            for j in 0..nb {
                let b_start = j * dim;
                let b_vec = &b[b_start..b_start + dim];
                result.push(self.compute(a_vec, b_vec));
            }
        }
        
        result
    }
}

/// Hamming 距离计算（用于二进制向量）
pub struct HammingDistance;

impl Distance for HammingDistance {
    #[inline]
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        // 对于二进制向量，计算不同位的数量
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");
        
        let mut distance = 0usize;
        for i in 0..a.len() {
            if (a[i] >= 0.0) != (b[i] >= 0.0) {
                distance += 1;
            }
        }
        distance as f32
    }
    
    #[inline]
    fn compute_batch(&self, a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
        let na = a.len() / dim;
        let nb = b.len() / dim;
        
        let mut result = Vec::with_capacity(na * nb);
        
        for i in 0..na {
            let a_start = i * dim;
            let a_vec = &a[a_start..a_start + dim];
            
            for j in 0..nb {
                let b_start = j * dim;
                let b_vec = &b[b_start..b_start + dim];
                result.push(self.compute(a_vec, b_vec));
            }
        }
        
        result
    }
}

/// 根据 MetricType 获取距离计算器
pub fn get_distance_calculator(metric: MetricType) -> Box<dyn Distance> {
    match metric {
        MetricType::L2 => Box::new(L2Distance),
        MetricType::IP => Box::new(InnerProductDistance),
        MetricType::COSINE => Box::new(CosineDistance),
        MetricType::Hamming => Box::new(HammingDistance),
        MetricType::Jaccard => Box::new(HammingDistance), // Jaccard 用 Hamming 近似
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_distance() {
        let l2 = L2Distance;
        
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        
        assert!((l2.compute(&a, &b) - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_inner_product() {
        let ip = InnerProductDistance;
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        assert_eq!(ip.compute(&a, &b), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }
    
    #[test]
    fn test_cosine_distance() {
        let cos = CosineDistance;
        
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        
        assert!((cos.compute(&a, &b) - 0.0).abs() < 1e-6);
        
        let c = vec![-1.0, 0.0];
        // 夹角 180 度，距离应该是 2
        assert!((cos.compute(&a, &c) - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_hamming_distance() {
        let h = HammingDistance;
        
        // 二进制向量：1.0 表示 bit 1, -1.0 表示 bit 0
        let a = vec![1.0, 1.0, -1.0, -1.0];
        let b = vec![1.0, -1.0, 1.0, -1.0];
        
        // 比较符号：1vs1=same, 1vs-1=diff, -1vs1=diff, -1vs-1=same -> 2 different
        assert_eq!(h.compute(&a, &b), 2.0);
    }
    
    #[test]
    fn test_distance_batch() {
        let l2 = L2Distance;
        
        let a = vec![0.0, 0.0, 3.0, 4.0];  // 2 vectors
        let b = vec![3.0, 4.0, 0.0, 0.0];  // 2 vectors
        
        let result = l2.compute_batch(&a, &b, 2);
        
        // a[0]=[0,0] vs b[0]=[3,4], b[1]=[0,0] -> dist=5, 0
        // a[1]=[3,4] vs b[0]=[3,4], b[1]=[0,0] -> dist=0, 5
        assert_eq!(result.len(), 4);
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!((result[3] - 5.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_metric_type_from_str() {
        assert_eq!(MetricType::from_str("l2"), Some(MetricType::L2));
        assert_eq!(MetricType::from_str("L2"), Some(MetricType::L2));
        assert_eq!(MetricType::from_str("ip"), Some(MetricType::IP));
        assert_eq!(MetricType::from_str("cosine"), Some(MetricType::COSINE));
        assert_eq!(MetricType::from_str("unknown"), None);
    }
}
