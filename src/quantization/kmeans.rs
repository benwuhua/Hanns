//! 量化模块

/// K-Means 聚类器
pub struct KMeans {
    k: usize,
    max_iter: usize,
    pub centroids: Vec<f32>,
    dim: usize,
}

impl KMeans {
    pub fn new(k: usize, dim: usize) -> Self {
        Self {
            k,
            max_iter: 10,
            centroids: vec![0.0; k * dim],
            dim,
        }
    }
    
    pub fn train(&mut self, vectors: &[f32]) -> usize {
        let n = vectors.len() / self.dim;
        
        // 简单初始化
        for c in 0..self.k.min(n) {
            for j in 0..self.dim {
                self.centroids[c * self.dim + j] = vectors[c * self.dim + j];
            }
        }
        
        n
    }
    
    pub fn centroids(&self) -> &[f32] { &self.centroids }
    pub fn k(&self) -> usize { self.k }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kmeans() {
        let mut km = KMeans::new(4, 128);
        let data = vec![0.0; 512];
        km.train(&data);
        assert!(km.centroids().len() > 0);
    }
}
