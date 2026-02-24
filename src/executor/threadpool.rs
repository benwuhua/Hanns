//! Thread pool executor

use std::thread;

use rayon::prelude::*;

/// Parallel executor for vector operations
pub struct Executor {
    num_threads: usize,
}

impl Executor {
    /// Create a new executor
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads }
    }

    /// Get the default executor
    pub fn default_executor() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        Self { num_threads: num_cpus }
    }

    /// Execute a parallel search across multiple query vectors
    pub fn parallel_search<F, R>(
        &self,
        queries: &[f32],
        dim: usize,
        f: F,
    ) -> Vec<R>
    where
        F: Fn(&[f32]) -> R + Send + Sync,
        R: Send,
    {
        queries
            .par_chunks(dim)
            .map(f)
            .collect()
    }

    /// Execute parallel batch add
    pub fn parallel_add<F, R>(
        &self,
        vectors: &[f32],
        dim: usize,
        f: F,
    ) -> Vec<R>
    where
        F: Fn(&[f32]) -> R + Send + Sync,
        R: Send,
    {
        vectors
            .par_chunks(dim)
            .map(f)
            .collect()
    }

    /// Execute parallel distance computation
    pub fn parallel_distance<L>(
        &self,
        a: &[f32],
        b: &[f32],
        dim: usize,
        metric: L,
    ) -> Vec<f32>
    where
        L: Fn(&[f32], &[f32]) -> f32 + Send + Sync,
    {
        assert_eq!(a.len() % dim, 0);
        assert_eq!(b.len() % dim, 0);
        
        let num_a = a.len() / dim;
        let num_b = b.len() / dim;
        
        // Compute distances between all pairs
        (0..num_a)
            .into_par_iter()
            .flat_map(|i| {
                let a_vec = &a[i * dim..(i + 1) * dim];
                (0..num_b)
                    .map(|j| {
                        let b_vec = &b[j * dim..(j + 1) * dim];
                        metric(a_vec, b_vec)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Execute map-reduce style operation
    pub fn map_reduce<T, U, M>(
        &self,
        data: &[T],
        map: M,
    ) -> Vec<U>
    where
        T: Send + Sync,
        U: Send,
        M: Fn(&T) -> U + Send + Sync,
    {
        data.par_iter()
            .map(map)
            .collect()
    }
}

/// L2 distance
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Inner product
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine similarity
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = inner_product(a, b);
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_search() {
        let executor = Executor::new(2);
        let queries = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dim = 2;
        
        let results: Vec<usize> = executor.parallel_search(&queries, dim, |q| {
            q.len()
        });
        
        assert_eq!(results, vec![2, 2, 2]);
    }
}
