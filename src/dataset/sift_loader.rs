//! SIFT1M Dataset Loader
//!
//! Loads SIFT1M dataset from fvecs format.
//!
//! # Fvecs Format
//! - 4-byte dimension (little-endian u32)
//! - N vectors, each: 4-byte length prefix + dim * 4-byte floats
//!
//! # SIFT1M Dataset
//! - Base: 1,000,000 vectors x 128 dimensions
//! - Query: 10,000 vectors x 128 dimensions
//! - Ground truth: 10,000 queries x 100 nearest neighbors

use crate::dataset::Dataset;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

/// Load SIFT1M base dataset from fvecs file
///
/// # Arguments
/// * `path` - Path to the base.fvecs file
///
/// # Returns
/// * `Ok(Dataset)` - Loaded dataset with 1M vectors
/// * `Err` - I/O error or format error
pub fn load_sift_base<P: AsRef<Path>>(path: P) -> std::io::Result<Dataset> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let vectors = read_fvecs(&mut reader)?;
    let dim = 128;

    Ok(Dataset::from_vectors(vectors, dim))
}

/// Load SIFT1M query dataset from fvecs file
///
/// # Arguments
/// * `path` - Path to the query.fvecs file
///
/// # Returns
/// * `Ok(Dataset)` - Loaded dataset with 10K query vectors
pub fn load_sift_query<P: AsRef<Path>>(path: P) -> std::io::Result<Dataset> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let vectors = read_fvecs(&mut reader)?;
    let dim = 128;

    Ok(Dataset::from_vectors(vectors, dim))
}

/// Load ground truth from ivecs file
///
/// Ground truth format: each query has 100 nearest neighbor IDs
///
/// # Arguments
/// * `path` - Path to the groundtruth.ivecs file
///
/// # Returns
/// * `Ok(Vec<Vec<i32>>)` - Ground truth neighbors for each query
/// * `Err` - I/O error or format error
pub fn load_sift_ground_truth<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vec<i32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    read_ivecs(&mut reader)
}

/// Read vectors from fvecs format
///
/// Format: for each vector
/// - 4 bytes: dimension (u32, little-endian)
/// - dim * 4 bytes: f32 values (little-endian)
fn read_fvecs<R: Read>(reader: &mut R) -> std::io::Result<Vec<f32>> {
    let mut all_vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        // Try to read dimension
        match reader.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut vec_data = vec![0f32; dim];
        for value in vec_data.iter_mut().take(dim) {
            let mut float_buf = [0u8; 4];
            reader.read_exact(&mut float_buf)?;
            *value = f32::from_le_bytes(float_buf);
        }

        all_vectors.extend(vec_data);
    }

    Ok(all_vectors)
}

/// Read integer vectors from ivecs format
///
/// Format: for each vector
/// - 4 bytes: dimension (u32, little-endian)
/// - dim * 4 bytes: i32 values (little-endian)
fn read_ivecs<R: Read>(reader: &mut R) -> std::io::Result<Vec<Vec<i32>>> {
    let mut all_vectors = Vec::new();
    let mut dim_buf = [0u8; 4];

    loop {
        // Try to read dimension
        match reader.read_exact(&mut dim_buf) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }

        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut vec_data = vec![0i32; dim];
        for value in vec_data.iter_mut().take(dim) {
            let mut int_buf = [0u8; 4];
            reader.read_exact(&mut int_buf)?;
            *value = i32::from_le_bytes(int_buf);
        }

        all_vectors.push(vec_data);
    }

    Ok(all_vectors)
}

/// Load complete SIFT1M dataset (base, query, ground truth)
///
/// # Arguments
/// * `base_dir` - Directory containing the SIFT1M files
///
/// # Returns
/// * `Ok(SiftDataset)` - Complete dataset
/// * `Err` - I/O error
fn resolve_sift1m_path(base_dir: &Path, candidates: &[&str]) -> PathBuf {
    candidates
        .iter()
        .map(|name| base_dir.join(name))
        .find(|path| path.exists())
        .unwrap_or_else(|| base_dir.join(candidates[0]))
}

pub fn load_sift1m_complete<P: AsRef<Path>>(base_dir: P) -> std::io::Result<SiftDataset> {
    let base_dir = base_dir.as_ref();

    let base_path = resolve_sift1m_path(base_dir, &["base.fvecs", "sift_base.fvecs"]);
    let query_path = resolve_sift1m_path(base_dir, &["query.fvecs", "sift_query.fvecs"]);
    let gt_path = resolve_sift1m_path(base_dir, &["groundtruth.ivecs", "sift_groundtruth.ivecs"]);

    let base = load_sift_base(&base_path)?;
    let query = load_sift_query(&query_path)?;
    let ground_truth = load_sift_ground_truth(&gt_path)?;

    Ok(SiftDataset {
        base,
        query,
        ground_truth,
    })
}

/// Complete SIFT1M dataset
pub struct SiftDataset {
    /// Base vectors for indexing (1M x 128D)
    pub base: Dataset,
    /// Query vectors for search (10K x 128D)
    pub query: Dataset,
    /// Ground truth neighbors for recall calculation (10K x 100)
    pub ground_truth: Vec<Vec<i32>>,
}

impl SiftDataset {
    /// Get number of base vectors
    pub fn num_base(&self) -> usize {
        self.base.num_vectors()
    }

    /// Get number of query vectors
    pub fn num_query(&self) -> usize {
        self.query.num_vectors()
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.base.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Cursor;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_test_fvecs(path: &Path, vectors: &[Vec<f32>]) {
        let mut data = Vec::new();
        for vector in vectors {
            data.extend_from_slice(&(vector.len() as u32).to_le_bytes());
            for value in vector {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, data).unwrap();
    }

    fn write_test_ivecs(path: &Path, vectors: &[Vec<i32>]) {
        let mut data = Vec::new();
        for vector in vectors {
            data.extend_from_slice(&(vector.len() as u32).to_le_bytes());
            for value in vector {
                data.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, data).unwrap();
    }

    fn unique_temp_dir() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("knowhere-rs-sift-loader-{nanos}"))
    }

    #[test]
    fn test_read_fvecs() {
        // Create test fvecs data: 2 vectors of dim 3
        let mut data = Vec::new();

        // Vector 1: [1.0, 2.0, 3.0]
        data.extend_from_slice(&3u32.to_le_bytes()); // dim
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());

        // Vector 2: [4.0, 5.0, 6.0]
        data.extend_from_slice(&3u32.to_le_bytes()); // dim
        data.extend_from_slice(&4.0f32.to_le_bytes());
        data.extend_from_slice(&5.0f32.to_le_bytes());
        data.extend_from_slice(&6.0f32.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let vectors = read_fvecs(&mut cursor).unwrap();

        assert_eq!(vectors.len(), 6);
        assert_eq!(vectors, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_read_ivecs() {
        // Create test ivecs data: 2 vectors of dim 3
        let mut data = Vec::new();

        // Vector 1: [0, 1, 2]
        data.extend_from_slice(&3u32.to_le_bytes()); // dim
        data.extend_from_slice(&0i32.to_le_bytes());
        data.extend_from_slice(&1i32.to_le_bytes());
        data.extend_from_slice(&2i32.to_le_bytes());

        // Vector 2: [100, 200, 300]
        data.extend_from_slice(&3u32.to_le_bytes()); // dim
        data.extend_from_slice(&100i32.to_le_bytes());
        data.extend_from_slice(&200i32.to_le_bytes());
        data.extend_from_slice(&300i32.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let vectors = read_ivecs(&mut cursor).unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], vec![0, 1, 2]);
        assert_eq!(vectors[1], vec![100, 200, 300]);
    }

    #[test]
    fn test_load_sift1m_complete_accepts_prefixed_layout() {
        let dir = unique_temp_dir();
        fs::create_dir_all(&dir).unwrap();

        let vector = vec![1.0f32; 128];
        write_test_fvecs(&dir.join("sift_base.fvecs"), std::slice::from_ref(&vector));
        write_test_fvecs(&dir.join("sift_query.fvecs"), &[vector]);
        write_test_ivecs(&dir.join("sift_groundtruth.ivecs"), &[vec![7, 8, 9]]);

        let dataset = load_sift1m_complete(&dir).unwrap();

        assert_eq!(dataset.num_base(), 1);
        assert_eq!(dataset.num_query(), 1);
        assert_eq!(dataset.dim(), 128);
        assert_eq!(dataset.ground_truth, vec![vec![7, 8, 9]]);

        fs::remove_dir_all(&dir).unwrap();
    }
}
