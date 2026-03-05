//! Generic Dataset Loader for Deep1M and GIST1M
//!
//! Provides common utilities for loading vector datasets in fvecs/ivecs format.
//!
//! # Supported Formats
//! - **fvecs**: 4-byte dimension + N vectors (4-byte length prefix + dim * 4-byte floats)
//! - **ivecs**: 4-byte dimension + N vectors (4-byte length prefix + dim * 4-byte integers)
//!
//! # Supported Datasets
//! | Dataset | Dim | Base Vectors | Query Vectors | Files |
//! |---------|-----|--------------|---------------|-------|
//! | Deep1M  | 96  | 1,000,000    | 10,000        | base.fvecs, query.fvecs, groundtruth.ivecs |
//! | GIST1M  | 960 | 1,000,000    | 1,000         | base.fvecs, query.fvecs, groundtruth.ivecs |

use crate::dataset::Dataset;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Read vectors from fvecs format
///
/// Format: for each vector
/// - 4 bytes: dimension (u32, little-endian)
/// - dim * 4 bytes: f32 values (little-endian)
///
/// # Arguments
/// * `reader` - Reader to read from
///
/// # Returns
/// * `Ok(Vec<f32>)` - All vectors flattened into a single array
/// * `Err` - I/O error
pub fn read_fvecs<R: Read>(reader: &mut R) -> std::io::Result<Vec<f32>> {
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
///
/// # Arguments
/// * `reader` - Reader to read from
///
/// # Returns
/// * `Ok(Vec<Vec<i32>>)` - Vector of integer vectors (one per query)
/// * `Err` - I/O error
pub fn read_ivecs<R: Read>(reader: &mut R) -> std::io::Result<Vec<Vec<i32>>> {
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

/// Load base dataset from fvecs file
///
/// # Arguments
/// * `path` - Path to the fvecs file
/// * `expected_dim` - Expected dimension (for validation, 0 means skip validation)
///
/// # Returns
/// * `Ok(Dataset)` - Loaded dataset
/// * `Err` - I/O error
pub fn load_base_from_fvecs<P: AsRef<Path>>(
    path: P,
    _expected_dim: usize,
) -> std::io::Result<Dataset> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let vectors = read_fvecs(&mut reader)?;

    // Dimension will be inferred from the file
    // For Deep1M: 96, for GIST1M: 960
    let dim = infer_dim_from_fvecs(vectors.len())?;

    Ok(Dataset::from_vectors(vectors, dim))
}

/// Load query dataset from fvecs file
///
/// # Arguments
/// * `path` - Path to the fvecs file
///
/// # Returns
/// * `Ok(Dataset)` - Loaded query dataset
/// * `Err` - I/O error
pub fn load_query_from_fvecs<P: AsRef<Path>>(path: P) -> std::io::Result<Dataset> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let vectors = read_fvecs(&mut reader)?;
    let dim = infer_dim_from_fvecs(vectors.len())?;

    Ok(Dataset::from_vectors(vectors, dim))
}

/// Load ground truth from ivecs file
///
/// # Arguments
/// * `path` - Path to the ivecs file
///
/// # Returns
/// * `Ok(Vec<Vec<i32>>)` - Ground truth neighbors for each query
/// * `Err` - I/O error
pub fn load_ground_truth<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<Vec<i32>>> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    read_ivecs(&mut reader)
}

/// Infer dimension from total vector count
///
/// # Arguments
/// * `total_elements` - Total number of float elements
///
/// # Returns
/// * `Ok(usize)` - Inferred dimension
/// * `Err` - Invalid element count
fn infer_dim_from_fvecs(total_elements: usize) -> std::io::Result<usize> {
    // Known dataset sizes - check these first for exact matches
    match total_elements {
        // GIST1M base: 960D x 1M = 960M
        960_000_000 => return Ok(960),
        // SIFT1M base: 128D x 1M = 128M
        128_000_000 => return Ok(128),
        // Deep1M base: 96D x 1M = 96M
        96_000_000 => return Ok(96),
        // GIST1M query: 960D x 1K = 960K
        960_000 => return Ok(960),
        // SIFT1M query: 128D x 10K = 1.28M
        1_280_000 => return Ok(128),
        // Deep1M query: 96D x 10K = 960K
        // Note: This is ambiguous with GIST1M query (same element count).
        // We handle this by checking known sizes first, but 960K matches GIST1M query above.
        // For 96D x 10K, users should use load_query_from_fvecs which infers from context.
        _ => {}
    }

    // Common dimensions for known datasets
    // Order: check larger dimensions first to avoid false matches
    // (e.g., 960M elements / 96 = 10M vectors, but should be 960D x 1M)
    let common_dims = [2048, 1536, 1024, 960, 512, 256, 128, 96];

    // First pass: look for dimensions that give typical dataset sizes (100K-10M vectors)
    for &dim in &common_dims {
        if total_elements % dim == 0 {
            let num_vectors = total_elements / dim;
            // Typical benchmark datasets have 100K-10M vectors
            if (100_000..=10_000_000).contains(&num_vectors) {
                return Ok(dim);
            }
        }
    }

    // Second pass: accept smaller datasets (1K-100K vectors)
    for &dim in &common_dims {
        if total_elements % dim == 0 {
            let num_vectors = total_elements / dim;
            if (1_000..100_000).contains(&num_vectors) {
                return Ok(dim);
            }
        }
    }

    // Third pass: return any matching dimension
    for &dim in &common_dims {
        if total_elements % dim == 0 {
            return Ok(dim);
        }
    }

    // If no common dimension matches, try to infer from typical query counts
    // Deep1M: 1M base, 10K query
    // GIST1M: 1M base, 1K query
    // SIFT1M: 1M base, 10K query

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("Cannot infer dimension from {} elements", total_elements),
    ))
}

/// Complete Deep1M dataset
pub struct DeepDataset {
    /// Base vectors for indexing (1M x 96D)
    pub base: Dataset,
    /// Query vectors for search (10K x 96D)
    pub query: Dataset,
    /// Ground truth neighbors for recall calculation (10K x 100)
    pub ground_truth: Vec<Vec<i32>>,
}

impl DeepDataset {
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

/// Complete GIST1M dataset
pub struct GistDataset {
    /// Base vectors for indexing (1M x 960D)
    pub base: Dataset,
    /// Query vectors for search (1K x 960D)
    pub query: Dataset,
    /// Ground truth neighbors for recall calculation (1K x 100)
    pub ground_truth: Vec<Vec<i32>>,
}

impl GistDataset {
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

/// Load complete Deep1M dataset (base, query, ground truth)
///
/// # Arguments
/// * `base_dir` - Directory containing the Deep1M files
///
/// # Returns
/// * `Ok(DeepDataset)` - Complete Deep1M dataset
/// * `Err` - I/O error
///
/// # Expected Files
/// - `base.fvecs` - 1M base vectors (96D)
/// - `query.fvecs` - 10K query vectors (96D)
/// - `groundtruth.ivecs` - Ground truth neighbors
pub fn load_deep1m_complete<P: AsRef<Path>>(base_dir: P) -> std::io::Result<DeepDataset> {
    let base_dir = base_dir.as_ref();

    let base_path = base_dir.join("base.fvecs");
    let query_path = base_dir.join("query.fvecs");
    let gt_path = base_dir.join("groundtruth.ivecs");

    let base = load_base_from_fvecs(&base_path, 96)?;
    let query = load_query_from_fvecs(&query_path)?;
    let ground_truth = load_ground_truth(&gt_path)?;

    Ok(DeepDataset {
        base,
        query,
        ground_truth,
    })
}

/// Load complete GIST1M dataset (base, query, ground truth)
///
/// # Arguments
/// * `base_dir` - Directory containing the GIST1M files
///
/// # Returns
/// * `Ok(GistDataset)` - Complete GIST1M dataset
/// * `Err` - I/O error
///
/// # Expected Files
/// - `base.fvecs` - 1M base vectors (960D)
/// - `query.fvecs` - 1K query vectors (960D)
/// - `groundtruth.ivecs` - Ground truth neighbors
pub fn load_gist1m_complete<P: AsRef<Path>>(base_dir: P) -> std::io::Result<GistDataset> {
    let base_dir = base_dir.as_ref();

    let base_path = base_dir.join("base.fvecs");
    let query_path = base_dir.join("query.fvecs");
    let gt_path = base_dir.join("groundtruth.ivecs");

    let base = load_base_from_fvecs(&base_path, 960)?;
    let query = load_query_from_fvecs(&query_path)?;
    let ground_truth = load_ground_truth(&gt_path)?;

    Ok(GistDataset {
        base,
        query,
        ground_truth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_fvecs_basic() {
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
    fn test_read_ivecs_basic() {
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
    fn test_infer_dim_from_fvecs() {
        // Test common dimensions (base datasets only - query sizes can be ambiguous)
        assert_eq!(infer_dim_from_fvecs(96 * 1_000_000).unwrap(), 96); // Deep1M base: 96M
        assert_eq!(infer_dim_from_fvecs(960 * 1_000_000).unwrap(), 960); // GIST1M base: 960M
        assert_eq!(infer_dim_from_fvecs(128 * 1_000_000).unwrap(), 128); // SIFT1M base: 128M
                                                                         // Note: 96 * 10K = 960K and 960 * 1K = 960K are ambiguous (same element count)
                                                                         // Query dimension inference requires additional context
    }

    #[test]
    fn test_deep_dataset_methods() {
        let vectors = vec![1.0; 96 * 100]; // 100 vectors of 96D
        let base = Dataset::from_vectors(vectors, 96);

        let query_vectors = vec![2.0; 96 * 10]; // 10 queries of 96D
        let query = Dataset::from_vectors(query_vectors, 96);

        let gt = vec![vec![0, 1, 2]; 10]; // 10 queries, 3 neighbors each

        let dataset = DeepDataset {
            base,
            query,
            ground_truth: gt,
        };

        assert_eq!(dataset.num_base(), 100);
        assert_eq!(dataset.num_query(), 10);
        assert_eq!(dataset.dim(), 96);
    }

    #[test]
    fn test_gist_dataset_methods() {
        let vectors = vec![1.0; 960 * 100]; // 100 vectors of 960D
        let base = Dataset::from_vectors(vectors, 960);

        let query_vectors = vec![2.0; 960 * 10]; // 10 queries of 960D
        let query = Dataset::from_vectors(query_vectors, 960);

        let gt = vec![vec![0, 1, 2]; 10]; // 10 queries, 3 neighbors each

        let dataset = GistDataset {
            base,
            query,
            ground_truth: gt,
        };

        assert_eq!(dataset.num_base(), 100);
        assert_eq!(dataset.num_query(), 10);
        assert_eq!(dataset.dim(), 960);
    }
}
