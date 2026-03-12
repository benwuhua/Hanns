//! HDF5 Dataset Loader for ann-benchmarks format
//!
//! Supports loading standard HDF5 datasets from ann-benchmarks:
//! - GloVe, SIFT, Deep, GIST in HDF5 format
//!
//! # Dataset Format
//! HDF5 files from ann-benchmarks contain:
//! - `train`: Training vectors (N x D)
//! - `test`: Query vectors (Q x D)
//! - `neighbors`: Ground truth neighbor IDs (Q x K)
//! - `distances`: Ground truth distances (Q x K)
//!
//! # Feature Flag
//! Requires `hdf5` feature to be enabled:
//! ```toml
//! [dependencies]
//! knowhere-rs = { version = "0.1", features = ["hdf5"] }
//! ```
//!
//! And HDF5 C library installed on your system:
//! - macOS: `brew install hdf5`
//! - Ubuntu: `apt-get install libhdf5-dev`

#![cfg(feature = "hdf5")]

use std::path::Path;
use thiserror::Error;

use crate::dataset::Dataset;

/// HDF5 dataset loading errors
#[derive(Error, Debug)]
pub enum Hdf5LoaderError {
    #[error("HDF5 error: {0}")]
    Hdf5Error(String),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("Invalid dataset shape: {0}")]
    InvalidShape(String),

    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for HDF5 loader operations
pub type Result<T> = std::result::Result<T, Hdf5LoaderError>;

/// HDF5 Dataset container
///
/// Contains all components loaded from an ann-benchmarks HDF5 file
#[derive(Clone)]
pub struct Hdf5Dataset {
    /// Training vectors (N x D)
    pub train: Dataset,
    /// Query/test vectors (Q x D)
    pub test: Dataset,
    /// Ground truth neighbor IDs (Q x K)
    pub neighbors: Vec<Vec<i32>>,
    /// Ground truth distances (Q x K)
    pub distances: Vec<Vec<f32>>,
}

impl Hdf5Dataset {
    /// Create a new Hdf5Dataset
    pub fn new(
        train: Dataset,
        test: Dataset,
        neighbors: Vec<Vec<i32>>,
        distances: Vec<Vec<f32>>,
    ) -> Self {
        Self {
            train,
            test,
            neighbors,
            distances,
        }
    }

    /// Get number of training vectors
    pub fn num_train(&self) -> usize {
        self.train.num_vectors()
    }

    /// Get number of query vectors
    pub fn num_test(&self) -> usize {
        self.test.num_vectors()
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.train.dim()
    }

    /// Get number of ground truth neighbors per query
    pub fn num_neighbors(&self) -> usize {
        self.neighbors.first().map(|v| v.len()).unwrap_or(0)
    }
}

/// Load dataset from HDF5 file
///
/// # Arguments
/// * `path` - Path to HDF5 file
///
/// # Returns
/// * `Ok(Hdf5Dataset)` - Loaded dataset with train, test, neighbors, distances
/// * `Err(Hdf5LoaderError)` - Error if file not found or invalid format
///
/// # Example
/// ```no_run
/// use knowhere_rs::dataset::load_hdf5_dataset;
///
/// let dataset = load_hdf5_dataset("glove-100.hdf5").unwrap();
/// println!("Train: {}, Test: {}", dataset.num_train(), dataset.num_test());
/// ```
pub fn load_hdf5_dataset<P: AsRef<Path>>(path: P) -> Result<Hdf5Dataset> {
    let path = path.as_ref();

    // Check if file exists
    if !path.exists() {
        return Err(Hdf5LoaderError::FileNotFound(path.display().to_string()));
    }

    // Open HDF5 file
    let file = hdf5::File::open(path).map_err(|e| Hdf5LoaderError::Hdf5Error(e.to_string()))?;

    // Load train dataset
    let train_dataset = file
        .dataset("train")
        .map_err(|_| Hdf5LoaderError::DatasetNotFound("train".to_string()))?;

    let train_data = read_f32_dataset(&train_dataset, "train")?;
    let train_dim = get_dataset_dim(&train_dataset, "train")?;
    let train = Dataset::from_vectors(train_data, train_dim);

    // Load test dataset
    let test_dataset = file
        .dataset("test")
        .map_err(|_| Hdf5LoaderError::DatasetNotFound("test".to_string()))?;

    let test_data = read_f32_dataset(&test_dataset, "test")?;
    let test_dim = get_dataset_dim(&test_dataset, "test")?;
    let test = Dataset::from_vectors(test_data, test_dim);

    // Verify dimensions match
    if train_dim != test_dim {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "Train dim {} != Test dim {}",
            train_dim, test_dim
        )));
    }

    // Load neighbors (ground truth)
    let neighbors_dataset = file
        .dataset("neighbors")
        .map_err(|_| Hdf5LoaderError::DatasetNotFound("neighbors".to_string()))?;

    let neighbors = read_i32_2d(&neighbors_dataset, "neighbors")?;

    // Load distances (ground truth)
    let distances_dataset = file
        .dataset("distances")
        .map_err(|_| Hdf5LoaderError::DatasetNotFound("distances".to_string()))?;

    let distances = read_f32_2d(&distances_dataset, "distances")?;

    // Verify neighbors and distances have same shape
    if neighbors.len() != distances.len() {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "Neighbors rows {} != Distances rows {}",
            neighbors.len(),
            distances.len()
        )));
    }

    if !neighbors.is_empty() && !distances.is_empty() && neighbors[0].len() != distances[0].len() {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "Neighbors cols {} != Distances cols {}",
            neighbors[0].len(),
            distances[0].len()
        )));
    }

    // Verify test queries match ground truth rows
    if test.num_vectors() != neighbors.len() {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "Test vectors {} != Neighbor rows {}",
            test.num_vectors(),
            neighbors.len()
        )));
    }

    Ok(Hdf5Dataset::new(train, test, neighbors, distances))
}

/// Read 1D/2D f32 dataset and return flattened vector
fn read_f32_dataset(dataset: &hdf5::Dataset, name: &str) -> Result<Vec<f32>> {
    // Read into ndarray and convert to Vec<f32>
    let array: ndarray::ArrayD<f32> = dataset
        .read()
        .map_err(|e| Hdf5LoaderError::Hdf5Error(format!("{}: {}", name, e)))?;

    Ok(array.into_raw_vec())
}

/// Read 2D i64 dataset and return as Vec<Vec<i32>>
fn read_i32_2d(dataset: &hdf5::Dataset, name: &str) -> Result<Vec<Vec<i32>>> {
    let shape = dataset.shape();

    if shape.len() != 2 {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "{}: expected 2D array, got {}D",
            name,
            shape.len()
        )));
    }

    let rows = shape[0];
    let cols = shape[1];

    // Read into ndarray and convert
    let array: ndarray::ArrayD<i32> = dataset
        .read()
        .map_err(|e| Hdf5LoaderError::Hdf5Error(format!("{}: {}", name, e)))?;

    let data = array.into_raw_vec();

    // Reshape into 2D vector
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        result.push(data[start..end].to_vec());
    }

    Ok(result)
}

/// Read 2D f32 dataset and return as Vec<Vec<f32>>
fn read_f32_2d(dataset: &hdf5::Dataset, name: &str) -> Result<Vec<Vec<f32>>> {
    let shape = dataset.shape();

    if shape.len() != 2 {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "{}: expected 2D array, got {}D",
            name,
            shape.len()
        )));
    }

    let rows = shape[0];
    let cols = shape[1];

    // Read into ndarray and convert
    let array: ndarray::ArrayD<f32> = dataset
        .read()
        .map_err(|e| Hdf5LoaderError::Hdf5Error(format!("{}: {}", name, e)))?;

    let data = array.into_raw_vec();

    // Reshape into 2D vector
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let start = i * cols;
        let end = start + cols;
        result.push(data[start..end].to_vec());
    }

    Ok(result)
}

/// Get dimension from dataset shape
fn get_dataset_dim(dataset: &hdf5::Dataset, name: &str) -> Result<usize> {
    let shape = dataset.shape();

    if shape.is_empty() {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "{}: empty shape",
            name
        )));
    }

    if shape.len() != 2 {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "{}: expected 2D (N, D), got {}D",
            name,
            shape.len()
        )));
    }

    // Dimension is the second element (N, D)
    Ok(shape[1])
}

/// Load dataset with custom dataset names
///
/// Some HDF5 files use different names (e.g., "base" instead of "train")
///
/// # Arguments
/// * `path` - Path to HDF5 file
/// * `train_name` - Name of training dataset (default: "train")
/// * `test_name` - Name of test dataset (default: "test")
/// * `neighbors_name` - Name of neighbors dataset (default: "neighbors")
/// * `distances_name` - Name of distances dataset (default: "distances")
pub fn load_hdf5_dataset_custom<P: AsRef<Path>>(
    path: P,
    train_name: &str,
    test_name: &str,
    neighbors_name: &str,
    distances_name: &str,
) -> Result<Hdf5Dataset> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(Hdf5LoaderError::FileNotFound(path.display().to_string()));
    }

    let file = hdf5::File::open(path).map_err(|e| Hdf5LoaderError::Hdf5Error(e.to_string()))?;

    // Load train
    let train_dataset = file
        .dataset(train_name)
        .map_err(|_| Hdf5LoaderError::DatasetNotFound(train_name.to_string()))?;

    let train_data = read_f32_dataset(&train_dataset, train_name)?;
    let train_dim = get_dataset_dim(&train_dataset, train_name)?;
    let train = Dataset::from_vectors(train_data, train_dim);

    // Load test
    let test_dataset = file
        .dataset(test_name)
        .map_err(|_| Hdf5LoaderError::DatasetNotFound(test_name.to_string()))?;

    let test_data = read_f32_dataset(&test_dataset, test_name)?;
    let test_dim = get_dataset_dim(&test_dataset, test_name)?;
    let test = Dataset::from_vectors(test_data, test_dim);

    if train_dim != test_dim {
        return Err(Hdf5LoaderError::InvalidShape(format!(
            "Train dim {} != Test dim {}",
            train_dim, test_dim
        )));
    }

    // Load neighbors
    let neighbors_dataset = file
        .dataset(neighbors_name)
        .map_err(|_| Hdf5LoaderError::DatasetNotFound(neighbors_name.to_string()))?;

    let neighbors = read_i32_2d(&neighbors_dataset, neighbors_name)?;

    // Load distances
    let distances_dataset = file
        .dataset(distances_name)
        .map_err(|_| Hdf5LoaderError::DatasetNotFound(distances_name.to_string()))?;

    let distances = read_f32_2d(&distances_dataset, distances_name)?;

    Ok(Hdf5Dataset::new(train, test, neighbors, distances))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_dataset_not_found() {
        let result = load_hdf5_dataset("/nonexistent/path/file.hdf5");
        assert!(matches!(result, Err(Hdf5LoaderError::FileNotFound(_))));
    }

    #[test]
    fn test_hdf5_dataset_struct() {
        let train = Dataset::from_vectors(vec![1.0, 2.0, 3.0, 4.0], 2);
        let test = Dataset::from_vectors(vec![5.0, 6.0], 2);
        let neighbors = vec![vec![0, 1], vec![1, 0]];
        let distances = vec![vec![0.0, 1.0], vec![0.5, 1.5]];

        let dataset = Hdf5Dataset::new(train, test, neighbors, distances);

        assert_eq!(dataset.num_train(), 2);
        assert_eq!(dataset.num_test(), 1);
        assert_eq!(dataset.dim(), 2);
        assert_eq!(dataset.num_neighbors(), 2);
    }
}
