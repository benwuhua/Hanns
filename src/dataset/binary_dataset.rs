//! Binary Dataset for binary vectors (e.g., for BinaryHNSW)
//!
//! Supports vectors of u8 bits (each bit is a dimension)

use crate::bitset::BitsetView;

/// Binary dataset for bit vectors
#[derive(Clone)]
pub struct BinaryDataset {
    /// Binary vector data (each u8 contains 8 dimensions)
    vectors: Vec<u8>,
    /// Dimension in bits
    dim_bits: usize,
    /// Dimension in bytes (dim_bits / 8 rounded up)
    dim_bytes: usize,
    /// Number of vectors
    num_vectors: usize,
    /// Vector IDs (optional)
    ids: Option<Vec<i64>>,
    /// Soft delete bitmap
    deleted: Option<BitsetView>,
}

impl BinaryDataset {
    /// Create binary dataset from u8 vector data
    pub fn from_vectors(vectors: Vec<u8>, dim_bits: usize) -> Self {
        let dim_bytes = (dim_bits + 7) / 8; // Round up to nearest byte
        let num_vectors = vectors.len() / dim_bytes;
        
        Self {
            vectors,
            dim_bits,
            dim_bytes,
            num_vectors,
            ids: None,
            deleted: None,
        }
    }

    /// Create binary dataset with IDs
    pub fn from_vectors_with_ids(
        vectors: Vec<u8>,
        dim_bits: usize,
        ids: Vec<i64>,
    ) -> Self {
        let dim_bytes = (dim_bits + 7) / 8;
        let num_vectors = vectors.len() / dim_bytes;
        assert_eq!(ids.len(), num_vectors, "IDs count must match vector count");

        Self {
            vectors,
            dim_bits,
            dim_bytes,
            num_vectors,
            ids: Some(ids),
            deleted: None,
        }
    }

    /// Get number of vectors
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Get dimension in bits
    pub fn dim_bits(&self) -> usize {
        self.dim_bits
    }

    /// Get dimension in bytes
    pub fn dim_bytes(&self) -> usize {
        self.dim_bytes
    }

    /// Get binary vector data
    pub fn vectors(&self) -> &[u8] {
        &self.vectors
    }

    /// Get a specific binary vector
    pub fn get_vector(&self, idx: usize) -> Option<&[u8]> {
        if idx >= self.num_vectors {
            return None;
        }

        let start = idx * self.dim_bytes;
        let end = start + self.dim_bytes;
        Some(&self.vectors[start..end])
    }

    /// Get IDs
    pub fn ids(&self) -> Option<&[i64]> {
        self.ids.as_deref()
    }

    /// Set IDs
    pub fn set_ids(&mut self, ids: Vec<i64>) {
        assert_eq!(ids.len(), self.num_vectors);
        self.ids = Some(ids);
    }

    /// Get data size in bytes
    pub fn data_size(&self) -> usize {
        self.vectors.len()
    }

    /// Set soft delete bitmap
    pub fn set_deleted(&mut self, deleted: BitsetView) {
        self.deleted = Some(deleted);
    }

    /// Get soft delete bitmap
    pub fn deleted(&self) -> Option<&BitsetView> {
        self.deleted.as_ref()
    }

    /// Check if vector is deleted
    pub fn is_deleted(&self, idx: usize) -> bool {
        self.deleted.as_ref().map(|d| d.get(idx)).unwrap_or(false)
    }

    /// Get number of valid (non-deleted) vectors
    pub fn num_valid_vectors(&self) -> usize {
        if let Some(deleted) = &self.deleted {
            self.num_vectors - deleted.count()
        } else {
            self.num_vectors
        }
    }
}

impl Default for BinaryDataset {
    fn default() -> Self {
        Self {
            vectors: Vec::new(),
            dim_bits: 0,
            dim_bytes: 0,
            num_vectors: 0,
            ids: None,
            deleted: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_dataset_basic() {
        // 16-bit vectors: 1010101010101010 (0xAA) and 0101010101010101 (0x55)
        let vectors = vec![0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55];
        let dataset = BinaryDataset::from_vectors(vectors, 16);

        assert_eq!(dataset.num_vectors(), 3);
        assert_eq!(dataset.dim_bits(), 16);
        assert_eq!(dataset.dim_bytes(), 2);
        assert_eq!(dataset.data_size(), 6);

        // Check first vector
        let vec0 = dataset.get_vector(0).unwrap();
        assert_eq!(vec0, &[0xAA, 0x55]);

        // Check second vector
        let vec1 = dataset.get_vector(1).unwrap();
        assert_eq!(vec1, &[0xAA, 0x55]);
    }

    #[test]
    fn test_binary_dataset_with_ids() {
        let vectors = vec![0xFF, 0x00, 0x0F, 0xF0];
        let ids = vec![100, 200];
        let dataset = BinaryDataset::from_vectors_with_ids(vectors, 16, ids);

        assert_eq!(dataset.num_vectors(), 2);
        assert_eq!(dataset.ids(), Some(&[100i64, 200][..]));
    }

    #[test]
    fn test_binary_dataset_deleted() {
        let vectors = vec![0xFF, 0x00, 0x0F, 0xF0, 0xAA, 0x55];
        let mut dataset = BinaryDataset::from_vectors(vectors, 16);
        assert_eq!(dataset.num_vectors(), 3);
        assert_eq!(dataset.num_valid_vectors(), 3);

        // Verify default is no deletion
        assert!(!dataset.is_deleted(0));
        assert!(!dataset.is_deleted(1));
        assert!(!dataset.is_deleted(2));
    }
}
