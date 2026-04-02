//! Legal Matrix for Index × DataType × Metric Combinations
//!
//! Centralized validation layer that enforces legal combinations at entry points.
//! Based on C++ knowhere/index/index_table.h

use super::{DataType, IndexType, MetricType};
use std::collections::HashSet;

/// Legal combination of (IndexType, DataType, MetricType)
type LegalKey = (IndexType, DataType, MetricType);

/// Lazy-initialized legal matrix
static LEGAL_MATRIX: std::sync::OnceLock<LegalMatrix> = std::sync::OnceLock::new();

/// Legal matrix structure
#[derive(Debug)]
pub struct LegalMatrix {
    /// Legal (index, datatype) combinations
    legal_index_datatype: HashSet<(IndexType, DataType)>,
    /// Legal (index, datatype, metric) combinations
    legal_combinations: HashSet<LegalKey>,
    /// Indexes that support mmap
    mmap_supported: HashSet<IndexType>,
}

impl LegalMatrix {
    /// Initialize the legal matrix with C++ knowhere parity rules
    fn new() -> Self {
        let mut legal_index_datatype = HashSet::new();
        let mut legal_combinations = HashSet::new();
        let mut mmap_supported = HashSet::new();

        // === Binary IVF ===
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::BinFlat,
            DataType::Binary,
            vec![MetricType::Hamming],
        );
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::BinIvfFlat,
            DataType::Binary,
            vec![MetricType::Hamming],
        );

        // === Flat (IDMAP) ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::Flat,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-Flat ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfFlat,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-Flat-CC ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfFlatCc,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-PQ ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfPq,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === ScaNN ===
        #[cfg(feature = "scann")]
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::Scann,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-SQ8 ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfSq8,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-SQ-CC ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfSqCc,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === IVF-USQ ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfUsq,
                *dt,
                vec![MetricType::L2],
            );
        }

        // === IVF-TurboQuant ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::IvfTurboQuant,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === HNSW ===
        for dt in &[
            DataType::Float,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int8,
        ] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::Hnsw,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }
        // HNSW with binary
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::Hnsw,
            DataType::Binary,
            vec![MetricType::Hamming],
        );

        // === HNSW-SQ ===
        for dt in &[
            DataType::Float,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int8,
        ] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::HnswSq,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === HNSW-PQ ===
        for dt in &[
            DataType::Float,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int8,
        ] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::HnswPq,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === HNSW-PRQ ===
        for dt in &[
            DataType::Float,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Int8,
        ] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::HnswPrq,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === DiskANN ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::DiskAnn,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === AISAQ ===
        for dt in &[DataType::Float, DataType::Float16, DataType::BFloat16] {
            add_legal(
                &mut legal_index_datatype,
                &mut legal_combinations,
                IndexType::Aisaq,
                *dt,
                vec![MetricType::L2, MetricType::Ip, MetricType::Cosine],
            );
        }

        // === Sparse Indexes ===
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::SparseInverted,
            DataType::SparseFloat,
            vec![MetricType::Ip],
        );
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::SparseWand,
            DataType::SparseFloat,
            vec![MetricType::Ip],
        );
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::SparseWandCc,
            DataType::SparseFloat,
            vec![MetricType::Ip],
        );
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::SparseInvertedCc,
            DataType::SparseFloat,
            vec![MetricType::Ip],
        );

        // === MinHash-LSH ===
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::MinHashLsh,
            DataType::Binary,
            vec![MetricType::Hamming],
        );

        // === Binary HNSW ===
        add_legal(
            &mut legal_index_datatype,
            &mut legal_combinations,
            IndexType::BinaryHnsw,
            DataType::Binary,
            vec![MetricType::Hamming],
        );

        // === Mmap-supported indexes ===
        mmap_supported.extend(vec![
            IndexType::BinFlat,
            IndexType::BinIvfFlat,
            IndexType::Flat,
            IndexType::IvfFlat,
            IndexType::IvfPq,
            #[cfg(feature = "scann")]
            IndexType::Scann,
            IndexType::IvfSq8,
            IndexType::IvfSqCc,
            IndexType::IvfUsq,
            IndexType::Hnsw,
            IndexType::HnswSq,
            IndexType::HnswPq,
            IndexType::HnswPrq,
            IndexType::SparseInverted,
            IndexType::SparseWand,
        ]);

        LegalMatrix {
            legal_index_datatype,
            legal_combinations,
            mmap_supported,
        }
    }

    /// Get or initialize the legal matrix
    pub fn instance() -> &'static Self {
        LEGAL_MATRIX.get_or_init(Self::new)
    }

    /// Check if (index_type, data_type) is legal
    pub fn is_legal_index_datatype(&self, index_type: IndexType, data_type: DataType) -> bool {
        self.legal_index_datatype.contains(&(index_type, data_type))
    }

    /// Check if (index_type, data_type, metric_type) is legal
    pub fn is_legal_combination(
        &self,
        index_type: IndexType,
        data_type: DataType,
        metric_type: MetricType,
    ) -> bool {
        self.legal_combinations
            .contains(&(index_type, data_type, metric_type))
    }

    /// Check if index supports mmap
    pub fn supports_mmap(&self, index_type: IndexType) -> bool {
        self.mmap_supported.contains(&index_type)
    }

    /// Get all legal data types for an index
    pub fn legal_data_types(&self, index_type: IndexType) -> Vec<DataType> {
        self.legal_index_datatype
            .iter()
            .filter(|(idx, _)| *idx == index_type)
            .map(|(_, dt)| *dt)
            .collect()
    }

    /// Get all legal metrics for (index, datatype)
    pub fn legal_metrics(&self, index_type: IndexType, data_type: DataType) -> Vec<MetricType> {
        self.legal_combinations
            .iter()
            .filter(|(idx, dt, _)| *idx == index_type && *dt == data_type)
            .map(|(_, _, mt)| *mt)
            .collect()
    }
}

/// Helper to add legal combinations
fn add_legal(
    legal_index_datatype: &mut HashSet<(IndexType, DataType)>,
    legal_combinations: &mut HashSet<LegalKey>,
    index_type: IndexType,
    data_type: DataType,
    metrics: Vec<MetricType>,
) {
    legal_index_datatype.insert((index_type, data_type));
    for metric in metrics {
        legal_combinations.insert((index_type, data_type, metric));
    }
}

/// Validate index configuration
///
/// Returns Ok(()) if legal, Err with description if illegal
pub fn validate_index_config(
    index_type: IndexType,
    data_type: DataType,
    metric_type: MetricType,
) -> Result<(), String> {
    let matrix = LegalMatrix::instance();

    // Check (index, datatype) combination
    if !matrix.is_legal_index_datatype(index_type, data_type) {
        let legal_types = matrix.legal_data_types(index_type);
        return Err(format!(
            "Illegal combination: index_type={:?} does not support data_type={:?}. Legal data types: {:?}",
            index_type, data_type, legal_types
        ));
    }

    // Check (index, datatype, metric) combination
    if !matrix.is_legal_combination(index_type, data_type, metric_type) {
        let legal_metrics = matrix.legal_metrics(index_type, data_type);
        return Err(format!(
            "Illegal combination: ({:?}, {:?}) does not support metric={:?}. Legal metrics: {:?}",
            index_type, data_type, metric_type, legal_metrics
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legal_matrix_hnsw_float() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.is_legal_combination(IndexType::Hnsw, DataType::Float, MetricType::L2));
        assert!(matrix.is_legal_combination(IndexType::Hnsw, DataType::Float, MetricType::Ip));
        assert!(matrix.is_legal_combination(IndexType::Hnsw, DataType::Float, MetricType::Cosine));
    }

    #[test]
    fn test_legal_matrix_hnsw_binary() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.is_legal_combination(
            IndexType::Hnsw,
            DataType::Binary,
            MetricType::Hamming
        ));
        // Binary HNSW should not support L2
        assert!(!matrix.is_legal_combination(IndexType::Hnsw, DataType::Binary, MetricType::L2));
    }

    #[test]
    fn test_legal_matrix_ivf_sq8() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.is_legal_combination(IndexType::IvfSq8, DataType::Float, MetricType::L2));
        assert!(matrix.is_legal_index_datatype(IndexType::IvfSq8, DataType::Float16));
        // IVF-SQ8 should not support binary
        assert!(!matrix.is_legal_index_datatype(IndexType::IvfSq8, DataType::Binary));
    }

    #[test]
    fn test_legal_matrix_sparse() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.is_legal_combination(
            IndexType::SparseInverted,
            DataType::SparseFloat,
            MetricType::Ip
        ));
        // Sparse should not support L2
        assert!(!matrix.is_legal_combination(
            IndexType::SparseInverted,
            DataType::SparseFloat,
            MetricType::L2
        ));
    }

    #[test]
    fn test_legal_matrix_diskann() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.is_legal_combination(IndexType::DiskAnn, DataType::Float, MetricType::L2));
        assert!(matrix.is_legal_index_datatype(IndexType::DiskAnn, DataType::Float16));
        // DiskANN does not support Int8
        assert!(!matrix.is_legal_index_datatype(IndexType::DiskAnn, DataType::Int8));
    }

    #[test]
    fn test_validate_index_config() {
        // Legal combination
        assert!(validate_index_config(IndexType::Hnsw, DataType::Float, MetricType::L2).is_ok());

        // Illegal datatype
        assert!(
            validate_index_config(IndexType::IvfSq8, DataType::Binary, MetricType::L2).is_err()
        );

        // Illegal metric
        assert!(validate_index_config(
            IndexType::SparseInverted,
            DataType::SparseFloat,
            MetricType::L2
        )
        .is_err());
    }

    #[test]
    fn test_validate_index_config_ivf_exrabitq_l2_only() {
        assert!(
            validate_index_config(IndexType::IvfUsq, DataType::Float, MetricType::L2).is_ok()
        );
        assert!(
            validate_index_config(IndexType::IvfUsq, DataType::Float, MetricType::Ip).is_err()
        );
        assert!(validate_index_config(
            IndexType::IvfUsq,
            DataType::Binary,
            MetricType::Hamming
        )
        .is_err());
    }

    #[test]
    fn test_mmap_support() {
        let matrix = LegalMatrix::instance();
        assert!(matrix.supports_mmap(IndexType::Hnsw));
        assert!(matrix.supports_mmap(IndexType::IvfFlat));
        assert!(!matrix.supports_mmap(IndexType::MinHashLsh)); // Not in mmap list
    }

    #[test]
    fn test_get_legal_data_types() {
        let matrix = LegalMatrix::instance();
        let types = matrix.legal_data_types(IndexType::Hnsw);
        assert!(types.contains(&DataType::Float));
        assert!(types.contains(&DataType::Float16));
        assert!(types.contains(&DataType::BFloat16));
        assert!(types.contains(&DataType::Int8));
        assert!(types.contains(&DataType::Binary));
    }

    #[test]
    fn test_get_legal_metrics() {
        let matrix = LegalMatrix::instance();
        let metrics = matrix.legal_metrics(IndexType::Hnsw, DataType::Float);
        assert!(metrics.contains(&MetricType::L2));
        assert!(metrics.contains(&MetricType::Ip));
        assert!(metrics.contains(&MetricType::Cosine));
        assert!(!metrics.contains(&MetricType::Hamming));
    }
}
