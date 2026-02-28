//! Faiss binding layer

pub mod index;
pub mod raw;
pub mod mem_index;
pub mod hnsw;
pub mod hnsw_complete;
pub mod ivfpq_complete;
pub mod diskann_complete;
pub mod hnsw_build;
pub mod hnsw_parallel;
pub mod hnsw_search;
pub mod ivf;
pub mod ivfpq;
pub mod ivf_sq8;
pub mod ivf_flat;
pub mod pq;
pub mod pq_simd;
pub mod diskann_beam;
pub mod diskann;
pub mod binary;
pub mod annoy;
pub mod hnsw_quantized;
pub mod hnsw_prq;
pub mod sparse;
pub mod scann;
pub mod ivf_rabitq;
pub mod ivf_flat_cc;
pub mod ivf_sq_cc;
pub mod sparse_inverted;
pub mod binary_hnsw;

pub use index::FaissIndex;
pub use mem_index::MemIndex;
pub use hnsw::HnswIndex;
pub use hnsw_search::HnswSearcher;
pub use ivf::IvfIndex;
pub use ivfpq::IvfPqIndex;
pub use ivf_sq8::IvfSq8Index;
pub use ivf_flat::IvfFlatIndex;
pub use ivf_flat_cc::IvfFlatCcIndex;
pub use ivf_sq_cc::IvfSqCcIndex;
pub use pq::PqEncoder;
pub use diskann::DiskAnnIndex;
pub use scann::{ScaNNIndex, ScaNNConfig};
pub use binary::BinaryIndex;
pub use sparse::{SparseIndex, SparseVector};
pub use hnsw_prq::{HnswPrqIndex, HnswPrqConfig};
pub use ivf_rabitq::{IvfRaBitqIndex, IvfRaBitqConfig};
pub use sparse_inverted::{SparseInvertedIndex, SparseInvertedSearcher, SparseMetricType, InvertedIndexAlgo, ApproxSearchParams};
pub use binary_hnsw::BinaryHnswIndex;

#[cfg(feature = "ffi")]
pub mod rabitq_ffi;
