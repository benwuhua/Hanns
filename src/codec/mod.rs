//! Codec module - serialization for vectors and indices

pub mod index;
pub mod vector;

pub use index::IndexCodec;
pub use vector::VectorCodec;
