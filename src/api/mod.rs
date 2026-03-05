//! API module - public interfaces

pub mod admin;
pub mod index;
pub mod search;

pub use admin::Admin;
pub use index::{IndexConfig, IndexParams, IndexType, MetricType};
pub use search::{
    BitsetPredicate, IdsPredicate, Predicate, RangePredicate, RangeSearchResult, SearchRequest,
    SearchResult,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum KnowhereError {
    #[error("Faiss error: {0}")]
    Faiss(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid argument: {0}")]
    InvalidArg(String),

    #[error("Index not found: {0}")]
    NotFound(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Index not trained: {0}")]
    IndexNotTrained(String),
}

pub type Result<T> = std::result::Result<T, KnowhereError>;
