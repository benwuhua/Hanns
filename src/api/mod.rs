//! API module - public interfaces

pub mod search;
pub mod index;
pub mod admin;

pub use search::{SearchRequest, SearchResult, RangeSearchResult, Predicate, RangePredicate, IdsPredicate, BitsetPredicate};
pub use index::{IndexType, MetricType, IndexConfig, IndexParams};
pub use admin::Admin;

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
