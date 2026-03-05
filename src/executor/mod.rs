//! Executor module - concurrent task execution

pub mod concurrent;
pub mod threadpool;

pub use concurrent::ConcurrentSearcher;
pub use threadpool::{cosine_similarity, inner_product, l2_distance, Executor};
