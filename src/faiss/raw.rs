//! Raw Faiss FFI bindings
//! 
//! This module provides unsafe FFI bindings to Faiss C API.
//! In production, use `cxx` or `rust-cpp` for safer bindings.

use std::ffi::c_char;

/// Faiss return codes
pub const FAISS_SUCCESS: i32 = 0;
pub const FAISS_ERR_INVALID_ARG: i32 = -1;
pub const FAISS_ERR_IO: i32 = -2;
pub const FAISS_ERR_INTERNAL: i32 = -3;

/// Index types
pub const FAISS_INDEX_FLAT: i32 = 0;
pub const FAISS_INDEX_IVFFLAT: i32 = 1;
pub const FAISS_INDEX_IVFPQ: i32 = 2;
pub const FAISS_INDEX_HNSW: i32 = 3;
pub const FAISS_INDEX_LSH: i32 = 32;

/// Metric types
pub const FAISS_METRIC_L2: i32 = 1;
pub const FAISS_METRIC_INNER_PRODUCT: i32 = 2;

/// Opaque Faiss index type
#[repr(C)]
pub struct FaissIndex {
    _priv: [u8; 0],
}

/// Create a new flat index
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_new(d: i32, metric: i32) -> *mut FaissIndex;
}

/// Free an index
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_free(index: *mut FaissIndex);
}

/// Get the number of vectors
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_ntotal(index: *const FaissIndex) -> i64;
}

/// Get the dimension
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_d(index: *const FaissIndex) -> i32;
}

/// Add vectors to the index
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_add(
        index: *mut FaissIndex,
        x: *const f32,
        n: i64,
    ) -> i32;
}

/// Search the index
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_search(
        index: *mut FaissIndex,
        x: *const f32,
        k: i32,
        distances: *mut f32,
        labels: *mut i64,
    ) -> i32;
}

/// Train the index
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_train(
        index: *mut FaissIndex,
        x: *const f32,
        n: i64,
    ) -> i32;
}

/// Reset the index (remove all vectors)
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_reset(index: *mut FaissIndex) -> i32;
}

/// Write index to file
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_write(
        index: *mut FaissIndex,
        path: *const c_char,
    ) -> i32;
}

/// Read index from file
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_read(
        path: *const c_char,
    ) -> *mut FaissIndex;
}

/// Check if index is trained
#[cfg(feature = "ffi")]
extern "C" {
    pub fn faiss_index_is_trained(index: *const FaissIndex) -> bool;
}
