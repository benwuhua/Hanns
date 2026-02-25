//! Faiss FFI bridge using CXX
//! 
//! This module provides C++ bindings to Faiss library.
//! Requires Faiss to be installed: brew install faiss (macOS) or libfaiss-dev (Linux)

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        // Index management
        type FaissIndex;
        
        fn faiss_index_new_flat(dim: i32, metric: i32) -> *mut FaissIndex;
        fn faiss_index_free(index: *mut FaissIndex);
        fn faiss_index_ntotal(index: *const FaissIndex) -> i64;
        fn faiss_index_d(index: *const FaissIndex) -> i32;
        fn faiss_index_is_trained(index: *const FaissIndex) -> bool;
        
        // Index operations
        fn faiss_index_train(index: *mut FaissIndex, x: *const f32, n: i64) -> i32;
        fn faiss_index_add(index: *mut FaissIndex, x: *const f32, n: i64) -> i32;
        fn faiss_index_search(
            index: *mut FaissIndex,
            x: *const f32,
            k: i32,
            distances: *mut f32,
            labels: *mut i64,
        ) -> i32;
        fn faiss_index_reset(index: *mut FaissIndex) -> i32;
        
        // Serialization
        fn faiss_index_write(index: *mut FaissIndex, path: *const c_char) -> i32;
        fn faiss_index_read(path: *const c_char) -> *mut FaissIndex;
    }
}

use std::os::raw::c_char;

// Re-export for use in FaissIndex
pub use ffi::*;
