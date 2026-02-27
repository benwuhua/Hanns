//! MinHash-LSH C API Bindings
//! 
//! C API for MinHash-LSH index operations.
//! Compatible with C++ knowhere MinHash-LSH interface.

use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::ptr;
use crate::index::minhash_lsh::{MinHashLSHIndex, KVPair};
use crate::bitset::BitsetView;

/// Opaque pointer type for MinHash-LSH index
pub type CMinHashLSHIndex = *mut c_void;

/// MinHash-LSH build parameters
#[repr(C)]
pub struct CMinHashLSHBuildParams {
    pub data_path: *const c_char,
    pub index_file_path: *const c_char,
    pub band: usize,
    pub block_size: usize,
    pub with_raw_data: bool,
    pub mh_vec_element_size: usize,
    pub mh_vec_length: usize,
}

/// MinHash-LSH load parameters
#[repr(C)]
pub struct CMinHashLSHLoadParams {
    pub index_file_path: *const c_char,
    pub hash_code_in_memory: bool,
    pub global_bloom_filter: bool,
    pub false_positive_prob: f32,
}

/// MinHash-LSH search parameters
#[repr(C)]
pub struct CMinHashLSHSearchParams {
    pub k: usize,
    pub refine_k: usize,
    pub search_with_jaccard: bool,
    pub bitset: *const c_void,
}

/// C API error codes
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CMinHashError {
    Success = 0,
    InvalidArg = 1,
    InternalError = 2,
    NotImplemented = 3,
    IoError = 4,
}

/// Build MinHash-LSH index from data file
/// 
/// # Safety
/// - `params` must be a valid pointer
/// - `data_path` and `index_file_path` must be valid C strings
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_build(params: *const CMinHashLSHBuildParams) -> i32 {
    if params.is_null() {
        return CMinHashError::InvalidArg as i32;
    }

    let params_ref = &*params;
    
    // Read data from file (simplified - in production would load binary vectors)
    let data = match load_binary_data(params_ref.data_path) {
        Ok(d) => d,
        Err(_) => return CMinHashError::IoError as i32,
    };

    let mut index = MinHashLSHIndex::new();
    
    match index.build(
        &data,
        params_ref.mh_vec_length,
        params_ref.mh_vec_element_size,
        params_ref.band,
        params_ref.with_raw_data,
    ) {
        Ok(_) => {
            // Save index to file
            let index_path = match CStr::from_ptr(params_ref.index_file_path).to_str() {
                Ok(s) => s,
                Err(_) => return CMinHashError::InvalidArg as i32,
            };
            
            match index.save(index_path) {
                Ok(_) => CMinHashError::Success as i32,
                Err(_) => CMinHashError::IoError as i32,
            }
        }
        Err(_) => CMinHashError::InternalError as i32,
    }
}

/// Load MinHash-LSH index from file
/// 
/// # Safety
/// - `params` must be a valid pointer
/// - Returns opaque pointer to index
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_load(params: *const CMinHashLSHLoadParams) -> CMinHashLSHIndex {
    if params.is_null() {
        return ptr::null_mut();
    }

    let params_ref = &*params;
    
    let index_path = match CStr::from_ptr(params_ref.index_file_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let mut index = MinHashLSHIndex::new();
    
    match index.load(index_path) {
        Ok(_) => {
            let boxed = Box::new(index);
            Box::into_raw(boxed) as CMinHashLSHIndex
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Save MinHash-LSH index to file
/// 
/// # Safety
/// - `index` must be a valid pointer returned by load or build
/// - `path` must be a valid C string
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_save(index: CMinHashLSHIndex, path: *const c_char) -> i32 {
    if index.is_null() || path.is_null() {
        return CMinHashError::InvalidArg as i32;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => return CMinHashError::InvalidArg as i32,
    };

    match index_ref.save(path_str) {
        Ok(_) => CMinHashError::Success as i32,
        Err(_) => CMinHashError::IoError as i32,
    }
}

/// Search for nearest neighbors
/// 
/// # Safety
/// - `index` must be a valid pointer
/// - `query` must point to `mh_vec_length * mh_vec_element_size` bytes
/// - `distances` must point to `k` floats
/// - `labels` must point to `k` int64s
/// - `params` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_search(
    index: CMinHashLSHIndex,
    query: *const c_char,
    distances: *mut f32,
    labels: *mut i64,
    params: *const CMinHashLSHSearchParams,
) -> i32 {
    if index.is_null() || query.is_null() || distances.is_null() || labels.is_null() || params.is_null() {
        return CMinHashError::InvalidArg as i32;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    let params_ref = &*params;
    
    let query_size = index_ref.count() * index_ref.count(); // Simplified
    let query_slice = std::slice::from_raw_parts(query as *const u8, query_size);
    
    let bitset = if params_ref.bitset.is_null() {
        None
    } else {
        Some(&*(params_ref.bitset as *const BitsetView))
    };

    match index_ref.search(query_slice, params_ref.k, bitset) {
        Ok((ids, dists)) => {
            ptr::copy_nonoverlapping(dists.as_ptr(), distances, dists.len());
            ptr::copy_nonoverlapping(ids.as_ptr(), labels, ids.len());
            CMinHashError::Success as i32
        }
        Err(_) => CMinHashError::InternalError as i32,
    }
}

/// Batch search for multiple queries
/// 
/// # Safety
/// - `index` must be a valid pointer
/// - `queries` must point to `nq * mh_vec_length * mh_vec_element_size` bytes
/// - `distances` must point to `nq * k` floats
/// - `labels` must point to `nq * k` int64s
/// - `params` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_batch_search(
    index: CMinHashLSHIndex,
    queries: *const c_char,
    nq: usize,
    distances: *mut f32,
    labels: *mut i64,
    params: *const CMinHashLSHSearchParams,
) -> i32 {
    if index.is_null() || queries.is_null() || distances.is_null() || labels.is_null() || params.is_null() {
        return CMinHashError::InvalidArg as i32;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    let params_ref = &*params;
    
    let vec_size = index_ref.count() * index_ref.count(); // Simplified
    let queries_slice = std::slice::from_raw_parts(queries as *const u8, nq * vec_size);

    match index_ref.batch_search(queries_slice, nq, params_ref.k, None) {
        Ok((all_ids, all_dists)) => {
            for (i, (ids, dists)) in all_ids.iter().zip(all_dists.iter()).enumerate() {
                ptr::copy_nonoverlapping(
                    dists.as_ptr(),
                    distances.add(i * params_ref.k),
                    dists.len(),
                );
                ptr::copy_nonoverlapping(
                    ids.as_ptr(),
                    labels.add(i * params_ref.k),
                    ids.len(),
                );
            }
            CMinHashError::Success as i32
        }
        Err(_) => CMinHashError::InternalError as i32,
    }
}

/// Get vectors by IDs
/// 
/// # Safety
/// - `index` must be a valid pointer
/// - `ids` must point to `n` int64s
/// - `data` must point to `n * mh_vec_length * mh_vec_element_size` bytes
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_get_vector_by_ids(
    index: CMinHashLSHIndex,
    ids: *const i64,
    n: usize,
    data: *mut c_char,
) -> i32 {
    if index.is_null() || ids.is_null() || data.is_null() {
        return CMinHashError::InvalidArg as i32;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    let ids_slice = std::slice::from_raw_parts(ids, n);

    match index_ref.get_vector_by_ids(ids_slice) {
        Ok(vectors) => {
            ptr::copy_nonoverlapping(vectors.as_ptr(), data as *mut u8, vectors.len());
            CMinHashError::Success as i32
        }
        Err(_) => CMinHashError::NotImplemented as i32,
    }
}

/// Check if index has raw data
/// 
/// # Safety
/// - `index` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_has_raw_data(index: CMinHashLSHIndex) -> bool {
    if index.is_null() {
        return false;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    index_ref.has_raw_data()
}

/// Get number of vectors in index
/// 
/// # Safety
/// - `index` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_count(index: CMinHashLSHIndex) -> usize {
    if index.is_null() {
        return 0;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    index_ref.count()
}

/// Get index size in bytes
/// 
/// # Safety
/// - `index` must be a valid pointer
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_size(index: CMinHashLSHIndex) -> usize {
    if index.is_null() {
        return 0;
    }

    let index_ref = &*(index as *const MinHashLSHIndex);
    index_ref.size()
}

/// Free MinHash-LSH index
/// 
/// # Safety
/// - `index` must be a valid pointer or null
#[no_mangle]
pub unsafe extern "C" fn knowhere_minhash_lsh_free(index: CMinHashLSHIndex) {
    if !index.is_null() {
        let _ = Box::from_raw(index as *mut MinHashLSHIndex);
    }
}

/// Helper function to load binary vector data from file
unsafe fn load_binary_data(path: *const c_char) -> Result<Vec<u8>, std::io::Error> {
    use std::fs::File;
    use std::io::Read;
    
    let path_str = CStr::from_ptr(path).to_str()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid path"))?;
    
    let mut file = File::open(path_str)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_null_pointer_handling() {
        // Test with null pointers
        unsafe {
            assert_eq!(knowhere_minhash_lsh_build(ptr::null()), CMinHashError::InvalidArg as i32);
            assert_eq!(knowhere_minhash_lsh_load(ptr::null()), ptr::null_mut());
            assert_eq!(knowhere_minhash_lsh_save(ptr::null_mut(), ptr::null()), CMinHashError::InvalidArg as i32);
            assert_eq!(knowhere_minhash_lsh_has_raw_data(ptr::null_mut()), false);
            assert_eq!(knowhere_minhash_lsh_count(ptr::null_mut()), 0);
            assert_eq!(knowhere_minhash_lsh_size(ptr::null_mut()), 0);
            
            // Free should not panic with null
            knowhere_minhash_lsh_free(ptr::null_mut());
        }
    }
}
