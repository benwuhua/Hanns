//! Memory-Mapped Vector Storage
//! 
//! Using memmap2 for DiskANN and large indexes

use std::fs::OpenOptions;
use std::path::Path;

use memmap2::{Mmap, MmapMut};
use crate::api::{KnowhereError, Result};

/// Read-only memory mapped float array
pub struct MmapFloatArray {
    mmap: Mmap,
    dim: usize,
    count: usize,
}

impl MmapFloatArray {
    /// Open a file containing flat f32 vectors
    pub fn open<P: AsRef<Path>>(path: P, dim: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)?;
            
        let mmap = unsafe { Mmap::map(&file)? };
        
        let bytes_len = mmap.len();
        if bytes_len % (4 * dim) != 0 {
            return Err(KnowhereError::InvalidArg(
                format!("File size {} is not a multiple of vector size ({} bytes)", bytes_len, 4 * dim)
            ));
        }
        
        let count = bytes_len / (4 * dim);
        
        Ok(Self {
            mmap,
            dim,
            count,
        })
    }
    
    /// Get number of vectors
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get vector by index
    pub fn get_vector(&self, index: usize) -> &[f32] {
        if index >= self.count {
            panic!("Index out of bounds");
        }
        
        let start = index * self.dim * 4;
        let end = start + self.dim * 4;
        
        let bytes = &self.mmap[start..end];
        
        // Safety: the file was validated to contain f32
        unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.dim
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_read() {
        let mut file = NamedTempFile::new().unwrap();
        
        // Write 2 vectors of dim 3
        let vec1 = [1.0f32, 2.0, 3.0];
        let vec2 = [4.0f32, 5.0, 6.0];
        
        let bytes1: &[u8] = unsafe { std::slice::from_raw_parts(vec1.as_ptr() as *const u8, 12) };
        let bytes2: &[u8] = unsafe { std::slice::from_raw_parts(vec2.as_ptr() as *const u8, 12) };
        
        file.write_all(bytes1).unwrap();
        file.write_all(bytes2).unwrap();
        file.flush().unwrap();
        
        let mmap_arr = MmapFloatArray::open(file.path(), 3).unwrap();
        assert_eq!(mmap_arr.len(), 2);
        
        let v1 = mmap_arr.get_vector(0);
        assert_eq!(v1, &[1.0, 2.0, 3.0]);
        
        let v2 = mmap_arr.get_vector(1);
        assert_eq!(v2, &[4.0, 5.0, 6.0]);
    }
}