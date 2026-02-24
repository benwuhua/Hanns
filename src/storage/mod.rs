//! Storage layer

pub mod mem;
pub mod disk;

pub use mem::MemStorage;
pub use disk::DiskStorage;

use crate::api::{KnowhereError, Result};

/// Storage trait
pub trait Storage: Send + Sync {
    /// Read data
    fn read(&self, key: &[u8]) -> Result<Vec<u8>>;
    
    /// Write data
    fn write(&self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// Delete data
    fn delete(&self, key: &[u8]) -> Result<()>;
    
    /// Check if key exists
    fn exists(&self, key: &[u8]) -> Result<bool>;
    
    /// List keys with prefix
    fn list(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>>;
}
