//! In-memory storage

use std::collections::HashMap;
use std::sync::RwLock;

use super::Storage;
use crate::api::{KnowhereError, Result};

/// In-memory storage implementation
pub struct MemStorage {
    data: RwLock<HashMap<Vec<u8>, Vec<u8>>>,
}

impl MemStorage {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    /// Get the number of keys
    pub fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemStorage {
    fn read(&self, key: &[u8]) -> Result<Vec<u8>> {
        let data = self.data.read()
            .map_err(|e| KnowhereError::Storage(e.to_string()))?;
        
        data.get(key)
            .cloned()
            .ok_or_else(|| KnowhereError::NotFound(format!("{:?}", key)))
    }

    fn write(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let mut data = self.data.write()
            .map_err(|e| KnowhereError::Storage(e.to_string()))?;
        
        data.insert(key.to_vec(), value.to_vec());
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        let mut data = self.data.write()
            .map_err(|e| KnowhereError::Storage(e.to_string()))?;
        
        data.remove(key);
        Ok(())
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        let data = self.data.read()
            .map_err(|e| KnowhereError::Storage(e.to_string()))?;
        
        Ok(data.contains_key(key))
    }

    fn list(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        let data = self.data.read()
            .map_err(|e| KnowhereError::Storage(e.to_string()))?;
        
        let keys: Vec<Vec<u8>> = data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_storage() {
        let storage = MemStorage::new();
        
        storage.write(b"key1", b"value1").unwrap();
        assert!(storage.exists(b"key1").unwrap());
        
        let val = storage.read(b"key1").unwrap();
        assert_eq!(val, b"value1");
        
        storage.delete(b"key1").unwrap();
        assert!(!storage.exists(b"key1").unwrap());
    }
}
