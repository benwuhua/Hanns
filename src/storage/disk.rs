//! Disk-based storage

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write as IoWrite};
use std::path::{Path, PathBuf};

use crate::api::{KnowhereError, Result};
use super::Storage;

/// Disk storage implementation
pub struct DiskStorage {
    base_path: PathBuf,
}

impl DiskStorage {
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let path = base_path.as_ref().to_path_buf();
        
        if !path.exists() {
            fs::create_dir_all(&path)?;
        }
        
        Ok(Self { base_path: path })
    }

    fn key_to_path(&self, key: &[u8]) -> PathBuf {
        // Simple hex encoding for file names
        let hex: String = key.iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        self.base_path.join(hex)
    }
}

impl Storage for DiskStorage {
    fn read(&self, key: &[u8]) -> Result<Vec<u8>> {
        let path = self.key_to_path(key);
        
        if !path.exists() {
            return Err(KnowhereError::NotFound(format!("{:?}", key)));
        }
        
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn write(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        
        file.write_all(value)?;
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        
        if path.exists() {
            fs::remove_file(&path)?;
        }
        
        Ok(())
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        let path = self.key_to_path(key);
        Ok(path.exists())
    }

    fn list(&self, prefix: &[u8]) -> Result<Vec<Vec<u8>>> {
        let prefix_hex: String = prefix.iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        
        let mut keys = Vec::new();
        
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            
            if name.starts_with(&prefix_hex) {
                // Decode hex back to bytes
                let key: Vec<u8> = (0..name.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&name[i..i+2], 16).unwrap())
                    .collect();
                keys.push(key);
            }
        }
        
        Ok(keys)
    }
}
