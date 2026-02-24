//! Admin API - index management

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use crate::api::{IndexConfig, IndexType, KnowhereError, MetricType, Result};
use crate::faiss::MemIndex;

/// Index registry - manages all indices
pub struct IndexRegistry {
    indices: RwLock<HashMap<String, MemIndex>>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        Self {
            indices: RwLock::new(HashMap::new()),
        }
    }

    pub fn create_index(&self, name: &str, config: &IndexConfig) -> Result<()> {
        let index = MemIndex::new(config)?;
        
        let mut indices = self.indices.write()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        indices.insert(name.to_string(), index);
        Ok(())
    }

    pub fn get_index(&self, name: &str) -> Result<MemIndex> {
        let indices = self.indices.read()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        indices.get(name)
            .cloned()
            .ok_or_else(|| KnowhereError::NotFound(name.to_string()))
    }

    pub fn drop_index(&self, name: &str) -> Result<()> {
        let mut indices = self.indices.write()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        indices.remove(name)
            .ok_or_else(|| KnowhereError::NotFound(name.to_string()))?;
        
        Ok(())
    }

    pub fn list_indices(&self) -> Result<Vec<String>> {
        let indices = self.indices.read()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        Ok(indices.keys().cloned().collect())
    }

    pub fn save_index(&self, name: &str, path: &Path) -> Result<()> {
        let indices = self.indices.read()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        let index = indices.get(name)
            .ok_or_else(|| KnowhereError::NotFound(name.to_string()))?;
        
        index.save(path)
    }

    pub fn load_index(&self, name: &str, path: &Path, dim: usize) -> Result<()> {
        let mut indices = self.indices.write()
            .map_err(|e| KnowhereError::InvalidArg(e.to_string()))?;
        
        let config = IndexConfig::new(IndexType::Flat, MetricType::L2, dim);
        let mut index = MemIndex::new(&config)?;
        index.load(path)?;
        
        indices.insert(name.to_string(), index);
        Ok(())
    }
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Admin interface
pub struct Admin {
    registry: IndexRegistry,
}

impl Admin {
    pub fn new() -> Self {
        Self {
            registry: IndexRegistry::new(),
        }
    }

    pub fn create_collection(&self, name: &str, dim: usize) -> Result<()> {
        let config = IndexConfig::new(
            IndexType::Flat,
            MetricType::L2,
            dim,
        );
        self.registry.create_index(name, &config)
    }

    pub fn create_index(&self, collection: &str, index_type: IndexType, metric: MetricType) -> Result<()> {
        let config = IndexConfig::new(index_type, metric, 0); // dim will be set from collection
        self.registry.create_index(collection, &config)
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        self.registry.drop_index(name)
    }

    pub fn list_collections(&self) -> Result<Vec<String>> {
        self.registry.list_indices()
    }

    pub fn flush(&self, name: &str, path: &str) -> Result<()> {
        let p = Path::new(path);
        self.registry.save_index(name, p)
    }

    pub fn load(&self, name: &str, path: &str, dim: usize) -> Result<()> {
        let p = Path::new(path);
        self.registry.load_index(name, p, dim)
    }
}

impl Default for Admin {
    fn default() -> Self {
        Self::new()
    }
}
