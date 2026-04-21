use crate::api::{KnowhereError, Result};
use crate::kernel::AnnRuntime;

use super::{IndexArtifactReader, IndexArtifactWriter};
use std::collections::HashMap;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadMode {
    OwnedMemory,
    Mmap,
    PageCache,
    Lazy,
}

pub trait AnnSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()>;
}

pub trait AnnSnapshotLoader {
    fn supported_load_modes(&self) -> &'static [LoadMode] {
        &[LoadMode::OwnedMemory]
    }

    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>>;
}

pub fn require_owned_memory_load_mode(mode: LoadMode, loader_name: &str) -> Result<()> {
    if mode == LoadMode::OwnedMemory {
        return Ok(());
    }
    Err(KnowhereError::InvalidArg(format!(
        "{loader_name} supports only LoadMode::OwnedMemory; requested {mode:?}"
    )))
}

#[derive(Default)]
pub struct AnnSnapshotRegistry {
    loaders: HashMap<String, Box<dyn AnnSnapshotLoader>>,
}

impl AnnSnapshotRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_loader(
        &mut self,
        variant: impl Into<String>,
        loader: Box<dyn AnnSnapshotLoader>,
    ) -> Result<()> {
        let variant = variant.into();
        if self.loaders.contains_key(&variant) {
            return Err(crate::api::KnowhereError::InvalidArg(format!(
                "snapshot loader already registered for variant {variant:?}"
            )));
        }
        self.loaders.insert(variant, loader);
        Ok(())
    }

    pub fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        let manifest = reader.manifest()?;
        let loader = self.loaders.get(&manifest.variant).ok_or_else(|| {
            crate::api::KnowhereError::InvalidArg(format!(
                "no snapshot loader registered for variant {:?}",
                manifest.variant
            ))
        })?;
        loader.load_snapshot(reader, mode)
    }

    pub fn supported_load_modes_for_variant(&self, variant: &str) -> Result<&'static [LoadMode]> {
        let loader = self.loaders.get(variant).ok_or_else(|| {
            crate::api::KnowhereError::InvalidArg(format!(
                "no snapshot loader registered for variant {variant:?}"
            ))
        })?;
        Ok(loader.supported_load_modes())
    }

    pub fn supported_load_modes(&self, reader: &dyn IndexArtifactReader) -> Result<Vec<LoadMode>> {
        let manifest = reader.manifest()?;
        let loader_modes = self.supported_load_modes_for_variant(&manifest.variant)?;
        Ok(manifest
            .supported_load_modes
            .iter()
            .copied()
            .filter(|mode| loader_modes.contains(mode))
            .collect())
    }

    pub fn registered_variants(&self) -> Vec<&str> {
        let mut variants = self.loaders.keys().map(String::as_str).collect::<Vec<_>>();
        variants.sort_unstable();
        variants
    }
}
