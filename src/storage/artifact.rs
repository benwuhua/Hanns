use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::api::{KnowhereError, Result};

use super::IndexManifest;

pub trait IndexArtifactReader {
    fn manifest(&self) -> Result<&IndexManifest>;
    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>>;
    fn section_len(&self, name: &str) -> Result<u64>;
    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>>;
}

pub trait IndexArtifactWriter {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()>;
    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()>;
}

#[derive(Debug, Default, Clone)]
pub struct MemoryArtifactStore {
    manifest: Option<IndexManifest>,
    sections: BTreeMap<String, Vec<u8>>,
}

impl IndexArtifactReader for MemoryArtifactStore {
    fn manifest(&self) -> Result<&IndexManifest> {
        self.manifest
            .as_ref()
            .ok_or_else(|| KnowhereError::Codec("missing manifest".to_string()))
    }

    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        self.sections
            .get(name)
            .map(|bytes| Cow::Borrowed(bytes.as_slice()))
            .ok_or_else(|| KnowhereError::Codec(format!("missing section: {name}")))
    }

    fn section_len(&self, name: &str) -> Result<u64> {
        Ok(self.read_section(name)?.len() as u64)
    }

    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let section = self.read_section(name)?;
        let offset = offset as usize;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| KnowhereError::Codec("section range overflow".to_string()))?;
        let bytes = section
            .get(offset..end)
            .ok_or_else(|| KnowhereError::Codec(format!("section range out of bounds: {name}")))?;
        Ok(bytes.to_vec())
    }
}

impl IndexArtifactWriter for MemoryArtifactStore {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.sections.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }

    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()> {
        self.manifest = Some(manifest.clone());
        Ok(())
    }
}
