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

/// Read a sectioned artifact through caller-owned storage callbacks.
///
/// This adapter lets hosts such as Lance or pgvector keep their native page,
/// object-store, or buffer-cache management while satisfying Hanns snapshot
/// loaders through the common [`IndexArtifactReader`] contract.
pub struct CallbackArtifactReader<SectionLen, ReadRange> {
    manifest: IndexManifest,
    section_len: SectionLen,
    read_range: ReadRange,
}

impl<SectionLen, ReadRange> CallbackArtifactReader<SectionLen, ReadRange> {
    /// Create a reader from a manifest, a section length callback, and a range read callback.
    pub fn new(manifest: IndexManifest, section_len: SectionLen, read_range: ReadRange) -> Self {
        Self {
            manifest,
            section_len,
            read_range,
        }
    }
}

impl<SectionLen, ReadRange> IndexArtifactReader for CallbackArtifactReader<SectionLen, ReadRange>
where
    SectionLen: for<'a> Fn(&'a str) -> Result<u64>,
    ReadRange: for<'a> Fn(&'a str, u64, usize) -> Result<Vec<u8>>,
{
    fn manifest(&self) -> Result<&IndexManifest> {
        Ok(&self.manifest)
    }

    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let section_len = (self.section_len)(name)?;
        let section_len = usize::try_from(section_len).map_err(|_| {
            KnowhereError::Codec(format!("section {name} length does not fit usize"))
        })?;
        (self.read_range)(name, 0, section_len).map(Cow::Owned)
    }

    fn section_len(&self, name: &str) -> Result<u64> {
        (self.section_len)(name)
    }

    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        (self.read_range)(name, offset, len)
    }
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
