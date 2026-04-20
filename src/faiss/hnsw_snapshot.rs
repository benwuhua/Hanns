use crate::api::{MetricType, Result};
use crate::index::Index;
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, IndexArtifactReader, IndexArtifactWriter, IndexManifest,
    LoadMode, SectionDescriptor,
};

use super::{HnswIndex, HnswRuntime};

pub const HNSW_SNAPSHOT_SECTION: &str = "hnsw.bytes";

pub struct HnswSnapshot {
    manifest: IndexManifest,
    bytes: Vec<u8>,
}

impl HnswSnapshot {
    pub fn from_index(index: &HnswIndex) -> Result<Self> {
        let bytes = index.serialize_to_bytes()?;
        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Hnsw,
            variant: "hnsw_blob_v1".to_string(),
            dim: index.dim(),
            metric: metric_name(index.metric_type()).to_string(),
            count: index.ntotal(),
            sections: vec![SectionDescriptor {
                name: HNSW_SNAPSHOT_SECTION.to_string(),
                len: bytes.len() as u64,
                checksum: None,
            }],
        };

        Ok(Self { manifest, bytes })
    }
}

impl AnnSnapshot for HnswSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        writer.write_section(HNSW_SNAPSHOT_SECTION, &self.bytes)?;
        writer.finish_manifest(&self.manifest)
    }
}

pub struct HnswSnapshotLoader;

impl AnnSnapshotLoader for HnswSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        let _ = reader.manifest()?;

        match mode {
            LoadMode::OwnedMemory => {}
            LoadMode::Mmap | LoadMode::PageCache | LoadMode::Lazy => {
                // This compatibility bridge stores one serialized HNSW blob, so planned
                // storage modes load through the same owned-memory path until the
                // sectioned HNSW layout exists.
            }
        }

        let bytes = reader.read_section(HNSW_SNAPSHOT_SECTION)?;
        let index = HnswIndex::deserialize_from_bytes(bytes.as_ref())?;
        Ok(Box::new(HnswRuntime::new(index)))
    }
}

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Ip => "ip",
        MetricType::Cosine => "cosine",
        MetricType::Hamming => "hamming",
    }
}
