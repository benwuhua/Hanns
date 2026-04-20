use crate::api::{KnowhereError, MetricType, Result, SqMode};
use crate::index::Index;
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, IndexArtifactReader, IndexArtifactWriter, IndexManifest,
    LoadMode, SectionDescriptor,
};

use serde::Serialize;

use super::{hnsw::HnswSqExportMeta, HnswIndex, HnswRuntime};

pub const HNSW_SNAPSHOT_SECTION: &str = "hnsw.bytes";
const HNSW_SNAPSHOT_VARIANT: &str = "hnsw_blob_v1";
pub const HNSW_SECTIONS_SNAPSHOT_VARIANT: &str = "hnsw_sections_v1";

pub const HNSW_META_SECTION: &str = "hnsw.meta.json";
pub const HNSW_VECTORS_SECTION: &str = "hnsw.vectors.f32";
pub const HNSW_IDS_SECTION: &str = "hnsw.ids.i64";
pub const HNSW_LEVELS_SECTION: &str = "hnsw.levels.u32";
pub const HNSW_NEIGHBOR_OFFSETS_SECTION: &str = "hnsw.neighbors.offsets.u64";
pub const HNSW_NEIGHBOR_IDS_SECTION: &str = "hnsw.neighbors.ids.i64";
pub const HNSW_NEIGHBOR_DISTS_SECTION: &str = "hnsw.neighbors.dists.f32";
pub const HNSW_DELETED_IDS_SECTION: &str = "hnsw.deleted.ids.i64";
pub const HNSW_SQ_CODES_SECTION: &str = "hnsw.sq.codes.u8";
pub const HNSW_SQ_META_SECTION: &str = "hnsw.sq.meta.json";

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
            variant: HNSW_SNAPSHOT_VARIANT.to_string(),
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

pub struct HnswSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

#[derive(Serialize)]
struct HnswSectionMetadata {
    version: u32,
    dim: usize,
    metric: &'static str,
    m: usize,
    m_max0: usize,
    ef_search: usize,
    ef_construction: usize,
    max_level: usize,
    level_multiplier: f32,
    count: usize,
    entry_point: Option<i64>,
    sq_mode: &'static str,
}

#[derive(Serialize)]
struct HnswSqSectionMetadata {
    dim: usize,
    bit: usize,
    quantizer_type: u8,
    min_val: f32,
    max_val: f32,
    scale: f32,
    offset: f32,
}

impl From<HnswSqExportMeta> for HnswSqSectionMetadata {
    fn from(meta: HnswSqExportMeta) -> Self {
        Self {
            dim: meta.dim,
            bit: meta.bit,
            quantizer_type: meta.quantizer_type,
            min_val: meta.min_val,
            max_val: meta.max_val,
            scale: meta.scale,
            offset: meta.offset,
        }
    }
}

impl HnswSectionedSnapshot {
    pub fn from_index(index: &HnswIndex) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        let mut sections = Vec::new();

        let meta = HnswSectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type),
            m: export.m,
            m_max0: export.m_max0,
            ef_search: export.ef_search,
            ef_construction: export.ef_construction,
            max_level: export.max_level,
            level_multiplier: export.level_multiplier,
            count: export.count,
            entry_point: export.entry_point,
            sq_mode: sq_mode_name(export.sq_mode),
        };
        sections.push((
            HNSW_META_SECTION.to_string(),
            encode_json(&meta, HNSW_META_SECTION)?,
        ));
        sections.push((HNSW_IDS_SECTION.to_string(), encode_i64s(&export.ids)));
        sections.push((
            HNSW_VECTORS_SECTION.to_string(),
            encode_f32s(&export.vectors),
        ));
        sections.push((HNSW_LEVELS_SECTION.to_string(), encode_u32s(&export.levels)));
        sections.push((
            HNSW_NEIGHBOR_OFFSETS_SECTION.to_string(),
            encode_u64s(&export.neighbor_offsets),
        ));
        sections.push((
            HNSW_NEIGHBOR_IDS_SECTION.to_string(),
            encode_i64s(&export.neighbor_ids),
        ));
        sections.push((
            HNSW_NEIGHBOR_DISTS_SECTION.to_string(),
            encode_f32s(&export.neighbor_dists),
        ));
        sections.push((
            HNSW_DELETED_IDS_SECTION.to_string(),
            encode_i64s(&export.deleted_ids),
        ));

        if let Some(sq_meta) = export.sq_meta {
            let sq_meta = HnswSqSectionMetadata::from(sq_meta);
            sections.push((
                HNSW_SQ_META_SECTION.to_string(),
                encode_json(&sq_meta, HNSW_SQ_META_SECTION)?,
            ));
        }

        if !export.sq_codes.is_empty() {
            sections.push((HNSW_SQ_CODES_SECTION.to_string(), export.sq_codes));
        }

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Hnsw,
            variant: HNSW_SECTIONS_SNAPSHOT_VARIANT.to_string(),
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            count: export.count,
            sections: sections
                .iter()
                .map(|(name, bytes)| SectionDescriptor {
                    name: name.clone(),
                    len: bytes.len() as u64,
                    checksum: None,
                })
                .collect(),
        };

        Ok(Self { manifest, sections })
    }
}

impl AnnSnapshot for HnswSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
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
        let manifest = reader.manifest()?;
        if manifest.version != 1
            || manifest.family != IndexFamily::Hnsw
            || manifest.variant != HNSW_SNAPSHOT_VARIANT
        {
            return Err(KnowhereError::Codec(format!(
                "invalid HNSW snapshot manifest: expected version 1 family Hnsw variant {HNSW_SNAPSHOT_VARIANT}, got version {} family {:?} variant {:?}",
                manifest.version, manifest.family, manifest.variant
            )));
        }

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

fn sq_mode_name(sq_mode: SqMode) -> &'static str {
    match sq_mode {
        SqMode::None => "none",
        SqMode::SQ8 => "sq8",
        SqMode::SQ8Refine => "sq8_refine",
    }
}

fn encode_json<T: Serialize>(value: &T, section_name: &str) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value)
        .map_err(|error| KnowhereError::Codec(format!("encode {section_name}: {error}")))
}

fn encode_i64s(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_u64s(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
