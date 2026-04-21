use crate::api::{KnowhereError, MetricType, Result};
use crate::kernel::IndexFamily;
use crate::storage::{AnnSnapshot, IndexArtifactWriter, IndexManifest, SectionDescriptor};

use serde::{Deserialize, Serialize};

use super::IvfFlatIndex;

pub const IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT: &str = "ivf_flat_sections_v1";

pub const IVF_FLAT_META_SECTION: &str = "ivf_flat.meta.json";
pub const IVF_FLAT_CENTROIDS_SECTION: &str = "ivf_flat.centroids.f32";
pub const IVF_FLAT_LIST_OFFSETS_SECTION: &str = "ivf_flat.list_offsets.u64";
pub const IVF_FLAT_LIST_SIZES_SECTION: &str = "ivf_flat.list_sizes.u64";
pub const IVF_FLAT_LIST_IDS_SECTION: &str = "ivf_flat.list_ids.i64";
pub const IVF_FLAT_LIST_VECTORS_SECTION: &str = "ivf_flat.list_vectors.f32";
pub const IVF_FLAT_IDS_SECTION: &str = "ivf_flat.ids.i64";
pub const IVF_FLAT_VECTORS_SECTION: &str = "ivf_flat.vectors.f32";

#[derive(Deserialize, Serialize)]
struct IvfFlatSectionMetadata {
    version: u32,
    dim: usize,
    metric: String,
    nlist: usize,
    nprobe: usize,
    count: usize,
    next_id: i64,
    trained: bool,
}

pub struct IvfFlatSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl IvfFlatSectionedSnapshot {
    pub fn from_index(index: &IvfFlatIndex) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        let mut sections = Vec::new();

        let meta = IvfFlatSectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            nlist: export.nlist,
            nprobe: export.nprobe,
            count: export.count,
            next_id: export.next_id,
            trained: export.trained,
        };
        sections.push((
            IVF_FLAT_META_SECTION.to_string(),
            encode_json(&meta, IVF_FLAT_META_SECTION)?,
        ));
        sections.push((
            IVF_FLAT_CENTROIDS_SECTION.to_string(),
            encode_f32s(&export.centroids),
        ));
        sections.push((
            IVF_FLAT_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&export.list_offsets),
        ));
        sections.push((
            IVF_FLAT_LIST_SIZES_SECTION.to_string(),
            encode_u64s(&export.list_sizes),
        ));
        sections.push((
            IVF_FLAT_LIST_IDS_SECTION.to_string(),
            encode_i64s(&export.list_ids),
        ));
        sections.push((
            IVF_FLAT_LIST_VECTORS_SECTION.to_string(),
            encode_f32s(&export.list_vectors),
        ));
        sections.push((IVF_FLAT_IDS_SECTION.to_string(), encode_i64s(&export.ids)));
        sections.push((
            IVF_FLAT_VECTORS_SECTION.to_string(),
            encode_f32s(&export.vectors),
        ));

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Ivf,
            variant: IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT.to_string(),
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

impl AnnSnapshot for IvfFlatSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
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

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
