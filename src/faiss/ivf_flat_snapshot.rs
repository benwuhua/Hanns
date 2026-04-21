use crate::api::{KnowhereError, MetricType, Result};
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::ivf_flat::{IvfFlatIndex, IvfFlatSectionedExport};
use super::IvfRuntime;

pub const IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT: &str = "ivf_flat_sections_v1";

pub const IVF_FLAT_META_SECTION: &str = "ivf_flat.meta.json";
pub const IVF_FLAT_CENTROIDS_SECTION: &str = "ivf_flat.centroids.f32";
pub const IVF_FLAT_LIST_OFFSETS_SECTION: &str = "ivf_flat.list_offsets.u64";
pub const IVF_FLAT_LIST_SIZES_SECTION: &str = "ivf_flat.list_sizes.u64";
pub const IVF_FLAT_LIST_IDS_SECTION: &str = "ivf_flat.list_ids.i64";
pub const IVF_FLAT_LIST_VECTORS_SECTION: &str = "ivf_flat.list_vectors.f32";
pub const IVF_FLAT_IDS_SECTION: &str = "ivf_flat.ids.i64";
pub const IVF_FLAT_VECTORS_SECTION: &str = "ivf_flat.vectors.f32";

const REQUIRED_IVF_FLAT_SECTIONS: [&str; 8] = [
    IVF_FLAT_META_SECTION,
    IVF_FLAT_CENTROIDS_SECTION,
    IVF_FLAT_LIST_OFFSETS_SECTION,
    IVF_FLAT_LIST_SIZES_SECTION,
    IVF_FLAT_LIST_IDS_SECTION,
    IVF_FLAT_LIST_VECTORS_SECTION,
    IVF_FLAT_IDS_SECTION,
    IVF_FLAT_VECTORS_SECTION,
];

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

pub fn save_ivf_flat_sectioned_snapshot(
    index: &IvfFlatIndex,
    root: impl AsRef<Path>,
) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = IvfFlatSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_ivf_flat_sectioned_snapshot(root: impl AsRef<Path>) -> Result<IvfFlatIndex> {
    let store = FileArtifactStore::new(root)?;
    load_ivf_flat_index_from_artifact(&store)
}

pub fn load_ivf_flat_index_from_artifact(reader: &dyn IndexArtifactReader) -> Result<IvfFlatIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1 || manifest.family != IndexFamily::Ivf {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-Flat snapshot manifest: expected version 1 family Ivf, got version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }

    match manifest.variant.as_str() {
        IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT => load_sectioned_ivf_flat(reader, manifest),
        variant => Err(KnowhereError::Codec(format!(
            "invalid IVF-Flat snapshot manifest: expected variant {IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT}, got {variant:?}"
        ))),
    }
}

pub struct IvfFlatSnapshotLoader;

impl AnnSnapshotLoader for IvfFlatSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        _mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        Ok(Box::new(IvfRuntime::flat(
            load_ivf_flat_index_from_artifact(reader)?,
        )))
    }
}

fn load_sectioned_ivf_flat(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<IvfFlatIndex> {
    validate_manifest_sections(reader, manifest)?;

    let meta_bytes = reader.read_section(IVF_FLAT_META_SECTION)?;
    let meta: IvfFlatSectionMetadata =
        serde_json::from_slice(meta_bytes.as_ref()).map_err(|error| {
            KnowhereError::Codec(format!("decode {IVF_FLAT_META_SECTION}: {error}"))
        })?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-Flat sectioned metadata: unsupported version {}",
            meta.version
        )));
    }
    if manifest.dim != meta.dim || manifest.count != meta.count || manifest.metric != meta.metric {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-Flat sectioned manifest metadata: manifest dim/count/metric = {}/{}/{:?}, metadata = {}/{}/{:?}",
            manifest.dim, manifest.count, manifest.metric, meta.dim, meta.count, meta.metric
        )));
    }

    let export = IvfFlatSectionedExport {
        dim: meta.dim,
        count: meta.count,
        nlist: meta.nlist,
        nprobe: meta.nprobe,
        metric_type: parse_metric_name(&meta.metric)?,
        next_id: meta.next_id,
        trained: meta.trained,
        centroids: decode_f32s(
            reader.read_section(IVF_FLAT_CENTROIDS_SECTION)?.as_ref(),
            IVF_FLAT_CENTROIDS_SECTION,
        )?,
        list_offsets: decode_u64s(
            reader.read_section(IVF_FLAT_LIST_OFFSETS_SECTION)?.as_ref(),
            IVF_FLAT_LIST_OFFSETS_SECTION,
        )?,
        list_sizes: decode_u64s(
            reader.read_section(IVF_FLAT_LIST_SIZES_SECTION)?.as_ref(),
            IVF_FLAT_LIST_SIZES_SECTION,
        )?,
        list_ids: decode_i64s(
            reader.read_section(IVF_FLAT_LIST_IDS_SECTION)?.as_ref(),
            IVF_FLAT_LIST_IDS_SECTION,
        )?,
        list_vectors: decode_f32s(
            reader.read_section(IVF_FLAT_LIST_VECTORS_SECTION)?.as_ref(),
            IVF_FLAT_LIST_VECTORS_SECTION,
        )?,
        ids: decode_i64s(
            reader.read_section(IVF_FLAT_IDS_SECTION)?.as_ref(),
            IVF_FLAT_IDS_SECTION,
        )?,
        vectors: decode_f32s(
            reader.read_section(IVF_FLAT_VECTORS_SECTION)?.as_ref(),
            IVF_FLAT_VECTORS_SECTION,
        )?,
    };

    IvfFlatIndex::from_sectioned_snapshot_export(export)
}

fn validate_manifest_sections(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<()> {
    use std::collections::HashSet;

    let mut seen = HashSet::new();
    for descriptor in &manifest.sections {
        if !seen.insert(descriptor.name.as_str()) {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-Flat sectioned manifest: duplicate section descriptor {}",
                descriptor.name
            )));
        }

        let actual_len = reader.section_len(&descriptor.name).map_err(|error| {
            KnowhereError::Codec(format!(
                "invalid IVF-Flat sectioned manifest: descriptor {} references unreadable section: {error}",
                descriptor.name
            ))
        })?;
        if descriptor.len != actual_len {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-Flat sectioned manifest: descriptor length for {} is {}, actual section length is {}",
                descriptor.name, descriptor.len, actual_len
            )));
        }
    }

    for section_name in REQUIRED_IVF_FLAT_SECTIONS {
        if !seen.contains(section_name) {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-Flat sectioned manifest: missing required section descriptor {section_name}"
            )));
        }
    }

    Ok(())
}

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Ip => "ip",
        MetricType::Cosine => "cosine",
        MetricType::Hamming => "hamming",
    }
}

fn parse_metric_name(name: &str) -> Result<MetricType> {
    match name {
        "l2" => Ok(MetricType::L2),
        "ip" => Ok(MetricType::Ip),
        "cosine" => Ok(MetricType::Cosine),
        "hamming" => Ok(MetricType::Hamming),
        other => Err(KnowhereError::Codec(format!(
            "unsupported IVF-Flat sectioned metric: {other}"
        ))),
    }
}

fn decode_i64s(bytes: &[u8], section_name: &str) -> Result<Vec<i64>> {
    if bytes.len() % 8 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section_name}: length {} is not a multiple of 8",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_u64s(bytes: &[u8], section_name: &str) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section_name}: length {} is not a multiple of 8",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_f32s(bytes: &[u8], section_name: &str) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section_name}: length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
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
