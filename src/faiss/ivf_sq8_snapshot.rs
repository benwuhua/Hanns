use crate::api::{KnowhereError, MetricType, Result};
use crate::kernel::IndexFamily;
use crate::storage::{
    AnnSnapshot, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter, IndexManifest,
    SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::ivf_sq8::{IvfSq8Index, IvfSq8SectionedExport};

pub const IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT: &str = "ivf_sq8_sections_v1";

pub const IVF_SQ8_META_SECTION: &str = "ivf_sq8.meta.json";
pub const IVF_SQ8_CENTROIDS_SECTION: &str = "ivf_sq8.centroids.f32";
pub const IVF_SQ8_LIST_OFFSETS_SECTION: &str = "ivf_sq8.list_offsets.u64";
pub const IVF_SQ8_LIST_SIZES_SECTION: &str = "ivf_sq8.list_sizes.u64";
pub const IVF_SQ8_LIST_IDS_SECTION: &str = "ivf_sq8.list_ids.i64";
pub const IVF_SQ8_LIST_ROWS_SECTION: &str = "ivf_sq8.list_rows.u64";
pub const IVF_SQ8_LIST_CODES_SECTION: &str = "ivf_sq8.list_codes.u8";
pub const IVF_SQ8_IDS_SECTION: &str = "ivf_sq8.ids.i64";
pub const IVF_SQ8_VECTORS_SECTION: &str = "ivf_sq8.vectors.f32";

const REQUIRED_IVF_SQ8_SECTIONS: [&str; 9] = [
    IVF_SQ8_META_SECTION,
    IVF_SQ8_CENTROIDS_SECTION,
    IVF_SQ8_LIST_OFFSETS_SECTION,
    IVF_SQ8_LIST_SIZES_SECTION,
    IVF_SQ8_LIST_IDS_SECTION,
    IVF_SQ8_LIST_ROWS_SECTION,
    IVF_SQ8_LIST_CODES_SECTION,
    IVF_SQ8_IDS_SECTION,
    IVF_SQ8_VECTORS_SECTION,
];

#[derive(Deserialize, Serialize)]
struct IvfSq8SectionMetadata {
    version: u32,
    dim: usize,
    metric: String,
    nlist: usize,
    nprobe: usize,
    count: usize,
    next_id: i64,
    trained: bool,
    quantizer_bit: usize,
    quantizer_min: f32,
    quantizer_max: f32,
    quantizer_scale: f32,
    quantizer_offset: f32,
}

pub struct IvfSq8SectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl IvfSq8SectionedSnapshot {
    pub fn from_index(index: &IvfSq8Index) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        let mut sections = Vec::new();

        let meta = IvfSq8SectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            nlist: export.nlist,
            nprobe: export.nprobe,
            count: export.count,
            next_id: export.next_id,
            trained: export.trained,
            quantizer_bit: export.quantizer_bit,
            quantizer_min: export.quantizer_min_val,
            quantizer_max: export.quantizer_max_val,
            quantizer_scale: export.quantizer_scale,
            quantizer_offset: export.quantizer_offset,
        };
        sections.push((
            IVF_SQ8_META_SECTION.to_string(),
            encode_json(&meta, IVF_SQ8_META_SECTION)?,
        ));
        sections.push((
            IVF_SQ8_CENTROIDS_SECTION.to_string(),
            encode_f32s(&export.centroids),
        ));
        sections.push((
            IVF_SQ8_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&export.list_offsets),
        ));
        sections.push((
            IVF_SQ8_LIST_SIZES_SECTION.to_string(),
            encode_u64s(&export.list_sizes),
        ));
        sections.push((
            IVF_SQ8_LIST_IDS_SECTION.to_string(),
            encode_i64s(&export.list_ids),
        ));
        sections.push((
            IVF_SQ8_LIST_ROWS_SECTION.to_string(),
            encode_u64s(&export.list_rows),
        ));
        sections.push((
            IVF_SQ8_LIST_CODES_SECTION.to_string(),
            export.list_codes.clone(),
        ));
        sections.push((IVF_SQ8_IDS_SECTION.to_string(), encode_i64s(&export.ids)));
        sections.push((
            IVF_SQ8_VECTORS_SECTION.to_string(),
            encode_f32s(&export.vectors),
        ));

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Ivf,
            variant: IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT.to_string(),
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

impl AnnSnapshot for IvfSq8SectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_ivf_sq8_sectioned_snapshot(index: &IvfSq8Index, root: impl AsRef<Path>) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = IvfSq8SectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_ivf_sq8_sectioned_snapshot(root: impl AsRef<Path>) -> Result<IvfSq8Index> {
    let store = FileArtifactStore::new(root)?;
    load_ivf_sq8_index_from_artifact(&store)
}

pub fn load_ivf_sq8_index_from_artifact(reader: &dyn IndexArtifactReader) -> Result<IvfSq8Index> {
    let manifest = reader.manifest()?;
    if manifest.version != 1 || manifest.family != IndexFamily::Ivf {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-SQ8 snapshot manifest: expected version 1 family Ivf, got version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }

    match manifest.variant.as_str() {
        IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT => load_sectioned_ivf_sq8(reader, manifest),
        variant => Err(KnowhereError::Codec(format!(
            "invalid IVF-SQ8 snapshot manifest: expected variant {IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT}, got {variant:?}"
        ))),
    }
}

fn load_sectioned_ivf_sq8(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<IvfSq8Index> {
    validate_manifest_sections(reader, manifest)?;

    let meta_bytes = reader.read_section(IVF_SQ8_META_SECTION)?;
    let meta: IvfSq8SectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|error| KnowhereError::Codec(format!("decode {IVF_SQ8_META_SECTION}: {error}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-SQ8 sectioned metadata: unsupported version {}",
            meta.version
        )));
    }
    if manifest.dim != meta.dim || manifest.count != meta.count || manifest.metric != meta.metric {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-SQ8 sectioned manifest metadata: manifest dim/count/metric = {}/{}/{:?}, metadata = {}/{}/{:?}",
            manifest.dim, manifest.count, manifest.metric, meta.dim, meta.count, meta.metric
        )));
    }

    let export = IvfSq8SectionedExport {
        dim: meta.dim,
        count: meta.count,
        nlist: meta.nlist,
        nprobe: meta.nprobe,
        metric_type: parse_metric_name(&meta.metric)?,
        next_id: meta.next_id,
        trained: meta.trained,
        quantizer_bit: meta.quantizer_bit,
        quantizer_min_val: meta.quantizer_min,
        quantizer_max_val: meta.quantizer_max,
        quantizer_scale: meta.quantizer_scale,
        quantizer_offset: meta.quantizer_offset,
        centroids: decode_f32s(
            reader.read_section(IVF_SQ8_CENTROIDS_SECTION)?.as_ref(),
            IVF_SQ8_CENTROIDS_SECTION,
        )?,
        list_offsets: decode_u64s(
            reader.read_section(IVF_SQ8_LIST_OFFSETS_SECTION)?.as_ref(),
            IVF_SQ8_LIST_OFFSETS_SECTION,
        )?,
        list_sizes: decode_u64s(
            reader.read_section(IVF_SQ8_LIST_SIZES_SECTION)?.as_ref(),
            IVF_SQ8_LIST_SIZES_SECTION,
        )?,
        list_ids: decode_i64s(
            reader.read_section(IVF_SQ8_LIST_IDS_SECTION)?.as_ref(),
            IVF_SQ8_LIST_IDS_SECTION,
        )?,
        list_rows: decode_u64s(
            reader.read_section(IVF_SQ8_LIST_ROWS_SECTION)?.as_ref(),
            IVF_SQ8_LIST_ROWS_SECTION,
        )?,
        list_codes: reader
            .read_section(IVF_SQ8_LIST_CODES_SECTION)?
            .into_owned(),
        ids: decode_i64s(
            reader.read_section(IVF_SQ8_IDS_SECTION)?.as_ref(),
            IVF_SQ8_IDS_SECTION,
        )?,
        vectors: decode_f32s(
            reader.read_section(IVF_SQ8_VECTORS_SECTION)?.as_ref(),
            IVF_SQ8_VECTORS_SECTION,
        )?,
    };

    IvfSq8Index::from_sectioned_snapshot_export(export)
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
                "invalid IVF-SQ8 sectioned manifest: duplicate section descriptor {}",
                descriptor.name
            )));
        }

        let actual_len = reader.section_len(&descriptor.name).map_err(|error| {
            KnowhereError::Codec(format!(
                "invalid IVF-SQ8 sectioned manifest: descriptor {} references unreadable section: {error}",
                descriptor.name
            ))
        })?;
        if descriptor.len != actual_len {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-SQ8 sectioned manifest: descriptor length for {} is {}, actual section length is {}",
                descriptor.name, descriptor.len, actual_len
            )));
        }
    }

    for section_name in REQUIRED_IVF_SQ8_SECTIONS {
        if !seen.contains(section_name) {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-SQ8 sectioned manifest: missing required section descriptor {section_name}"
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
            "unsupported IVF-SQ8 sectioned metric: {other}"
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
