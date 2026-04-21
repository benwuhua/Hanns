use crate::api::{KnowhereError, MetricType, Result};
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::ivfpq::{IvfPqIndex, IvfPqSectionedExport};
use super::IvfRuntime;

pub const IVF_PQ_SECTIONS_SNAPSHOT_VARIANT: &str = "ivf_pq_sections_v1";

pub const IVF_PQ_META_SECTION: &str = "ivf_pq.meta.json";
pub const IVF_PQ_CENTROIDS_SECTION: &str = "ivf_pq.centroids.f32";
pub const IVF_PQ_PQ_CENTROIDS_SECTION: &str = "ivf_pq.pq_centroids.f32";
pub const IVF_PQ_LIST_OFFSETS_SECTION: &str = "ivf_pq.list_offsets.u64";
pub const IVF_PQ_LIST_SIZES_SECTION: &str = "ivf_pq.list_sizes.u64";
pub const IVF_PQ_LIST_IDS_SECTION: &str = "ivf_pq.list_ids.i64";
pub const IVF_PQ_LIST_CODES_SECTION: &str = "ivf_pq.list_codes.u8";
pub const IVF_PQ_IDS_SECTION: &str = "ivf_pq.ids.i64";
pub const IVF_PQ_VECTORS_SECTION: &str = "ivf_pq.vectors.f32";

const REQUIRED_IVF_PQ_SECTIONS: [&str; 9] = [
    IVF_PQ_META_SECTION,
    IVF_PQ_CENTROIDS_SECTION,
    IVF_PQ_PQ_CENTROIDS_SECTION,
    IVF_PQ_LIST_OFFSETS_SECTION,
    IVF_PQ_LIST_SIZES_SECTION,
    IVF_PQ_LIST_IDS_SECTION,
    IVF_PQ_LIST_CODES_SECTION,
    IVF_PQ_IDS_SECTION,
    IVF_PQ_VECTORS_SECTION,
];

#[derive(Deserialize, Serialize)]
struct IvfPqSectionMetadata {
    version: u32,
    dim: usize,
    metric: String,
    nlist: usize,
    nprobe: usize,
    m: usize,
    nbits_per_idx: usize,
    code_size: usize,
    count: usize,
    next_id: i64,
    trained: bool,
    use_opq: bool,
    imported_pq_centroids: bool,
}

pub struct IvfPqSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl IvfPqSectionedSnapshot {
    pub fn from_index(index: &IvfPqIndex) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        let mut sections = Vec::new();

        let meta = IvfPqSectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            nlist: export.nlist,
            nprobe: export.nprobe,
            m: export.m,
            nbits_per_idx: export.nbits_per_idx,
            code_size: export.code_size,
            count: export.count,
            next_id: export.next_id,
            trained: export.trained,
            use_opq: export.use_opq,
            imported_pq_centroids: export.imported_pq_centroids,
        };
        sections.push((
            IVF_PQ_META_SECTION.to_string(),
            encode_json(&meta, IVF_PQ_META_SECTION)?,
        ));
        sections.push((
            IVF_PQ_CENTROIDS_SECTION.to_string(),
            encode_f32s(&export.centroids),
        ));
        sections.push((
            IVF_PQ_PQ_CENTROIDS_SECTION.to_string(),
            encode_f32s(&export.pq_centroids),
        ));
        sections.push((
            IVF_PQ_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&export.list_offsets),
        ));
        sections.push((
            IVF_PQ_LIST_SIZES_SECTION.to_string(),
            encode_u64s(&export.list_sizes),
        ));
        sections.push((
            IVF_PQ_LIST_IDS_SECTION.to_string(),
            encode_i64s(&export.list_ids),
        ));
        sections.push((
            IVF_PQ_LIST_CODES_SECTION.to_string(),
            export.list_codes.clone(),
        ));
        sections.push((IVF_PQ_IDS_SECTION.to_string(), encode_i64s(&export.ids)));
        sections.push((
            IVF_PQ_VECTORS_SECTION.to_string(),
            encode_f32s(&export.vectors),
        ));

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Ivf,
            variant: IVF_PQ_SECTIONS_SNAPSHOT_VARIANT.to_string(),
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

impl AnnSnapshot for IvfPqSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_ivf_pq_sectioned_snapshot(index: &IvfPqIndex, root: impl AsRef<Path>) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = IvfPqSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_ivf_pq_sectioned_snapshot(root: impl AsRef<Path>) -> Result<IvfPqIndex> {
    let store = FileArtifactStore::new(root)?;
    load_ivf_pq_index_from_artifact(&store)
}

pub fn load_ivf_pq_index_from_artifact(reader: &dyn IndexArtifactReader) -> Result<IvfPqIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1 || manifest.family != IndexFamily::Ivf {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-PQ snapshot manifest: expected version 1 family Ivf, got version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }

    match manifest.variant.as_str() {
        IVF_PQ_SECTIONS_SNAPSHOT_VARIANT => load_sectioned_ivf_pq(reader, manifest),
        variant => Err(KnowhereError::Codec(format!(
            "invalid IVF-PQ snapshot manifest: expected variant {IVF_PQ_SECTIONS_SNAPSHOT_VARIANT}, got {variant:?}"
        ))),
    }
}

pub struct IvfPqSnapshotLoader;

impl AnnSnapshotLoader for IvfPqSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        _mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        Ok(Box::new(IvfRuntime::pq(load_ivf_pq_index_from_artifact(
            reader,
        )?)))
    }
}

fn load_sectioned_ivf_pq(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<IvfPqIndex> {
    validate_manifest_sections(reader, manifest)?;

    let meta_bytes = reader.read_section(IVF_PQ_META_SECTION)?;
    let meta: IvfPqSectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|error| KnowhereError::Codec(format!("decode {IVF_PQ_META_SECTION}: {error}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-PQ sectioned metadata: unsupported version {}",
            meta.version
        )));
    }
    if manifest.dim != meta.dim || manifest.count != meta.count || manifest.metric != meta.metric {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-PQ sectioned manifest metadata: manifest dim/count/metric = {}/{}/{:?}, metadata = {}/{}/{:?}",
            manifest.dim, manifest.count, manifest.metric, meta.dim, meta.count, meta.metric
        )));
    }

    let export = IvfPqSectionedExport {
        dim: meta.dim,
        count: meta.count,
        nlist: meta.nlist,
        nprobe: meta.nprobe,
        m: meta.m,
        nbits_per_idx: meta.nbits_per_idx,
        code_size: meta.code_size,
        metric_type: parse_metric_name(&meta.metric)?,
        next_id: meta.next_id,
        trained: meta.trained,
        use_opq: meta.use_opq,
        imported_pq_centroids: meta.imported_pq_centroids,
        centroids: decode_f32s(
            reader.read_section(IVF_PQ_CENTROIDS_SECTION)?.as_ref(),
            IVF_PQ_CENTROIDS_SECTION,
        )?,
        pq_centroids: decode_f32s(
            reader.read_section(IVF_PQ_PQ_CENTROIDS_SECTION)?.as_ref(),
            IVF_PQ_PQ_CENTROIDS_SECTION,
        )?,
        list_offsets: decode_u64s(
            reader.read_section(IVF_PQ_LIST_OFFSETS_SECTION)?.as_ref(),
            IVF_PQ_LIST_OFFSETS_SECTION,
        )?,
        list_sizes: decode_u64s(
            reader.read_section(IVF_PQ_LIST_SIZES_SECTION)?.as_ref(),
            IVF_PQ_LIST_SIZES_SECTION,
        )?,
        list_ids: decode_i64s(
            reader.read_section(IVF_PQ_LIST_IDS_SECTION)?.as_ref(),
            IVF_PQ_LIST_IDS_SECTION,
        )?,
        list_codes: reader.read_section(IVF_PQ_LIST_CODES_SECTION)?.into_owned(),
        ids: decode_i64s(
            reader.read_section(IVF_PQ_IDS_SECTION)?.as_ref(),
            IVF_PQ_IDS_SECTION,
        )?,
        vectors: decode_f32s(
            reader.read_section(IVF_PQ_VECTORS_SECTION)?.as_ref(),
            IVF_PQ_VECTORS_SECTION,
        )?,
    };

    IvfPqIndex::from_sectioned_snapshot_export(export)
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
                "invalid IVF-PQ sectioned manifest: duplicate section descriptor {}",
                descriptor.name
            )));
        }

        let actual_len = reader.section_len(&descriptor.name).map_err(|error| {
            KnowhereError::Codec(format!(
                "invalid IVF-PQ sectioned manifest: descriptor {} references unreadable section: {error}",
                descriptor.name
            ))
        })?;
        if descriptor.len != actual_len {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-PQ sectioned manifest: descriptor length for {} is {}, actual section length is {}",
                descriptor.name, descriptor.len, actual_len
            )));
        }
    }

    for section_name in REQUIRED_IVF_PQ_SECTIONS {
        if !seen.contains(section_name) {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-PQ sectioned manifest: missing required section descriptor {section_name}"
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
            "unsupported IVF-PQ sectioned metric: {other}"
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
