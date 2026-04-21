use crate::api::{KnowhereError, MetricType, Result};
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::ivf_usq::{IvfUsqIndex, IvfUsqSectionedExport};
use super::IvfRuntime;

pub const IVF_USQ_SECTIONS_SNAPSHOT_VARIANT: &str = "ivf_usq_sections_v1";

pub const IVF_USQ_META_SECTION: &str = "ivf_usq.meta.json";
pub const IVF_USQ_CENTROIDS_SECTION: &str = "ivf_usq.centroids.f32";
pub const IVF_USQ_QUANTIZER_CENTROID_SECTION: &str = "ivf_usq.quantizer_centroid.f32";
pub const IVF_USQ_LIST_OFFSETS_SECTION: &str = "ivf_usq.list_offsets.u64";
pub const IVF_USQ_LIST_SIZES_SECTION: &str = "ivf_usq.list_sizes.u64";
pub const IVF_USQ_LIST_IDS_SECTION: &str = "ivf_usq.list_ids.i64";
pub const IVF_USQ_PACKED_BITS_SECTION: &str = "ivf_usq.packed_bits.u8";
pub const IVF_USQ_SIGN_BITS_SECTION: &str = "ivf_usq.sign_bits.u8";
pub const IVF_USQ_NORMS_SECTION: &str = "ivf_usq.norms.f32";
pub const IVF_USQ_NORMS_SQ_SECTION: &str = "ivf_usq.norms_sq.f32";
pub const IVF_USQ_VMAXS_SECTION: &str = "ivf_usq.vmaxs.f32";
pub const IVF_USQ_QUANT_QUALITIES_SECTION: &str = "ivf_usq.quant_qualities.f32";

const REQUIRED_IVF_USQ_SECTIONS: [&str; 12] = [
    IVF_USQ_META_SECTION,
    IVF_USQ_CENTROIDS_SECTION,
    IVF_USQ_QUANTIZER_CENTROID_SECTION,
    IVF_USQ_LIST_OFFSETS_SECTION,
    IVF_USQ_LIST_SIZES_SECTION,
    IVF_USQ_LIST_IDS_SECTION,
    IVF_USQ_PACKED_BITS_SECTION,
    IVF_USQ_SIGN_BITS_SECTION,
    IVF_USQ_NORMS_SECTION,
    IVF_USQ_NORMS_SQ_SECTION,
    IVF_USQ_VMAXS_SECTION,
    IVF_USQ_QUANT_QUALITIES_SECTION,
];

#[derive(Deserialize, Serialize)]
struct IvfUsqSectionMetadata {
    version: u32,
    dim: usize,
    metric: String,
    nlist: usize,
    nprobe: usize,
    bits_per_dim: usize,
    use_high_accuracy_scan: bool,
    rotation_seed: u64,
    rerank_k: usize,
    ntotal: usize,
    trained: bool,
    code_bytes: usize,
    sign_bytes: usize,
    padded_dim: usize,
}

pub struct IvfUsqSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl IvfUsqSectionedSnapshot {
    pub fn from_index(index: &IvfUsqIndex) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        let mut sections = Vec::new();

        let meta = IvfUsqSectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            nlist: export.nlist,
            nprobe: export.nprobe,
            bits_per_dim: export.bits_per_dim,
            use_high_accuracy_scan: export.use_high_accuracy_scan,
            rotation_seed: export.rotation_seed,
            rerank_k: export.rerank_k,
            ntotal: export.ntotal,
            trained: export.trained,
            code_bytes: export.code_bytes,
            sign_bytes: export.sign_bytes,
            padded_dim: export.padded_dim,
        };
        sections.push((
            IVF_USQ_META_SECTION.to_string(),
            encode_json(&meta, IVF_USQ_META_SECTION)?,
        ));
        sections.push((
            IVF_USQ_CENTROIDS_SECTION.to_string(),
            encode_f32s(&export.centroids),
        ));
        sections.push((
            IVF_USQ_QUANTIZER_CENTROID_SECTION.to_string(),
            encode_f32s(&export.quantizer_centroid),
        ));
        sections.push((
            IVF_USQ_LIST_OFFSETS_SECTION.to_string(),
            encode_u64s(&export.list_offsets),
        ));
        sections.push((
            IVF_USQ_LIST_SIZES_SECTION.to_string(),
            encode_u64s(&export.list_sizes),
        ));
        sections.push((
            IVF_USQ_LIST_IDS_SECTION.to_string(),
            encode_i64s(&export.list_ids),
        ));
        sections.push((
            IVF_USQ_PACKED_BITS_SECTION.to_string(),
            export.packed_bits.clone(),
        ));
        sections.push((
            IVF_USQ_SIGN_BITS_SECTION.to_string(),
            export.sign_bits.clone(),
        ));
        sections.push((
            IVF_USQ_NORMS_SECTION.to_string(),
            encode_f32s(&export.norms),
        ));
        sections.push((
            IVF_USQ_NORMS_SQ_SECTION.to_string(),
            encode_f32s(&export.norms_sq),
        ));
        sections.push((
            IVF_USQ_VMAXS_SECTION.to_string(),
            encode_f32s(&export.vmaxs),
        ));
        sections.push((
            IVF_USQ_QUANT_QUALITIES_SECTION.to_string(),
            encode_f32s(&export.quant_qualities),
        ));

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::Ivf,
            variant: IVF_USQ_SECTIONS_SNAPSHOT_VARIANT.to_string(),
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            count: export.ntotal,
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

impl AnnSnapshot for IvfUsqSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_ivf_usq_sectioned_snapshot(index: &IvfUsqIndex, root: impl AsRef<Path>) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = IvfUsqSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_ivf_usq_sectioned_snapshot(root: impl AsRef<Path>) -> Result<IvfUsqIndex> {
    let store = FileArtifactStore::new(root)?;
    load_ivf_usq_index_from_artifact(&store)
}

pub fn load_ivf_usq_index_from_artifact(reader: &dyn IndexArtifactReader) -> Result<IvfUsqIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1 || manifest.family != IndexFamily::Ivf {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-USQ snapshot manifest: expected version 1 family Ivf, got version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }

    match manifest.variant.as_str() {
        IVF_USQ_SECTIONS_SNAPSHOT_VARIANT => load_sectioned_ivf_usq(reader, manifest),
        variant => Err(KnowhereError::Codec(format!(
            "invalid IVF-USQ snapshot manifest: expected variant {IVF_USQ_SECTIONS_SNAPSHOT_VARIANT}, got {variant:?}"
        ))),
    }
}

pub struct IvfUsqSnapshotLoader;

impl AnnSnapshotLoader for IvfUsqSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        _mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        Ok(Box::new(IvfRuntime::usq(load_ivf_usq_index_from_artifact(
            reader,
        )?)))
    }
}

fn load_sectioned_ivf_usq(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<IvfUsqIndex> {
    validate_manifest_sections(reader, manifest)?;

    let meta_bytes = reader.read_section(IVF_USQ_META_SECTION)?;
    let meta: IvfUsqSectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|error| KnowhereError::Codec(format!("decode {IVF_USQ_META_SECTION}: {error}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-USQ sectioned metadata: unsupported version {}",
            meta.version
        )));
    }
    if manifest.dim != meta.dim || manifest.count != meta.ntotal || manifest.metric != meta.metric {
        return Err(KnowhereError::Codec(format!(
            "invalid IVF-USQ sectioned manifest metadata: manifest dim/count/metric = {}/{}/{:?}, metadata = {}/{}/{:?}",
            manifest.dim, manifest.count, manifest.metric, meta.dim, meta.ntotal, meta.metric
        )));
    }

    let export = IvfUsqSectionedExport {
        dim: meta.dim,
        nlist: meta.nlist,
        nprobe: meta.nprobe,
        metric_type: parse_metric_name(&meta.metric)?,
        bits_per_dim: meta.bits_per_dim,
        use_high_accuracy_scan: meta.use_high_accuracy_scan,
        rotation_seed: meta.rotation_seed,
        rerank_k: meta.rerank_k,
        ntotal: meta.ntotal,
        trained: meta.trained,
        code_bytes: meta.code_bytes,
        sign_bytes: meta.sign_bytes,
        padded_dim: meta.padded_dim,
        centroids: decode_f32s(
            reader.read_section(IVF_USQ_CENTROIDS_SECTION)?.as_ref(),
            IVF_USQ_CENTROIDS_SECTION,
        )?,
        quantizer_centroid: decode_f32s(
            reader
                .read_section(IVF_USQ_QUANTIZER_CENTROID_SECTION)?
                .as_ref(),
            IVF_USQ_QUANTIZER_CENTROID_SECTION,
        )?,
        list_offsets: decode_u64s(
            reader.read_section(IVF_USQ_LIST_OFFSETS_SECTION)?.as_ref(),
            IVF_USQ_LIST_OFFSETS_SECTION,
        )?,
        list_sizes: decode_u64s(
            reader.read_section(IVF_USQ_LIST_SIZES_SECTION)?.as_ref(),
            IVF_USQ_LIST_SIZES_SECTION,
        )?,
        list_ids: decode_i64s(
            reader.read_section(IVF_USQ_LIST_IDS_SECTION)?.as_ref(),
            IVF_USQ_LIST_IDS_SECTION,
        )?,
        packed_bits: reader
            .read_section(IVF_USQ_PACKED_BITS_SECTION)?
            .into_owned(),
        sign_bits: reader.read_section(IVF_USQ_SIGN_BITS_SECTION)?.into_owned(),
        norms: decode_f32s(
            reader.read_section(IVF_USQ_NORMS_SECTION)?.as_ref(),
            IVF_USQ_NORMS_SECTION,
        )?,
        norms_sq: decode_f32s(
            reader.read_section(IVF_USQ_NORMS_SQ_SECTION)?.as_ref(),
            IVF_USQ_NORMS_SQ_SECTION,
        )?,
        vmaxs: decode_f32s(
            reader.read_section(IVF_USQ_VMAXS_SECTION)?.as_ref(),
            IVF_USQ_VMAXS_SECTION,
        )?,
        quant_qualities: decode_f32s(
            reader
                .read_section(IVF_USQ_QUANT_QUALITIES_SECTION)?
                .as_ref(),
            IVF_USQ_QUANT_QUALITIES_SECTION,
        )?,
    };

    IvfUsqIndex::from_sectioned_snapshot_export(export)
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
                "invalid IVF-USQ sectioned manifest: duplicate section descriptor {}",
                descriptor.name
            )));
        }

        let actual_len = reader.section_len(&descriptor.name).map_err(|error| {
            KnowhereError::Codec(format!(
                "invalid IVF-USQ sectioned manifest: descriptor {} references unreadable section: {error}",
                descriptor.name
            ))
        })?;
        if descriptor.len != actual_len {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-USQ sectioned manifest: descriptor length for {} is {}, actual section length is {}",
                descriptor.name, descriptor.len, actual_len
            )));
        }
    }

    for section_name in REQUIRED_IVF_USQ_SECTIONS {
        if !seen.contains(section_name) {
            return Err(KnowhereError::Codec(format!(
                "invalid IVF-USQ sectioned manifest: missing required section descriptor {section_name}"
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
            "unsupported IVF-USQ sectioned metric: {other}"
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
