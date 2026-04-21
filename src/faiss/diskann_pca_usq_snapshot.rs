use crate::api::{KnowhereError, MetricType, Result};
use crate::faiss::diskann_pca_usq::{
    DiskAnnPcaUsqConfig, DiskAnnPcaUsqIndex, DiskAnnPcaUsqPcaExport, DiskAnnPcaUsqSectionedExport,
};
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    AnnSnapshot, AnnSnapshotLoader, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter,
    IndexManifest, LoadMode, MemoryArtifactStore, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::pqflash_snapshot::{load_pqflash_index_from_artifact, PqFlashSectionedSnapshot};

pub const DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT: &str = "diskann_pca_usq_sections_v1";
pub const DISKANN_PCA_USQ_META_SECTION: &str = "diskann_pca_usq.meta.json";
pub const DISKANN_PCA_USQ_QUANTIZER_CENTROID_SECTION: &str =
    "diskann_pca_usq.quantizer_centroid.f32";
pub const DISKANN_PCA_USQ_PACKED_BITS_SECTION: &str = "diskann_pca_usq.packed_bits.u8";
pub const DISKANN_PCA_USQ_SIGN_BITS_SECTION: &str = "diskann_pca_usq.sign_bits.u8";
pub const DISKANN_PCA_USQ_NORMS_SECTION: &str = "diskann_pca_usq.norms.f32";
pub const DISKANN_PCA_USQ_NORMS_SQ_SECTION: &str = "diskann_pca_usq.norms_sq.f32";
pub const DISKANN_PCA_USQ_VMAXS_SECTION: &str = "diskann_pca_usq.vmaxs.f32";
pub const DISKANN_PCA_USQ_QUANT_QUALITIES_SECTION: &str = "diskann_pca_usq.quant_qualities.f32";
pub const DISKANN_PCA_USQ_PCA_MEAN_SECTION: &str = "diskann_pca_usq.pca.mean.f32";
pub const DISKANN_PCA_USQ_PCA_COMPONENTS_SECTION: &str = "diskann_pca_usq.pca.components.f32";
pub const DISKANN_PCA_USQ_INNER_MANIFEST_SECTION: &str = "diskann_pca_usq.inner.manifest.json";
pub const DISKANN_PCA_USQ_INNER_PREFIX: &str = "diskann_pca_usq.inner.";

#[derive(Deserialize, Serialize)]
struct DiskAnnPcaUsqSectionMetadata {
    version: u32,
    metric: String,
    n: usize,
    d_in: usize,
    d_proj: usize,
    config: DiskAnnPcaUsqConfigSerde,
    code_bytes: usize,
    sign_bytes: usize,
    has_pca: bool,
    pca_d_in: usize,
    pca_d_out: usize,
}

#[derive(Deserialize, Serialize)]
struct DiskAnnPcaUsqConfigSerde {
    base: crate::faiss::diskann_aisaq::AisaqConfig,
    pca_dim: usize,
    bits_per_dim: u8,
    rotation_seed: u64,
    rerank_k: usize,
}

impl From<&DiskAnnPcaUsqConfig> for DiskAnnPcaUsqConfigSerde {
    fn from(config: &DiskAnnPcaUsqConfig) -> Self {
        Self {
            base: config.base.clone(),
            pca_dim: config.pca_dim,
            bits_per_dim: config.bits_per_dim,
            rotation_seed: config.rotation_seed,
            rerank_k: config.rerank_k,
        }
    }
}

impl From<DiskAnnPcaUsqConfigSerde> for DiskAnnPcaUsqConfig {
    fn from(config: DiskAnnPcaUsqConfigSerde) -> Self {
        Self {
            base: config.base,
            pca_dim: config.pca_dim,
            bits_per_dim: config.bits_per_dim,
            rotation_seed: config.rotation_seed,
            rerank_k: config.rerank_k,
        }
    }
}

pub struct DiskAnnPcaUsqSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl DiskAnnPcaUsqSectionedSnapshot {
    pub fn from_index(index: &DiskAnnPcaUsqIndex) -> Result<Self> {
        Self::from_export(index.export_sectioned_snapshot()?)
    }

    pub fn from_export(export: DiskAnnPcaUsqSectionedExport) -> Result<Self> {
        let inner_snapshot = PqFlashSectionedSnapshot::from_export(export.inner.clone())?;
        let (inner_manifest, inner_sections) = inner_snapshot.into_manifest_and_sections();
        let meta = DiskAnnPcaUsqSectionMetadata {
            version: 1,
            metric: metric_name(export.metric_type).to_string(),
            n: export.n,
            d_in: export.d_in,
            d_proj: export.d_proj,
            config: DiskAnnPcaUsqConfigSerde::from(&export.config),
            code_bytes: export.code_bytes,
            sign_bytes: export.sign_bytes,
            has_pca: export.pca.is_some(),
            pca_d_in: export.pca.as_ref().map_or(0, |pca| pca.d_in),
            pca_d_out: export.pca.as_ref().map_or(0, |pca| pca.d_out),
        };

        let mut sections = vec![
            (
                DISKANN_PCA_USQ_META_SECTION.to_string(),
                encode_json(&meta, DISKANN_PCA_USQ_META_SECTION)?,
            ),
            (
                DISKANN_PCA_USQ_QUANTIZER_CENTROID_SECTION.to_string(),
                encode_f32s(&export.quantizer_centroid),
            ),
            (
                DISKANN_PCA_USQ_PACKED_BITS_SECTION.to_string(),
                export.packed_bits,
            ),
            (
                DISKANN_PCA_USQ_SIGN_BITS_SECTION.to_string(),
                export.sign_bits,
            ),
            (
                DISKANN_PCA_USQ_NORMS_SECTION.to_string(),
                encode_f32s(&export.norms),
            ),
            (
                DISKANN_PCA_USQ_NORMS_SQ_SECTION.to_string(),
                encode_f32s(&export.norms_sq),
            ),
            (
                DISKANN_PCA_USQ_VMAXS_SECTION.to_string(),
                encode_f32s(&export.vmaxs),
            ),
            (
                DISKANN_PCA_USQ_QUANT_QUALITIES_SECTION.to_string(),
                encode_f32s(&export.quant_qualities),
            ),
            (
                DISKANN_PCA_USQ_PCA_MEAN_SECTION.to_string(),
                encode_f32s(export.pca.as_ref().map_or(&[], |pca| pca.mean.as_slice())),
            ),
            (
                DISKANN_PCA_USQ_PCA_COMPONENTS_SECTION.to_string(),
                encode_f32s(
                    export
                        .pca
                        .as_ref()
                        .map_or(&[], |pca| pca.components.as_slice()),
                ),
            ),
            (
                DISKANN_PCA_USQ_INNER_MANIFEST_SECTION.to_string(),
                encode_json(&inner_manifest, DISKANN_PCA_USQ_INNER_MANIFEST_SECTION)?,
            ),
        ];
        for (name, bytes) in inner_sections {
            sections.push((format!("{DISKANN_PCA_USQ_INNER_PREFIX}{name}"), bytes));
        }

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::DiskAnn,
            variant: DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT.to_string(),
            dim: export.d_in,
            metric: metric_name(export.metric_type).to_string(),
            count: export.n,
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

impl AnnSnapshot for DiskAnnPcaUsqSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_diskann_pca_usq_sectioned_snapshot(
    index: &DiskAnnPcaUsqIndex,
    root: impl AsRef<Path>,
) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = DiskAnnPcaUsqSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_diskann_pca_usq_sectioned_snapshot(
    root: impl AsRef<Path>,
) -> Result<DiskAnnPcaUsqIndex> {
    let store = FileArtifactStore::new(root)?;
    load_diskann_pca_usq_index_from_artifact(&store)
}

pub fn load_diskann_pca_usq_index_from_artifact(
    reader: &dyn IndexArtifactReader,
) -> Result<DiskAnnPcaUsqIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1
        || manifest.family != IndexFamily::DiskAnn
        || manifest.variant != DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT
    {
        return Err(KnowhereError::Codec(format!(
            "invalid DiskAnnPcaUsq snapshot manifest: version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }
    let meta_bytes = reader.read_section(DISKANN_PCA_USQ_META_SECTION)?;
    let meta: DiskAnnPcaUsqSectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|e| KnowhereError::Codec(format!("decode {DISKANN_PCA_USQ_META_SECTION}: {e}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "unsupported DiskAnnPcaUsq sectioned metadata version {}",
            meta.version
        )));
    }
    let quantizer_centroid = decode_f32s(
        reader
            .read_section(DISKANN_PCA_USQ_QUANTIZER_CENTROID_SECTION)?
            .as_ref(),
        DISKANN_PCA_USQ_QUANTIZER_CENTROID_SECTION,
    )?;
    let packed_bits = reader
        .read_section(DISKANN_PCA_USQ_PACKED_BITS_SECTION)?
        .into_owned();
    let sign_bits = reader
        .read_section(DISKANN_PCA_USQ_SIGN_BITS_SECTION)?
        .into_owned();
    let norms = decode_f32s(
        reader.read_section(DISKANN_PCA_USQ_NORMS_SECTION)?.as_ref(),
        DISKANN_PCA_USQ_NORMS_SECTION,
    )?;
    let norms_sq = decode_f32s(
        reader
            .read_section(DISKANN_PCA_USQ_NORMS_SQ_SECTION)?
            .as_ref(),
        DISKANN_PCA_USQ_NORMS_SQ_SECTION,
    )?;
    let vmaxs = decode_f32s(
        reader.read_section(DISKANN_PCA_USQ_VMAXS_SECTION)?.as_ref(),
        DISKANN_PCA_USQ_VMAXS_SECTION,
    )?;
    let quant_qualities = decode_f32s(
        reader
            .read_section(DISKANN_PCA_USQ_QUANT_QUALITIES_SECTION)?
            .as_ref(),
        DISKANN_PCA_USQ_QUANT_QUALITIES_SECTION,
    )?;

    let pca = if meta.has_pca {
        Some(DiskAnnPcaUsqPcaExport {
            d_in: meta.pca_d_in,
            d_out: meta.pca_d_out,
            mean: decode_f32s(
                reader
                    .read_section(DISKANN_PCA_USQ_PCA_MEAN_SECTION)?
                    .as_ref(),
                DISKANN_PCA_USQ_PCA_MEAN_SECTION,
            )?,
            components: decode_f32s(
                reader
                    .read_section(DISKANN_PCA_USQ_PCA_COMPONENTS_SECTION)?
                    .as_ref(),
                DISKANN_PCA_USQ_PCA_COMPONENTS_SECTION,
            )?,
        })
    } else {
        None
    };

    let inner_manifest_bytes = reader.read_section(DISKANN_PCA_USQ_INNER_MANIFEST_SECTION)?;
    let inner_manifest: IndexManifest = serde_json::from_slice(inner_manifest_bytes.as_ref())
        .map_err(|e| {
            KnowhereError::Codec(format!(
                "decode {DISKANN_PCA_USQ_INNER_MANIFEST_SECTION}: {e}"
            ))
        })?;
    let mut inner_store = MemoryArtifactStore::default();
    for descriptor in &inner_manifest.sections {
        let section = format!("{DISKANN_PCA_USQ_INNER_PREFIX}{}", descriptor.name);
        let bytes = reader.read_section(&section)?;
        inner_store.write_section(&descriptor.name, bytes.as_ref())?;
    }
    inner_store.finish_manifest(&inner_manifest)?;

    let export = DiskAnnPcaUsqSectionedExport {
        metric_type: parse_metric_name(&meta.metric)?,
        n: meta.n,
        d_in: meta.d_in,
        d_proj: meta.d_proj,
        config: meta.config.into(),
        quantizer_centroid,
        code_bytes: meta.code_bytes,
        sign_bytes: meta.sign_bytes,
        packed_bits,
        sign_bits,
        norms,
        norms_sq,
        vmaxs,
        quant_qualities,
        pca,
        inner: load_pqflash_index_from_artifact(&inner_store)?.export_sectioned_snapshot()?,
    };
    DiskAnnPcaUsqIndex::from_sectioned_snapshot_export(export)
}

pub struct DiskAnnPcaUsqSnapshotLoader;

impl AnnSnapshotLoader for DiskAnnPcaUsqSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        _mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        Ok(Box::new(load_diskann_pca_usq_index_from_artifact(reader)?))
    }
}

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Cosine => "cosine",
        MetricType::Ip => "ip",
        MetricType::Hamming => "hamming",
    }
}

fn parse_metric_name(metric: &str) -> Result<MetricType> {
    match metric {
        "l2" => Ok(MetricType::L2),
        "cosine" => Ok(MetricType::Cosine),
        "ip" => Ok(MetricType::Ip),
        "hamming" => Ok(MetricType::Hamming),
        other => Err(KnowhereError::Codec(format!(
            "unsupported DiskAnnPcaUsq metric: {other}"
        ))),
    }
}

fn encode_json<T: Serialize>(value: &T, section: &str) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(value)
        .map_err(|e| KnowhereError::Codec(format!("encode {section}: {e}")))
}

fn encode_f32s(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32s(bytes: &[u8], section: &str) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section}: length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
