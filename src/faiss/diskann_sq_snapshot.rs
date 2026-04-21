use crate::api::{KnowhereError, MetricType, Result};
use crate::faiss::diskann_sq::{
    DiskAnnSqConfig, DiskAnnSqIndex, DiskAnnSqPcaExport, DiskAnnSqSectionedExport,
};
use crate::kernel::IndexFamily;
use crate::quantization::sq::QuantizerType;
use crate::storage::{
    AnnSnapshot, FileArtifactStore, IndexArtifactReader, IndexArtifactWriter, IndexManifest,
    MemoryArtifactStore, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::pqflash_snapshot::{load_pqflash_index_from_artifact, PqFlashSectionedSnapshot};

pub const DISKANN_SQ_SECTIONS_SNAPSHOT_VARIANT: &str = "diskann_sq_sections_v1";
pub const DISKANN_SQ_META_SECTION: &str = "diskann_sq.meta.json";
pub const DISKANN_SQ_SQ_CODES_SECTION: &str = "diskann_sq.sq.codes.u8";
pub const DISKANN_SQ_PCA_MEAN_SECTION: &str = "diskann_sq.pca.mean.f32";
pub const DISKANN_SQ_PCA_COMPONENTS_SECTION: &str = "diskann_sq.pca.components.f32";
pub const DISKANN_SQ_INNER_MANIFEST_SECTION: &str = "diskann_sq.inner.manifest.json";
pub const DISKANN_SQ_INNER_PREFIX: &str = "diskann_sq.inner.";

#[derive(Deserialize, Serialize)]
struct DiskAnnSqSectionMetadata {
    version: u32,
    metric: String,
    n: usize,
    d_in: usize,
    d_sq: usize,
    config: DiskAnnSqConfigSerde,
    sq_dim: usize,
    sq_bit: usize,
    sq_quantizer_type: QuantizerType,
    sq_min_val: f32,
    sq_max_val: f32,
    sq_scale: f32,
    sq_offset: f32,
    has_pca: bool,
    pca_d_in: usize,
    pca_d_out: usize,
}

#[derive(Deserialize, Serialize)]
struct DiskAnnSqConfigSerde {
    base: crate::faiss::diskann_aisaq::AisaqConfig,
    use_pca: bool,
    pca_dim: usize,
    rerank_k: usize,
}

impl From<&DiskAnnSqConfig> for DiskAnnSqConfigSerde {
    fn from(config: &DiskAnnSqConfig) -> Self {
        Self {
            base: config.base.clone(),
            use_pca: config.use_pca,
            pca_dim: config.pca_dim,
            rerank_k: config.rerank_k,
        }
    }
}

impl From<DiskAnnSqConfigSerde> for DiskAnnSqConfig {
    fn from(config: DiskAnnSqConfigSerde) -> Self {
        Self {
            base: config.base,
            use_pca: config.use_pca,
            pca_dim: config.pca_dim,
            rerank_k: config.rerank_k,
        }
    }
}

pub struct DiskAnnSqSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl DiskAnnSqSectionedSnapshot {
    pub fn from_index(index: &DiskAnnSqIndex) -> Result<Self> {
        Self::from_export(index.export_sectioned_snapshot()?)
    }

    pub fn from_export(export: DiskAnnSqSectionedExport) -> Result<Self> {
        let inner_snapshot = PqFlashSectionedSnapshot::from_export(export.inner.clone())?;
        let (inner_manifest, inner_sections) = inner_snapshot.into_manifest_and_sections();
        let meta = DiskAnnSqSectionMetadata {
            version: 1,
            metric: metric_name(export.metric_type).to_string(),
            n: export.n,
            d_in: export.d_in,
            d_sq: export.d_sq,
            config: DiskAnnSqConfigSerde::from(&export.config),
            sq_dim: export.sq_dim,
            sq_bit: export.sq_bit,
            sq_quantizer_type: export.sq_quantizer_type,
            sq_min_val: export.sq_min_val,
            sq_max_val: export.sq_max_val,
            sq_scale: export.sq_scale,
            sq_offset: export.sq_offset,
            has_pca: export.pca.is_some(),
            pca_d_in: export.pca.as_ref().map_or(0, |pca| pca.d_in),
            pca_d_out: export.pca.as_ref().map_or(0, |pca| pca.d_out),
        };

        let mut sections = vec![
            (
                DISKANN_SQ_META_SECTION.to_string(),
                encode_json(&meta, DISKANN_SQ_META_SECTION)?,
            ),
            (DISKANN_SQ_SQ_CODES_SECTION.to_string(), export.sq_codes),
            (
                DISKANN_SQ_PCA_MEAN_SECTION.to_string(),
                encode_f32s(export.pca.as_ref().map_or(&[], |pca| pca.mean.as_slice())),
            ),
            (
                DISKANN_SQ_PCA_COMPONENTS_SECTION.to_string(),
                encode_f32s(
                    export
                        .pca
                        .as_ref()
                        .map_or(&[], |pca| pca.components.as_slice()),
                ),
            ),
            (
                DISKANN_SQ_INNER_MANIFEST_SECTION.to_string(),
                encode_json(&inner_manifest, DISKANN_SQ_INNER_MANIFEST_SECTION)?,
            ),
        ];
        for (name, bytes) in inner_sections {
            sections.push((format!("{DISKANN_SQ_INNER_PREFIX}{name}"), bytes));
        }

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::DiskAnn,
            variant: DISKANN_SQ_SECTIONS_SNAPSHOT_VARIANT.to_string(),
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

impl AnnSnapshot for DiskAnnSqSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_diskann_sq_sectioned_snapshot(
    index: &DiskAnnSqIndex,
    root: impl AsRef<Path>,
) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = DiskAnnSqSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_diskann_sq_sectioned_snapshot(root: impl AsRef<Path>) -> Result<DiskAnnSqIndex> {
    let store = FileArtifactStore::new(root)?;
    load_diskann_sq_index_from_artifact(&store)
}

pub fn load_diskann_sq_index_from_artifact(
    reader: &dyn IndexArtifactReader,
) -> Result<DiskAnnSqIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1
        || manifest.family != IndexFamily::DiskAnn
        || manifest.variant != DISKANN_SQ_SECTIONS_SNAPSHOT_VARIANT
    {
        return Err(KnowhereError::Codec(format!(
            "invalid DiskAnnSq snapshot manifest: version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }

    let meta_bytes = reader.read_section(DISKANN_SQ_META_SECTION)?;
    let meta: DiskAnnSqSectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|e| KnowhereError::Codec(format!("decode {DISKANN_SQ_META_SECTION}: {e}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "unsupported DiskAnnSq sectioned metadata version {}",
            meta.version
        )));
    }
    let sq_codes = reader
        .read_section(DISKANN_SQ_SQ_CODES_SECTION)?
        .into_owned();
    let expected_sq = meta.n.checked_mul(meta.d_sq).ok_or_else(|| {
        KnowhereError::Codec("invalid DiskAnnSq sectioned snapshot: SQ code overflow".to_string())
    })?;
    if sq_codes.len() != expected_sq {
        return Err(KnowhereError::Codec(format!(
            "invalid DiskAnnSq sectioned snapshot: sq_codes len {} != n * d_sq {}",
            sq_codes.len(),
            expected_sq
        )));
    }
    let pca = if meta.has_pca {
        let mean = decode_f32s(
            reader.read_section(DISKANN_SQ_PCA_MEAN_SECTION)?.as_ref(),
            DISKANN_SQ_PCA_MEAN_SECTION,
        )?;
        let components = decode_f32s(
            reader
                .read_section(DISKANN_SQ_PCA_COMPONENTS_SECTION)?
                .as_ref(),
            DISKANN_SQ_PCA_COMPONENTS_SECTION,
        )?;
        Some(DiskAnnSqPcaExport {
            d_in: meta.pca_d_in,
            d_out: meta.pca_d_out,
            mean,
            components,
        })
    } else {
        None
    };

    let inner_manifest_bytes = reader.read_section(DISKANN_SQ_INNER_MANIFEST_SECTION)?;
    let inner_manifest: IndexManifest = serde_json::from_slice(inner_manifest_bytes.as_ref())
        .map_err(|e| {
            KnowhereError::Codec(format!("decode {DISKANN_SQ_INNER_MANIFEST_SECTION}: {e}"))
        })?;
    let mut inner_store = MemoryArtifactStore::default();
    for descriptor in &inner_manifest.sections {
        let section = format!("{DISKANN_SQ_INNER_PREFIX}{}", descriptor.name);
        let bytes = reader.read_section(&section)?;
        inner_store.write_section(&descriptor.name, bytes.as_ref())?;
    }
    inner_store.finish_manifest(&inner_manifest)?;

    let export = DiskAnnSqSectionedExport {
        metric_type: parse_metric_name(&meta.metric)?,
        n: meta.n,
        d_in: meta.d_in,
        d_sq: meta.d_sq,
        config: meta.config.into(),
        sq_dim: meta.sq_dim,
        sq_bit: meta.sq_bit,
        sq_quantizer_type: meta.sq_quantizer_type,
        sq_min_val: meta.sq_min_val,
        sq_max_val: meta.sq_max_val,
        sq_scale: meta.sq_scale,
        sq_offset: meta.sq_offset,
        sq_codes,
        pca,
        inner: load_pqflash_index_from_artifact(&inner_store)?.export_sectioned_snapshot()?,
    };
    DiskAnnSqIndex::from_sectioned_snapshot_export(export)
}

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "l2",
        MetricType::Ip => "ip",
        MetricType::Cosine => "cosine",
        MetricType::Hamming => "hamming",
    }
}

fn parse_metric_name(metric: &str) -> Result<MetricType> {
    match metric {
        "l2" => Ok(MetricType::L2),
        "ip" => Ok(MetricType::Ip),
        "cosine" => Ok(MetricType::Cosine),
        "hamming" => Ok(MetricType::Hamming),
        other => Err(KnowhereError::Codec(format!(
            "unsupported DiskAnnSq metric: {other}"
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
