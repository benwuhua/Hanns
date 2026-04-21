use crate::api::{KnowhereError, MetricType, Result};
use crate::faiss::diskann_aisaq::{AisaqConfig, FlashLayout, PQFlashIndex};
use crate::kernel::{AnnRuntime, IndexFamily};
use crate::storage::{
    require_owned_memory_load_mode, AnnSnapshot, AnnSnapshotLoader, FileArtifactStore,
    IndexArtifactReader, IndexArtifactWriter, IndexManifest, LoadMode, SectionDescriptor,
};

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::diskann_aisaq::{
    validate_pqflash_hvq_sectioned, validate_pqflash_sq8_sectioned, PQFlashSectionedExport,
    PQFlashSectionedHvqExport, PQFlashSectionedPqExport, PQFlashSectionedSq8Export,
};

pub const PQFLASH_SECTIONS_SNAPSHOT_VARIANT: &str = "pqflash_sections_v1";

pub const PQFLASH_META_SECTION: &str = "pqflash.meta.json";
pub const PQFLASH_VECTORS_SECTION: &str = "pqflash.vectors.f32";
pub const PQFLASH_NODE_IDS_SECTION: &str = "pqflash.node_ids.i64";
pub const PQFLASH_NEIGHBOR_COUNTS_SECTION: &str = "pqflash.neighbor_counts.u32";
pub const PQFLASH_NEIGHBOR_IDS_SECTION: &str = "pqflash.neighbor_ids.u32";
pub const PQFLASH_NODE_PQ_CODES_SECTION: &str = "pqflash.node_pq_codes.u8";
pub const PQFLASH_DELETED_ROWS_SECTION: &str = "pqflash.deleted_rows.u64";
pub const PQFLASH_PQ_CENTROIDS_SECTION: &str = "pqflash.pq_centroids.f32";
pub const PQFLASH_HVQ_META_SECTION: &str = "pqflash.hvq.meta.json";
pub const PQFLASH_HVQ_CODES_SECTION: &str = "pqflash.hvq.codes.u8";
pub const PQFLASH_SQ8_META_SECTION: &str = "pqflash.sq8.meta.json";
pub const PQFLASH_SQ8_CODES_SECTION: &str = "pqflash.sq8.codes.u8";

const REQUIRED_PQFLASH_SECTIONS: [&str; 12] = [
    PQFLASH_META_SECTION,
    PQFLASH_VECTORS_SECTION,
    PQFLASH_NODE_IDS_SECTION,
    PQFLASH_NEIGHBOR_COUNTS_SECTION,
    PQFLASH_NEIGHBOR_IDS_SECTION,
    PQFLASH_NODE_PQ_CODES_SECTION,
    PQFLASH_DELETED_ROWS_SECTION,
    PQFLASH_PQ_CENTROIDS_SECTION,
    PQFLASH_HVQ_META_SECTION,
    PQFLASH_HVQ_CODES_SECTION,
    PQFLASH_SQ8_META_SECTION,
    PQFLASH_SQ8_CODES_SECTION,
];

#[derive(Deserialize, Serialize)]
struct PqFlashSectionMetadata {
    version: u32,
    dim: usize,
    metric: String,
    count: usize,
    trained: bool,
    flat_stride: usize,
    pq_code_size: usize,
    config: AisaqConfig,
    flash_layout: FlashLayout,
    entry_points: Vec<u32>,
    pq_m: Option<usize>,
    pq_nbits: Option<usize>,
    has_hvq: bool,
    has_sq8: bool,
}

#[derive(Deserialize, Serialize)]
struct PqFlashHvqMetadata {
    dim: usize,
    nbits: u8,
    rotation_matrix: Vec<f32>,
    scale: f32,
    offset: f32,
    centroid: Vec<f32>,
    rotated_centroid: Vec<f32>,
}

#[derive(Deserialize, Serialize)]
struct PqFlashSq8Metadata {
    dim: usize,
    bit: usize,
    min_val: f32,
    max_val: f32,
    scale: f32,
    offset: f32,
}

pub struct PqFlashSectionedSnapshot {
    manifest: IndexManifest,
    sections: Vec<(String, Vec<u8>)>,
}

impl PqFlashSectionedSnapshot {
    pub fn from_index(index: &PQFlashIndex) -> Result<Self> {
        let export = index.export_sectioned_snapshot()?;
        Self::from_export(export)
    }

    pub fn from_export(export: PQFlashSectionedExport) -> Result<Self> {
        let meta = PqFlashSectionMetadata {
            version: 1,
            dim: export.dim,
            metric: metric_name(export.metric_type).to_string(),
            count: export.count,
            trained: export.trained,
            flat_stride: export.flat_stride,
            pq_code_size: export.pq_code_size,
            config: export.config.clone(),
            flash_layout: export.flash_layout.clone(),
            entry_points: export.entry_points.clone(),
            pq_m: export.pq.as_ref().map(|pq| pq.m),
            pq_nbits: export.pq.as_ref().map(|pq| pq.nbits),
            has_hvq: export.hvq.is_some(),
            has_sq8: export.sq8.is_some(),
        };

        let mut sections = vec![
            (
                PQFLASH_META_SECTION.to_string(),
                encode_json(&meta, PQFLASH_META_SECTION)?,
            ),
            (
                PQFLASH_VECTORS_SECTION.to_string(),
                encode_f32s(&export.vectors),
            ),
            (
                PQFLASH_NODE_IDS_SECTION.to_string(),
                encode_i64s(&export.node_ids),
            ),
            (
                PQFLASH_NEIGHBOR_COUNTS_SECTION.to_string(),
                encode_u32s(&export.neighbor_counts),
            ),
            (
                PQFLASH_NEIGHBOR_IDS_SECTION.to_string(),
                encode_u32s(&export.neighbor_ids),
            ),
            (
                PQFLASH_NODE_PQ_CODES_SECTION.to_string(),
                export.node_pq_codes.clone(),
            ),
            (
                PQFLASH_DELETED_ROWS_SECTION.to_string(),
                encode_u64s(&export.deleted_rows),
            ),
            (
                PQFLASH_PQ_CENTROIDS_SECTION.to_string(),
                encode_f32s(
                    export
                        .pq
                        .as_ref()
                        .map(|pq| pq.centroids.as_slice())
                        .unwrap_or(&[]),
                ),
            ),
            (
                PQFLASH_HVQ_META_SECTION.to_string(),
                encode_json(
                    &export.hvq.as_ref().map(|hvq| PqFlashHvqMetadata {
                        dim: hvq.dim,
                        nbits: hvq.nbits,
                        rotation_matrix: hvq.rotation_matrix.clone(),
                        scale: hvq.scale,
                        offset: hvq.offset,
                        centroid: hvq.centroid.clone(),
                        rotated_centroid: hvq.rotated_centroid.clone(),
                    }),
                    PQFLASH_HVQ_META_SECTION,
                )?,
            ),
            (
                PQFLASH_HVQ_CODES_SECTION.to_string(),
                export
                    .hvq
                    .as_ref()
                    .map(|hvq| hvq.codes.clone())
                    .unwrap_or_default(),
            ),
            (
                PQFLASH_SQ8_META_SECTION.to_string(),
                encode_json(
                    &export.sq8.as_ref().map(|sq8| PqFlashSq8Metadata {
                        dim: sq8.dim,
                        bit: sq8.bit,
                        min_val: sq8.min_val,
                        max_val: sq8.max_val,
                        scale: sq8.scale,
                        offset: sq8.offset,
                    }),
                    PQFLASH_SQ8_META_SECTION,
                )?,
            ),
            (
                PQFLASH_SQ8_CODES_SECTION.to_string(),
                export
                    .sq8
                    .as_ref()
                    .map(|sq8| sq8.codes.clone())
                    .unwrap_or_default(),
            ),
        ];

        let manifest = IndexManifest {
            version: 1,
            family: IndexFamily::DiskAnn,
            variant: PQFLASH_SECTIONS_SNAPSHOT_VARIANT.to_string(),
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

        // Keep a stable section order matching REQUIRED_PQFLASH_SECTIONS.
        sections.shrink_to_fit();
        Ok(Self { manifest, sections })
    }

    pub fn into_manifest_and_sections(self) -> (IndexManifest, Vec<(String, Vec<u8>)>) {
        (self.manifest, self.sections)
    }
}

impl AnnSnapshot for PqFlashSectionedSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()> {
        for (name, bytes) in &self.sections {
            writer.write_section(name, bytes)?;
        }
        writer.finish_manifest(&self.manifest)
    }
}

pub fn save_pqflash_sectioned_snapshot(index: &PQFlashIndex, root: impl AsRef<Path>) -> Result<()> {
    let mut store = FileArtifactStore::new(root)?;
    let snapshot = PqFlashSectionedSnapshot::from_index(index)?;
    snapshot.write_snapshot(&mut store)
}

pub fn load_pqflash_sectioned_snapshot(root: impl AsRef<Path>) -> Result<PQFlashIndex> {
    let store = FileArtifactStore::new(root)?;
    load_pqflash_index_from_artifact(&store)
}

pub fn load_pqflash_index_from_artifact(reader: &dyn IndexArtifactReader) -> Result<PQFlashIndex> {
    let manifest = reader.manifest()?;
    if manifest.version != 1 || manifest.family != IndexFamily::DiskAnn {
        return Err(KnowhereError::Codec(format!(
            "invalid PQFlash snapshot manifest: expected version 1 family DiskAnn, got version {} family {:?} variant {:?}",
            manifest.version, manifest.family, manifest.variant
        )));
    }
    match manifest.variant.as_str() {
        PQFLASH_SECTIONS_SNAPSHOT_VARIANT => load_sectioned_pqflash(reader, manifest),
        variant => Err(KnowhereError::Codec(format!(
            "invalid PQFlash snapshot manifest: expected variant {PQFLASH_SECTIONS_SNAPSHOT_VARIANT}, got {variant:?}"
        ))),
    }
}

pub struct PqFlashSnapshotLoader;

impl AnnSnapshotLoader for PqFlashSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>> {
        require_owned_memory_load_mode(mode, "PqFlashSnapshotLoader")?;
        Ok(Box::new(load_pqflash_index_from_artifact(reader)?))
    }
}

fn load_sectioned_pqflash(
    reader: &dyn IndexArtifactReader,
    manifest: &IndexManifest,
) -> Result<PQFlashIndex> {
    validate_manifest_sections(reader, manifest)?;
    let meta_bytes = reader.read_section(PQFLASH_META_SECTION)?;
    let meta: PqFlashSectionMetadata = serde_json::from_slice(meta_bytes.as_ref())
        .map_err(|e| KnowhereError::Codec(format!("decode {PQFLASH_META_SECTION}: {e}")))?;
    if meta.version != 1 {
        return Err(KnowhereError::Codec(format!(
            "invalid PQFlash metadata version {}",
            meta.version
        )));
    }
    if manifest.dim != meta.dim || manifest.count != meta.count || manifest.metric != meta.metric {
        return Err(KnowhereError::Codec(format!(
            "invalid PQFlash manifest metadata mismatch: manifest {}/{}/{:?}, metadata {}/{}/{:?}",
            manifest.dim, manifest.count, manifest.metric, meta.dim, meta.count, meta.metric
        )));
    }
    let pq_centroids = decode_f32s(
        reader.read_section(PQFLASH_PQ_CENTROIDS_SECTION)?.as_ref(),
        PQFLASH_PQ_CENTROIDS_SECTION,
    )?;
    let pq = match (meta.pq_m, meta.pq_nbits, meta.pq_code_size) {
        (Some(m), Some(nbits), code_size) if code_size > 0 => Some(PQFlashSectionedPqExport {
            m,
            nbits,
            dim: meta.dim,
            centroids: pq_centroids,
        }),
        _ if pq_centroids.is_empty() => None,
        _ => {
            return Err(KnowhereError::Codec(
                "invalid PQFlash metadata: pq_centroids present without PQ config".to_string(),
            ));
        }
    };
    let hvq_meta: Option<PqFlashHvqMetadata> =
        serde_json::from_slice(reader.read_section(PQFLASH_HVQ_META_SECTION)?.as_ref())
            .map_err(|e| KnowhereError::Codec(format!("decode {PQFLASH_HVQ_META_SECTION}: {e}")))?;
    let hvq_codes = reader.read_section(PQFLASH_HVQ_CODES_SECTION)?.into_owned();
    let hvq = match (meta.has_hvq, hvq_meta) {
        (true, Some(hvq_meta)) => {
            let hvq = PQFlashSectionedHvqExport {
                dim: hvq_meta.dim,
                nbits: hvq_meta.nbits,
                rotation_matrix: hvq_meta.rotation_matrix,
                scale: hvq_meta.scale,
                offset: hvq_meta.offset,
                centroid: hvq_meta.centroid,
                rotated_centroid: hvq_meta.rotated_centroid,
                codes: hvq_codes,
            };
            validate_pqflash_hvq_sectioned(meta.count, meta.dim, &hvq)?;
            Some(hvq)
        }
        (false, None) if hvq_codes.is_empty() => None,
        _ => {
            return Err(KnowhereError::Codec(
                "invalid PQFlash HVQ section state".to_string(),
            ));
        }
    };
    let sq8_meta: Option<PqFlashSq8Metadata> =
        serde_json::from_slice(reader.read_section(PQFLASH_SQ8_META_SECTION)?.as_ref())
            .map_err(|e| KnowhereError::Codec(format!("decode {PQFLASH_SQ8_META_SECTION}: {e}")))?;
    let sq8_codes = reader.read_section(PQFLASH_SQ8_CODES_SECTION)?.into_owned();
    let sq8 = match (meta.has_sq8, sq8_meta) {
        (true, Some(sq8_meta)) => {
            let sq8 = PQFlashSectionedSq8Export {
                dim: sq8_meta.dim,
                bit: sq8_meta.bit,
                min_val: sq8_meta.min_val,
                max_val: sq8_meta.max_val,
                scale: sq8_meta.scale,
                offset: sq8_meta.offset,
                codes: sq8_codes,
            };
            validate_pqflash_sq8_sectioned(meta.count, meta.dim, &sq8)?;
            Some(sq8)
        }
        (false, None) if sq8_codes.is_empty() => None,
        _ => {
            return Err(KnowhereError::Codec(
                "invalid PQFlash SQ8 section state".to_string(),
            ));
        }
    };

    let export = PQFlashSectionedExport {
        config: meta.config,
        metric_type: parse_metric_name(&meta.metric)?,
        dim: meta.dim,
        flash_layout: meta.flash_layout,
        flat_stride: meta.flat_stride,
        pq_code_size: meta.pq_code_size,
        entry_points: meta.entry_points,
        trained: meta.trained,
        count: meta.count,
        vectors: decode_f32s(
            reader.read_section(PQFLASH_VECTORS_SECTION)?.as_ref(),
            PQFLASH_VECTORS_SECTION,
        )?,
        node_ids: decode_i64s(
            reader.read_section(PQFLASH_NODE_IDS_SECTION)?.as_ref(),
            PQFLASH_NODE_IDS_SECTION,
        )?,
        neighbor_counts: decode_u32s(
            reader
                .read_section(PQFLASH_NEIGHBOR_COUNTS_SECTION)?
                .as_ref(),
            PQFLASH_NEIGHBOR_COUNTS_SECTION,
        )?,
        neighbor_ids: decode_u32s(
            reader.read_section(PQFLASH_NEIGHBOR_IDS_SECTION)?.as_ref(),
            PQFLASH_NEIGHBOR_IDS_SECTION,
        )?,
        node_pq_codes: reader
            .read_section(PQFLASH_NODE_PQ_CODES_SECTION)?
            .into_owned(),
        deleted_rows: decode_u64s(
            reader.read_section(PQFLASH_DELETED_ROWS_SECTION)?.as_ref(),
            PQFLASH_DELETED_ROWS_SECTION,
        )?,
        pq,
        hvq,
        sq8,
    };

    PQFlashIndex::from_sectioned_snapshot_export(export)
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
                "invalid PQFlash sectioned manifest: duplicate section descriptor {}",
                descriptor.name
            )));
        }
        let actual_len = reader.section_len(&descriptor.name).map_err(|error| {
            KnowhereError::Codec(format!(
                "invalid PQFlash sectioned manifest: descriptor {} references unreadable section: {error}",
                descriptor.name
            ))
        })?;
        if descriptor.len != actual_len {
            return Err(KnowhereError::Codec(format!(
                "invalid PQFlash sectioned manifest: descriptor length for {} is {}, actual section length is {}",
                descriptor.name, descriptor.len, actual_len
            )));
        }
    }

    for section in REQUIRED_PQFLASH_SECTIONS {
        if !seen.contains(section) {
            return Err(KnowhereError::Codec(format!(
                "invalid PQFlash sectioned manifest: missing required section descriptor {section}"
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
            "unsupported PQFlash metric: {other}"
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

fn encode_i64s(values: &[i64]) -> Vec<u8> {
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

fn encode_u64s(values: &[u64]) -> Vec<u8> {
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

fn decode_i64s(bytes: &[u8], section: &str) -> Result<Vec<i64>> {
    if bytes.len() % 8 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section}: length {} is not a multiple of 8",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_u32s(bytes: &[u8], section: &str) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section}: length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_u64s(bytes: &[u8], section: &str) -> Result<Vec<u64>> {
    if bytes.len() % 8 != 0 {
        return Err(KnowhereError::Codec(format!(
            "decode {section}: length {} is not a multiple of 8",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}
