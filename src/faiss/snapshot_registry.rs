use crate::api::Result;
use crate::storage::AnnSnapshotRegistry;

use super::{
    DiskAnnPcaUsqSnapshotLoader, DiskAnnSqSnapshotLoader, HnswSnapshotLoader,
    IvfFlatSnapshotLoader, IvfPqSnapshotLoader, IvfSq8SnapshotLoader, IvfUsqSnapshotLoader,
    PqFlashSnapshotLoader, DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT,
    DISKANN_SQ_SECTIONS_SNAPSHOT_VARIANT, HNSW_SECTIONS_SNAPSHOT_VARIANT, HNSW_SNAPSHOT_VARIANT,
    IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT, IVF_PQ_SECTIONS_SNAPSHOT_VARIANT,
    IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT, IVF_USQ_SECTIONS_SNAPSHOT_VARIANT,
    PQFLASH_SECTIONS_SNAPSHOT_VARIANT,
};

pub fn default_ann_snapshot_registry() -> Result<AnnSnapshotRegistry> {
    let mut registry = AnnSnapshotRegistry::new();
    registry.register_loader(HNSW_SNAPSHOT_VARIANT, Box::new(HnswSnapshotLoader))?;
    registry.register_loader(HNSW_SECTIONS_SNAPSHOT_VARIANT, Box::new(HnswSnapshotLoader))?;
    registry.register_loader(
        IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(IvfFlatSnapshotLoader),
    )?;
    registry.register_loader(
        IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(IvfSq8SnapshotLoader),
    )?;
    registry.register_loader(
        IVF_PQ_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(IvfPqSnapshotLoader),
    )?;
    registry.register_loader(
        IVF_USQ_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(IvfUsqSnapshotLoader),
    )?;
    registry.register_loader(
        PQFLASH_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(PqFlashSnapshotLoader),
    )?;
    registry.register_loader(
        DISKANN_SQ_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(DiskAnnSqSnapshotLoader),
    )?;
    registry.register_loader(
        DISKANN_PCA_USQ_SECTIONS_SNAPSHOT_VARIANT,
        Box::new(DiskAnnPcaUsqSnapshotLoader),
    )?;
    Ok(registry)
}
