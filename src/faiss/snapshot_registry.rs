use crate::api::Result;
use crate::storage::AnnSnapshotRegistry;

use super::{
    HnswSnapshotLoader, IvfFlatSnapshotLoader, IvfPqSnapshotLoader, IvfSq8SnapshotLoader,
    IvfUsqSnapshotLoader, HNSW_SECTIONS_SNAPSHOT_VARIANT, HNSW_SNAPSHOT_VARIANT,
    IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT, IVF_PQ_SECTIONS_SNAPSHOT_VARIANT,
    IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT, IVF_USQ_SECTIONS_SNAPSHOT_VARIANT,
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
    Ok(registry)
}
