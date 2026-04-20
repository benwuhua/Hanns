use crate::api::Result;
use crate::kernel::AnnRuntime;

use super::{IndexArtifactReader, IndexArtifactWriter};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    OwnedMemory,
    Mmap,
    PageCache,
    Lazy,
}

pub trait AnnSnapshot {
    fn write_snapshot(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()>;
}

pub trait AnnSnapshotLoader {
    fn load_snapshot(
        &self,
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>>;
}
