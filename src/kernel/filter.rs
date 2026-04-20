pub trait RowFilter: Send + Sync {
    fn is_deleted(&self, row: usize) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoFilter;

impl RowFilter for NoFilter {
    #[inline]
    fn is_deleted(&self, _row: usize) -> bool {
        false
    }
}
