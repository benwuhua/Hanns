pub trait QuantizerModel: Send + Sync {
    type QueryState;

    fn dim(&self) -> usize;
    fn code_size(&self) -> usize;
    fn precompute_query(&self, query: &[f32]) -> Self::QueryState;
    fn score_code(&self, state: &Self::QueryState, code: &[u8]) -> f32;
}

pub trait EncodedVectorStore: Send + Sync {
    fn code(&self, row: usize) -> &[u8];
}

pub trait RerankStore: Send + Sync {
    fn raw_vector(&self, row: usize) -> Option<&[f32]>;
}
