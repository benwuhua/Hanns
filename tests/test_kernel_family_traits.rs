use hanns::kernel::{
    DiskGraphRuntimeConfig, EncodedVectorStore, IvfListScanner, IvfPartitionSelector, NodeReader,
    NodeRecord, QuantizerModel, RerankStore, SelectedPartitions,
};

struct DummyPartitionSelector;

impl IvfPartitionSelector for DummyPartitionSelector {
    fn select_partitions(&self, _query: &[f32], nprobe: usize) -> SelectedPartitions {
        SelectedPartitions {
            ids: vec![0, 2].into_iter().take(nprobe).collect(),
            distances: vec![0.1, 0.2].into_iter().take(nprobe).collect(),
        }
    }
}

struct DummyListScanner;

impl IvfListScanner for DummyListScanner {
    fn scan_list(
        &self,
        _query: &[f32],
        list_id: u32,
        ids_out: &mut [i64],
        dists_out: &mut [f32],
    ) -> hanns::api::Result<usize> {
        ids_out[0] = list_id as i64;
        dists_out[0] = list_id as f32;
        Ok(1)
    }
}

struct DummyQuantizer;

impl QuantizerModel for DummyQuantizer {
    type QueryState = f32;

    fn dim(&self) -> usize {
        4
    }

    fn code_size(&self) -> usize {
        2
    }

    fn precompute_query(&self, query: &[f32]) -> Self::QueryState {
        query.iter().sum()
    }

    fn score_code(&self, state: &Self::QueryState, code: &[u8]) -> f32 {
        *state + code.iter().map(|v| *v as f32).sum::<f32>()
    }
}

struct DummyCodes(Vec<u8>);

impl EncodedVectorStore for DummyCodes {
    fn code(&self, row: usize) -> &[u8] {
        let start = row * 2;
        &self.0[start..start + 2]
    }
}

struct DummyRerank;

impl RerankStore for DummyRerank {
    fn raw_vector(&self, row: usize) -> Option<&[f32]> {
        const VECTORS: [[f32; 2]; 2] = [[0.0, 1.0], [1.0, 0.0]];
        VECTORS.get(row).map(|v| v.as_slice())
    }
}

struct DummyNodeReader;

impl NodeReader for DummyNodeReader {
    fn read_node(&self, node_id: u32) -> hanns::api::Result<NodeRecord<'_>> {
        Ok(NodeRecord {
            node_id,
            neighbor_ids: &[1, 2],
            code: Some(&[7, 8]),
            raw_vector: None,
        })
    }
}

#[test]
fn ivf_kernel_traits_express_partition_selection_and_list_scan() {
    let selected = DummyPartitionSelector.select_partitions(&[0.0, 1.0], 1);
    assert_eq!(selected.ids, vec![0]);

    let mut ids = [-1_i64; 1];
    let mut dists = [f32::INFINITY; 1];
    let n = DummyListScanner
        .scan_list(&[0.0, 1.0], 3, &mut ids, &mut dists)
        .unwrap();
    assert_eq!(n, 1);
    assert_eq!(ids[0], 3);
}

#[test]
fn quantization_traits_separate_model_codes_and_rerank_store() {
    let quantizer = DummyQuantizer;
    let state = quantizer.precompute_query(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(quantizer.score_code(&state, &[5, 6]), 21.0);

    let codes = DummyCodes(vec![1, 2, 3, 4]);
    assert_eq!(codes.code(1), &[3, 4]);
    assert_eq!(DummyRerank.raw_vector(0).unwrap(), &[0.0, 1.0]);
}

#[test]
fn diskann_traits_express_page_node_reads() {
    let cfg = DiskGraphRuntimeConfig {
        beam_width: 8,
        search_list_size: 64,
    };
    assert_eq!(cfg.beam_width, 8);

    let node = DummyNodeReader.read_node(42).unwrap();
    assert_eq!(node.node_id, 42);
    assert_eq!(node.neighbor_ids, &[1, 2]);
    assert_eq!(node.code.unwrap(), &[7, 8]);
}
