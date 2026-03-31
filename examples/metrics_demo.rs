fn main() {
    #[cfg(feature = "metrics")]
    {
        use knowhere_rs::api::{
            DataType, IndexConfig, IndexParams, IndexType, MetricType, SearchRequest,
        };
        use knowhere_rs::faiss::HnswIndex;
        use knowhere_rs::metrics::{gather_metrics, init_metrics};

        let _ = init_metrics();

        let params = IndexParams::hnsw(100, 50, 0.5);
        let config = IndexConfig {
            index_type: IndexType::Hnsw,
            metric_type: MetricType::L2,
            data_type: DataType::Float,
            dim: 32,
            params,
        };

        let mut index = HnswIndex::new(&config).expect("create HNSW");

        let n = 500usize;
        let dim = 32usize;
        let data: Vec<f32> = (0..n * dim).map(|i| (i % 100) as f32).collect();

        index.train(&data).expect("train");
        index.add(&data, None).expect("add");

        let req = SearchRequest {
            top_k: 10,
            nprobe: 50,
            filter: None,
            params: None,
            radius: None,
        };
        let _ = index.search(&data[..dim], &req).expect("search");

        let output = gather_metrics();
        println!("{}", output);
        assert!(output.contains("knowhere_search"), "metrics missing");
        println!("metrics_demo: OK");
    }

    #[cfg(not(feature = "metrics"))]
    {
        println!("metrics feature not enabled");
    }
}
