use clap::Parser;
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, IndexType, MetricType};
use knowhere_rs::faiss::HnswIndex;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    output: PathBuf,
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 2,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(2),
            ef_construction: Some(8),
            ef_search: Some(8),
            num_threads: Some(1),
            ..Default::default()
        },
    };

    let vectors = vec![
        0.0, 0.0, 100.0, 0.0, 101.0, 0.0, 102.0, 0.0, 103.0, 1.0, 104.0, 1.0,
    ];

    let mut index = HnswIndex::new(&config)?;
    index.train(&vectors)?;
    index.add(&vectors, None)?;

    let report = index.graph_diagnosis_report();
    let output = serde_json::to_string_pretty(&report)?;

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(args.output, output)?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("hnsw_graph_diagnose failed: {err}");
        std::process::exit(1);
    }
}
