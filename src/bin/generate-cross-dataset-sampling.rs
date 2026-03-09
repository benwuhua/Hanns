use clap::Parser;
use knowhere_rs::benchmark::{generate_cross_dataset_artifact, CROSS_DATASET_OUTPUT_PATH};

#[derive(Parser, Debug)]
#[command(name = "generate-cross-dataset-sampling")]
#[command(about = "Generate BENCH-P2-003 cross-dataset sampling artifact")]
struct Args {
    /// Output JSON path
    #[arg(long, default_value = CROSS_DATASET_OUTPUT_PATH)]
    output: String,
}

fn main() {
    let args = Args::parse();

    match generate_cross_dataset_artifact(&args.output) {
        Ok(artifact) => {
            println!(
                "cross-dataset artifact generated: {} (rows={})",
                args.output,
                artifact.rows.len()
            );
        }
        Err(err) => {
            eprintln!("failed to generate cross-dataset artifact: {err}");
            std::process::exit(1);
        }
    }
}
