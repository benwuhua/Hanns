use clap::Parser;
use knowhere_rs::benchmark::{generate_recall_gated_baseline_report, DEFAULT_OUTPUT_PATH};

#[derive(Parser, Debug)]
#[command(name = "generate-recall-gated-baseline")]
#[command(about = "Generate recall-gated benchmark baseline report")]
struct Args {
    /// Output JSON path
    #[arg(long, default_value = DEFAULT_OUTPUT_PATH)]
    output: String,
}

fn main() {
    let args = Args::parse();

    match generate_recall_gated_baseline_report(&args.output) {
        Ok(report) => {
            println!(
                "baseline report generated: {} (rows={})",
                args.output,
                report.rows.len()
            );
        }
        Err(err) => {
            eprintln!("failed to generate baseline report: {err}");
            std::process::exit(1);
        }
    }
}
