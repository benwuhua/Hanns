use std::collections::HashSet;
use std::fs;
use std::path::Path;

use knowhere_rs::api::{IndexConfig, IndexParams, IndexType, MetricType, SearchRequest};
use knowhere_rs::faiss::{DiskAnnIndex, MemIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::json;

const DIMS: [usize; 6] = [4, 16, 32, 64, 96, 128];
const N: usize = 2_000;
const NQ: usize = 100;
const TOP_K: usize = 10;
const MAX_DEGREE: usize = 32;
const SEARCH_LIST_SIZE: usize = 64;
const BEAMWIDTH: usize = 8;
const SEED: u64 = 42;
const OUT_JSON: &str = "/tmp/diskann_dim_diag.json";
const STATUS_FILE: &str = "/tmp/codex_status_b.txt";

fn gen_vectors(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.r#gen::<f32>()).collect()
}

fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

fn compute_recall_at_10(gt_top10: &[Vec<i64>], pred_top10: &[Vec<i64>]) -> f64 {
    let mut hits = 0usize;
    for (gt, pred) in gt_top10.iter().zip(pred_top10.iter()) {
        let gt_set: HashSet<i64> = gt.iter().copied().collect();
        for id in pred.iter().take(TOP_K) {
            if gt_set.contains(id) {
                hits += 1;
            }
        }
    }
    hits as f64 / (gt_top10.len() * TOP_K) as f64
}

fn find_medoid(vectors: &[f32], dim: usize) -> usize {
    let n = vectors.len() / dim;
    let mut centroid = vec![0.0f32; dim];
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        for d in 0..dim {
            centroid[d] += v[d];
        }
    }
    for c in &mut centroid {
        *c /= n as f32;
    }

    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        let d = l2_sqr(&centroid, v);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

fn parse_saved_neighbor_degrees(path: &Path) -> Result<(usize, usize, usize, Vec<u32>), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {} failed: {e}", path.display()))?;
    if bytes.len() < 36 {
        return Err("saved index too short".to_string());
    }
    if &bytes[0..4] != b"DANN" {
        return Err("invalid magic".to_string());
    }

    let read_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let read_u64 = |off: usize| -> u64 {
        u64::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ])
    };

    let dim = read_u32(8) as usize;
    let count = read_u64(12) as usize;
    let max_degree = read_u32(20) as usize;
    let search_list_size = read_u32(24) as usize;
    let beamwidth = read_u32(28) as usize;

    let vectors_bytes = count
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| "overflow vectors_bytes".to_string())?;
    let ids_bytes = count
        .checked_mul(8)
        .ok_or_else(|| "overflow ids_bytes".to_string())?;
    let degree_off = 32usize
        .checked_add(vectors_bytes)
        .and_then(|v| v.checked_add(ids_bytes))
        .ok_or_else(|| "overflow degree offset".to_string())?;
    let degree_end = degree_off
        .checked_add(
            count
                .checked_mul(4)
                .ok_or_else(|| "overflow degrees len".to_string())?,
        )
        .ok_or_else(|| "overflow degree_end".to_string())?;
    if degree_end > bytes.len() {
        return Err(format!(
            "degree section out of bounds: degree_end={} total={}",
            degree_end,
            bytes.len()
        ));
    }

    let mut degrees = Vec::with_capacity(count);
    for i in 0..count {
        let off = degree_off + i * 4;
        degrees.push(read_u32(off));
    }

    let _ = max_degree;
    Ok((search_list_size, beamwidth, count, degrees))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DiskANN dim diagnostic ===");
    println!(
        "n={} queries={} top_k={} max_degree={} search_list_size={} beamwidth={} seed={}",
        N, NQ, TOP_K, MAX_DEGREE, SEARCH_LIST_SIZE, BEAMWIDTH, SEED
    );

    let mut result_rows = Vec::new();
    let mut status_rows = Vec::new();
    let mut root_cause_lines = Vec::new();

    for &dim in &DIMS {
        let base = gen_vectors(SEED, N, dim);
        let queries = gen_vectors(SEED.wrapping_add(1), NQ, dim);

        let flat_cfg = IndexConfig::new(IndexType::Flat, MetricType::L2, dim);
        let mut gt_index = MemIndex::new(&flat_cfg)?;
        gt_index.add(&base, None)?;

        let params = IndexParams {
            max_degree: Some(MAX_DEGREE),
            search_list_size: Some(SEARCH_LIST_SIZE),
            beamwidth: Some(BEAMWIDTH),
            ..Default::default()
        };
        let disk_cfg = IndexConfig {
            index_type: IndexType::DiskAnn,
            metric_type: MetricType::L2,
            data_type: knowhere_rs::api::DataType::Float,
            dim,
            params,
        };
        let mut disk = DiskAnnIndex::new(&disk_cfg)?;
        disk.train(&base)?;
        // Keep the same calling pattern as the reported scenario (train + add).
        let _ = disk.add(&base, None)?;

        let stats = disk.get_stats();
        let nonzero_degree_nodes = if stats.avg_degree > 0.0 {
            // More accurate count is parsed from saved file below.
            None
        } else {
            Some(0usize)
        };

        let req = SearchRequest {
            top_k: TOP_K,
            ..Default::default()
        };

        let mut gt_top10 = Vec::with_capacity(NQ);
        let mut disk_top10 = Vec::with_capacity(NQ);
        for qi in 0..NQ {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let gt = gt_index.search(q, &req)?;
            let pred = disk.search(q, &req)?;
            gt_top10.push(gt.ids.into_iter().take(TOP_K).collect());
            disk_top10.push(pred.ids.into_iter().take(TOP_K).collect());
        }
        let recall = compute_recall_at_10(&gt_top10, &disk_top10);
        println!(
            "dim={:>3} recall@10={:.4} ntotal={} avg_degree={:.2} min_degree={} max_degree={}",
            dim,
            recall,
            disk.ntotal(),
            stats.avg_degree,
            stats.min_degree,
            stats.max_degree
        );

        let mut extra = json!({});
        let diag_path = format!("/tmp/diskann_dim{}_diag.idx", dim);
        let save_path = Path::new(&diag_path);
        disk.save(save_path)?;
        let (saved_l, saved_bw, saved_count, saved_degrees) =
            parse_saved_neighbor_degrees(save_path).map_err(std::io::Error::other)?;
        let saved_nonzero = saved_degrees.iter().filter(|&&d| d > 0).count();

        if dim == 64 {
            let mut full_vectors = Vec::with_capacity(base.len() * 2);
            full_vectors.extend_from_slice(&base);
            full_vectors.extend_from_slice(&base);
            let entry_idx = find_medoid(&full_vectors, dim);
            let query0 = &queries[0..dim];
            let entry_vec = &full_vectors[entry_idx * dim..(entry_idx + 1) * dim];
            let entry_dist_sqr = l2_sqr(query0, entry_vec);
            let entry_degree = saved_degrees.get(entry_idx).copied().unwrap_or(0) as usize;
            let first_step_expand = entry_degree.min(saved_bw);
            let effective_l = req.nprobe.max(saved_l / 2);

            println!();
            println!("--- dim=64 focused diagnostics ---");
            println!(
                "entry_point_inferred_id={} entry_dist={:.6}",
                entry_idx,
                entry_dist_sqr.sqrt()
            );
            println!(
                "first_step: entry_degree={} beamwidth={} => first_step_candidates={}",
                entry_degree, saved_bw, first_step_expand
            );
            println!(
                "search effective L = max(req.nprobe={}, search_list_size/2={}) = {}",
                req.nprobe,
                saved_l / 2,
                effective_l
            );
            println!(
                "graph degree check: nonzero_nodes={} / {}",
                saved_nonzero, saved_count
            );
            println!("----------------------------------");
            println!();

            extra = json!({
                "entry_point_inferred_id": entry_idx,
                "entry_distance": entry_dist_sqr.sqrt(),
                "entry_degree": entry_degree,
                "first_step_candidates": first_step_expand,
                "effective_search_l": effective_l,
                "graph_nonzero_degree_nodes": saved_nonzero,
                "graph_total_nodes": saved_count
            });

            if effective_l < SEARCH_LIST_SIZE {
                root_cause_lines.push(format!(
                    "dim=64 uses default req.nprobe=0 => effective L={} (search_list_size/2), see src/faiss/diskann.rs:2020",
                    effective_l
                ));
            }
            if disk.ntotal() > N {
                root_cause_lines.push(format!(
                    "train() already builds {} nodes; extra add() increases ntotal to {} and changes id space, likely hurting recall-vs-GT id matching",
                    N,
                    disk.ntotal()
                ));
            }
        }

        result_rows.push(json!({
            "dim": dim,
            "recall": recall,
            "ntotal": disk.ntotal(),
            "avg_degree": stats.avg_degree,
            "min_degree": stats.min_degree,
            "max_degree": stats.max_degree,
            "graph_nonzero_degree_nodes": nonzero_degree_nodes.unwrap_or(saved_nonzero),
            "graph_total_nodes": saved_count,
            "search_list_size": saved_l,
            "beamwidth": saved_bw,
            "search_effective_l_default": req.nprobe.max(saved_l / 2),
            "diag": extra
        }));
        status_rows.push(format!("dim={}: recall@10={:.4}", dim, recall));
    }

    let output = json!({
        "config": {
            "n": N,
            "nq": NQ,
            "top_k": TOP_K,
            "max_degree": MAX_DEGREE,
            "search_list_size": SEARCH_LIST_SIZE,
            "beamwidth": BEAMWIDTH,
            "seed": SEED
        },
        "results": result_rows
    });
    fs::write(OUT_JSON, serde_json::to_string_pretty(&output)?)?;
    println!("saved {}", OUT_JSON);

    let mut status = String::new();
    status.push_str("DONE: DiskANN dim diagnostic complete. ");
    status.push_str(&status_rows.join("; "));
    if !root_cause_lines.is_empty() {
        status.push_str(". root-cause clues: ");
        status.push_str(&root_cause_lines.join(" | "));
    }
    status.push('\n');
    fs::write(STATUS_FILE, status)?;
    println!("saved {}", STATUS_FILE);

    Ok(())
}
