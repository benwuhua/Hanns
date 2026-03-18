use knowhere_rs::api::MetricType;
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::fs;

const N: usize = 5_000;
const DIM: usize = 32;
const TOP_K: usize = 10;
const DELETE_START: i64 = 1000;
const DELETE_END: i64 = 1100; // exclusive, 100 IDs

#[inline]
fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn brute_force_topk_filtered(
    vectors: &[f32],
    ids: &[i64],
    query: &[f32],
    dim: usize,
    top_k: usize,
    deleted: &HashSet<i64>,
) -> Vec<i64> {
    let mut scored = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        if deleted.contains(&id) {
            continue;
        }
        let v = &vectors[i * dim..(i + 1) * dim];
        scored.push((id, l2_sqr(query, v)));
    }
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}

fn has_deleted(ids: &[i64]) -> bool {
    ids.iter().any(|&id| (DELETE_START..DELETE_END).contains(&id))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..N * DIM).map(|_| rng.r#gen::<f32>()).collect();
    let external_ids: Vec<i64> = (1000..6000).collect();
    let query = &vectors[200 * DIM..201 * DIM];

    let config = AisaqConfig {
        max_degree: 16,
        search_list_size: 64,
        disk_pq_dims: 0,
        ..AisaqConfig::default()
    };

    let mut index = PQFlashIndex::new(config.clone(), MetricType::L2, DIM)?;
    index.train(&vectors)?;
    index.add_with_ids(&vectors, Some(&external_ids))?;

    for id in DELETE_START..DELETE_END {
        let ok = index.soft_delete(id);
        assert!(ok, "soft_delete failed for external id {}", id);
    }

    let res_before = index.search(query, TOP_K)?;
    assert!(
        !has_deleted(&res_before.ids),
        "deleted IDs appeared before consolidate: {:?}",
        res_before.ids
    );

    let removed = index.consolidate();
    let node_count = index.len();
    println!(
        "consolidate removed {} nodes, node_count={}",
        removed, node_count
    );
    assert_eq!(removed, 100, "expected to remove 100 nodes");
    assert_eq!(node_count, 4_900, "expected node_count=4900 after consolidate");

    let res_after = index.search(query, TOP_K)?;
    assert!(
        !has_deleted(&res_after.ids),
        "deleted IDs appeared after consolidate: {:?}",
        res_after.ids
    );

    let save_path = "/tmp/aisaq_delete_test.bin";
    let _ = fs::remove_dir_all(save_path);
    fs::create_dir_all(save_path)?;
    index.save(save_path)?;

    let loaded = PQFlashIndex::load(save_path)?;
    let res_loaded = loaded.search(query, TOP_K)?;
    assert!(
        !has_deleted(&res_loaded.ids),
        "deleted IDs appeared after load: {:?}",
        res_loaded.ids
    );

    let deleted_set: HashSet<i64> = (DELETE_START..DELETE_END).collect();
    let gt = brute_force_topk_filtered(
        &vectors,
        &external_ids,
        query,
        DIM,
        TOP_K,
        &deleted_set,
    );
    let ret: HashSet<i64> = res_loaded.ids.iter().copied().collect();
    let hit = gt.iter().filter(|id| ret.contains(id)).count();
    let recall = hit as f64 / TOP_K as f64;
    println!("recall@10 vs filtered brute-force: {:.3}", recall);
    assert!(recall > 0.9, "recall too low: {:.3}", recall);

    Ok(())
}
