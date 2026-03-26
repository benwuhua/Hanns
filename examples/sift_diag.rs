use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use knowhere_rs::api::MetricType;
use knowhere_rs::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};

const BASE_PATH: &str = "/data/work/datasets/sift-1m/base.fbin";
const QUERY_PATH: &str = "/data/work/datasets/sift-1m/query.fbin";
const GT_PATH: &str = "/data/work/datasets/sift-1m/gt.ibin";
const N_BASE: usize = 10_000;
const N_QUERY: usize = 100;
const TOP_K: usize = 10;

fn read_fbin_partial(path: &str, n: usize) -> (usize, Vec<f32>) {
    let mut file = File::open(path).expect("open fbin");
    let mut header = [0u8; 8];
    file.read_exact(&mut header).expect("read fbin header");
    let total = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let take = n.min(total);
    let mut raw = vec![0u8; take * dim * 4];
    file.read_exact(&mut raw).expect("read fbin payload");
    let data = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (dim, data)
}

fn read_ibin_partial(path: &str, n: usize) -> (usize, Vec<i32>) {
    let mut file = File::open(path).expect("open ibin");
    let mut header = [0u8; 8];
    file.read_exact(&mut header).expect("read ibin header");
    let total = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let gt_k = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let take = n.min(total);
    let mut raw = vec![0u8; take * gt_k * 4];
    file.read_exact(&mut raw).expect("read ibin payload");
    let data = raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (gt_k, data)
}

fn read_fbin(path: &str) -> (usize, usize, Vec<f32>) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .unwrap_or_else(|e| panic!("failed to read fbin header {path}: {e}"));
    let n_vecs = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; n_vecs * dim * 4];
    file.read_exact(&mut raw)
        .unwrap_or_else(|e| panic!("failed to read fbin payload {path}: {e}"));
    let data = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (n_vecs, dim, data)
}

fn read_ibin(path: &str) -> (usize, usize, Vec<u32>) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("failed to open {path}: {e}"));
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .unwrap_or_else(|e| panic!("failed to read ibin header {path}: {e}"));
    let n_vecs = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; n_vecs * k * 4];
    file.read_exact(&mut raw)
        .unwrap_or_else(|e| panic!("failed to read ibin payload {path}: {e}"));
    let data = raw
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    (n_vecs, k, data)
}

fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn brute_force_topk(base: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(i64, f32)> {
    let mut scored: Vec<(i64, f32)> = base
        .chunks_exact(dim)
        .enumerate()
        .map(|(idx, vector)| (idx as i64, l2_distance_sq(query, vector)))
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored
}

fn recall_at_k(results: &[i64], gt: &[i64], k: usize) -> f64 {
    let hits = results
        .iter()
        .take(k)
        .filter(|id| gt[..k].contains(id))
        .count();
    hits as f64 / k as f64
}

fn compute_recall_at_k(
    results: &[i64],
    gt: &[u32],
    n_queries: usize,
    k: usize,
    gt_k: usize,
) -> f64 {
    if n_queries == 0 || k == 0 || gt_k == 0 {
        return 0.0;
    }

    let mut total = 0usize;
    let mut hits = 0usize;
    for qi in 0..n_queries {
        let result_row = &results[qi * k..(qi + 1) * k];
        let gt_row = &gt[qi * gt_k..(qi + 1) * gt_k];
        for &gt_id in gt_row.iter().take(k) {
            total += 1;
            if result_row.iter().any(|&rid| rid >= 0 && rid as u32 == gt_id) {
                hits += 1;
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

fn sift10k_quick_diag() {
    if !Path::new(BASE_PATH).exists() || !Path::new(QUERY_PATH).exists() || !Path::new(GT_PATH).exists()
    {
        println!("SKIPPED: SIFT-1M data not found under /data/work/datasets/sift-1m/");
        return;
    }

    let data_start = Instant::now();
    let (base_dim, base) = read_fbin_partial(BASE_PATH, N_BASE);
    let (query_dim, queries) = read_fbin_partial(QUERY_PATH, N_QUERY);
    let (gt_k, gt_file) = read_ibin_partial(GT_PATH, N_QUERY);
    let load_secs = data_start.elapsed().as_secs_f64();

    assert_eq!(base_dim, query_dim, "base/query dim mismatch");
    let n_base = base.len() / base_dim;
    let n_query = queries.len() / query_dim;

    println!("=== SIFT-1M Quick Diagnostic ===");
    println!(
        "loaded base={} query={} dim={} gt_k={} in {:.2}s",
        n_base, n_query, base_dim, gt_k, load_secs
    );

    let cfg = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 512,
        max_degree: 48,
        random_init_edges: 8,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(cfg, MetricType::L2, base_dim).expect("new PQFlashIndex");

    let build_start = Instant::now();
    index.train(&base).expect("train");
    index.add(&base).expect("add");
    let build_secs = build_start.elapsed().as_secs_f64();
    println!("build: {:.2}s", build_secs);

    let mut recall_sum = 0.0;
    let mut q0_result_ids = Vec::new();
    let mut q0_result_dists = Vec::new();
    let mut q0_gt_ids = Vec::new();
    let mut q0_gt_dists = Vec::new();

    let search_start = Instant::now();
    for q_idx in 0..n_query {
        let q = &queries[q_idx * base_dim..(q_idx + 1) * base_dim];
        let result = index.search(q, TOP_K).expect("search");
        let exact = brute_force_topk(&base, base_dim, q, TOP_K);
        let exact_ids: Vec<i64> = exact.iter().map(|(id, _)| *id).collect();
        recall_sum += recall_at_k(&result.ids, &exact_ids, TOP_K);

        if q_idx == 0 {
            q0_result_ids = result.ids.clone();
            q0_result_dists = result.distances.clone();
            q0_gt_ids = exact_ids;
            q0_gt_dists = exact.iter().map(|(_, dist)| *dist).collect();
        }
    }
    let search_secs = search_start.elapsed().as_secs_f64();
    let recall = recall_sum / n_query as f64;
    let qps = n_query as f64 / search_secs.max(f64::EPSILON);

    println!("NoPQ recall@10 on first {} queries: {:.4}", n_query, recall);
    println!("NoPQ QPS: {:.0}", qps);
    println!("query[0] result IDs: {:?}", q0_result_ids);
    println!(
        "query[0] result dists: {:?}",
        q0_result_dists
            .iter()
            .map(|d| format!("{:.1}", d))
            .collect::<Vec<_>>()
    );
    println!("query[0] brute-force top-10 IDs over first {} base vectors: {:?}", n_base, q0_gt_ids);
    println!(
        "query[0] brute-force top-10 dists: {:?}",
        q0_gt_dists
            .iter()
            .map(|d| format!("{:.1}", d))
            .collect::<Vec<_>>()
    );

    let file_gt_row = &gt_file[..TOP_K.min(gt_k)];
    println!("query[0] gt.ibin top-10 IDs (full 1M reference): {:?}", file_gt_row);
}

fn sift1m_quick_recall() {
    if !Path::new(BASE_PATH).exists() || !Path::new(QUERY_PATH).exists() || !Path::new(GT_PATH).exists()
    {
        println!("SKIPPED: SIFT-1M data not found under /data/work/datasets/sift-1m/");
        return;
    }

    println!("=== SIFT-1M Quick Recall ===");

    let load_start = Instant::now();
    let (base_n, base_dim, base) = read_fbin(BASE_PATH);
    let (query_n, query_dim, queries) = read_fbin(QUERY_PATH);
    let (gt_n, gt_k, gt) = read_ibin(GT_PATH);
    let load_secs = load_start.elapsed().as_secs_f64();

    assert_eq!(base_dim, query_dim, "base/query dim mismatch");
    assert_eq!(query_n, gt_n, "query/gt count mismatch");
    println!(
        "loaded base={} query={} dim={} gt_k={} in {:.2}s",
        base_n, query_n, base_dim, gt_k, load_secs
    );

    let config = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 128,
        cache_all_on_load: true,
        random_init_edges: 0,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(config, MetricType::L2, base_dim).expect("new PQFlashIndex");

    let build_start = Instant::now();
    index.train(&base).expect("train");
    index.add(&base).expect("add");
    let build_secs = build_start.elapsed().as_secs_f64();

    let search_start = Instant::now();
    let result = index.search_batch(&queries, TOP_K).expect("search_batch");
    let search_secs = search_start.elapsed().as_secs_f64().max(f64::EPSILON);
    let qps = query_n as f64 / search_secs;
    let recall = compute_recall_at_k(&result.ids, &gt, query_n, TOP_K, gt_k);

    println!("build: {:.2}s", build_secs);
    println!("QPS: {:.0}", qps);
    println!("recall@10: {:.4}", recall);
}

fn sift1m_disk_mode() {
    if !Path::new(BASE_PATH).exists() || !Path::new(QUERY_PATH).exists() || !Path::new(GT_PATH).exists()
    {
        println!("SKIPPED: SIFT-1M data not found under /data/work/datasets/sift-1m/");
        return;
    }

    println!("=== SIFT-1M Disk Mode ===");

    let load_start = Instant::now();
    let (base_n, base_dim, base) = read_fbin(BASE_PATH);
    let (query_n, query_dim, queries) = read_fbin(QUERY_PATH);
    let (gt_n, gt_k, gt) = read_ibin(GT_PATH);
    let load_secs = load_start.elapsed().as_secs_f64();

    assert_eq!(base_dim, query_dim, "base/query dim mismatch");
    assert_eq!(query_n, gt_n, "query/gt count mismatch");
    println!(
        "loaded base={} query={} dim={} gt_k={} in {:.2}s",
        base_n, query_n, base_dim, gt_k, load_secs
    );

    let config = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 128,
        cache_all_on_load: false,
        random_init_edges: 0,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(config, MetricType::L2, base_dim).expect("new PQFlashIndex");

    let build_start = Instant::now();
    index.train(&base).expect("train");
    index.add(&base).expect("add");
    let build_secs = build_start.elapsed().as_secs_f64();
    println!("build: {:.2}s", build_secs);

    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let save_start = Instant::now();
    index.save(tmp_dir.path()).expect("save");
    let save_secs = save_start.elapsed().as_secs_f64();
    println!("save: {:.2}s", save_secs);

    drop(index);

    let load_start = Instant::now();
    let disk_index = PQFlashIndex::load(tmp_dir.path()).expect("load");
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("load (disk mode): {:.2}s", load_secs);

    let search_start = Instant::now();
    let result = disk_index.search_batch(&queries, TOP_K).expect("search_batch");
    let search_secs = search_start.elapsed().as_secs_f64().max(f64::EPSILON);
    let qps = query_n as f64 / search_secs;
    let recall = compute_recall_at_k(&result.ids, &gt, query_n, TOP_K, gt_k);

    println!("disk_mode QPS: {:.0}", qps);
    println!("disk_mode recall@10: {:.4}", recall);
    println!("(note: 首次搜索 mmap 冷启动，实际 SSD IO 场景)");
}

fn sift1m_disk_pq_uring() {
    if !Path::new(BASE_PATH).exists() || !Path::new(QUERY_PATH).exists() || !Path::new(GT_PATH).exists()
    {
        println!("SKIPPED: SIFT-1M data not found under /data/work/datasets/sift-1m/");
        return;
    }

    println!("=== SIFT-1M Disk PQ io_uring Mode ===");
    println!("index dir: /data/work/tmp/sift_pq_bench");

    let load_start = Instant::now();
    let (base_n, base_dim, base) = read_fbin(BASE_PATH);
    let (query_n, query_dim, queries) = read_fbin(QUERY_PATH);
    let (gt_n, gt_k, gt) = read_ibin(GT_PATH);
    let load_secs = load_start.elapsed().as_secs_f64();

    assert_eq!(base_dim, query_dim, "base/query dim mismatch");
    assert_eq!(query_n, gt_n, "query/gt count mismatch");
    println!(
        "loaded base={} query={} dim={} gt_k={} in {:.2}s",
        base_n, query_n, base_dim, gt_k, load_secs
    );

    let config = AisaqConfig {
        disk_pq_dims: 32,
        search_list_size: 128,
        cache_all_on_load: false,
        random_init_edges: 0,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(config, MetricType::L2, base_dim).expect("new PQFlashIndex");

    let build_start = Instant::now();
    index.train(&base).expect("train");
    index.add(&base).expect("add");
    let build_secs = build_start.elapsed().as_secs_f64();
    println!("build: {:.2}s", build_secs);

    let data_tmp = std::path::Path::new("/data/work/tmp/sift_pq_bench");
    std::fs::create_dir_all(data_tmp).expect("create bench dir");
    let tmp_dir_owned = data_tmp.to_path_buf();
    let save_start = Instant::now();
    index.save(&tmp_dir_owned).expect("save");
    let save_secs = save_start.elapsed().as_secs_f64();
    println!("save: {:.2}s", save_secs);

    drop(index);

    for group_size in [8usize, 16, 32] {
        let load_start = Instant::now();
        let mut disk_index = PQFlashIndex::load(&tmp_dir_owned).expect("load");
        disk_index.set_uring_group_size(group_size);
        let load_secs = load_start.elapsed().as_secs_f64();

        let search_start = Instant::now();
        let result = disk_index.search_batch(&queries, TOP_K).expect("search_batch");
        let search_secs = search_start.elapsed().as_secs_f64().max(f64::EPSILON);
        let qps = query_n as f64 / search_secs;
        let recall = compute_recall_at_k(&result.ids, &gt, query_n, TOP_K, gt_k);

        println!("group_size={} load (disk_pq_uring mode): {:.2}s", group_size, load_secs);
        println!("group_size={} disk_pq_uring QPS: {:.0}", group_size, qps);
        println!(
            "group_size={} disk_pq_uring recall@10: {:.4}",
            group_size, recall
        );
    }
    println!("(note: 需要 Linux + --features async-io 才会走 grouped io_uring storage 路径)");
}

fn sift1m_warm_disk_mode() {
    if !Path::new(BASE_PATH).exists() || !Path::new(QUERY_PATH).exists() || !Path::new(GT_PATH).exists()
    {
        println!("SKIPPED: SIFT-1M data not found under /data/work/datasets/sift-1m/");
        return;
    }

    println!("=== SIFT-1M Warm Disk Mode ===");

    let load_start = Instant::now();
    let (base_n, base_dim, base) = read_fbin(BASE_PATH);
    let (query_n, query_dim, queries) = read_fbin(QUERY_PATH);
    let (gt_n, gt_k, gt) = read_ibin(GT_PATH);
    let load_secs = load_start.elapsed().as_secs_f64();

    assert_eq!(base_dim, query_dim, "base/query dim mismatch");
    assert_eq!(query_n, gt_n, "query/gt count mismatch");
    println!(
        "loaded base={} query={} dim={} gt_k={} in {:.2}s",
        base_n, query_n, base_dim, gt_k, load_secs
    );

    let config = AisaqConfig {
        disk_pq_dims: 0,
        search_list_size: 128,
        cache_all_on_load: false,
        random_init_edges: 0,
        ..AisaqConfig::default()
    };
    let mut index = PQFlashIndex::new(config, MetricType::L2, base_dim).expect("new PQFlashIndex");

    let build_start = Instant::now();
    index.train(&base).expect("train");
    index.add(&base).expect("add");
    let build_secs = build_start.elapsed().as_secs_f64();
    println!("build: {:.2}s", build_secs);

    let tmp_dir = tempfile::tempdir().expect("tempdir");
    let save_start = Instant::now();
    index.save(tmp_dir.path()).expect("save");
    let save_secs = save_start.elapsed().as_secs_f64();
    println!("save: {:.2}s", save_secs);

    drop(index);

    let load_start = Instant::now();
    let mut disk_index = PQFlashIndex::load(tmp_dir.path()).expect("load");
    let load_secs = load_start.elapsed().as_secs_f64();
    println!("load (disk mode): {:.2}s", load_secs);

    let sample_queries: Vec<Vec<f32>> = queries
        .chunks_exact(base_dim)
        .take(100)
        .map(|query| query.to_vec())
        .collect();
    let warm_start = Instant::now();
    let cache_nodes = disk_index.generate_cache_list_from_sample_queries(&sample_queries, 4096);
    let warm_secs = warm_start.elapsed().as_secs_f64();
    println!("warm_time: {:.2}s", warm_secs);
    println!("cache_nodes: {}", cache_nodes);

    let search_start = Instant::now();
    let result = disk_index.search_batch(&queries, TOP_K).expect("search_batch");
    let search_secs = search_start.elapsed().as_secs_f64().max(f64::EPSILON);
    let qps = query_n as f64 / search_secs;
    let recall = compute_recall_at_k(&result.ids, &gt, query_n, TOP_K, gt_k);

    println!("warm_disk_mode QPS: {:.0}", qps);
    println!("warm_disk_mode recall@10: {:.4}", recall);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("1m") {
        sift1m_quick_recall();
    } else if args.get(1).map(|s| s.as_str()) == Some("1m_disk") {
        sift1m_disk_mode();
    } else if args.get(1).map(|s| s.as_str()) == Some("1m_disk_pq") {
        sift1m_disk_pq_uring();
    } else if args.get(1).map(|s| s.as_str()) == Some("1m_warm_disk") {
        sift1m_warm_disk_mode();
    } else {
        sift10k_quick_diag();
    }
}
