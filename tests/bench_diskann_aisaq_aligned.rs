#![cfg(feature = "long-tests")]
//! Constrained DiskANN/AISAQ aligned evidence lane for Hanns-vs-Zilliz.
//!
//! This benchmark intentionally emits `native_comparable=false`: Hanns'
//! `PQFlashIndex` is still an in-memory / simplified AISAQ skeleton, not the
//! native SSD DiskANN pipeline used by official Knowhere.  The lane exists to
//! collect top_k=100 / Recall@100 evidence without allowing a leadership claim.

use hanns::api::MetricType;
use hanns::benchmark::average_recall_at_k;
use hanns::dataset::load_sift1m_complete;
use hanns::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TOP_K: usize = 100;

#[derive(Debug, Clone)]
struct AlignedConfig {
    name: String,
    max_degree: usize,
    search_list_size: usize,
    beamwidth: usize,
    num_entry_points: usize,
}

impl AlignedConfig {
    fn to_aisaq_config(&self) -> AisaqConfig {
        AisaqConfig {
            max_degree: self.max_degree,
            search_list_size: self.search_list_size,
            beamwidth: self.beamwidth,
            num_entry_points: self.num_entry_points,
            ..AisaqConfig::default()
        }
    }
}

fn default_configs() -> Vec<AlignedConfig> {
    vec![
        AlignedConfig {
            name: "R48-L100-B8-EP1".to_string(),
            max_degree: 48,
            search_list_size: 100,
            beamwidth: 8,
            num_entry_points: 1,
        },
        AlignedConfig {
            name: "R48-L108-B8-EP1".to_string(),
            max_degree: 48,
            search_list_size: 108,
            beamwidth: 8,
            num_entry_points: 1,
        },
        AlignedConfig {
            name: "R48-L128-B8-EP1".to_string(),
            max_degree: 48,
            search_list_size: 128,
            beamwidth: 8,
            num_entry_points: 1,
        },
    ]
}

fn parse_configs() -> Vec<AlignedConfig> {
    let Ok(value) = env::var("AISAQ_ALIGNED_CONFIGS") else {
        return default_configs();
    };
    let configs = value
        .split(',')
        .filter_map(|entry| {
            let mut parts = entry.split(':').map(str::trim);
            let name = parts.next()?.to_string();
            let max_degree = parts.next()?.parse::<usize>().ok()?;
            let search_list_size = parts.next()?.parse::<usize>().ok()?;
            let beamwidth = parts.next()?.parse::<usize>().ok()?;
            let num_entry_points = parts.next()?.parse::<usize>().ok()?;
            Some(AlignedConfig {
                name,
                max_degree,
                search_list_size,
                beamwidth,
                num_entry_points,
            })
        })
        .collect::<Vec<_>>();
    if configs.is_empty() {
        default_configs()
    } else {
        configs
    }
}

fn parse_optional_usize(var: &str) -> Option<usize> {
    env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn disk_pq_dims_from_env() -> usize {
    parse_optional_usize("AISAQ_DISK_PQ_DIMS").unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
struct SearchOverrides {
    disk_pq_dims: usize,
    pq_candidate_expand_pct: usize,
    rerank_expand_pct: usize,
}

impl SearchOverrides {
    fn from_env() -> Self {
        let defaults = AisaqConfig::default();
        Self {
            disk_pq_dims: disk_pq_dims_from_env(),
            pq_candidate_expand_pct: parse_optional_usize("AISAQ_PQ_CANDIDATE_EXPAND_PCT")
                .unwrap_or(defaults.pq_candidate_expand_pct),
            rerank_expand_pct: parse_optional_usize("AISAQ_RERANK_EXPAND_PCT")
                .unwrap_or(defaults.rerank_expand_pct),
        }
    }

    fn apply_to(self, config: &mut AisaqConfig) {
        config.disk_pq_dims = self.disk_pq_dims;
        config.pq_candidate_expand_pct = self.pq_candidate_expand_pct.max(100);
        config.rerank_expand_pct = self.rerank_expand_pct.max(100);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchSurface {
    Memory,
    Mmap,
    PageCache,
}

impl SearchSurface {
    fn from_env() -> Self {
        let Ok(value) = env::var("AISAQ_SEARCH_SURFACE") else {
            return Self::Memory;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "mmap" => Self::Mmap,
            "page-cache" | "page_cache" | "paged" => Self::PageCache,
            "memory" | "" => Self::Memory,
            other => panic!("unsupported AISAQ_SEARCH_SURFACE={other}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Mmap => "mmap",
            Self::PageCache => "page_cache",
        }
    }
}

fn safe_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[test]
#[ignore = "SIFT1M remote benchmark lane; run on HannsDB-x86 for authority evidence"]
fn bench_diskann_aisaq_aligned_topk100() {
    let dataset_path = env::var("SIFT1M_PATH").unwrap_or_else(|_| "./data/sift".to_string());
    let dataset = load_sift1m_complete(&dataset_path).expect("load SIFT1M dataset");
    let nq = env::var("SIFT_NUM_QUERIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000)
        .min(dataset.num_query());
    let threads = query_threads();
    let search_surface = SearchSurface::from_env();
    let search_overrides = SearchOverrides::from_env();
    if search_surface == SearchSurface::PageCache {
        assert!(
            search_overrides.disk_pq_dims > 0,
            "AISAQ_SEARCH_SURFACE=page_cache requires AISAQ_DISK_PQ_DIMS>0; \
             NoPQ load() materializes storage into memory and cannot prove page-cache search"
        );
    }
    if let Some(expected_threads) = parse_optional_usize("AISAQ_EXPECT_THREADS") {
        assert_eq!(
            threads, expected_threads,
            "aligned AISAQ evidence lane must run with expected thread count"
        );
    }

    let base = dataset.base.vectors();
    let queries = dataset.query.vectors();
    let query_batch = &queries[..nq * dataset.dim()];
    let gt = dataset
        .ground_truth
        .iter()
        .take(nq)
        .cloned()
        .collect::<Vec<_>>();

    let run_set_id = format!(
        "diskann-aisaq-aligned-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_secs()
    );
    let mut rows = Vec::new();
    let work_root = env::var("AISAQ_ALIGNED_WORK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir().join(&run_set_id));

    for aligned in parse_configs() {
        let mut config = aligned.to_aisaq_config();
        search_overrides.apply_to(&mut config);
        let build_start = Instant::now();
        let mut index =
            PQFlashIndex::new(config, MetricType::L2, dataset.dim()).expect("create AISAQ");
        index.add(base).expect("add vectors to AISAQ");
        let build_s = build_start.elapsed().as_secs_f64();
        let mut persist_s = 0.0f64;
        let mut load_s = 0.0f64;
        let search_index = match search_surface {
            SearchSurface::Memory => index,
            SearchSurface::Mmap | SearchSurface::PageCache => {
                let surface_dir = work_root.join(format!(
                    "{}-{}",
                    search_surface.as_str(),
                    safe_path_component(&aligned.name)
                ));
                let _ = fs::remove_dir_all(&surface_dir);
                let persist_start = Instant::now();
                index.save(&surface_dir).expect("persist AISAQ index");
                persist_s = persist_start.elapsed().as_secs_f64();

                let load_start = Instant::now();
                let loaded = match search_surface {
                    SearchSurface::Mmap => {
                        PQFlashIndex::load_with_mmap(&surface_dir).expect("load mmap AISAQ index")
                    }
                    SearchSurface::PageCache => {
                        PQFlashIndex::load(&surface_dir).expect("load page-cache AISAQ index")
                    }
                    SearchSurface::Memory => unreachable!(),
                };
                load_s = load_start.elapsed().as_secs_f64();
                loaded
            }
        };

        let search_start = Instant::now();
        #[cfg(feature = "parallel")]
        let search_result = search_index
            .search_batch(query_batch, TOP_K)
            .expect("batch search AISAQ");
        #[cfg(not(feature = "parallel"))]
        let search_result = {
            let mut ids = Vec::with_capacity(nq * TOP_K);
            let mut distances = Vec::with_capacity(nq * TOP_K);
            for query in query_batch.chunks(dataset.dim()) {
                let result = search_index.search(query, TOP_K).expect("search AISAQ");
                ids.extend(result.ids);
                distances.extend(result.distances);
            }
            hanns::api::SearchResult::new(ids, distances, 0.0)
        };
        let search_s = search_start.elapsed().as_secs_f64();
        let qps = nq as f64 / search_s.max(1e-9);

        let results = search_result
            .ids
            .chunks(TOP_K)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();
        let recall_at_10 = average_recall_at_k(&results, &gt, 10);
        let recall_at_100 = average_recall_at_k(&results, &gt, TOP_K);
        let audit = search_index.scope_audit();
        match search_surface {
            SearchSurface::Memory => {}
            SearchSurface::Mmap => assert!(
                audit.uses_mmap_backed_pages,
                "AISAQ mmap evidence must prove uses_mmap_backed_pages=true"
            ),
            SearchSurface::PageCache => assert!(
                audit.has_page_cache,
                "AISAQ page_cache evidence must prove has_page_cache=true"
            ),
        }

        println!(
            "AISAQ aligned config={} surface={} disk_pq_dims={} pq_candidate_expand_pct={} rerank_expand_pct={} search_list_size={} build={build_s:.3}s persist={persist_s:.3}s load={load_s:.3}s qps={qps:.3} R@10={recall_at_10:.4} R@100={recall_at_100:.4} native_comparable={}",
            aligned.name,
            search_surface.as_str(),
            search_overrides.disk_pq_dims,
            search_overrides.pq_candidate_expand_pct,
            search_overrides.rerank_expand_pct,
            aligned.search_list_size,
            audit.native_comparable
        );

        rows.push(json!({
            "family": "DISKANN/AISAQ",
            "implementation": "hanns",
            "comparability_status": "constrained",
            "native_comparable": audit.native_comparable,
            "comparability_reason": audit.comparability_reason,
            "config_name": aligned.name,
            "search_surface": search_surface.as_str(),
            "disk_pq_dims": search_overrides.disk_pq_dims,
            "pq_candidate_expand_pct": search_overrides.pq_candidate_expand_pct,
            "rerank_expand_pct": search_overrides.rerank_expand_pct,
            "top_k": TOP_K,
            "recall_at_10": recall_at_10,
            "recall_at_100": recall_at_100,
            "max_degree": aligned.max_degree,
            "search_list_size": aligned.search_list_size,
            "beamwidth": aligned.beamwidth,
            "num_entry_points": aligned.num_entry_points,
            "threads": threads,
            "build_s": build_s,
            "persist_s": persist_s,
            "load_s": load_s,
            "search_s": search_s,
            "qps": qps,
            "scope_audit": {
                "dim": audit.dim,
                "node_count": audit.node_count,
                "entry_point_count": audit.entry_point_count,
                "uses_flash_layout": audit.uses_flash_layout,
                "uses_beam_search_io": audit.uses_beam_search_io,
                "uses_mmap_backed_pages": audit.uses_mmap_backed_pages,
                "has_page_cache": audit.has_page_cache,
                "native_comparable": audit.native_comparable,
                "comparability_reason": audit.comparability_reason,
            },
        }));
    }

    let payload = json!({
        "artifact_type": "diskann_aisaq_aligned_hanns_rows",
        "authority_surface": env::var("AUTHORITY_SURFACE").unwrap_or_else(|_| "local_non_authority_smoke".to_string()),
        "run_set_id": run_set_id,
        "comparability_status": "constrained",
        "native_comparable": false,
        "leadership_claim_allowed": false,
        "search_surface": search_surface.as_str(),
        "disk_pq_dims": search_overrides.disk_pq_dims,
        "pq_candidate_expand_pct": search_overrides.pq_candidate_expand_pct,
        "rerank_expand_pct": search_overrides.rerank_expand_pct,
        "dataset": {
            "path": dataset_path,
            "base_count": dataset.num_base(),
            "query_count_used": nq,
            "dim": dataset.dim(),
        },
        "hanns_commit": env::var("HANNS_COMMIT").unwrap_or_else(|_| "local".to_string()),
        "rows": rows,
    });

    let output_dir = env::var("AISAQ_ALIGNED_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("docs/parity"));
    fs::create_dir_all(&output_dir).expect("create output dir");
    let stem = payload["run_set_id"].as_str().expect("run_set_id");
    let json_path = output_dir.join(format!("hanns-knowhere-{stem}.json"));
    fs::write(&json_path, serde_json::to_string_pretty(&payload).unwrap()).unwrap();
    println!("wrote {}", json_path.display());
}

fn query_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}
