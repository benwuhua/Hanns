# AISAQ SIFT-1M Authority Numbers (2026-03-23)

**Platform**: x86 native (knowhere-x86-hk-proxy)  
**Dataset**: SIFT-1M (1M base vectors, dim=128, L2 metric)  
**Commits**: f811cf8 + 5fc548e (Vamana multi-pass refine fix)  
**Config**: AisaqConfig default, num_refine_passes=2, max_degree=48

## Results

### NoPQ (disk_pq_dims=0)

| search_list_size (L) | recall@10 | QPS |
|----------------------|-----------|-----|
| 64 | **0.952** ✅ | 5,828 |
| 128 | **0.979** ✅ | 2,888 |
| 200 | **0.986** ✅ | 1,946 |

build: train=0.00s  add=**1990.28s**  save=0.94s  load=0.91s

### PQ32 (disk_pq_dims=32, run_refine_pass=true, rerank_expand_pct=200)

| search_list_size (L) | recall@10 | QPS |
|----------------------|-----------|-----|
| 128 | **0.765** | 10,165 |

build: train=183.18s  add=**1936.70s**  save=1.01s  load=1.27s

## vs Baseline (before fix)

All configs: recall@10 = **0.162** (Phase 2 nodes unreachable)  
After fix: NoPQ recall@10 = **0.952–0.986** (+490–510%)

## Known Issues

- **Build time 1990s (33 min)**: 2 × vamana_refine_pass over 1M nodes.  
  Root cause: benchmark builds index 3× for NoPQ L sweep (shared build would be ~10 min).  
  Optimization task: P2 (AISAQ-OPT-001)
- **PQ32 recall 0.765**: PQ quantization ceiling (32 sub-codes × 8 bits), not a bug.
