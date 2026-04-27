# DiskANN/AISAQ page-cache expanded-candidate verdict (20260426T024300Z)

- Status: `non_comparable` (with numeric observation); leadership claim remains blocked.
- Authority surface: HannsDB-x86.
- Hanns candidate: `R48-L108-B8-EP1`, `search_surface=page_cache`, `disk_pq_dims=32`, `pq_candidate_expand_pct=400`, `rerank_expand_pct=400`.
- Page-cache proof: `scope_audit.has_page_cache=true`.
- Metrics: R@100=0.993680, QPS=624.911, build=124.187s, persist=62.215s, load=0.327s.
- Numeric observation: compared to official Zilliz Knowhere `AISAQ_S` near-0.95 row (R@100=0.9515, VPS=520.39, build=159.796s), Hanns is higher on recall/QPS/build_s when persist is kept as a separate recorded phase.
- Conclusion: this is useful optimization evidence, not a family leadership verdict, because `native_comparable=false` and PQFlashIndex is still not proven native-comparable to official SSD DiskANN/AISAQ.

Secondary row:
- `disk_pq_dims=64`: R@100=0.997348, QPS=578.785, build=151.763s.

Evidence:
- Hanns artifact: `docs/parity/raw/hanns-knowhere-diskann-aisaq-aligned-1777170795.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260426T023312Z_84693.log`, `docs/parity/raw/remote_background_20260426T023312Z_84693.status`
- Remote prefilter: `docs/parity/raw/test_20260426T023227Z_76776.log`, `docs/parity/raw/test_20260426T023227Z_76776.status`
