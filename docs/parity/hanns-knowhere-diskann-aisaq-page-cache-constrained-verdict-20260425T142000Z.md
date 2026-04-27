# DiskANN/AISAQ page-cache constrained verdict (20260425T142000Z)

- Status: `non_comparable`; leadership claim remains blocked.
- Authority surface: HannsDB-x86.
- Hanns row: `R48-L108-B8-EP1`, `search_surface=page_cache`, `disk_pq_dims=8`, `scope_audit.has_page_cache=true`.
- Metrics: R@100=0.433609, QPS=2625.189, build=99.943s, persist=61.945s, load=0.297s.
- Conclusion: this run proves the page-cache evidence binding but is not a performance win; it is below the official near-0.95 recall/QPS targets and remains non-native-comparable.

Evidence:
- Hanns artifact: `docs/parity/raw/hanns-knowhere-diskann-aisaq-aligned-1777126402.json`
- Hanns log/status: `docs/parity/raw/remote_background_20260425T141320Z_92861.log`, `docs/parity/raw/remote_background_20260425T141320Z_92861.status`
- Remote verification: `docs/parity/raw/test_20260425T141134Z_76106.log`, `docs/parity/raw/test_20260425T141134Z_76106.status`
