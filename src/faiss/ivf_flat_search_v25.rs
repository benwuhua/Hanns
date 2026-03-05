    /// Search (优化版本 V2.5：保留 map + extend，避免 flat_map 开销)
    /// OPT-003: 使用 Vec 直接索引替代 HashMap
    /// OPT-004: 性能优化（目标 QPS 1500+）
    /// 
    /// V2.5 优化（在 V2 基础上）：
    /// 1. 保留 l2_distance_sq（避免 sqrt）
    /// 2. 保留 select_nth_unstable_by（O(n) vs O(n log n)）
    /// 3. 保留 map + extend（比 flat_map 更快）
    /// 4. 预分配内存优化
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(crate::api::KnowhereError::InvalidArg(
                "index not trained".to_string(),
            ));
        }
        
        let top_k = req.top_k;
        let nprobe = if req.nprobe > 0 { req.nprobe } else { self.nprobe };
        
        // 优化 1: 选择最近的 nprobe 个簇（使用 select_nth_unstable）
        let mut cluster_dists: Vec<(usize, f32)> = (0..self.nlist)
            .map(|c| (c, l2_distance_sq(query, &self.centroids[c * self.dim..])))
            .collect();
        
        if nprobe < cluster_dists.len() {
            cluster_dists.select_nth_unstable_by(nprobe, |a, b| {
                a.1.partial_cmp(&b.1).unwrap()
            });
            cluster_dists.truncate(nprobe);
        }
        
        // 优化 2: 并行搜索（保留 map + extend 模式，避免 flat_map 开销）
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            
            // 预估结果数量
            let estimated_size: usize = cluster_dists.iter()
                .map(|(c, _)| self.invlist_ids[*c].len())
                .sum();
            
            // 并行搜索 - 保留 map 模式
            let partial_results: Vec<Vec<(i64, f32)>> = cluster_dists
                .par_iter()
                .map(|(cluster_id, _)| {
                    let ids = &self.invlist_ids[*cluster_id];
                    let vectors = &self.invlist_vectors[*cluster_id];
                    let list_len = ids.len();
                    
                    let mut results = Vec::with_capacity(list_len);
                    for i in 0..list_len {
                        let vector = &vectors[i * self.dim..(i + 1) * self.dim];
                        // 使用 l2_distance_sq（无 sqrt）
                        let dist = l2_distance_sq(query, vector);
                        results.push((ids[i], dist));
                    }
                    results
                })
                .collect();
            
            // 合并结果（预分配）
            let mut all_results: Vec<(i64, f32)> = Vec::with_capacity(estimated_size);
            for partial in partial_results {
                all_results.extend(partial);
            }
            
            // 优化 3: 使用 select_nth_unstable_by 选择 Top-K
            if top_k < all_results.len() {
                all_results.select_nth_unstable_by(top_k, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap()
                });
                all_results.truncate(top_k);
            } else if all_results.len() > 1 {
                all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
            
            let ids: Vec<i64> = all_results.iter().map(|(id, _)| *id).collect();
            let distances: Vec<f32> = all_results.iter().map(|(_, d)| *d).collect();
            
            Ok(SearchResult {
                ids,
                distances,
                elapsed_ms: 0.0,
                num_visited: all_results.len(),
            })
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            // 串行版本
            let estimated_size: usize = cluster_dists.iter()
                .map(|(c, _)| self.invlist_ids[*c].len())
                .sum();
            let mut all_results: Vec<(i64, f32)> = Vec::with_capacity(estimated_size);
            
            for (cluster_id, _) in cluster_dists {
                let ids = &self.invlist_ids[cluster_id];
                let vectors = &self.invlist_vectors[cluster_id];
                let list_len = ids.len();
                
                for i in 0..list_len {
                    let vector = &vectors[i * self.dim..(i + 1) * self.dim];
                    let dist = l2_distance_sq(query, vector);
                    all_results.push((ids[i], dist));
                }
            }
            
            // Top-K 选择
            if top_k < all_results.len() {
                all_results.select_nth_unstable_by(top_k, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap()
                });
                all_results.truncate(top_k);
            } else if all_results.len() > 1 {
                all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
            
            let ids: Vec<i64> = all_results.iter().map(|(id, _)| *id).collect();
            let distances: Vec<f32> = all_results.iter().map(|(_, d)| *d).collect();
            
            Ok(SearchResult {
                ids,
                distances,
                elapsed_ms: 0.0,
                num_visited: all_results.len(),
            })
        }
    }
