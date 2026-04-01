//! IVF-RaBitQ 索引 - 完整实现
//!
//! 结合倒排索引和 RaBitQ 量化，支持 32x 压缩的高效向量搜索

use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::api::{KnowhereError, MetricType, Result, SearchRequest, SearchResult};
use crate::bitset::BitsetView;
use crate::dataset::Dataset;
use crate::index::{
    AnnIterator, Index as IndexTrait, IndexError, SearchResult as IndexSearchResult,
};
use crate::quantization::{pick_refine_index, KMeans, RaBitQEncoder, RefineIndex, RefineType};

/// IVF-RaBitQ 索引配置
#[derive(Clone, Debug)]
pub struct IvfRaBitqConfig {
    pub dim: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub metric_type: MetricType,
    pub refine_type: Option<RefineType>,
    pub reorder_k: usize,
}

impl IvfRaBitqConfig {
    pub fn new(dim: usize, nlist: usize) -> Self {
        Self {
            dim,
            nlist,
            nprobe: 1,
            metric_type: MetricType::L2,
            refine_type: None,
            reorder_k: 0,
        }
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    pub fn with_metric(mut self, metric: MetricType) -> Self {
        self.metric_type = metric;
        self
    }

    pub fn with_refine(mut self, refine_type: RefineType, reorder_k: usize) -> Self {
        self.refine_type = Some(refine_type);
        self.reorder_k = reorder_k;
        self
    }
}

/// IVF-RaBitQ 索引条目
/// 存储：(id, binary_code, centroid_dist, inner_product, sum_xb)
/// - binary_code: 二进制编码
/// - centroid_dist: ||or - c|| (残差的 L2 范数)
/// - inner_product: dp_multiplier 校正因子
/// - sum_xb: 二进制位之和（C++ FactorsData.sum_xb）
pub type IvfRaBitqEntry = (i64, Vec<u8>, f32, f32, f32);

/// IVF-RaBitQ 索引
pub struct IvfRaBitqIndex {
    config: IvfRaBitqConfig,
    centroids: Vec<f32>,
    inverted_lists: Arc<RwLock<HashMap<usize, Vec<IvfRaBitqEntry>>>>,
    encoder: RaBitQEncoder,
    refine_index: Option<RefineIndex>,
    trained: bool,
    ntotal: usize,
}

impl IvfRaBitqIndex {
    /// 创建新索引
    pub fn new(config: IvfRaBitqConfig) -> Self {
        Self {
            centroids: Vec::new(),
            inverted_lists: Arc::new(RwLock::new(HashMap::new())),
            encoder: RaBitQEncoder::new(config.dim),
            refine_index: None,
            trained: false,
            ntotal: 0,
            config,
        }
    }

    /// 训练索引
    pub fn train(&mut self, data: &[f32]) -> Result<()> {
        let n = data.len() / self.config.dim;
        if n < self.config.nlist {
            return Err(KnowhereError::InvalidArg(format!(
                "训练数据不足：{} < {}",
                n, self.config.nlist
            )));
        }

        // 训练 RaBitQ 编码器
        self.encoder.train(data);

        // K-means 训练质心
        let mut km = KMeans::new(self.config.nlist, self.config.dim);
        if matches!(self.config.metric_type, MetricType::Ip | MetricType::Cosine) {
            km = km.with_metric(crate::quantization::kmeans::KMeansMetric::InnerProduct);
        }
        km.train(data);
        self.centroids = km.centroids().to_vec();

        self.trained = true;

        tracing::info!(
            "IVF-RaBitQ 索引训练完成：nlist={}, dim={}",
            self.config.nlist,
            self.config.dim
        );

        Ok(())
    }

    /// 添加向量
    pub fn add(&mut self, data: &[f32], ids: Option<&[i64]>) -> Result<usize> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg(
                "索引未训练，请先调用 train()".to_string(),
            ));
        }

        let n = data.len() / self.config.dim;
        if n == 0 {
            return Ok(0);
        }

        let mut lists = self.inverted_lists.write();
        let batch_ids: Vec<i64> = (0..n)
            .map(|i| ids.map(|ids| ids[i]).unwrap_or((self.ntotal + i) as i64))
            .collect();

        for i in 0..n {
            let vector = &data[i * self.config.dim..(i + 1) * self.config.dim];

            // 找到最近的质心
            let cluster = self.find_nearest_centroid(vector);

            // 使用质心进行残差编码 (返回 code, centroid_dist, ip)
            let centroid =
                &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];
            let (code, centroid_dist, ip, sum_xb) =
                self.encoder.encode_with_centroid(vector, centroid);

            let id = batch_ids[i];

            lists
                .entry(cluster)
                .or_default()
                .push((id, code, centroid_dist, ip, sum_xb));
        }

        if self.config.refine_type.is_some() {
            match &mut self.refine_index {
                Some(refine_index) => refine_index.append(data, &batch_ids)?,
                None => {
                    self.refine_index = pick_refine_index(
                        data,
                        self.config.dim,
                        &batch_ids,
                        self.config.metric_type,
                        self.config.refine_type,
                    )?;
                }
            }
        }

        self.ntotal += n;

        tracing::debug!("IVF-RaBitQ 添加 {} 个向量，总计 {}", n, self.ntotal);

        Ok(n)
    }

    /// 搜索
    pub fn search(&self, query: &[f32], req: &SearchRequest) -> Result<SearchResult> {
        if !self.trained {
            return Err(KnowhereError::InvalidArg("索引未训练".to_string()));
        }

        let n = query.len() / self.config.dim;
        if n == 0 {
            return Err(KnowhereError::InvalidArg("查询向量为空".to_string()));
        }

        let nprobe = req.nprobe.max(1).min(self.config.nlist);
        let top_k = req.top_k;
        let reorder_k = self.resolve_reorder_k(req).max(top_k);

        // 批量搜索
        let mut all_ids = Vec::with_capacity(n * top_k);
        let mut all_distances = Vec::with_capacity(n * top_k);
        let mut candidate_batches = Vec::with_capacity(n);

        for i in 0..n {
            let q = &query[i * self.config.dim..(i + 1) * self.config.dim];
            let filter_ref = req.filter.as_ref().map(|f| f.as_ref());
            candidate_batches.push(self.search_single(q, reorder_k, nprobe, filter_ref));
        }

        let final_batches = if let Some(refine_index) = &self.refine_index {
            refine_index.rerank_batch(query, &candidate_batches, top_k)?
        } else {
            candidate_batches
                .into_iter()
                .map(|mut candidates| {
                    candidates.truncate(top_k);
                    candidates
                })
                .collect()
        };

        for batch in final_batches {
            for (id, dist) in batch {
                all_ids.push(id);
                all_distances.push(dist);
            }
        }

        Ok(SearchResult::new(all_ids, all_distances, 0.0))
    }

    /// 单个查询搜索
    fn search_single(
        &self,
        query: &[f32],
        top_k: usize,
        nprobe: usize,
        filter: Option<&dyn crate::api::Predicate>,
    ) -> Vec<(i64, f32)> {
        // 先按质心距离排序，再按需自适应扩展 probe，避免候选不足导致召回阈值回归。
        let ranked_clusters = self.search_centroids_ranked(query);
        let mut probe_count = ranked_clusters.len();
        let mut processed_clusters = 0usize;

        // 经验候选预算：至少 top_k*8（上限 ntotal），在高压缩场景下可显著降低漏召回。
        let target_candidates = top_k.saturating_mul(8).min(self.ntotal.max(top_k));

        // 收集候选
        let mut candidates: Vec<(i64, f32)> = Vec::new();
        let lists = self.inverted_lists.read();
        let use_q8 = self.encoder.qb == 8;

        while processed_clusters < ranked_clusters.len() {
            for &cluster in ranked_clusters
                .iter()
                .take(probe_count)
                .skip(processed_clusters)
            {
                if let Some(list) = lists.get(&cluster) {
                    let centroid =
                        &self.centroids[cluster * self.config.dim..(cluster + 1) * self.config.dim];

                    if use_q8 {
                        // qb=8 模式：量化查询到 8-bit
                        let qq = self
                            .encoder
                            .build_distance_table_q8_with_centroid(query, centroid);

                        for &(id, ref code, data_centroid_dist, data_ip, data_sum_xb) in list {
                            if let Some(f) = filter {
                                if !f.evaluate(id) {
                                    continue;
                                }
                            }
                            let dist = self.encoder.compute_distance_q8(
                                &qq,
                                code,
                                data_centroid_dist,
                                data_ip,
                                data_sum_xb,
                            );
                            candidates.push((id, dist));
                        }
                    } else {
                        // qb=0 模式：不量化查询
                        let (query_rotated, _, query_residual_norm, _) = self
                            .encoder
                            .build_distance_table_with_centroid(query, centroid);

                        for &(id, ref code, data_centroid_dist, data_ip, data_sum_xb) in list {
                            if let Some(f) = filter {
                                if !f.evaluate(id) {
                                    continue;
                                }
                            }
                            let dist = self.encoder.compute_distance(
                                &query_rotated,
                                query_residual_norm,
                                code,
                                data_centroid_dist,
                                data_ip,
                                data_sum_xb,
                            );
                            candidates.push((id, dist));
                        }
                    }
                }
            }

            processed_clusters = probe_count;

            if candidates.len() >= target_candidates || probe_count == ranked_clusters.len() {
                break;
            }

            // 逐步扩展探测簇，避免一次性全扫。
            probe_count = (probe_count + nprobe.max(1)).min(ranked_clusters.len());
        }

        // 排序返回 top-k
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates.truncate(top_k);

        candidates
    }

    /// 查找最近的质心
    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        let mut min_dist = f32::MAX;
        let mut best = 0;

        for (i, centroid) in self.centroids.chunks(self.config.dim).enumerate() {
            let dist = self.l2_distance(vector, centroid);
            if dist < min_dist {
                min_dist = dist;
                best = i;
            }
        }

        best
    }

    /// 返回按距离排序的全部质心
    fn search_centroids_ranked(&self, query: &[f32]) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = (0..self.config.nlist)
            .map(|i| {
                let c = &self.centroids[i * self.config.dim..(i + 1) * self.config.dim];
                (i, self.l2_distance(query, c))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().map(|(i, _)| i).collect()
    }

    #[inline]
    fn l2_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum += diff * diff;
        }
        sum
    }

    fn resolve_reorder_k(&self, req: &SearchRequest) -> usize {
        #[derive(Deserialize)]
        struct RefineSearchParams {
            reorder_k: Option<usize>,
            refine_k: Option<usize>,
        }

        req.params
            .as_ref()
            .and_then(|params| serde_json::from_str::<RefineSearchParams>(params).ok())
            .and_then(|params| params.reorder_k.or(params.refine_k))
            .unwrap_or(self.config.reorder_k)
    }

    /// 检查是否有原始数据
    pub fn has_raw_data(&self) -> bool {
        matches!(
            self.refine_index.as_ref().map(|index| index.refine_type()),
            Some(RefineType::DataView)
        )
    }

    /// 返回向量数量
    pub fn count(&self) -> usize {
        self.ntotal
    }

    /// 返回索引大小（字节）
    pub fn size(&self) -> usize {
        let encoder_size = self.encoder.code_size() * std::mem::size_of::<f32>();
        let centroids_size = self.centroids.len() * std::mem::size_of::<f32>();

        let lists = self.inverted_lists.read();
        let codes_size: usize = lists
            .values()
            .flat_map(|list| list.iter().map(|(_, code, _, _, _)| code.len()))
            .sum();
        let ids_size: usize = lists
            .values()
            .flat_map(|list| list.iter().map(|_| std::mem::size_of::<i64>()))
            .sum();
        // 校正因子：每个向量 2 个 f32
        let factors_size: usize = lists
            .values()
            .flat_map(|list| list.iter().map(|_| 2 * std::mem::size_of::<f32>()))
            .sum();
        let refine_size = match &self.refine_index {
            Some(refine) => match refine.refine_type() {
                RefineType::DataView => refine.len() * self.config.dim * std::mem::size_of::<f32>(),
                RefineType::Uint8Quant => {
                    refine.len() * self.config.dim * std::mem::size_of::<u8>()
                }
                RefineType::Float16Quant | RefineType::Bfloat16Quant => {
                    refine.len() * self.config.dim * std::mem::size_of::<u16>()
                }
                RefineType::Sq4Quant => refine.len() * self.config.dim.div_ceil(2),
                RefineType::Sq6Quant => refine.len() * ((self.config.dim * 6).div_ceil(8)),
            },
            None => 0,
        };

        encoder_size + centroids_size + codes_size + ids_size + factors_size + refine_size
    }

    /// 保存索引
    pub fn save(&self, path: &Path) -> Result<()> {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // 写入头部
        writer.write_all(&(self.config.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.config.nlist as u32).to_le_bytes())?;
        writer.write_all(&(self.config.nprobe as u32).to_le_bytes())?;
        writer.write_all(&(self.ntotal as u32).to_le_bytes())?;
        writer.write_all(&(self.config.reorder_k as u32).to_le_bytes())?;
        writer.write_all(&[if self.trained { 1 } else { 0 }])?;
        writer.write_all(&[self.config.refine_type.map(|t| t as u8).unwrap_or(u8::MAX)])?;

        // 写入质心
        for &c in &self.centroids {
            writer.write_all(&c.to_le_bytes())?;
        }

        // RaBitQ 不再使用旋转矩阵（与 C++ Faiss 对齐）
        // 写入 0 表示没有旋转矩阵
        writer.write_all(&0u32.to_le_bytes())?;

        // 写入倒排列表
        let lists = self.inverted_lists.read();
        writer.write_all(&(lists.len() as u32).to_le_bytes())?;
        for (cluster, list) in lists.iter() {
            writer.write_all(&(*cluster as u32).to_le_bytes())?;
            writer.write_all(&(list.len() as u32).to_le_bytes())?;
            for (id, code, centroid_dist, ip, _sum_xb) in list {
                writer.write_all(&id.to_le_bytes())?;
                writer.write_all(&(code.len() as u32).to_le_bytes())?;
                writer.write_all(code)?;
                writer.write_all(&centroid_dist.to_le_bytes())?;
                writer.write_all(&ip.to_le_bytes())?;
            }
        }

        if let Some(refine) = &self.refine_index {
            refine.write_to(&mut writer)?;
        } else {
            writer.write_all(&[0u8])?;
        }

        writer.flush()?;

        tracing::info!("IVF-RaBitQ 索引保存到 {:?}", path);

        Ok(())
    }

    /// 加载索引
    pub fn load(path: &Path) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut buf4 = [0u8; 4];

        // 读取头部
        reader.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let nlist = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let nprobe = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let ntotal = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let reorder_k = u32::from_le_bytes(buf4) as usize;

        let mut trained_buf = [0u8; 1];
        reader.read_exact(&mut trained_buf)?;
        let trained = trained_buf[0] != 0;

        let mut refine_type_buf = [0u8; 1];
        reader.read_exact(&mut refine_type_buf)?;
        let refine_type = if refine_type_buf[0] == u8::MAX {
            None
        } else {
            Some(RefineType::from_u8(refine_type_buf[0])?)
        };

        // 读取质心
        let mut centroids = vec![0.0f32; nlist * dim];
        for c in &mut centroids {
            reader.read_exact(&mut buf4)?;
            *c = f32::from_le_bytes(buf4);
        }

        // 读取 encoder 旋转矩阵长度（兼容旧格式）
        reader.read_exact(&mut buf4)?;
        let rotation_len = u32::from_le_bytes(buf4) as usize;

        // 跳过旋转矩阵数据（如果有）
        if rotation_len > 0 {
            let mut _rotation_matrix = vec![0.0f32; rotation_len];
            for c in &mut _rotation_matrix {
                reader.read_exact(&mut buf4)?;
                *c = f32::from_le_bytes(buf4);
            }
        }

        // 创建 encoder（无需旋转矩阵，直接标记为已训练）
        let mut encoder = RaBitQEncoder::new(dim);
        encoder.train(&[]); // RaBitQ 无需训练数据

        // 读取倒排列表
        let mut inverted_lists: HashMap<usize, Vec<IvfRaBitqEntry>> = HashMap::new();

        reader.read_exact(&mut buf4)?;
        let num_lists = u32::from_le_bytes(buf4) as usize;

        for _ in 0..num_lists {
            reader.read_exact(&mut buf4)?;
            let cluster = u32::from_le_bytes(buf4) as usize;

            reader.read_exact(&mut buf4)?;
            let list_len = u32::from_le_bytes(buf4) as usize;

            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let mut buf8 = [0u8; 8];
                reader.read_exact(&mut buf8)?;
                let id = i64::from_le_bytes(buf8);

                reader.read_exact(&mut buf4)?;
                let code_len = u32::from_le_bytes(buf4) as usize;

                let mut code = vec![0u8; code_len];
                reader.read_exact(&mut code)?;

                reader.read_exact(&mut buf4)?;
                let centroid_dist = f32::from_le_bytes(buf4);

                reader.read_exact(&mut buf4)?;
                let ip = f32::from_le_bytes(buf4);

                // 计算二进制位之和（向后兼容：从 code 中计算）
                let sum_xb = code.iter().map(|b| b.count_ones() as f32).sum();

                list.push((id, code, centroid_dist, ip, sum_xb));
            }

            inverted_lists.insert(cluster, list);
        }

        let refine_index = RefineIndex::read_from(&mut reader, dim, MetricType::L2)?;

        let config = IvfRaBitqConfig {
            dim,
            nlist,
            nprobe,
            metric_type: MetricType::L2,
            refine_type,
            reorder_k,
        };

        tracing::info!("IVF-RaBitQ 索引从 {:?} 加载", path);

        Ok(Self {
            config,
            centroids,
            inverted_lists: Arc::new(RwLock::new(inverted_lists)),
            encoder,
            refine_index,
            trained,
            ntotal,
        })
    }

    /// 设置 nprobe
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.config.nprobe = nprobe.min(self.config.nlist);
    }

    pub fn config(&self) -> &IvfRaBitqConfig {
        &self.config
    }
}

/// Index trait implementation for IvfRaBitqIndex
///
/// This wrapper enables IvfRaBitqIndex to be used through the unified Index trait interface,
/// allowing consistent access to advanced features (AnnIterator, get_vector_by_ids, etc.)
/// across all index types.
impl IndexTrait for IvfRaBitqIndex {
    fn index_type(&self) -> &str {
        "IVF-RaBitQ"
    }

    fn dim(&self) -> usize {
        self.config.dim
    }

    fn count(&self) -> usize {
        self.ntotal
    }

    fn is_trained(&self) -> bool {
        self.trained
    }

    fn train(&mut self, dataset: &Dataset) -> std::result::Result<(), IndexError> {
        let vectors = dataset.vectors();
        self.train(vectors)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn add(&mut self, dataset: &Dataset) -> std::result::Result<usize, IndexError> {
        let vectors = dataset.vectors();
        let ids = dataset.ids();
        self.add(vectors, ids)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn search(
        &self,
        query: &Dataset,
        top_k: usize,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        let vectors = query.vectors();
        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };
        let api_result = self
            .search(vectors, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;
        Ok(IndexSearchResult::new(
            api_result.ids,
            api_result.distances,
            api_result.elapsed_ms,
        ))
    }

    fn search_with_bitset(
        &self,
        query: &Dataset,
        top_k: usize,
        bitset: &BitsetView,
    ) -> std::result::Result<IndexSearchResult, IndexError> {
        // IVF-RaBitQ doesn't have native bitset support, use default implementation
        let vectors = query.vectors();
        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };

        let api_result = self
            .search(vectors, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        // Filter out bitset-marked vectors
        let mut filtered_ids = Vec::new();
        let mut filtered_distances = Vec::new();

        for (id, dist) in api_result.ids.iter().zip(api_result.distances.iter()) {
            let idx = *id as usize;
            if idx >= bitset.len() || !bitset.get(idx) {
                filtered_ids.push(*id);
                filtered_distances.push(*dist);
            }
        }

        Ok(IndexSearchResult::new(
            filtered_ids,
            filtered_distances,
            api_result.elapsed_ms,
        ))
    }

    fn save(&self, path: &str) -> std::result::Result<(), IndexError> {
        let path = std::path::Path::new(path);
        self.save(path)
            .map_err(|e| IndexError::Unsupported(e.to_string()))
    }

    fn load(&mut self, path: &str) -> std::result::Result<(), IndexError> {
        let path = std::path::Path::new(path);
        let loaded = Self::load(path).map_err(|e| IndexError::Unsupported(e.to_string()))?;
        *self = loaded;
        Ok(())
    }

    fn has_raw_data(&self) -> bool {
        self.has_raw_data()
    }

    fn get_vector_by_ids(&self, _ids: &[i64]) -> std::result::Result<Vec<f32>, IndexError> {
        // RaBitQ is a lossy compression, doesn't store raw vectors
        Err(IndexError::Unsupported(
            "get_vector_by_ids not supported for RaBitQ (lossy compression)".into(),
        ))
    }

    fn create_ann_iterator(
        &self,
        query: &Dataset,
        _bitset: Option<&BitsetView>,
    ) -> std::result::Result<Box<dyn AnnIterator>, IndexError> {
        // IVF-RaBitQ doesn't support native iterator, fallback to search
        let top_k = self.ntotal.max(1000);
        let vectors = query.vectors();

        let req = SearchRequest {
            top_k,
            nprobe: self.config.nprobe,
            filter: None,
            params: None,
            radius: None,
        };

        let api_result = self
            .search(vectors, &req)
            .map_err(|e| IndexError::Unsupported(e.to_string()))?;

        // Create simple iterator from search results
        let results: Vec<(i64, f32)> = api_result
            .ids
            .into_iter()
            .zip(api_result.distances)
            .collect();

        Ok(Box::new(IvfRaBitqAnnIterator::new(results)))
    }
}

/// Simple ANN iterator for IVF-RaBitQ (fallback implementation)
pub struct IvfRaBitqAnnIterator {
    results: Vec<(i64, f32)>,
    pos: usize,
}

impl IvfRaBitqAnnIterator {
    pub fn new(results: Vec<(i64, f32)>) -> Self {
        Self { results, pos: 0 }
    }
}

impl AnnIterator for IvfRaBitqAnnIterator {
    fn next(&mut self) -> Option<(i64, f32)> {
        if self.pos >= self.results.len() {
            return None;
        }
        let result = self.results[self.pos];
        self.pos += 1;
        Some(result)
    }

    fn buffer_size(&self) -> usize {
        self.results.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ivf_rabitq_train_add() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);

        // 生成训练数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }

        index.train(&data).unwrap();
        assert!(index.trained);

        // 添加向量
        let added = index.add(&data, None).unwrap();
        assert_eq!(added, 100);
        assert_eq!(index.count(), 100);
    }

    #[test]
    fn test_ivf_rabitq_search() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);

        // 生成训练和添加数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }

        index.train(&data).unwrap();
        index.add(&data, None).unwrap();

        // 搜索
        let query = vec![0.5f32; 16];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };

        let results = index.search(&query, &req).unwrap();
        assert!(results.ids.len() <= 10);
        assert_eq!(results.ids.len(), results.distances.len());
    }

    #[test]
    fn test_ivf_rabitq_save_load() {
        let config = IvfRaBitqConfig::new(16, 4);
        let mut index = IvfRaBitqIndex::new(config);

        // 生成数据
        let mut data = vec![0.0f32; 100 * 16];
        for i in 0..100 {
            for j in 0..16 {
                data[i * 16 + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
            }
        }

        index.train(&data).unwrap();
        index.add(&data, None).unwrap();

        // 保存
        let dir = tempdir().unwrap();
        let path = dir.path().join("ivf_rabitq.bin");
        index.save(&path).unwrap();

        // 加载
        let loaded = IvfRaBitqIndex::load(&path).unwrap();

        assert_eq!(loaded.config.dim, index.config.dim);
        assert_eq!(loaded.config.nlist, index.config.nlist);
        assert_eq!(loaded.count(), index.count());
    }

    #[test]
    fn test_ivf_rabitq_with_ids() {
        let config = IvfRaBitqConfig::new(8, 2);
        let mut index = IvfRaBitqIndex::new(config);

        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ];

        let ids = vec![100, 200, 300, 400];

        index.train(&data).unwrap();
        index.add(&data, Some(&ids)).unwrap();

        let query = vec![0.1f32; 8];
        let req = SearchRequest {
            top_k: 2,
            nprobe: 2,
            filter: None,
            params: None,
            radius: None,
        };

        let results = index.search(&query, &req).unwrap();
        assert!(!results.ids.is_empty());
    }
}
