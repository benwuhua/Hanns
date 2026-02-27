//! IVF-RaBitQ 索引测试

use knowhere_rs::{IvfRaBitqIndex, IvfRaBitqConfig, SearchRequest};

fn generate_test_data(n: usize, dim: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        for j in 0..dim {
            data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.01;
        }
    }
    data
}

#[test]
fn test_ivf_rabitq_train_and_add() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 生成训练数据
    let data = generate_test_data(1000, dim);
    
    // 训练
    index.train(&data).expect("训练失败");
    
    // 添加向量
    let added = index.add(&data, None).expect("添加失败");
    assert_eq!(added, 1000);
    assert_eq!(index.count(), 1000);
}

#[test]
fn test_ivf_rabitq_search() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 生成数据
    let data = generate_test_data(1000, dim);
    
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    // 搜索
    let query = vec![0.5f32; dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 5,
        filter: None,
        params: None,
        radius: None,
    };
    
    let results = index.search(&query, &req).expect("搜索失败");
    
    assert!(!results.ids.is_empty());
    assert!(results.ids.len() <= 10);
    assert_eq!(results.ids.len(), results.distances.len());
}

#[test]
fn test_ivf_rabitq_batch_search() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 生成数据
    let data = generate_test_data(1000, dim);
    
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    // 批量搜索
    let nq = 10;
    let queries = vec![0.5f32; nq * dim];
    let mut all_results = Vec::new();
    
    for i in 0..nq {
        let query = &queries[i * dim..(i + 1) * dim];
        let req = SearchRequest {
            top_k: 10,
            nprobe: 5,
            filter: None,
            params: None,
            radius: None,
        };
        
        let results = index.search(query, &req).expect("搜索失败");
        all_results.push(results);
    }
    
    assert_eq!(all_results.len(), nq);
    for results in all_results {
        assert!(!results.ids.is_empty());
    }
}

#[test]
fn test_ivf_rabitq_save_and_load() {
    use tempfile::tempdir;
    
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 生成数据
    let data = generate_test_data(1000, dim);
    
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    let original_count = index.count();
    let _original_size = index.size();
    
    // 保存
    let dir = tempdir().expect("创建临时目录失败");
    let path = dir.path().join("ivf_rabitq_test.bin");
    
    index.save(&path).expect("保存失败");
    
    // 加载
    let loaded = IvfRaBitqIndex::load(&path).expect("加载失败");
    
    assert_eq!(loaded.config().dim, dim);
    assert_eq!(loaded.config().nlist, nlist);
    assert_eq!(loaded.count(), original_count);
    
    // 验证搜索
    let query = vec![0.5f32; dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 5,
        filter: None,
        params: None,
        radius: None,
    };
    
    let results = loaded.search(&query, &req).expect("搜索失败");
    assert!(!results.ids.is_empty());
}

#[test]
fn test_ivf_rabitq_with_custom_ids() {
    let dim = 32;
    let nlist = 4;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 生成数据
    let data = generate_test_data(100, dim);
    let ids: Vec<i64> = (1000..1100).collect();
    
    index.train(&data).expect("训练失败");
    index.add(&data, Some(&ids)).expect("添加失败");
    
    // 搜索并验证 IDs
    let query = vec![0.5f32; dim];
    let req = SearchRequest {
        top_k: 10,
        nprobe: 2,
        filter: None,
        params: None,
        radius: None,
    };
    
    let results = index.search(&query, &req).expect("搜索失败");
    
    // 验证返回的 IDs 在自定义范围内
    for &id in &results.ids {
        assert!(id >= 1000 && id < 1100, "ID {} 不在预期范围内", id);
    }
}

#[test]
fn test_ivf_rabitq_has_raw_data() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    let data = generate_test_data(100, dim);
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    // RaBitQ 是有损量化，不存储原始数据
    assert!(!index.has_raw_data());
}

#[test]
fn test_ivf_rabitq_size() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    let data = generate_test_data(1000, dim);
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    let size = index.size();
    assert!(size > 0, "索引大小应该大于 0");
    
    // 验证压缩率：原始数据大小 vs 索引大小
    let raw_size = 1000 * dim * std::mem::size_of::<f32>();
    let compression_ratio = raw_size as f64 / size as f64;
    
    // RaBitQ 应该提供显著的压缩（至少 2x）
    assert!(compression_ratio > 2.0, "压缩比 {:.2} 应该大于 2x", compression_ratio);
}

#[test]
fn test_ivf_rabitq_nprobe() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    let data = generate_test_data(1000, dim);
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    let query = vec![0.5f32; dim];
    
    // 不同 nprobe 的搜索
    let req_nprobe_1 = SearchRequest {
        top_k: 10,
        nprobe: 1,
        filter: None,
        params: None,
        radius: None,
    };
    
    let req_nprobe_5 = SearchRequest {
        top_k: 10,
        nprobe: 5,
        filter: None,
        params: None,
        radius: None,
    };
    
    let results_1 = index.search(&query, &req_nprobe_1).expect("搜索失败");
    let results_5 = index.search(&query, &req_nprobe_5).expect("搜索失败");
    
    // 更大的 nprobe 应该找到更多或相等的结果
    assert!(results_5.ids.len() >= results_1.ids.len());
}

#[test]
fn test_ivf_rabitq_empty_data() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 空数据添加
    let empty_data: Vec<f32> = vec![];
    let added = index.add(&empty_data, None).unwrap_or(0);
    assert_eq!(added, 0);
}

#[test]
fn test_ivf_rabitq_untrained() {
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 未训练时添加应该失败
    let data = generate_test_data(100, dim);
    let result = index.add(&data, None);
    assert!(result.is_err());
}

#[test]
fn test_ivf_rabitq_compression_ratio() {
    let dim = 128;
    let nlist = 20;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    let n = 10000;
    let data = generate_test_data(n, dim);
    
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    let raw_size = n * dim * std::mem::size_of::<f32>();
    let index_size = index.size();
    let compression_ratio = raw_size as f64 / index_size as f64;
    
    println!("原始大小：{} bytes", raw_size);
    println!("索引大小：{} bytes", index_size);
    println!("压缩比：{:.2}x", compression_ratio);
    
    // RaBitQ 应该提供高压缩率
    assert!(compression_ratio > 4.0, "压缩比应该大于 4x");
}

#[test]
fn test_ivf_rabitq_search_accuracy() {
    // 测试搜索准确性：查询应该找到相似的向量
    let dim = 64;
    let nlist = 10;
    let config = IvfRaBitqConfig::new(dim, nlist);
    let mut index = IvfRaBitqIndex::new(config);
    
    // 创建明显的聚类
    let mut data = vec![0.0f32; 100 * dim];
    
    // Cluster 1: 围绕 (0, 0, ..., 0)
    for i in 0..50 {
        for j in 0..dim {
            data[i * dim + j] = (i as f32) * 0.01;
        }
    }
    
    // Cluster 2: 围绕 (10, 10, ..., 10)
    for i in 50..100 {
        for j in 0..dim {
            data[i * dim + j] = 10.0 + (i as f32) * 0.01;
        }
    }
    
    index.train(&data).expect("训练失败");
    index.add(&data, None).expect("添加失败");
    
    // 查询应该找到 Cluster 1 的向量
    let query_cluster1 = vec![0.1f32; dim];
    let req = SearchRequest {
        top_k: 5,
        nprobe: 5,
        filter: None,
        params: None,
        radius: None,
    };
    
    let results = index.search(&query_cluster1, &req).expect("搜索失败");
    
    // 验证返回的 IDs 主要来自 Cluster 1 (IDs 0-49)
    let cluster1_count = results.ids.iter().filter(|&&id| id >= 0 && id < 50).count();
    assert!(cluster1_count >= 3, "应该主要找到 Cluster 1 的向量，但只找到 {}/{}", cluster1_count, results.ids.len());
}
