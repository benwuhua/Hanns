#!/usr/bin/env python3
"""测试 hanns Python 绑定"""

import numpy as np
import hanns

print(f"hanns version: {hanns.__version__}")

# 测试 1: 创建 Flat 索引
print("\n=== Test 1: Flat Index ===")
index = hanns.Index(
    index_type="flat",
    dimension=128,
    metric_type="l2"
)
print(f"Created Flat index, dimension: {index.dimension()}, type: {index.index_type()}")

# 测试 2: 创建 HNSW 索引
print("\n=== Test 2: HNSW Index ===")
hnsw = hanns.Index(
    index_type="hnsw",
    dimension=128,
    metric_type="l2",
    ef_construction=400,
    ef_search=128,
    m=16
)
print(f"Created HNSW index, type: {hnsw.index_type()}")

# 测试 3: 训练和添加向量
print("\n=== Test 3: Train and Add Vectors ===")
vectors = np.random.rand(100, 128).astype(np.float32)
ids = np.arange(100, dtype=np.int64)

# 先训练
hnsw.train(vectors)
print("Training completed")

# 再添加
count = hnsw.add(vectors, ids)
print(f"Added {count} vectors, total count: {hnsw.count()}")
assert hnsw.count() == 100, f"Expected count 100, got {hnsw.count()}"

# 测试 4: 搜索
print("\n=== Test 4: Search ===")
query = np.random.rand(1, 128).astype(np.float32)
result = hnsw.search(query, k=10)
print(f"Search result: {result.len()} neighbors")
print(f"Top 3 IDs: {result.ids[:3]}")
print(f"Top 3 distances: {result.distances[:3]}")
assert result.len() == 10, f"Expected 10 results, got {result.len()}"

# 测试 5: 序列化
print("\n=== Test 5: Serialization ===")
hnsw.save("/tmp/test_hnsw.bin")
print("Saved index to /tmp/test_hnsw.bin")
import os
assert os.path.exists("/tmp/test_hnsw.bin"), "Index file not created"
print(f"File size: {os.path.getsize('/tmp/test_hnsw.bin')} bytes")

# 测试 6: 错误处理
print("\n=== Test 6: Error Handling ===")
try:
    # 无效索引类型
    invalid = hanns.Index("invalid", 128, "l2")
    print("ERROR: Should have raised exception for invalid index type")
except ValueError as e:
    print(f"✓ Caught invalid index type: {e}")

try:
    # 无效度量类型
    invalid = hanns.Index("flat", 128, "invalid")
    print("ERROR: Should have raised exception for invalid metric type")
except ValueError as e:
    print(f"✓ Caught invalid metric type: {e}")

try:
    # 维度不匹配
    wrong_dim = np.random.rand(100, 64).astype(np.float32)
    hnsw.add(wrong_dim, ids)
    print("ERROR: Should have raised exception for dimension mismatch")
except ValueError as e:
    print(f"✓ Caught dimension mismatch: {e}")

# 测试 7: IVF-PQ 索引
print("\n=== Test 7: IVF-PQ Index ===")
ivf_pq = hanns.Index(
    index_type="ivf_pq",
    dimension=128,
    metric_type="l2",
    nlist=10,
    nprobe=4,
    m=8,
    nbits=8
)
print(f"Created IVF-PQ index, type: {ivf_pq.index_type()}")

# 训练和添加
ivf_pq.train(vectors)
print("IVF-PQ training completed")
count = ivf_pq.add(vectors, ids)
print(f"Added {count} vectors to IVF-PQ, total count: {ivf_pq.count()}")
assert ivf_pq.count() == 100, f"Expected count 100, got {ivf_pq.count()}"

# 搜索
result_ivf = ivf_pq.search(query, k=10)
print(f"IVF-PQ search result: {result_ivf.len()} neighbors")
assert result_ivf.len() == 10, f"Expected 10 results, got {result_ivf.len()}"

# 序列化 IVF-PQ
ivf_pq.save("/tmp/test_ivf_pq.bin")
print("Saved IVF-PQ index to /tmp/test_ivf_pq.bin")
assert os.path.exists("/tmp/test_ivf_pq.bin"), "IVF-PQ index file not created"

# 测试 8: load() 反序列化 - Flat
print("\n=== Test 8: Load Flat Index ===")
flat = hanns.Index(
    index_type="flat",
    dimension=128,
    metric_type="l2"
)
flat.train(vectors)
flat.add(vectors, ids)
flat.save("/tmp/test_flat.bin")

# 加载 Flat 索引
flat_loaded = hanns.Index.load("/tmp/test_flat.bin")
print(f"Loaded Flat index, count: {flat_loaded.count()}, type: {flat_loaded.index_type()}")
assert flat_loaded.count() == 100, f"Expected count 100, got {flat_loaded.count()}"
assert flat_loaded.index_type() == "flat", f"Expected type 'flat', got {flat_loaded.index_type()}"

# 验证搜索结果一致
result1 = flat.search(query, k=10)
result2 = flat_loaded.search(query, k=10)
assert result1.len() == result2.len(), "Search result count mismatch after load"
print(f"✓ Flat index load test passed")

# 测试 9: load() 反序列化 - HNSW
print("\n=== Test 9: Load HNSW Index ===")
hnsw_loaded = hanns.Index.load("/tmp/test_hnsw.bin")
print(f"Loaded HNSW index, count: {hnsw_loaded.count()}, type: {hnsw_loaded.index_type()}")
assert hnsw_loaded.count() == 100, f"Expected count 100, got {hnsw_loaded.count()}"
assert hnsw_loaded.index_type() == "hnsw", f"Expected type 'hnsw', got {hnsw_loaded.index_type()}"

# 验证搜索结果
result_hnsw1 = hnsw.search(query, k=10)
result_hnsw2 = hnsw_loaded.search(query, k=10)
assert result_hnsw1.len() == result_hnsw2.len(), "Search result count mismatch after load"
print(f"✓ HNSW index load test passed")

# 测试 10: load() 反序列化 - IVF-PQ
print("\n=== Test 10: Load IVF-PQ Index ===")
ivf_pq_loaded = hanns.Index.load("/tmp/test_ivf_pq.bin")
print(f"Loaded IVF-PQ index, count: {ivf_pq_loaded.count()}, type: {ivf_pq_loaded.index_type()}")
assert ivf_pq_loaded.count() == 100, f"Expected count 100, got {ivf_pq_loaded.count()}"
assert ivf_pq_loaded.index_type() == "ivf_pq", f"Expected type 'ivf_pq', got {ivf_pq_loaded.index_type()}"

# 验证搜索结果
result_ivf1 = ivf_pq.search(query, k=10)
result_ivf2 = ivf_pq_loaded.search(query, k=10)
assert result_ivf1.len() == result_ivf2.len(), "Search result count mismatch after load"
print(f"✓ IVF-PQ index load test passed")

print("\n=== All Tests Passed ===")
