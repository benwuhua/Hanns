#!/usr/bin/env python3
"""Test IVF-Flat Python binding"""
import numpy as np

try:
    import hanns
    print("✅ hanns imported successfully")
except ImportError:
    print("❌ hanns not installed, run: maturin develop --release")
    exit(1)

# Test 1: Create IVF-Flat index
print("\n📝 Test 1: Create IVF-Flat index")
try:
    index = hanns.Index(
        index_type="ivf_flat",
        dimension=128,
        metric_type="l2",
        nlist=100,
        nprobe=8
    )
    print(f"✅ IVF-Flat index created: {index.index_type()}, dim={index.dimension()}")
except Exception as e:
    print(f"❌ Failed to create IVF-Flat index: {e}")
    exit(1)

# Test 2: Train index
print("\n📝 Test 2: Train IVF-Flat index")
try:
    train_data = np.random.rand(10000, 128).astype(np.float32)
    index.train(train_data)
    print(f"✅ Index trained with {train_data.shape[0]} vectors")
except Exception as e:
    print(f"❌ Training failed: {e}")
    exit(1)

# Test 3: Add vectors
print("\n📝 Test 3: Add vectors")
try:
    add_data = np.random.rand(1000, 128).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    count = index.add(add_data, ids)
    print(f"✅ Added {count} vectors, total: {index.count()}")
except Exception as e:
    print(f"❌ Add failed: {e}")
    exit(1)

# Test 4: Search
print("\n📝 Test 4: Search")
try:
    query = np.random.rand(1, 128).astype(np.float32)
    result = index.search(query, k=10)
    print(f"✅ Search returned {len(result.ids)} results")
    print(f"   Top-5 IDs: {result.ids[:5]}")
    print(f"   Top-5 distances: {result.distances[:5]}")
except Exception as e:
    print(f"❌ Search failed: {e}")
    exit(1)

# Test 5: Save and load
print("\n📝 Test 5: Save and load")
try:
    index.save("test_ivf_flat.bin")
    print("✅ Index saved to test_ivf_flat.bin")
    
    index2 = hanns.Index.load("test_ivf_flat.bin")
    print(f"✅ Index loaded: {index2.index_type()}, count={index2.count()}")
except Exception as e:
    print(f"❌ Save/load failed: {e}")
    exit(1)

print("\n🎉 All tests passed!")
