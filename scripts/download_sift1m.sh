#!/bin/bash
# Download SIFT1M dataset for benchmarking
# Dataset sources:
# 1. ANN-Benchmarks HDF5 (preferred): http://ann-benchmarks.com/sift-128-euclidean.hdf5
# 2. Original TEXMEX: http://corpus-texmex.irisa.fr/

set -e

DATA_DIR="./data/sift"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading SIFT1M dataset from ANN-Benchmarks..."

# Download HDF5 format (easier to work with)
if [ ! -f "sift-128-euclidean.hdf5" ]; then
    curl -L http://ann-benchmarks.com/sift-128-euclidean.hdf5 -o sift-128-euclidean.hdf5
fi

echo "SIFT1M dataset ready:"
ls -lh sift-128-euclidean.hdf5

# Verify with Python
echo "Verifying..."
python3 - <<EOF
import h5py
import os

filepath = 'sift-128-euclidean.hdf5'
if not os.path.exists(filepath):
    print("ERROR: File not found")
    exit(1)

with h5py.File(filepath, 'r') as f:
    print("Keys:", list(f.keys()))
    if 'train' in f:
        print(f"Train: {f['train'].shape}")
    if 'test' in f:
        print(f"Test: {f['test'].shape}")
    if 'neighbors' in f:
        print(f"Neighbors: {f['neighbors'].shape}")
    
    # Verify expected sizes
    train_shape = f['train'].shape if 'train' in f else f['base'].shape
    test_shape = f['test'].shape if 'test' in f else f['query'].shape
    
    assert train_shape[0] == 1000000, f"Expected 1M train vectors, got {train_shape[0]}"
    assert test_shape[0] == 10000, f"Expected 10K test vectors, got {test_shape[0]}"
    assert train_shape[1] == 128, f"Expected 128 dimensions, got {train_shape[1]}"
    
print("✓ SIFT1M HDF5 dataset verified successfully!")
EOF
