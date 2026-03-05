#!/usr/bin/env python3
"""Convert HDF5 dataset to fvecs/ivecs format."""

import h5py
import numpy as np
import sys
import os

def write_fvecs(filename, vectors):
    """Write vectors to fvecs format."""
    with open(filename, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(vec.astype(np.float32).tobytes())

def write_ivecs(filename, vectors):
    """Write integer vectors to ivecs format."""
    with open(filename, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(np.int32(dim).tobytes())
            f.write(vec.astype(np.int32).tobytes())

def convert_hdf5_to_fvecs(hdf5_path, output_dir):
    """Convert HDF5 dataset to fvecs format."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        # Try common key names
        train = None
        test = None
        neighbors = None
        
        for key in f.keys():
            if 'train' in key:
                train = np.array(f[key])
            elif 'test' in key:
                test = np.array(f[key])
            elif 'neighbors' in key:
                neighbors = np.array(f[key])
        
        # Fallback to first three datasets
        if train is None or test is None:
            keys = list(f.keys())
            if len(keys) >= 2:
                train = np.array(f[keys[0]])
                test = np.array(f[keys[1]])
                if len(keys) >= 3:
                    neighbors = np.array(f[keys[2]])
        
        if train is not None:
            print(f"Train (base): {train.shape}")
            write_fvecs(os.path.join(output_dir, 'base.fvecs'), train)
            print(f"  → base.fvecs")
        
        if test is not None:
            print(f"Test (query): {test.shape}")
            write_fvecs(os.path.join(output_dir, 'query.fvecs'), test)
            print(f"  → query.fvecs")
        
        if neighbors is not None:
            print(f"Neighbors (ground truth): {neighbors.shape}")
            write_ivecs(os.path.join(output_dir, 'groundtruth.ivecs'), neighbors)
            print(f"  → groundtruth.ivecs")
    
    print(f"\n✓ Conversion complete!")
    print(f"Files saved to: {output_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_hdf5_to_fvecs.py <input.hdf5> [output_dir]")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(hdf5_path)
    
    convert_hdf5_to_fvecs(hdf5_path, output_dir)
