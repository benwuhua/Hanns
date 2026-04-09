#!/bin/bash
# 构建脚本

set -e

echo "=== Building knowhere-rs ==="

# 检查 Rust
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust not found"
    exit 1
fi

# Debug or Release
BUILD_TYPE=${1:-release}

echo "Building $BUILD_TYPE..."

# 构建
cargo build --$BUILD_TYPE

# 静态库位置
LIB_PATH="target/$BUILD_TYPE"

if [ -f "$LIB_PATH/libknowhere_rs.rlib" ]; then
    echo "Static lib: $LIB_PATH/libknowhere_rs.rlib"
fi

if [ -f "$LIB_PATH/libknowhere_rs.so" ]; then
    echo "Shared lib: $LIB_PATH/libknowhere_rs.so"
fi

echo "=== Build complete ==="

# 运行测试
echo "Running tests..."
cargo test

echo "=== Done ==="
