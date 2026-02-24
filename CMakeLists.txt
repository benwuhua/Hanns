cmake_minimum_required(VERSION 3.16)
project(knowhere-rs VERSION 0.1.0 LANGUAGES C Rust)

# 设置 Rust 工具链
if(NOT CMAKE_RUST_COMPILER)
    find_program(CMAKE_RUST_COMPILER rustc)
endif()
if(NOT CMAKE_RUSTC_VERSION)
    execute_process(COMMAND ${CMAKE_RUST_COMPILER} --version OUTPUT_VARIABLE RUSTC_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" CMAKE_RUSTC_VERSION ${RUSTC_VERSION})
endif()

message(STATUS "Rust: ${CMAKE_RUST_COMPILER} ${CMAKE_RUSTC_VERSION}")

# 编译选项
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=armv8.2+neon")
set(CMAKE_RUST_FLAGS_RELEASE "-C opt-level=3 -C target-cpu=generic+neon")

# 构建库
add_library(knowhere_rs STATIC IMPORTED)

# 配置 Cargo
set(CRATE_NAME "knowhere-rs")
set(CRATE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# 构建命令
add_custom_target(build
    COMMAND cargo build --release --crate-type=staticlib,cdylib
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building knowhere-rs"
)

# 安装
install(TARGETS knowhere_rs LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

# 测试
enable_testing()
add_test(NAME knowhere_tests COMMAND cargo test)
