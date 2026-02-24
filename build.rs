//! Build script for knowhere-rs with Faiss support

fn main() {
    #[cfg(feature = "faiss-cxx")]
    {
        // Generate C++ bindings
        cxx_build::bridge("src/faiss/ffi.rs")
            .flag("-std=c++17")
            .flag("-O3")
            .include("/opt/homebrew/include")
            .include("/usr/local/include")
            .compile("knowhere-faiss");
            
        println!("cargo:rerun-if-changed=src/faiss/ffi.rs");
    }
    
    println!("cargo:rerun-if-changed=src/");
}
