//! Build script for knowhere-rs with Faiss support

fn main() {
    #[cfg(feature = "faiss-cxx")]
    {
        // Generate C++ bindings using cxx-build
        // Note: This requires faiss library to be installed
        // On macOS: brew install faiss
        // On Linux: apt-get install libfaiss-dev
        let result = cxx_build::bridge("src/faiss/ffi.rs")
            .flag("-std=c++17")
            .flag("-O3")
            .include("/opt/homebrew/include")
            .include("/usr/local/include")
            .include("/usr/include")
            .compile("knowhere-faiss");
        
        match result {
            Ok(_) => println!("cargo:rerun-if-changed=src/faiss/ffi.rs"),
            Err(e) => {
                // If Faiss is not installed, continue without it
                println!("cargo:warning=Faiss not found, FFI will be stub: {}", e);
            }
        }
    }
    
    println!("cargo:rerun-if-changed=src/");
}
