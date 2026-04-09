//! Build script for hanns with Faiss support
#[cfg(feature = "faiss-cxx")]
fn first_existing_include<'a>(paths: &'a [&'a str]) -> Option<&'a str> {
    paths
        .iter()
        .copied()
        .find(|path| std::path::Path::new(path).exists())
}

fn main() {
    #[cfg(feature = "faiss-cxx")]
    {
        let include_paths = [
            "/opt/homebrew/include",
            "/usr/local/include",
            "/usr/include",
        ];
        let mut bridge = cxx_build::bridge("src/faiss/ffi.rs");

        bridge.flag("-std=c++17").flag("-O3");

        if let Some(include_path) = first_existing_include(&include_paths) {
            bridge.include(include_path);
        }

        bridge.compile("knowhere-faiss");
        println!("cargo:rerun-if-changed=src/faiss/ffi.rs");
    }

    println!("cargo:rerun-if-changed=src/");
}
