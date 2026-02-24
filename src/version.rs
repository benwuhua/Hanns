//! 版本信息

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// 构建信息
pub struct BuildInfo {
    pub version: &'static str,
    pub name: &'static str,
    pub rustc: &'static str,
}

impl BuildInfo {
    pub fn new() -> Self {
        Self {
            version: VERSION,
            name: NAME,
            rustc: env!("CARGO_PKG_RUST_VERSION"),
        }
    }
}

impl Default for BuildInfo {
    fn default() -> Self { Self::new() }
}

impl std::fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} v{} (rust {})", self.name, self.version, self.rustc)
    }
}

/// 配置
#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub metric: String,
    pub index_type: String,
}

impl Config {
    pub fn new(dim: usize) -> Self {
        Self { dim, metric: "L2".into(), index_type: "HNSW".into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        let info = BuildInfo::new();
        assert!(!info.version.is_empty());
    }
    
    #[test]
    fn test_config() {
        let cfg = Config::new(128);
        assert_eq!(cfg.dim, 128);
    }
}
