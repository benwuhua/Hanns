use crate::kernel::IndexFamily;

use super::snapshot::LoadMode;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexManifest {
    pub version: u32,
    pub family: IndexFamily,
    pub variant: String,
    pub dim: usize,
    pub metric: String,
    pub count: usize,
    #[serde(default = "default_supported_load_modes")]
    pub supported_load_modes: Vec<LoadMode>,
    #[serde(default)]
    pub features: ManifestFeatures,
    pub sections: Vec<SectionDescriptor>,
}

impl IndexManifest {
    pub fn supports_load_mode(&self, mode: LoadMode) -> bool {
        self.supported_load_modes.contains(&mode)
    }

    pub fn has_raw_vectors(&self) -> bool {
        self.features.raw_vectors
    }

    pub fn has_graph_payload(&self) -> bool {
        self.features.graph_payload
    }

    pub fn has_quantized_payload(&self) -> bool {
        self.features.quantized_payload
    }

    pub fn has_compressed_vectors(&self) -> bool {
        self.features.compressed_vectors
    }
}

pub fn default_supported_load_modes() -> Vec<LoadMode> {
    vec![LoadMode::OwnedMemory]
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ManifestFeatures {
    #[serde(default)]
    pub raw_vectors: bool,
    #[serde(default)]
    pub graph_payload: bool,
    #[serde(default)]
    pub quantized_payload: bool,
    #[serde(default)]
    pub compressed_vectors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SectionDescriptor {
    pub name: String,
    pub len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}
