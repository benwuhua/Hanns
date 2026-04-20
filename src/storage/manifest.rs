use crate::kernel::IndexFamily;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexManifest {
    pub version: u32,
    pub family: IndexFamily,
    pub variant: String,
    pub dim: usize,
    pub metric: String,
    pub count: usize,
    pub sections: Vec<SectionDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SectionDescriptor {
    pub name: String,
    pub len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}
