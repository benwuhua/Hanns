use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::api::{KnowhereError, Result};

use super::{IndexArtifactReader, IndexArtifactWriter, IndexManifest};

const MANIFEST_FILE: &str = "manifest.json";
const SECTIONS_DIR: &str = "sections";

#[derive(Debug, Clone)]
pub struct FileArtifactStore {
    root: PathBuf,
    manifest: Option<IndexManifest>,
}

impl FileArtifactStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest = match fs::read(root.join(MANIFEST_FILE)) {
            Ok(bytes) => Some(
                serde_json::from_slice(&bytes)
                    .map_err(|error| KnowhereError::Codec(format!("decode manifest: {error}")))?,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(codec_io("read manifest", error)),
        };

        Ok(Self { root, manifest })
    }

    fn manifest_path(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE)
    }

    fn sections_dir(&self) -> PathBuf {
        self.root.join(SECTIONS_DIR)
    }

    fn section_path(&self, name: &str) -> Result<PathBuf> {
        validate_section_name(name)?;
        Ok(self.sections_dir().join(name))
    }
}

impl IndexArtifactReader for FileArtifactStore {
    fn manifest(&self) -> Result<&IndexManifest> {
        self.manifest
            .as_ref()
            .ok_or_else(|| KnowhereError::Codec("missing manifest".to_string()))
    }

    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let path = self.section_path(name)?;
        match fs::read(path) {
            Ok(bytes) => Ok(Cow::Owned(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(KnowhereError::Codec(format!("missing section: {name}")))
            }
            Err(error) => Err(codec_io("read section", error)),
        }
    }

    fn section_len(&self, name: &str) -> Result<u64> {
        let path = self.section_path(name)?;
        match fs::metadata(path) {
            Ok(metadata) => Ok(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(KnowhereError::Codec(format!("missing section: {name}")))
            }
            Err(error) => Err(codec_io("read section metadata", error)),
        }
    }

    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let path = self.section_path(name)?;
        let section_len = self.section_len(name)?;
        let len = len as u64;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| KnowhereError::Codec("section range overflow".to_string()))?;
        if end > section_len {
            return Err(KnowhereError::Codec(format!(
                "section range out of bounds: {name}"
            )));
        }

        let mut file = File::open(path).map_err(|error| codec_io("open section", error))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| codec_io("seek section", error))?;
        let mut bytes = vec![0; len as usize];
        file.read_exact(&mut bytes)
            .map_err(|error| codec_io("read section range", error))?;
        Ok(bytes)
    }
}

impl IndexArtifactWriter for FileArtifactStore {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let path = self.section_path(name)?;
        fs::create_dir_all(self.sections_dir())
            .map_err(|error| codec_io("create sections directory", error))?;
        fs::write(path, bytes).map_err(|error| codec_io("write section", error))?;
        Ok(())
    }

    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()> {
        fs::create_dir_all(&self.root)
            .map_err(|error| codec_io("create artifact directory", error))?;
        let bytes = serde_json::to_vec_pretty(manifest)
            .map_err(|error| KnowhereError::Codec(format!("encode manifest: {error}")))?;
        fs::write(self.manifest_path(), bytes)
            .map_err(|error| codec_io("write manifest", error))?;
        self.manifest = Some(manifest.clone());
        Ok(())
    }
}

fn validate_section_name(name: &str) -> Result<()> {
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(KnowhereError::Codec(format!(
            "invalid section name: {name}"
        )));
    }

    Ok(())
}

fn codec_io(context: &str, error: std::io::Error) -> KnowhereError {
    KnowhereError::Codec(format!("{context}: {error}"))
}
