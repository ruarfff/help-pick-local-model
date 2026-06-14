use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Serialize, de::DeserializeOwned};

#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    pub fn new() -> Result<Self> {
        let dirs = ProjectDirs::from("dev", "ruarfff", "mlx-model-picker")
            .context("failed to locate a local cache directory")?;
        let root = dirs.cache_dir().to_path_buf();
        Ok(Self { root })
    }

    pub async fn read_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let path = self.root.join(key);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error).with_context(|| format!("failed to read {path:?}")),
        };
        let value = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse cached JSON at {path:?}"))?;
        Ok(Some(value))
    }

    pub async fn write_json<T: Serialize>(&self, key: &str, value: &T) -> Result<()> {
        tokio::fs::create_dir_all(&self.root)
            .await
            .with_context(|| format!("failed to create cache directory {:?}", self.root))?;
        let path = self.root.join(key);
        let bytes = serde_json::to_vec_pretty(value).context("failed to encode cache JSON")?;
        tokio::fs::write(&path, bytes)
            .await
            .with_context(|| format!("failed to write cache file {path:?}"))?;
        Ok(())
    }

    pub fn path(&self) -> &PathBuf {
        &self.root
    }
}
