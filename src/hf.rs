use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache::Cache;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfModel {
    pub id: String,
    #[serde(default)]
    pub downloads: Option<u64>,
    #[serde(default)]
    pub likes: Option<u64>,
    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<DateTime<Utc>>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(skip)]
    pub config: Option<ModelConfig>,
}

impl HfModel {
    pub fn minimal(id: &str) -> Self {
        Self {
            id: id.to_string(),
            downloads: None,
            likes: None,
            last_modified: None,
            tags: Vec::new(),
            config: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub raw: Value,
}

#[derive(Debug, Clone)]
pub struct HfClient {
    client: reqwest::Client,
    cache: Cache,
    refresh: bool,
}

impl HfClient {
    pub fn new(cache: Cache, refresh: bool) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("mlx-model-picker/0.1")
            .build()
            .context("failed to build Hugging Face HTTP client")?;
        Ok(Self {
            client,
            cache,
            refresh,
        })
    }

    pub async fn fetch_models(&self, author: &str, limit: usize) -> Result<Vec<HfModel>> {
        let cache_key = format!("models-{author}-{limit}.json");
        if !self.refresh
            && let Some(models) = self.cache.read_json::<Vec<HfModel>>(&cache_key).await?
        {
            return Ok(models);
        }

        let url = format!(
            "https://huggingface.co/api/models?author={author}&sort=downloads&limit={limit}"
        );
        let models = self
            .client
            .get(url)
            .send()
            .await
            .context("failed to fetch Hugging Face model list")?
            .error_for_status()
            .context("Hugging Face model list request failed")?
            .json::<Vec<HfModel>>()
            .await
            .context("failed to decode Hugging Face model list")?;
        self.cache.write_json(&cache_key, &models).await?;
        Ok(models)
    }

    pub async fn fetch_config(&self, repo_id: &str) -> Result<Option<ModelConfig>> {
        let cache_key = format!("config-{}.json", repo_id.replace('/', "__"));
        if !self.refresh
            && let Some(config) = self.cache.read_json::<ModelConfig>(&cache_key).await?
        {
            return Ok(Some(config));
        }

        let url = format!("https://huggingface.co/{repo_id}/resolve/main/config.json");
        let response = self.client.get(url).send().await;
        let response = match response {
            Ok(response) => response,
            Err(_) => return Ok(None),
        };
        if !response.status().is_success() {
            return Ok(None);
        }
        let raw = match response.json::<Value>().await {
            Ok(raw) => raw,
            Err(_) => return Ok(None),
        };
        let config = ModelConfig {
            model_type: raw
                .get("model_type")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            architectures: raw
                .get("architectures")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            raw,
        };
        self.cache.write_json(&cache_key, &config).await?;
        Ok(Some(config))
    }

    pub async fn attach_configs(&self, models: Vec<HfModel>, limit: usize) -> Vec<HfModel> {
        let fetch_count = models.len().min(limit);
        let mut configured: Vec<HfModel> = stream::iter(models.into_iter().enumerate())
            .map(|(index, mut model)| async move {
                if index < fetch_count {
                    model.config = self.fetch_config(&model.id).await.ok().flatten();
                }
                model
            })
            .buffer_unordered(8)
            .collect()
            .await;
        configured.sort_by_key(|model| std::cmp::Reverse(model.downloads.unwrap_or(0)));
        configured
    }
}
