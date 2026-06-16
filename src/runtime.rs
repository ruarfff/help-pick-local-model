use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use crate::{hf::ModelConfig, parser::ParsedModel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeChoice {
    Auto,
    #[clap(name = "mlx-lm")]
    MlxLm,
    #[clap(name = "mlx-vlm")]
    MlxVlm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeKind {
    MlxLm,
    MlxVlm,
}

impl RuntimeKind {
    pub fn command(self) -> &'static str {
        match self {
            Self::MlxLm => "mlx_lm.server",
            Self::MlxVlm => "mlx_vlm.server",
        }
    }
}

impl std::fmt::Display for RuntimeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MlxLm => write!(f, "mlx-lm"),
            Self::MlxVlm => write!(f, "mlx-vlm"),
        }
    }
}

impl std::fmt::Display for RuntimeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::MlxLm => write!(f, "mlx-lm"),
            Self::MlxVlm => write!(f, "mlx-vlm"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeDecision {
    pub runtime: Option<RuntimeKind>,
    pub inferred_runtime: Option<RuntimeKind>,
    pub is_compatible: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyStatus {
    pub checked: bool,
    pub python: bool,
    pub uv: bool,
    pub pip: bool,
    pub mlx: bool,
    pub mlx_lm: bool,
    pub mlx_vlm: bool,
    pub mlx_lm_server: bool,
    pub mlx_vlm_server: bool,
    pub warnings: Vec<String>,
}

impl DependencyStatus {
    pub fn not_checked() -> Self {
        Self {
            checked: false,
            python: false,
            uv: false,
            pip: false,
            mlx: false,
            mlx_lm: false,
            mlx_vlm: false,
            mlx_lm_server: false,
            mlx_vlm_server: false,
            warnings: vec!["local runtime dependency check skipped".to_string()],
        }
    }

    pub fn assume_available() -> Self {
        Self {
            checked: true,
            python: true,
            uv: true,
            pip: true,
            mlx: true,
            mlx_lm: true,
            mlx_vlm: true,
            mlx_lm_server: true,
            mlx_vlm_server: true,
            warnings: Vec::new(),
        }
    }

    pub fn supports(&self, runtime: Option<RuntimeKind>) -> bool {
        if !self.checked {
            return true;
        }
        match runtime {
            Some(RuntimeKind::MlxLm) => self.mlx && self.mlx_lm && self.mlx_lm_server,
            Some(RuntimeKind::MlxVlm) => self.mlx && self.mlx_vlm && self.mlx_vlm_server,
            None => false,
        }
    }

    pub fn summary(&self) -> String {
        if !self.checked {
            return "not checked".to_string();
        }
        let missing = self.missing();
        if missing.is_empty() {
            "ready".to_string()
        } else {
            format!("missing {}", missing.join(", "))
        }
    }

    pub fn missing(&self) -> Vec<&'static str> {
        [
            ("python", self.python),
            ("uv", self.uv),
            ("pip", self.pip),
            ("mlx", self.mlx),
            ("mlx_lm", self.mlx_lm),
            ("mlx_vlm", self.mlx_vlm),
            ("mlx_lm.server", self.mlx_lm_server),
            ("mlx_vlm.server", self.mlx_vlm_server),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect()
    }
}

pub fn infer_runtime(
    parsed: &ParsedModel,
    config: Option<&ModelConfig>,
    choice: RuntimeChoice,
) -> RuntimeDecision {
    let mut warnings = Vec::new();
    let inferred = infer_runtime_kind(parsed, config, &mut warnings);
    let forced = match choice {
        RuntimeChoice::Auto => inferred,
        RuntimeChoice::MlxLm => Some(RuntimeKind::MlxLm),
        RuntimeChoice::MlxVlm => Some(RuntimeKind::MlxVlm),
    };
    let is_compatible = choice == RuntimeChoice::Auto || inferred.is_none() || forced == inferred;
    if !is_compatible {
        warnings.push(format!(
            "forced runtime {choice} does not match inferred runtime {}",
            inferred
                .map(|runtime| runtime.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ));
    }

    RuntimeDecision {
        runtime: if is_compatible { forced } else { inferred },
        inferred_runtime: inferred,
        is_compatible,
        warnings,
    }
}

fn infer_runtime_kind(
    parsed: &ParsedModel,
    config: Option<&ModelConfig>,
    warnings: &mut Vec<String>,
) -> Option<RuntimeKind> {
    if let Some(config) = config {
        if let Some(model_type) = config.model_type.as_deref() {
            let normalized = model_type.to_lowercase();
            if normalized == "gemma4_unified" {
                warnings.push("config model_type gemma4_unified should use mlx-vlm".to_string());
                return Some(RuntimeKind::MlxVlm);
            }
            if is_vlm_marker(&normalized) {
                return Some(RuntimeKind::MlxVlm);
            }
            if matches!(
                normalized.as_str(),
                "llama" | "mistral" | "qwen2" | "qwen3" | "gemma" | "gemma2" | "gemma3"
            ) {
                return Some(RuntimeKind::MlxLm);
            }
        }
        if config
            .architectures
            .iter()
            .any(|architecture| is_vlm_marker(&architecture.to_lowercase()))
        {
            return Some(RuntimeKind::MlxVlm);
        }
    } else {
        warnings.push("config.json unavailable; runtime inferred from name".to_string());
    }

    let lower = parsed.model_id.to_lowercase();
    if is_vlm_marker(&lower) || parsed.is_diffusion {
        return Some(RuntimeKind::MlxVlm);
    }
    if matches!(
        parsed.family.as_str(),
        "llama" | "mistral" | "qwen" | "gemma"
    ) {
        return Some(RuntimeKind::MlxLm);
    }
    warnings.push("unknown runtime compatibility".to_string());
    None
}

fn is_vlm_marker(value: &str) -> bool {
    ["vlm", "vision", "multimodal", "image", "audio", "clip"]
        .iter()
        .any(|marker| value.contains(marker))
}

pub async fn check_dependencies() -> DependencyStatus {
    let python_bin = find_command(&["python3", "python"]).await;
    let uv = command_exists("uv").await;
    let pip = command_exists("pip").await || command_exists("pip3").await;
    let mut status = DependencyStatus {
        checked: true,
        python: python_bin.is_some(),
        uv,
        pip,
        mlx: false,
        mlx_lm: false,
        mlx_vlm: false,
        mlx_lm_server: false,
        mlx_vlm_server: false,
        warnings: Vec::new(),
    };
    if let Some(python) = python_bin {
        let modules = python_module_status(&python).await;
        status.mlx = modules.contains(&"mlx".to_string());
        status.mlx_lm = modules.contains(&"mlx_lm".to_string());
        status.mlx_vlm = modules.contains(&"mlx_vlm".to_string());
        status.mlx_lm_server = modules.contains(&"mlx_lm.server".to_string());
        status.mlx_vlm_server = modules.contains(&"mlx_vlm.server".to_string());
    }
    if !status.missing().is_empty() {
        status
            .warnings
            .push("install missing packages with: uv pip install -U mlx mlx-lm mlx-vlm huggingface_hub hf_xet".to_string());
    }
    status
}

async fn find_command(names: &[&str]) -> Option<String> {
    for name in names {
        if command_exists(name).await {
            return Some((*name).to_string());
        }
    }
    None
}

async fn command_exists(name: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .is_ok_and(|status| status.success())
}

async fn python_module_status(python: &str) -> Vec<String> {
    let script = r#"
import importlib.util
for mod in ["mlx", "mlx_lm", "mlx_vlm", "mlx_lm.server", "mlx_vlm.server"]:
    if importlib.util.find_spec(mod):
        print(mod)
"#;
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;
    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(ToString::to_string)
            .collect(),
        _ => Vec::new(),
    }
}
