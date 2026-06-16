use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

static TOTAL_PARAMS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*b\b").unwrap());
static ACTIVE_PARAMS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\ba(\d+(?:\.\d+)?)\s*b\b").unwrap());
static EFFECTIVE_PARAMS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\be(\d+(?:\.\d+)?)\s*b\b").unwrap());
static QWEN_GENERATION_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)qwen\s*(\d+(?:\.\d+)?)").unwrap());
static PARAM_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^[ae]?\d+(?:\.\d+)?b$").unwrap());

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParsedModel {
    pub repo_id: String,
    pub author: Option<String>,
    pub model_id: String,
    pub family: String,
    pub generation: Option<String>,
    pub total_params_b: Option<f64>,
    pub active_params_b: Option<f64>,
    pub effective_params_b: Option<f64>,
    pub is_instruction_tuned: bool,
    pub is_coding: bool,
    pub is_assistant: bool,
    pub is_diffusion: bool,
    pub is_base: bool,
    pub quantization: Option<String>,
    pub is_qat: bool,
    pub is_optiq: bool,
    pub unknown_tokens: Vec<String>,
}

pub fn parse_model_id(repo_id: &str) -> ParsedModel {
    let (author, model_id) = repo_id
        .split_once('/')
        .map(|(author, model)| (Some(author.to_string()), model.to_string()))
        .unwrap_or((None, repo_id.to_string()));
    let lower = model_id.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect();

    let family = infer_family(&lower);
    let generation = infer_generation(&family, &tokens, &lower);
    let total_params_b = capture_param(&lower, &TOTAL_PARAMS_RE);
    let active_params_b = capture_param(&lower, &ACTIVE_PARAMS_RE);
    let effective_params_b = capture_param(&lower, &EFFECTIVE_PARAMS_RE);
    let quantization = infer_quantization(&lower);
    let is_instruction_tuned = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "it" | "instruct" | "instruction" | "chat" | "chatml"
        )
    });
    let is_coding = tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "code" | "coder" | "coding" | "codestral" | "deepcoder"
        )
    });
    let is_assistant = tokens
        .iter()
        .any(|token| matches!(token.as_str(), "assistant" | "draft" | "speculative"));
    let is_diffusion = lower.contains("diffusion");
    let is_qat = tokens.iter().any(|token| token == "qat");
    let is_optiq = tokens.iter().any(|token| token == "optiq");
    let is_base = !is_instruction_tuned
        && !is_assistant
        && !is_diffusion
        && tokens.iter().any(|token| token == "base");
    let unknown_tokens = tokens
        .iter()
        .filter(|token| {
            !is_known_token(
                token,
                &family,
                generation.as_deref(),
                quantization.as_deref(),
            )
        })
        .cloned()
        .collect();

    ParsedModel {
        repo_id: repo_id.to_string(),
        author,
        model_id,
        family,
        generation,
        total_params_b,
        active_params_b,
        effective_params_b,
        is_instruction_tuned,
        is_coding,
        is_assistant,
        is_diffusion,
        is_base,
        quantization,
        is_qat,
        is_optiq,
        unknown_tokens,
    }
}

fn infer_family(lower: &str) -> String {
    for (needle, family) in [
        ("qwen", "qwen"),
        ("gemma", "gemma"),
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("phi", "phi"),
        ("deepseek", "deepseek"),
        ("mixtral", "mistral"),
    ] {
        if lower.contains(needle) {
            return family.to_string();
        }
    }
    "unknown".to_string()
}

fn infer_generation(family: &str, tokens: &[String], lower: &str) -> Option<String> {
    match family {
        "qwen" => QWEN_GENERATION_RE
            .captures(lower)
            .and_then(|captures| captures.get(1).map(|value| value.as_str().to_string())),
        "llama" => tokens
            .iter()
            .position(|token| token == "llama")
            .and_then(|index| tokens.get(index + 1))
            .filter(|token| token.chars().next().is_some_and(|c| c.is_ascii_digit()))
            .cloned(),
        _ => tokens
            .iter()
            .position(|token| token == family)
            .and_then(|index| tokens.get(index + 1))
            .filter(|token| token.chars().all(|c| c.is_ascii_digit() || c == '.'))
            .cloned(),
    }
}

fn capture_param(lower: &str, re: &Regex) -> Option<f64> {
    re.captures(lower)
        .and_then(|captures| captures.get(1))
        .and_then(|value| value.as_str().parse::<f64>().ok())
}

fn infer_quantization(lower: &str) -> Option<String> {
    for quant in ["4bit", "8bit", "bf16", "fp16", "nvfp4", "mxfp8"] {
        if lower.contains(quant) {
            return Some(quant.to_string());
        }
    }
    Regex::new(r"(?i)\b(\d+)bit\b")
        .ok()
        .and_then(|re| re.captures(lower))
        .and_then(|captures| {
            captures
                .get(1)
                .map(|value| format!("{}bit", value.as_str()))
        })
}

fn is_known_token(
    token: &str,
    family: &str,
    generation: Option<&str>,
    quantization: Option<&str>,
) -> bool {
    token == family
        || generation == Some(token)
        || quantization == Some(token)
        || matches!(
            token,
            "it" | "instruct"
                | "instruction"
                | "chat"
                | "chatml"
                | "code"
                | "coder"
                | "coding"
                | "assistant"
                | "draft"
                | "speculative"
                | "diffusion"
                | "qat"
                | "optiq"
                | "base"
        )
        || PARAM_TOKEN_RE.is_match(token)
}
