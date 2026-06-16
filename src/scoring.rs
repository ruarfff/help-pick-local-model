use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    hf::HfModel,
    machine::MachineInfo,
    parser::{ParsedModel, parse_model_id},
    runtime::{DependencyStatus, RuntimeChoice, RuntimeDecision, RuntimeKind, infer_runtime},
    use_case::{UseCaseId, UseCaseProfile},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitStatus {
    Excellent,
    Good,
    Tight,
    Borderline,
    Reject,
}

impl std::fmt::Display for FitStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Excellent => write!(f, "Excellent"),
            Self::Good => write!(f, "Good"),
            Self::Tight => write!(f, "Tight"),
            Self::Borderline => write!(f, "Borderline"),
            Self::Reject => write!(f, "Reject"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEstimate {
    pub model_weights_gb: f64,
    pub kv_cache_gb: f64,
    pub runtime_headroom_gb: f64,
    pub required_gb: f64,
    pub fit: FitStatus,
}

#[derive(Debug, Clone)]
pub struct PickerOptions {
    pub family: Option<String>,
    pub top: usize,
    pub context_tokens: u64,
    pub concurrent_sessions: u64,
    pub runtime: RuntimeChoice,
    pub use_case: UseCaseId,
    pub include_base: bool,
    pub include_assistant: bool,
    pub include_diffusion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub total: f64,
    pub items: Vec<ScoreItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreItem {
    pub label: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedModel {
    pub model: HfModel,
    pub use_case: UseCaseId,
    pub parsed: ParsedModel,
    pub memory: MemoryEstimate,
    pub runtime: RuntimeDecision,
    pub score: ScoreBreakdown,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub rejected: bool,
}

impl RankedModel {
    pub fn suggested_command(&self) -> Option<String> {
        self.runtime.runtime.map(|runtime| {
            format!(
                "{} --model {} --port 8080",
                runtime.command(),
                self.model.id
            )
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub use_case: UseCaseId,
    pub recommended: Option<RankedModel>,
    pub ranked: Vec<RankedModel>,
    pub rejected: Vec<RankedModel>,
}

pub fn estimate_memory(
    total_params_b: Option<f64>,
    active_params_b: Option<f64>,
    quantization: Option<&str>,
    context_tokens: u64,
    concurrent_sessions: u64,
    total_ram_gb: f64,
) -> MemoryEstimate {
    let params_b = active_params_b.or(total_params_b).unwrap_or(7.0);
    let total_b = total_params_b.unwrap_or(params_b);
    let quantization = quantization.map(str::to_lowercase);
    let bytes_per_param = match quantization.as_deref() {
        Some("4bit") | Some("nvfp4") => 0.55,
        Some("8bit") | Some("mxfp8") => 1.20,
        Some("bf16") | Some("fp16") => 2.05,
        Some(value) if value.ends_with("bit") => value
            .trim_end_matches("bit")
            .parse::<f64>()
            .map(|bits| (bits / 8.0) * 1.15)
            .unwrap_or(0.85),
        _ => 0.85,
    };
    let model_weights_gb = total_b * bytes_per_param;
    let context_scale = context_tokens as f64 / 16_000.0;
    let kv_cache_gb = params_b * 0.08 * context_scale * concurrent_sessions.max(1) as f64;
    let runtime_headroom_gb = 4.0;
    let required_gb = model_weights_gb + kv_cache_gb + runtime_headroom_gb;
    let fit = classify_fit(required_gb, total_ram_gb);

    MemoryEstimate {
        model_weights_gb,
        kv_cache_gb,
        runtime_headroom_gb,
        required_gb,
        fit,
    }
}

pub fn classify_fit(required_gb: f64, total_ram_gb: f64) -> FitStatus {
    let ratio = required_gb / total_ram_gb.max(1.0);
    if ratio <= 0.50 {
        FitStatus::Excellent
    } else if ratio <= 0.65 {
        FitStatus::Good
    } else if ratio <= 0.75 {
        FitStatus::Tight
    } else if ratio <= 0.85 {
        FitStatus::Borderline
    } else {
        FitStatus::Reject
    }
}

pub fn rank_models(
    models: Vec<HfModel>,
    machine: &MachineInfo,
    options: &PickerOptions,
    deps: &DependencyStatus,
) -> RankedResult {
    let mut all: Vec<RankedModel> = models
        .into_iter()
        .map(|model| score_model(model, machine, options, deps))
        .collect();

    all.sort_by(|left, right| {
        right
            .score
            .total
            .partial_cmp(&left.score.total)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.model.downloads.cmp(&left.model.downloads))
    });

    let (mut ranked, rejected): (Vec<_>, Vec<_>) =
        all.into_iter().partition(|candidate| !candidate.rejected);
    let recommended = ranked.first().cloned();
    ranked.truncate(options.top);

    RankedResult {
        use_case: options.use_case,
        recommended,
        ranked,
        rejected,
    }
}

pub fn score_model(
    model: HfModel,
    machine: &MachineInfo,
    options: &PickerOptions,
    deps: &DependencyStatus,
) -> RankedModel {
    let parsed = parse_model_id(&model.id);
    let memory = estimate_memory(
        parsed.total_params_b,
        parsed.active_params_b,
        parsed.quantization.as_deref(),
        options.context_tokens,
        options.concurrent_sessions,
        machine.total_memory_gb,
    );
    let runtime = infer_runtime(&parsed, model.config.as_ref(), options.runtime);
    let mut items = Vec::new();
    let mut reasons = Vec::new();
    let mut warnings = runtime.warnings.clone();
    let mut rejected = false;

    if let Some(family) = options.family.as_deref()
        && parsed.family != family
    {
        rejected = true;
        warnings.push(format!("filtered out because family is {}", parsed.family));
    }
    if parsed.is_base && !options.include_base {
        rejected = true;
        warnings.push("base model excluded by default".to_string());
    }
    if parsed.is_assistant && !options.include_assistant {
        rejected = true;
        warnings.push("assistant/draft model excluded by default".to_string());
    }
    if parsed.is_diffusion && !options.include_diffusion {
        rejected = true;
        warnings.push("diffusion model excluded by default".to_string());
    }
    if memory.fit == FitStatus::Reject {
        rejected = true;
        warnings.push(format!(
            "estimated memory {:.1} GB exceeds safe fit threshold",
            memory.required_gb
        ));
    }
    if !runtime.is_compatible {
        rejected = true;
        warnings.push("runtime choice is incompatible with inferred runtime".to_string());
    }
    if !deps.supports(runtime.runtime) {
        warnings.push("local runtime dependencies appear incomplete".to_string());
    }
    let profile = UseCaseProfile::for_id(options.use_case);
    let profile_score = profile.score_adjustments(&parsed, &runtime);
    for (label, value) in profile_score.items {
        items.push(ScoreItem {
            label: label.to_string(),
            value,
        });
    }
    for reason in profile_score.reasons {
        reasons.push(reason.to_string());
    }
    for warning in profile_score.warnings {
        warnings.push(warning.to_string());
    }

    add_score(
        &mut items,
        &mut reasons,
        parsed.is_instruction_tuned,
        18.0,
        "instruction tuned",
    );
    if let Some(params) = parsed.total_params_b {
        let value = (params.ln() * 8.0).min(30.0);
        items.push(ScoreItem {
            label: "model size".to_string(),
            value,
        });
        reasons.push(format!("{params:.0}B total parameters"));
    }
    if parsed.is_qat {
        items.push(ScoreItem {
            label: "QAT".to_string(),
            value: 5.0,
        });
        reasons.push("QAT quantization signal".to_string());
    }
    if parsed.is_optiq {
        items.push(ScoreItem {
            label: "OptiQ".to_string(),
            value: 5.0,
        });
        reasons.push("OptiQ optimization signal".to_string());
    }
    if parsed
        .active_params_b
        .zip(parsed.total_params_b)
        .is_some_and(|(active, total)| active < total)
    {
        items.push(ScoreItem {
            label: "MoE active params".to_string(),
            value: 6.0,
        });
        reasons.push("MoE uses fewer active params than total params".to_string());
    }
    match memory.fit {
        FitStatus::Excellent => add_fixed(&mut items, &mut reasons, 16.0, "excellent memory fit"),
        FitStatus::Good => add_fixed(&mut items, &mut reasons, 10.0, "good memory fit"),
        FitStatus::Tight => add_fixed(&mut items, &mut reasons, 2.0, "tight memory fit"),
        FitStatus::Borderline => {
            add_fixed(&mut items, &mut reasons, -10.0, "borderline memory fit")
        }
        FitStatus::Reject => add_fixed(&mut items, &mut reasons, -100.0, "memory rejected"),
    }
    if parsed.quantization.as_deref() == Some("8bit")
        && matches!(memory.fit, FitStatus::Excellent | FitStatus::Good)
    {
        add_fixed(&mut items, &mut reasons, 4.0, "8-bit fits comfortably");
    }
    if runtime.is_compatible && runtime.runtime.is_some() {
        add_fixed(&mut items, &mut reasons, 10.0, "runtime compatible");
    } else {
        add_fixed(
            &mut items,
            &mut reasons,
            -80.0,
            "runtime incompatible or unknown",
        );
    }
    if !deps.supports(runtime.runtime) {
        add_fixed(
            &mut items,
            &mut reasons,
            -15.0,
            "missing local runtime dependency",
        );
    }
    if let Some(downloads) = model.downloads {
        items.push(ScoreItem {
            label: "downloads confidence".to_string(),
            value: ((downloads as f64 + 1.0).ln() / 2.0).min(8.0),
        });
    }
    if let Some(likes) = model.likes {
        items.push(ScoreItem {
            label: "likes confidence".to_string(),
            value: ((likes as f64 + 1.0).ln() / 3.0).min(5.0),
        });
    }
    if model
        .last_modified
        .is_some_and(|modified| Utc::now() - modified < Duration::days(180))
    {
        items.push(ScoreItem {
            label: "recent update".to_string(),
            value: 2.0,
        });
    }
    if model.config.is_none() {
        add_fixed(&mut items, &mut reasons, -3.0, "config unavailable");
    }
    if parsed.is_base && !options.include_base {
        add_fixed(&mut items, &mut reasons, -40.0, "base model excluded");
    }
    if parsed.is_assistant && !options.include_assistant {
        add_fixed(
            &mut items,
            &mut reasons,
            -30.0,
            "assistant/draft model excluded",
        );
    }
    if parsed.is_diffusion && !options.include_diffusion {
        add_fixed(&mut items, &mut reasons, -40.0, "diffusion model excluded");
    }

    let total = items.iter().map(|item| item.value).sum();
    RankedModel {
        model,
        use_case: options.use_case,
        parsed,
        memory,
        runtime,
        score: ScoreBreakdown { total, items },
        reasons,
        warnings,
        rejected,
    }
}

fn add_score(
    items: &mut Vec<ScoreItem>,
    reasons: &mut Vec<String>,
    condition: bool,
    value: f64,
    label: &str,
) {
    if condition {
        add_fixed(items, reasons, value, label);
    }
}

fn add_fixed(items: &mut Vec<ScoreItem>, reasons: &mut Vec<String>, value: f64, label: &str) {
    items.push(ScoreItem {
        label: label.to_string(),
        value,
    });
    if value > 0.0 {
        reasons.push(label.to_string());
    }
}

pub fn runtime_label(runtime: Option<RuntimeKind>) -> String {
    runtime
        .map(|runtime| runtime.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
