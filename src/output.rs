use comfy_table::{Cell, Table, presets::UTF8_FULL};
use serde::Serialize;

use crate::{
    machine::MachineInfo,
    runtime::DependencyStatus,
    scoring::{RankedModel, RankedResult, runtime_label},
    use_case::{UseCaseId, UseCaseProfile},
};

#[derive(Debug, Serialize)]
pub struct JsonOutput<'a> {
    pub use_case: UseCaseId,
    pub machine: &'a MachineInfo,
    pub runtime_status: &'a DependencyStatus,
    pub recommended: &'a Option<RankedModel>,
    pub ranked: &'a Vec<RankedModel>,
    pub rejected: &'a Vec<RankedModel>,
}

pub fn print_human(machine: &MachineInfo, deps: &DependencyStatus, result: &RankedResult) {
    let profile = UseCaseProfile::for_id(result.use_case);
    println!("Use case: {} ({})", profile.id(), profile.description());
    println!();
    println!("Machine:");
    println!("  OS: {}", machine.os.as_deref().unwrap_or("unknown macOS"));
    println!("  Arch: {}", machine.arch);
    println!("  Chip: {}", machine.chip.as_deref().unwrap_or("unknown"));
    println!("  Total memory: {:.1} GB", machine.total_memory_gb);
    println!(
        "  Estimated LLM budget: {:.1} GB",
        machine.estimated_llm_budget_gb
    );
    println!("  Local runtime status: {}", deps.summary());
    println!();

    if let Some(recommended) = &result.recommended {
        println!("Recommended:");
        println!("  {}", recommended.model.id);
        println!();
        println!("Why:");
        for reason in recommended.reasons.iter().take(6) {
            println!("  - {reason}");
        }
        let warnings = merged_warnings(recommended, deps);
        if !warnings.is_empty() {
            println!();
            println!("Warnings:");
            for warning in warnings {
                println!("  - {warning}");
            }
        }
        if let Some(command) = recommended.suggested_command() {
            println!();
            println!("Suggested command:");
            println!("  {command}");
        }
        print_profile_note(profile);
    } else {
        println!("Recommended:");
        println!("  No compatible model found.");
    }
    println!();
    println!("Ranked candidates:");
    println!("{}", ranked_table(&result.ranked));
}

pub fn print_explain(candidate: &RankedModel, deps: &DependencyStatus) {
    let profile = UseCaseProfile::for_id(candidate.use_case);
    println!("Model:");
    println!("  {}", candidate.model.id);
    println!();
    println!("Use case: {} ({})", profile.id(), profile.description());
    println!();
    println!("Parsed name metadata:");
    println!("  Family: {}", candidate.parsed.family);
    println!(
        "  Params: {}",
        candidate
            .parsed
            .total_params_b
            .map(|value| format!("{value:.1}B"))
            .unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "  Active params: {}",
        candidate
            .parsed
            .active_params_b
            .map(|value| format!("{value:.1}B"))
            .unwrap_or_else(|| "n/a".to_string())
    );
    println!(
        "  Quantization: {}",
        candidate
            .parsed
            .quantization
            .as_deref()
            .unwrap_or("unknown")
    );
    println!(
        "  Instruction tuned: {}",
        candidate.parsed.is_instruction_tuned
    );
    println!("  Coding oriented: {}", candidate.parsed.is_coding);
    println!("  Assistant/draft: {}", candidate.parsed.is_assistant);
    println!("  Diffusion: {}", candidate.parsed.is_diffusion);
    println!("  QAT: {}", candidate.parsed.is_qat);
    println!("  OptiQ: {}", candidate.parsed.is_optiq);
    println!();
    println!("Config metadata:");
    if let Some(config) = &candidate.model.config {
        println!(
            "  model_type: {}",
            config.model_type.as_deref().unwrap_or("unknown")
        );
        println!("  architectures: {}", config.architectures.join(", "));
    } else {
        println!("  config.json unavailable");
    }
    println!();
    println!("Memory estimate:");
    println!("  Weights: {:.1} GB", candidate.memory.model_weights_gb);
    println!("  KV cache: {:.1} GB", candidate.memory.kv_cache_gb);
    println!(
        "  Runtime headroom: {:.1} GB",
        candidate.memory.runtime_headroom_gb
    );
    println!("  Required: {:.1} GB", candidate.memory.required_gb);
    println!("  Fit: {}", candidate.memory.fit);
    println!();
    println!("Runtime decision:");
    println!("  Runtime: {}", runtime_label(candidate.runtime.runtime));
    println!("  Compatible: {}", candidate.runtime.is_compatible);
    println!();
    println!("Dependency status:");
    println!("  {}", deps.summary());
    println!();
    println!("Score breakdown:");
    for item in &candidate.score.items {
        println!("  {:+.1} {}", item.value, item.label);
    }
    println!("  = {:.1}", candidate.score.total);
    let warnings = merged_warnings(candidate, deps);
    if !warnings.is_empty() {
        println!();
        println!("Warnings/rejection reasons:");
        for warning in warnings {
            println!("  - {warning}");
        }
    }
    if let Some(command) = candidate.suggested_command() {
        println!();
        println!("Suggested server command:");
        println!("  {command}");
    }
    print_profile_note(profile);
}

pub fn json_string<'a>(
    machine: &'a MachineInfo,
    deps: &'a DependencyStatus,
    result: &'a RankedResult,
) -> serde_json::Result<String> {
    serde_json::to_string_pretty(&JsonOutput {
        use_case: result.use_case,
        machine,
        runtime_status: deps,
        recommended: &result.recommended,
        ranked: &result.ranked,
        rejected: &result.rejected,
    })
}

fn print_profile_note(profile: UseCaseProfile) {
    let note = profile.output_note();
    if !note.is_empty() {
        println!();
        println!("{note}");
    }
}

fn ranked_table(ranked: &[RankedModel]) -> Table {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        "#", "Model", "Family", "Params", "Quant", "Runtime", "Fit", "Score",
    ]);
    for (index, candidate) in ranked.iter().enumerate() {
        table.add_row(vec![
            Cell::new(index + 1),
            Cell::new(&candidate.model.id),
            Cell::new(&candidate.parsed.family),
            Cell::new(
                candidate
                    .parsed
                    .total_params_b
                    .map(|value| format!("{value:.0}B"))
                    .unwrap_or_else(|| "?".to_string()),
            ),
            Cell::new(candidate.parsed.quantization.as_deref().unwrap_or("?")),
            Cell::new(runtime_label(candidate.runtime.runtime)),
            Cell::new(candidate.memory.fit.to_string()),
            Cell::new(format!("{:.1}", candidate.score.total)),
        ]);
    }
    table
}

fn merged_warnings(candidate: &RankedModel, deps: &DependencyStatus) -> Vec<String> {
    let mut warnings = candidate.warnings.clone();
    warnings.extend(deps.warnings.clone());
    warnings.sort();
    warnings.dedup();
    warnings
}
