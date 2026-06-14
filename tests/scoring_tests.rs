use mlx_model_picker::hf::HfModel;
use mlx_model_picker::machine::MachineInfo;
use mlx_model_picker::runtime::{DependencyStatus, RuntimeChoice, RuntimeKind};
use mlx_model_picker::scoring::{FitStatus, PickerOptions, estimate_memory, rank_models};

fn test_machine(total_memory_gb: f64) -> MachineInfo {
    MachineInfo {
        os: Some("macOS test".to_string()),
        arch: "arm64".to_string(),
        chip: Some("Apple M3 Max".to_string()),
        total_memory_gb,
        available_memory_gb: Some(total_memory_gb * 0.5),
        estimated_llm_budget_gb: total_memory_gb - 8.0_f64.max(total_memory_gb * 0.25),
        free_disk_gb: Some(500.0),
        is_apple_silicon: true,
    }
}

#[test]
fn memory_estimate_and_fit_classification_follow_thresholds() {
    let excellent = estimate_memory(Some(7.0), None, Some("4bit"), 16_000, 1, 64.0);
    let tight = estimate_memory(Some(31.0), None, Some("8bit"), 16_000, 1, 64.0);
    let reject = estimate_memory(Some(70.0), None, Some("bf16"), 16_000, 1, 64.0);

    assert_eq!(excellent.fit, FitStatus::Excellent);
    assert_eq!(tight.fit, FitStatus::Tight);
    assert_eq!(reject.fit, FitStatus::Reject);
}

#[test]
fn ranking_prefers_coding_instruction_model_over_base() {
    let machine = test_machine(64.0);
    let options = PickerOptions {
        family: None,
        top: 10,
        context_tokens: 16_000,
        concurrent_sessions: 1,
        runtime: RuntimeChoice::Auto,
        include_base: false,
        include_assistant: false,
        include_diffusion: false,
    };
    let deps = DependencyStatus::assume_available();
    let models = vec![
        HfModel::minimal("mlx-community/Some-Base-7B-4bit"),
        HfModel::minimal("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
    ];

    let ranked = rank_models(models, &machine, &options, &deps);

    assert_eq!(
        ranked.recommended.as_ref().map(|r| r.model.id.as_str()),
        Some("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
    );
}

#[test]
fn forced_runtime_keeps_vlm_only_model_out_of_top_slot() {
    let machine = test_machine(64.0);
    let options = PickerOptions {
        family: Some("gemma".to_string()),
        top: 10,
        context_tokens: 16_000,
        concurrent_sessions: 1,
        runtime: RuntimeChoice::MlxLm,
        include_base: false,
        include_assistant: false,
        include_diffusion: false,
    };
    let deps = DependencyStatus::assume_available();
    let mut vlm = HfModel::minimal("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    vlm.config = Some(mlx_model_picker::hf::ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    });
    let models = vec![
        vlm,
        HfModel::minimal("mlx-community/gemma-3-27b-it-qat-4bit"),
    ];

    let ranked = rank_models(models, &machine, &options, &deps);

    assert_ne!(
        ranked.recommended.as_ref().map(|r| r.model.id.as_str()),
        Some("mlx-community/gemma-4-12B-it-OptiQ-4bit")
    );
    assert_eq!(
        ranked.recommended.as_ref().and_then(|r| r.runtime.runtime),
        Some(RuntimeKind::MlxLm)
    );
}
