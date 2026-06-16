use mlx_model_picker::hf::HfModel;
use mlx_model_picker::machine::MachineInfo;
use mlx_model_picker::output::json_string;
use mlx_model_picker::runtime::{DependencyStatus, RuntimeChoice, RuntimeKind};
use mlx_model_picker::scoring::{PickerOptions, rank_models, score_model};
use mlx_model_picker::use_case::{UseCaseId, UseCaseProfile};

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

fn options(use_case: UseCaseId) -> PickerOptions {
    PickerOptions {
        family: None,
        top: 10,
        context_tokens: 16_000,
        concurrent_sessions: 1,
        runtime: RuntimeChoice::Auto,
        use_case,
        include_base: false,
        include_assistant: false,
        include_diffusion: false,
    }
}

#[test]
fn use_case_aliases_map_to_profiles() {
    assert_eq!(UseCaseId::parse("coding-agent").unwrap(), UseCaseId::Coding);
    assert_eq!(UseCaseId::parse("openclaw").unwrap(), UseCaseId::OpenClaw);
    assert_eq!(UseCaseId::parse("vision").unwrap(), UseCaseId::Vision);
    assert!(UseCaseId::parse("spreadsheet").is_err());
}

#[test]
fn coding_profile_adds_explainable_coder_score() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let candidate = score_model(
        HfModel::minimal("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        &machine,
        &options(UseCaseId::Coding),
        &deps,
    );

    assert!(candidate.score.items.iter().any(|item| {
        item.label == "coding profile: coder marker" && (item.value - 14.0).abs() < f64::EPSILON
    }));
    assert!(
        candidate
            .reasons
            .iter()
            .any(|reason| reason == "matches coding profile")
    );
}

#[test]
fn chat_profile_does_not_apply_generic_coding_bonus() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let candidate = score_model(
        HfModel::minimal("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
        &machine,
        &options(UseCaseId::Chat),
        &deps,
    );

    assert!(
        candidate
            .score
            .items
            .iter()
            .all(|item| item.label != "coding-oriented name")
    );
    assert!(
        candidate
            .reasons
            .iter()
            .all(|reason| reason != "coding-oriented name")
    );
}

#[test]
fn openclaw_profile_warns_about_vlm_only_models() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let mut model = HfModel::minimal("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    model.config = Some(mlx_model_picker::hf::ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    });

    let candidate = score_model(model, &machine, &options(UseCaseId::OpenClaw), &deps);

    assert_eq!(candidate.runtime.runtime, Some(RuntimeKind::MlxVlm));
    assert!(candidate.score.items.iter().any(|item| {
        item.label == "openclaw profile: VLM-only model needs mlx-vlm, not mlx-lm"
            && item.value < 0.0
    }));
    assert!(candidate.warnings.iter().any(|warning| {
        warning.contains("OpenClaw works best with a text model served through mlx-lm")
    }));
    assert!(
        UseCaseProfile::for_id(UseCaseId::OpenClaw)
            .output_note()
            .contains("http://localhost:8080/v1")
    );
}

#[test]
fn vision_profile_prefers_vlm_models() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let mut vlm = HfModel::minimal("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    vlm.config = Some(mlx_model_picker::hf::ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    });
    let text = HfModel::minimal("mlx-community/gemma-3-27b-it-qat-4bit");

    let ranked = rank_models(
        vec![text, vlm],
        &machine,
        &options(UseCaseId::Vision),
        &deps,
    );

    assert_eq!(
        ranked
            .recommended
            .as_ref()
            .map(|model| model.model.id.as_str()),
        Some("mlx-community/gemma-4-12B-it-OptiQ-4bit")
    );
}

#[test]
fn forced_vlm_runtime_does_not_make_unknown_models_look_vision_compatible() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let mut vlm = HfModel::minimal("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    vlm.config = Some(mlx_model_picker::hf::ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    });
    let mut forced_options = options(UseCaseId::Vision);
    forced_options.runtime = RuntimeChoice::MlxVlm;

    let ranked = rank_models(
        vec![
            HfModel::minimal("mlx-community/deepseek-coder-33b-instruct-hf-4bit-mlx"),
            vlm,
        ],
        &machine,
        &forced_options,
        &deps,
    );

    assert_eq!(
        ranked
            .recommended
            .as_ref()
            .map(|model| model.model.id.as_str()),
        Some("mlx-community/gemma-4-12B-it-OptiQ-4bit")
    );
}

#[test]
fn json_output_includes_active_use_case() {
    let machine = test_machine(64.0);
    let deps = DependencyStatus::assume_available();
    let ranked = rank_models(
        vec![HfModel::minimal(
            "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        )],
        &machine,
        &options(UseCaseId::OpenClaw),
        &deps,
    );

    let json = json_string(&machine, &deps, &ranked).unwrap();
    let payload: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(payload["use_case"], "openclaw");
}
