use mlx_model_picker::hf::ModelConfig;
use mlx_model_picker::parser::parse_model_id;
use mlx_model_picker::runtime::{RuntimeChoice, RuntimeKind, infer_runtime};

#[test]
fn gemma4_unified_maps_to_vlm() {
    let parsed = parse_model_id("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    let config = ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    };

    let decision = infer_runtime(&parsed, Some(&config), RuntimeChoice::Auto);

    assert_eq!(decision.runtime, Some(RuntimeKind::MlxVlm));
    assert!(
        decision
            .warnings
            .iter()
            .any(|w| w.contains("gemma4_unified"))
    );
}

#[test]
fn common_text_models_map_to_mlx_lm() {
    let parsed = parse_model_id("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");
    let config = ModelConfig {
        model_type: Some("qwen3".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "qwen3" }),
    };

    let decision = infer_runtime(&parsed, Some(&config), RuntimeChoice::Auto);

    assert_eq!(decision.runtime, Some(RuntimeKind::MlxLm));
}

#[test]
fn forced_mlx_lm_rejects_vlm_only_config() {
    let parsed = parse_model_id("mlx-community/gemma-4-12B-it-OptiQ-4bit");
    let config = ModelConfig {
        model_type: Some("gemma4_unified".to_string()),
        architectures: vec![],
        raw: serde_json::json!({ "model_type": "gemma4_unified" }),
    };

    let decision = infer_runtime(&parsed, Some(&config), RuntimeChoice::MlxLm);

    assert_eq!(decision.runtime, Some(RuntimeKind::MlxVlm));
    assert!(!decision.is_compatible);
}
