use mlx_model_picker::parser::parse_model_id;

#[test]
fn parses_gemma4_optiq_four_bit() {
    let parsed = parse_model_id("mlx-community/gemma-4-12B-it-OptiQ-4bit");

    assert_eq!(parsed.author.as_deref(), Some("mlx-community"));
    assert_eq!(parsed.family, "gemma");
    assert_eq!(parsed.generation.as_deref(), Some("4"));
    assert_eq!(parsed.total_params_b, Some(12.0));
    assert_eq!(parsed.quantization.as_deref(), Some("4bit"));
    assert!(parsed.is_instruction_tuned);
    assert!(parsed.is_optiq);
}

#[test]
fn parses_moe_active_params() {
    let parsed = parse_model_id("mlx-community/gemma-4-26b-a4b-it-4bit");

    assert_eq!(parsed.total_params_b, Some(26.0));
    assert_eq!(parsed.active_params_b, Some(4.0));
    assert_eq!(parsed.quantization.as_deref(), Some("4bit"));
}

#[test]
fn parses_qat_quantization_and_markers() {
    let parsed = parse_model_id("mlx-community/gemma-3-27b-it-qat-4bit");

    assert_eq!(parsed.generation.as_deref(), Some("3"));
    assert_eq!(parsed.total_params_b, Some(27.0));
    assert!(parsed.is_qat);
    assert_eq!(parsed.quantization.as_deref(), Some("4bit"));
}

#[test]
fn parses_eight_bit_and_bf16_variants() {
    let eight = parse_model_id("mlx-community/gemma-4-31b-it-8bit");
    let bf16 = parse_model_id("mlx-community/gemma-4-12B-it-assistant-bf16");

    assert_eq!(eight.quantization.as_deref(), Some("8bit"));
    assert_eq!(bf16.quantization.as_deref(), Some("bf16"));
    assert!(bf16.is_assistant);
}

#[test]
fn parses_generic_bit_quantization_variants() {
    let three = parse_model_id("mlx-community/Qwen3-Coder-30B-A3B-Instruct-3bit");
    let six = parse_model_id("mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit");

    assert_eq!(three.quantization.as_deref(), Some("3bit"));
    assert_eq!(six.quantization.as_deref(), Some("6bit"));
}

#[test]
fn parses_diffusion_and_coder_names() {
    let diffusion = parse_model_id("mlx-community/diffusiongemma-26B-A4B-it-4bit");
    let coder = parse_model_id("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit");

    assert!(diffusion.is_diffusion);
    assert_eq!(diffusion.active_params_b, Some(4.0));
    assert_eq!(coder.family, "qwen");
    assert!(coder.is_coding);
    assert!(coder.is_instruction_tuned);
    assert_eq!(coder.active_params_b, Some(3.0));
}

#[test]
fn unknown_names_do_not_panic() {
    let parsed = parse_model_id("mlx-community/a-very-strange-model-name");

    assert_eq!(parsed.author.as_deref(), Some("mlx-community"));
    assert_eq!(parsed.model_id, "a-very-strange-model-name");
    assert_eq!(parsed.family, "unknown");
    assert!(parsed.unknown_tokens.len() >= 3);
}
