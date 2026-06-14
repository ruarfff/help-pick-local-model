use anyhow::{Result, bail};
use clap::Parser;

use crate::{
    cache::Cache,
    hf::{HfClient, HfModel},
    machine::detect_machine,
    output::{json_string, print_explain, print_human},
    parser::parse_model_id,
    runtime::{RuntimeChoice, check_dependencies},
    scoring::{PickerOptions, rank_models, score_model},
};

#[derive(Debug, Parser)]
#[command(name = "mlx-model-picker")]
#[command(about = "Pick a local MLX model for your Mac")]
pub struct Args {
    #[arg(long)]
    pub family: Option<String>,

    #[arg(long, default_value = "mlx-community")]
    pub author: String,

    #[arg(long, default_value_t = 1000)]
    pub limit: usize,

    #[arg(long, default_value_t = 16_000)]
    pub context: u64,

    #[arg(long, default_value_t = 1)]
    pub concurrent: u64,

    #[arg(long, value_enum, default_value_t = RuntimeChoice::Auto)]
    pub runtime: RuntimeChoice,

    #[arg(long)]
    pub check_local_runtime: bool,

    #[arg(long)]
    pub no_runtime_check: bool,

    #[arg(long)]
    pub json: bool,

    #[arg(long, default_value_t = 10)]
    pub top: usize,

    #[arg(long)]
    pub refresh: bool,

    #[arg(long)]
    pub include_base: bool,

    #[arg(long)]
    pub include_assistant: bool,

    #[arg(long)]
    pub include_diffusion: bool,

    #[arg(long)]
    pub explain: Option<String>,

    #[arg(long, default_value = "coding-agent", hide = true)]
    pub workload: String,
}

pub async fn run() -> Result<()> {
    let args = Args::parse();
    if !cfg!(target_os = "macos") {
        bail!("mlx-model-picker is macOS-only for now.");
    }
    run_with_args(args).await
}

async fn run_with_args(args: Args) -> Result<()> {
    let machine = detect_machine().await?;
    let deps = if args.check_local_runtime && !args.no_runtime_check {
        check_dependencies().await
    } else {
        crate::runtime::DependencyStatus::not_checked()
    };
    let cache = Cache::new()?;
    let client = HfClient::new(cache, args.refresh)?;
    let options = PickerOptions {
        family: args.family.as_ref().map(|family| family.to_lowercase()),
        top: args.top,
        context_tokens: args.context,
        concurrent_sessions: args.concurrent,
        runtime: args.runtime,
        include_base: args.include_base,
        include_assistant: args.include_assistant,
        include_diffusion: args.include_diffusion,
    };

    if let Some(model_id) = args.explain.as_deref() {
        let mut model = HfModel::minimal(model_id);
        model.config = client.fetch_config(model_id).await?;
        let candidate = score_model(model, &machine, &options, &deps);
        if args.json {
            println!("{}", serde_json::to_string_pretty(&candidate)?);
        } else {
            print_explain(&candidate, &deps);
        }
        return Ok(());
    }

    let models = client.fetch_models(&args.author, args.limit).await?;
    let models = prefilter_models(models, &options);
    let models = client.attach_configs(models, 200).await;
    let result = rank_models(models, &machine, &options, &deps);
    if args.json {
        println!("{}", json_string(&machine, &deps, &result)?);
    } else {
        print_human(&machine, &deps, &result);
    }
    Ok(())
}

fn prefilter_models(models: Vec<HfModel>, options: &PickerOptions) -> Vec<HfModel> {
    models
        .into_iter()
        .filter(|model| {
            let parsed = parse_model_id(&model.id);
            options
                .family
                .as_deref()
                .is_none_or(|family| parsed.family == family)
                && (options.include_base || !parsed.is_base)
                && (options.include_assistant || !parsed.is_assistant)
                && (options.include_diffusion || !parsed.is_diffusion)
        })
        .collect()
}
