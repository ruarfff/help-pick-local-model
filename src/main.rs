use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    mlx_model_picker::cli::run().await
}
