# mlx-model-picker

`mlx-model-picker` is a macOS-only Rust CLI that helps pick a local MLX model
from Hugging Face's `mlx-community` organization.

It combines model names, config metadata, memory fit, runtime compatibility, and
optional local dependency checks. The first MVP favors explainable heuristics
over benchmark-style precision.

## Usage

From a fresh clone on macOS:

1. Install Rust if you do not already have it.

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustc --version
   cargo --version
   ```

2. Clone the repo and enter it.

   ```sh
   git clone <repo-url>
   cd help-pick-local-model
   ```

3. Build the CLI.

   ```sh
   cargo build
   ```

4. Confirm the CLI starts.

   ```sh
   cargo run -- --help
   ```

5. Run a first recommendation.

   ```sh
   cargo run -- --family gemma --top 5
   ```

   The first run fetches public model metadata from Hugging Face and caches it
   locally. Later runs reuse the cache unless you pass `--refresh`.

6. Explain a specific model.

   ```sh
   cargo run -- --explain mlx-community/gemma-4-12B-it-OptiQ-4bit
   ```

   For models whose config reports `model_type = gemma4_unified`, the tool
   should recommend `mlx_vlm.server` instead of `mlx_lm.server`.

7. Optionally check local MLX runtime dependencies.

   ```sh
   cargo run -- --check-local-runtime --family qwen --top 5
   ```

   If packages are missing, install them with:

   ```sh
   uv pip install -U mlx mlx-lm mlx-vlm huggingface_hub hf_xet
   ```

Common commands:

```sh
cargo run -- --family gemma --top 5
cargo run -- --family qwen --top 5
cargo run -- --runtime mlx-vlm --check-local-runtime
cargo run -- --json
cargo run -- --explain mlx-community/gemma-4-12B-it-OptiQ-4bit
```

The default author is `mlx-community`, the default context is `16000`, and the
default runtime mode is `auto`.

## Development

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
./scripts/qa.sh
```

`./scripts/qa.sh` builds the app and runs live smoke checks against this
machine, including JSON validation and the `gemma4_unified` runtime explanation.
