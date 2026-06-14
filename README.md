# mlx-model-picker

`mlx-model-picker` is a macOS-only Rust CLI that helps pick a local MLX model
from Hugging Face's `mlx-community` organization.

It combines model names, config metadata, memory fit, runtime compatibility, and
optional local dependency checks. The first MVP favors explainable heuristics
over benchmark-style precision.

## Usage

```sh
mlx-model-picker --family gemma --top 5
mlx-model-picker --runtime mlx-vlm --check-local-runtime
mlx-model-picker --json
mlx-model-picker --explain mlx-community/gemma-4-12B-it-OptiQ-4bit
```

The default author is `mlx-community`, the default context is `16000`, and the
default runtime mode is `auto`.

## Development

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
