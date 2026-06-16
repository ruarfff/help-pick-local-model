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
cargo run -- --use-case coding --family qwen --top 5
cargo run -- --use-case openclaw --top 5
cargo run -- --use-case vision --top 5
cargo run -- --family qwen --top 5
cargo run -- --runtime mlx-vlm --check-local-runtime
cargo run -- --json
cargo run -- --explain mlx-community/gemma-4-12B-it-OptiQ-4bit
```

The default author is `mlx-community`, the default context is `16000`, and the
default runtime mode is `auto`. The default use case is `coding`; choose
`openclaw`, `chat`, or `vision` when you want recommendation weights and notes
tailored to that workflow.

## Development

```sh
cargo fmt --check
cargo clippy -- -D warnings
cargo test
./scripts/qa.sh
```

`./scripts/qa.sh` builds the app and runs live smoke checks against this
machine, including JSON validation and the `gemma4_unified` runtime explanation.

## Release

Releases are built by GitHub Actions when a version tag is pushed. The current
release workflow runs on macOS, builds the release binary, packages it as
`mlx-model-picker-macos.tar.gz`, and attaches that archive to a GitHub Release.

Before creating a release:

1. Make sure the working tree is clean.

   ```sh
   git status --short
   ```

2. Update the version in `Cargo.toml`.

   ```toml
   version = "0.1.0"
   ```

3. Run the local verification checks.

   ```sh
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ./scripts/qa.sh
   ```

4. Commit the version/docs changes.

   ```sh
   git add Cargo.toml Cargo.lock README.md
   git commit -m "chore: release v0.1.0"
   ```

5. Create and push a version tag.

   ```sh
   git tag v0.1.0
   git push origin main
   git push origin v0.1.0
   ```

6. Watch the `Release` workflow in GitHub Actions. When it finishes, confirm the
   GitHub Release has the `mlx-model-picker-macos.tar.gz` asset attached.

To test the released archive on another Mac:

```sh
tar -xzf mlx-model-picker-macos.tar.gz
./mlx-model-picker --help
./mlx-model-picker --family gemma --top 5
```

Notes:

- Tags only need to match `v*` for the current workflow, but semantic versions
  like `v0.1.0` are recommended.
- The current workflow builds one macOS archive from `macos-latest`. Separate
  Apple Silicon, Intel, universal binaries, signing, and checksums are planned
  release polish rather than part of the MVP workflow.
