# mlx-model-picker

`mlx-model-picker` is a macOS-only CLI that helps you choose a local MLX model
from Hugging Face's `mlx-community` organization.

It looks at your Mac, model metadata, estimated memory fit, MLX runtime
compatibility, and optional local dependency checks. The recommendations are
explainable heuristics, not benchmark claims.

## Install

Download the latest release archive from the project's GitHub Releases page.
For now, this is the only supported install path.

```sh
tar -xzf mlx-model-picker-macos.tar.gz
./mlx-model-picker --help
```

During early development, release binaries are not yet signed or notarized. If
macOS shows a warning that `"mlx-model-picker" Not Opened`, remove the
quarantine flag after downloading:

```sh
xattr -d com.apple.quarantine ./mlx-model-picker
./mlx-model-picker --help
```

You can run the binary from the extracted directory, or move it somewhere on
your `PATH`:

```sh
chmod +x mlx-model-picker
sudo mv mlx-model-picker /usr/local/bin/
mlx-model-picker --help
```

## First Run

Ask for a recommendation:

```sh
mlx-model-picker --top 5
```

The first run fetches public model metadata from Hugging Face and caches it
locally. Later runs reuse the cache unless you pass `--refresh`.

The output includes:

- your Mac summary
- the recommended model
- reasons for the recommendation
- warnings
- a suggested local server command
- a ranked table of alternatives

## Choose A Use Case

The default use case is `coding`.

```sh
mlx-model-picker --use-case coding --top 5
mlx-model-picker --use-case chat --top 5
mlx-model-picker --use-case vision --top 5
mlx-model-picker --use-case openclaw --top 5
```

Use cases adjust scoring and notes:

- `coding`: local coding assistant or coding agent
- `chat`: general local assistant
- `vision`: multimodal/image workflows, usually `mlx-vlm`
- `openclaw`: local model served for OpenClaw-style agent use

For OpenClaw, start the suggested local server, then point OpenClaw at:

```text
http://localhost:8080/v1
```

## Filter Results

Filter by model family:

```sh
mlx-model-picker --family qwen --top 5
mlx-model-picker --family gemma --top 5
```

Change context or concurrent session assumptions:

```sh
mlx-model-picker --context 32000 --concurrent 2 --top 5
```

Force a runtime when you already know what you want:

```sh
mlx-model-picker --runtime mlx-lm --top 5
mlx-model-picker --runtime mlx-vlm --use-case vision --top 5
```

## Explain A Model

Use `--explain` to inspect one model in detail:

```sh
mlx-model-picker --explain mlx-community/gemma-4-12B-it-OptiQ-4bit
```

Explain mode shows parsed model-name metadata, config metadata when available,
memory estimate, runtime decision, score breakdown, warnings, and suggested
server command.

For example, some Gemma 4 models expose `model_type = gemma4_unified`. Those
should use `mlx_vlm.server`, not `mlx_lm.server`, and the tool calls that out.

## Check Local Runtime Dependencies

The picker can check whether common local MLX dependencies are installed:

```sh
mlx-model-picker --check-local-runtime --top 5
```

It checks for tools and Python modules such as `python`, `uv`, `pip`, `mlx`,
`mlx_lm`, and `mlx_vlm`.

If packages are missing, install them with:

```sh
uv pip install -U mlx mlx-lm mlx-vlm huggingface_hub hf_xet
```

## JSON Output

Use JSON when you want to script around the recommendation:

```sh
mlx-model-picker --json --top 5
mlx-model-picker --json --use-case openclaw --top 5
```

The JSON includes the active `use_case`, machine info, recommended model,
ranked candidates, and rejected candidates.

## Common Commands

```sh
mlx-model-picker --top 5
mlx-model-picker --use-case coding --family qwen --top 5
mlx-model-picker --use-case chat --top 5
mlx-model-picker --use-case vision --top 5
mlx-model-picker --use-case openclaw --top 5
mlx-model-picker --check-local-runtime --top 5
mlx-model-picker --refresh --top 5
mlx-model-picker --json --top 5
```

## Current Limitations

- macOS only
- recommendations are heuristics, not benchmarks
- public Hugging Face models only; no token is required
- release binary is the only supported install path for now
- compatibility data is inferred from model names and `config.json` when
  available
