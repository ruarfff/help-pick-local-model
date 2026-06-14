#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "mlx-model-picker is macOS-only for now."
  exit 1
fi

cargo build

bin="target/debug/mlx-model-picker"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

help_out="$tmp_dir/help.txt"
json_out="$tmp_dir/output.json"
gemma_out="$tmp_dir/gemma.txt"
explain_out="$tmp_dir/explain.txt"

"$bin" --help >"$help_out"
grep -Fq "Pick a local MLX model for your Mac" "$help_out"
grep -Fq -- "--runtime <RUNTIME>" "$help_out"

"$bin" --json --top 3 >"$json_out"
python3 -m json.tool "$json_out" >/dev/null
python3 - "$json_out" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

for key in ["machine", "recommended", "ranked", "rejected"]:
    if key not in payload:
        raise SystemExit(f"missing JSON key: {key}")

if not isinstance(payload["ranked"], list):
    raise SystemExit("ranked must be a list")
PY

"$bin" --family gemma --top 5 >"$gemma_out"
grep -Fq "Machine:" "$gemma_out"
grep -Fq "Recommended:" "$gemma_out"
grep -Fq "Ranked candidates:" "$gemma_out"
grep -Fq "gemma" "$gemma_out"

"$bin" --explain mlx-community/gemma-4-12B-it-OptiQ-4bit >"$explain_out"
grep -Fq "gemma4_unified" "$explain_out"
grep -Fq "Runtime: mlx-vlm" "$explain_out"
grep -Fq "mlx_vlm.server --model mlx-community/gemma-4-12B-it-OptiQ-4bit --port 8080" "$explain_out"

echo "QA passed"
