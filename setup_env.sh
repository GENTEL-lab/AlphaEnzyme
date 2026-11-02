#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv venv --python 3.11
uv sync --frozen "$@"

echo "AlphaEnzyme environment installed in $(pwd)/.venv"
echo "Activate with: source .venv/bin/activate"
echo "For feature generation, rerun: ./setup_env.sh --extra preprocessing"
