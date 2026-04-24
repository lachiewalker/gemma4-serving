#!/usr/bin/env bash
# Install uv and vLLM nightly (CUDA 12.9) for Gemma 4 support.
# Run once on a fresh RunPod L40S instance.
set -euo pipefail

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create a Python 3.12 venv
uv venv .venv --python 3.12

source .venv/bin/activate

# vLLM nightly with CUDA 12.9 — required for Gemma 4 support
# (Gemma 4 is not yet in a stable vLLM release as of 2026-04)
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

# Client and utility dependencies
uv pip install openai python-dotenv pillow pymupdf

echo ""
echo "Setup complete. Activate the venv with: source .venv/bin/activate"
echo "Copy .env.example to .env and add your HF_TOKEN before running serve.sh"
