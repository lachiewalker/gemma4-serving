#!/usr/bin/env bash
# Launch the Gemma 4 31B FP8 server on a single L40S (48 GB VRAM).
#
# Model: RedHatAI/gemma-4-31B-it-FP8-block
#   - Block FP8 quantized weights + activations (~50% VRAM vs BF16)
#   - Vision tower kept in full precision
#   - 99.9-100.3% accuracy recovery vs unquantized
#
# All capabilities are enabled: text, images, reasoning, tool use.
set -euo pipefail

if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example to .env and set HF_TOKEN." >&2
    exit 1
fi

source .env
source .venv/bin/activate

# Use vLLM's tool_chat_template_gemma4.jinja rather than the model's bundled
# chat_template.jinja. The vLLM template adds the <|thinking|> and <|tool_call|>
# token structures that --reasoning-parser gemma4 and --tool-call-parser gemma4 require.
CHAT_TEMPLATE=$(python -c "
import vllm, os
base = os.path.dirname(vllm.__file__)
p = os.path.join(base, 'entrypoints', 'chat_templates', 'tool_chat_template_gemma4.jinja')
# Fallback: examples directory
if not os.path.exists(p):
    p = os.path.join(base, '..', 'examples', 'tool_chat_template_gemma4.jinja')
print(os.path.abspath(p))
")

echo "Using chat template: $CHAT_TEMPLATE"
echo "Starting vLLM server for ${MODEL_ID:-RedHatAI/gemma-4-31B-it-FP8-block} ..."

vllm serve "${MODEL_ID:-RedHatAI/gemma-4-31B-it-FP8-block}" \
    --kv-cache-dtype fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --chat-template "$CHAT_TEMPLATE" \
    --limit-mm-per-prompt '{"image": 4, "audio": 0}' \
    --async-scheduling \
    --host 0.0.0.0 \
    --port "${PORT:-8000}"
