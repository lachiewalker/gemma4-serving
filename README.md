# gemma4-serving

Serve **Gemma 4 31B FP8** on a single RunPod L40S (48 GB VRAM) using [vLLM](https://github.com/vllm-project/vllm).

Based on the official [vLLM Gemma 4 recipe](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md).

## Model

[`RedHatAI/gemma-4-31B-it-FP8-block`](https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-block)

Block-FP8 quantized (weights + activations). Vision tower kept in full precision. ~50% VRAM reduction vs BF16 with 99.9–100.3% accuracy recovery. Fits on a single L40S at `max-model-len 32768`.

---

## On the RunPod instance

```bash
# 1. Clone the repo
git clone <repo-url> gemma4-serving && cd gemma4-serving

# 2. Install vLLM nightly (CUDA 12.9) + server-side dependencies
bash setup.sh

# 3. Set your HuggingFace token
cp .env.example .env
# Edit .env and set HF_TOKEN

# 4. Start the server
bash serve.sh
```

The server listens on `0.0.0.0:8000`.

---

## On your local machine

### 1. Install client dependencies (no vLLM needed)

```bash
uv sync
```

### 2. Configure the server URL

```bash
cp .env.example .env
# Set VLLM_BASE_URL to your server's public address, e.g.:
# VLLM_BASE_URL=http://your-server:8000/v1
```

### 3. Run examples

```bash
python examples/text_inference.py
python examples/image_inference.py path/to/image.jpg
python examples/reasoning.py
python examples/tool_use.py
python examples/reasoning_with_tools.py
python examples/document_ocr.py path/to/document.pdf
```

### 4. Benchmark

```bash
# Quick balanced test (16 prompts, 4 concurrent)
python src/benchmark.py

# Decode-heavy — stresses KV cache
python src/benchmark.py --profile decode-heavy --num-prompts 32

# Prompt-heavy — stresses prefill
python src/benchmark.py --profile prompt-heavy --num-prompts 32

# Higher concurrency
python src/benchmark.py --concurrency 8 --num-prompts 64
```

Output:
```
Successful requests:        16 / 16
Wall time (s):              12.34
Request throughput (req/s): 1.30
Output token throughput:    1302.4 tok/s

Time to first token (ms)
  Mean:   842.1
  Median: 810.3
  P99:    1203.7
```

---

## Document OCR

Per-page calls with structured output — each page is a separate API request processed in parallel.

```python
from src.ocr import ocr_document

result = ocr_document("report.pdf", max_workers=4)
for i, page in enumerate(result.pages, 1):
    for block in page.blocks:
        print(f"[p{i}] [{block.type.value}] {block.text}")
```

Extracted block types: `title`, `heading_1/2/3`, `paragraph`, `caption`, `header`, `footer`, `page_number`, `figure`, `table`, `list_item`, `code`, `footnote`.

---

## Server configuration

Key flags in `serve.sh`:

| Flag | Value | Purpose |
|------|-------|---------|
| `--kv-cache-dtype fp8` | fp8 | ~50% KV cache memory reduction |
| `--max-model-len` | 32768 | Context window (reduce to 16384 if OOM) |
| `--gpu-memory-utilization` | 0.92 | VRAM fraction for model + KV cache |
| `--reasoning-parser gemma4` | gemma4 | Enable thinking/reasoning mode |
| `--tool-call-parser gemma4` | gemma4 | Enable function calling |
| `--enable-auto-tool-choice` | — | Auto-detect tool calls |
| `--limit-mm-per-prompt image=4,audio=0` | — | 4 images per request, no audio encoder |
| `--async-scheduling` | — | Overlap scheduling with decoding for throughput |

## Vision token budget

Per-request image detail can be tuned at inference time (70 / 140 / 280 / 560 / 1120 tokens per image, default 280):

```python
client.chat.completions.create(
    ...,
    extra_body={"mm_processor_kwargs": {"max_soft_tokens": 560}},
)
```

## Requirements

**RunPod instance**
- Ubuntu 22.04+, CUDA 12.x driver
- 48 GB VRAM (L40S, A6000, or similar)
- ~64 GB system RAM
- HuggingFace token with access to `RedHatAI/gemma-4-31B-it-FP8-block`

**Local machine**
- Python 3.12+
- `uv sync`
