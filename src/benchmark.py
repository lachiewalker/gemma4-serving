"""Throughput benchmark — pure Python, no vLLM required locally.

Sends a configurable number of concurrent chat completion requests to the
running server and reports throughput and latency statistics.

Usage:
    python src/benchmark.py                          # balanced, 16 prompts
    python src/benchmark.py --profile decode-heavy --num-prompts 32
    python src/benchmark.py --concurrency 8
"""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx

from src.client import BASE_URL, MODEL_ID

# Workload profiles from the vLLM Gemma 4 guide
PROFILES = {
    "prompt-heavy": {"input_words": 1200, "max_tokens": 1000},
    "decode-heavy": {"input_words": 150,  "max_tokens": 8000},
    "balanced":     {"input_words": 150,  "max_tokens": 1000},
}

# Filler prompt padded to approximate input token counts.
# ~0.75 tokens per word is a reasonable English approximation.
_FILLER = (
    "The quick brown fox jumps over the lazy dog. " * 200
)


def _make_prompt(word_count: int) -> str:
    words = _FILLER.split()
    padded = " ".join((words * ((word_count // len(words)) + 1))[:word_count])
    return f"{padded}\n\nSummarise the above text in detail."


@dataclass
class RequestResult:
    success: bool
    ttft_s: float        # time to first token (seconds)
    total_s: float       # total request duration (seconds)
    output_tokens: int


async def _single_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    ttft_s = 0.0
    output_tokens = 0
    t_start = time.perf_counter()

    try:
        async with client.stream("POST", f"{BASE_URL}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                if output_tokens == 0:
                    ttft_s = time.perf_counter() - t_start
                output_tokens += 1

        return RequestResult(
            success=True,
            ttft_s=ttft_s,
            total_s=time.perf_counter() - t_start,
            output_tokens=output_tokens,
        )
    except Exception as exc:
        print(f"  Request failed: {exc}")
        return RequestResult(
            success=False,
            ttft_s=0.0,
            total_s=time.perf_counter() - t_start,
            output_tokens=0,
        )


async def _run(profile: str, num_prompts: int, concurrency: int) -> None:
    cfg = PROFILES[profile]
    prompt = _make_prompt(cfg["input_words"])
    max_tokens = cfg["max_tokens"]

    print(f"Server:      {BASE_URL}")
    print(f"Model:       {MODEL_ID}")
    print(f"Profile:     {profile}  (~{cfg['input_words']*1.3:.0f} input tokens / {max_tokens} max output tokens)")
    print(f"Requests:    {num_prompts}  (concurrency {concurrency})")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def bounded(client: httpx.AsyncClient) -> RequestResult:
        async with semaphore:
            return await _single_request(client, prompt, max_tokens)

    # Generous timeout — large decode-heavy requests can take a while
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)

    t_wall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [bounded(client) for _ in range(num_prompts)]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            r = await coro
            results.append(r)
            status = "OK" if r.success else "FAIL"
            print(f"  [{i:>3}/{num_prompts}] {status}  ttft={r.ttft_s*1000:.0f}ms  total={r.total_s:.1f}s  out_tokens={r.output_tokens}")
    wall_s = time.perf_counter() - t_wall_start

    ok = [r for r in results if r.success]
    if not ok:
        print("\nAll requests failed.")
        return

    total_out_tokens = sum(r.output_tokens for r in ok)
    ttfts = [r.ttft_s * 1000 for r in ok]  # ms

    print()
    print("=" * 50)
    print("Benchmark result")
    print("=" * 50)
    print(f"Successful requests:        {len(ok)} / {num_prompts}")
    print(f"Wall time (s):              {wall_s:.2f}")
    print(f"Request throughput (req/s): {len(ok) / wall_s:.2f}")
    print(f"Output token throughput:    {total_out_tokens / wall_s:.1f} tok/s")
    print()
    print("Time to first token (ms)")
    print(f"  Mean:   {statistics.mean(ttfts):.1f}")
    print(f"  Median: {statistics.median(ttfts):.1f}")
    print(f"  P99:    {sorted(ttfts)[int(len(ttfts) * 0.99)]:.1f}")
    print("=" * 50)


def run_benchmark(profile: str = "balanced", num_prompts: int = 16, concurrency: int = 4) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {list(PROFILES)}")
    asyncio.run(_run(profile, num_prompts, concurrency))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 server throughput")
    parser.add_argument("--profile", default="balanced", choices=list(PROFILES))
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests in flight")
    args = parser.parse_args()
    run_benchmark(args.profile, args.num_prompts, args.concurrency)
