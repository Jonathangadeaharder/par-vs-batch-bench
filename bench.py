#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx", "mlx-lm"]
# ///
# pyright: reportMissingImports=false
"""Benchmark: sequential vs batched vs parallel LLM inference on MLX.

Measures the three axes a serving system actually trades off:

  - throughput  (tok/s, aggregate)
  - latency     (TTFT = time-to-first-token, ms)
  - memory      (peak MLX allocation, GB)

Strategies, all using the SAME greedy (argmax) decode so the comparison is
apples-to-apples:

  A) Sequential : one prompt at a time, fresh KV cache each (latency-optimal)
  B) Batched    : N prompts share one forward pass per step (throughput-optimal)
  C) Parallel   : N processes, each its own model copy (memory-bound)

Model load time is excluded from every measurement.

Run:  uv run bench.py --model mlx-community/Qwen3-0.6B-4bit
"""

import argparse
import gc
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MEDIUM_PROMPT = (
    "Write a comprehensive analysis of the impact of artificial intelligence "
    "on modern healthcare, covering diagnostic imaging, drug discovery, "
    "personalized medicine, and ethical considerations."
)


def get_ram_gb() -> float:
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(r.stdout.strip()) / (1024**3)
    except Exception:
        return 0.0


def make_prompts(n: int, text: str = MEDIUM_PROMPT) -> list[str]:
    return [text] * n


# ---------------------------------------------------------------------------
# Core decode primitive: one greedy batch with KV cache.
# Returns (tokens_generated, ttft_seconds). cache is mutated in place by model().
# ---------------------------------------------------------------------------
def gen_batch(model, tokenizer, ids: list[list[int]], max_tokens: int) -> tuple[int, float]:
    eos_id = tokenizer.eos_token_id
    B = len(ids)
    # ponytail: prompts are duplicated (all equal length), so no attention mask
    # is needed for the batched prefill. Differing lengths would need left-pad +
    # masking (mlx_lm BatchKVCache) — out of scope for this throughput study.
    L = len(ids[0])
    assert all(len(t) == L for t in ids), "batched path requires equal-length prompts"

    x = mx.array(ids)  # (B, L)
    cache = make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(x, cache=cache)            # prefill, populates cache
    next_tok = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_tok)                          # force compute: honest TTFT (MLX is lazy)
    ttft = time.perf_counter() - t0

    finished = [False] * B
    total = 0
    for _ in range(max_tokens):
        toks = next_tok.tolist()
        active = False
        for i in range(B):
            if finished[i]:
                continue
            if toks[i] == eos_id:
                finished[i] = True
            else:
                total += 1
                active = True
        if not active:
            break
        logits = model(next_tok.reshape(B, 1), cache=cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_tok)
    return total, ttft


def measure(fn):
    """Run fn under peak-memory + wall-clock instrumentation.
    Returns (result, wall_s, peak_gb)."""
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    result = fn()
    wall = time.perf_counter() - t0
    return result, wall, mx.get_peak_memory() / 1e9


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
class ModelHandle:
    def __init__(self, path: str):
        self.model, self.tokenizer = load(path)

    def _encode(self, prompts: list[str]) -> list[list[int]]:
        return [self.tokenizer.encode(p) for p in prompts]

    def sequential(self, prompts: list[str], max_tokens: int) -> tuple[int, float]:
        ids = self._encode(prompts)
        total, ttfts = 0, []
        for t in ids:
            tok, ttft = gen_batch(self.model, self.tokenizer, [t], max_tokens)
            total += tok
            ttfts.append(ttft)
        return total, ttfts[0]

    def batched(self, prompts: list[str], bs: int, max_tokens: int) -> tuple[int, float]:
        ids = self._encode(prompts)
        total, ttfts = 0, []
        for i in range(0, len(ids), bs):
            tok, ttft = gen_batch(self.model, self.tokenizer, ids[i : i + bs], max_tokens)
            total += tok
            ttfts.append(ttft)
        return total, ttfts[0]


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def run_inproc(label: str, param: int, fn, repeats: int) -> dict:
    """Run fn (-> (tokens, ttft_s)) repeats+1 times; discard the first as a
    per-shape warmup (MLX JIT-compiles per tensor shape), report medians."""
    tps, ttfts, peaks = [], [], []
    for j in range(repeats + 1):
        (total, ttft), wall, peak = measure(fn)
        if j == 0:
            continue
        tps.append(total / wall)
        ttfts.append(ttft * 1000)
        peaks.append(peak)
    return {"mode": label, "param": param, "throughput": median(tps),
            "ttft_ms": median(ttfts), "peak_gb": median(peaks), "repeats": repeats}


def bench_sequential(handle: ModelHandle, n: int, max_tokens: int, repeats: int) -> dict:
    return run_inproc("sequential", 1, lambda: handle.sequential(make_prompts(n), max_tokens), repeats)


def bench_batched(handle: ModelHandle, n: int, bs: int, max_tokens: int, repeats: int) -> dict:
    return run_inproc("batched", bs, lambda: handle.batched(make_prompts(n), bs, max_tokens), repeats)


def _parallel_worker(model_path, ps, max_tokens, out_q):
    mx.reset_peak_memory()
    model, tokenizer = load(model_path)
    ids = [tokenizer.encode(p) for p in ps]
    gen_batch(model, tokenizer, [ids[0]], max_tokens)  # per-shape warmup, discarded
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    total, ttfts = 0, []
    for t in ids:
        tok, ttft = gen_batch(model, tokenizer, [t], max_tokens)
        total += tok
        ttfts.append(ttft)
    out_q.put({"gen_s": time.perf_counter() - t0, "tokens": total,
               "ttft_ms": ttfts[0] * 1000, "peak_gb": mx.get_peak_memory() / 1e9})


def _parallel_once(model_path: str, n: int, workers: int, max_tokens: int) -> dict:
    chunks = [make_prompts(n)[i::workers] for i in range(workers)]
    out_q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_parallel_worker, args=(model_path, chunks[w], max_tokens, out_q))
             for w in range(workers)]
    for p in procs:
        p.start()
    res = [out_q.get() for _ in range(workers)]  # drain before join (avoid pipe-buffer deadlock)
    for p in procs:
        p.join()
    gen_s = max(r["gen_s"] for r in res)  # load excluded, matching other modes
    return {"throughput": sum(r["tokens"] for r in res) / gen_s,
            "ttft_ms": max(r["ttft_ms"] for r in res),
            "peak_gb": sum(r["peak_gb"] for r in res)}  # aggregate across processes


def bench_parallel(model_path: str, n: int, workers: int, max_tokens: int, repeats: int) -> dict:
    s = [_parallel_once(model_path, n, workers, max_tokens) for _ in range(repeats)]
    return {"mode": "parallel", "param": workers,
            "throughput": median([r["throughput"] for r in s]),
            "ttft_ms": median([r["ttft_ms"] for r in s]),
            "peak_gb": median([r["peak_gb"] for r in s]), "repeats": repeats}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--num-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--repeats", type=int, default=3, help="measured runs per config (median reported)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"RAM: {get_ram_gb():.0f} GB  CPUs: {os.cpu_count()}  model: {Path(args.model).name}")
    print(f"workload: {args.num_prompts} prompts x {args.max_tokens} tokens, greedy decode, "
          f"median of {args.repeats} (per-shape warmup discarded)\n")

    handle = ModelHandle(args.model)
    results = [bench_sequential(handle, args.num_prompts, args.max_tokens, args.repeats)]
    single_peak = results[0]["peak_gb"]

    for bs in args.batch_sizes:
        if bs > args.num_prompts:
            continue
        results.append(bench_batched(handle, args.num_prompts, bs, args.max_tokens, args.repeats))

    del handle
    gc.collect()
    mx.clear_cache()

    ram_budget = get_ram_gb() - 6  # leave headroom for OS
    for nw in args.worker_counts:
        if single_peak and nw * single_peak > ram_budget:
            print(f"  SKIP parallel x{nw}: ~{nw * single_peak:.1f}GB > {ram_budget:.0f}GB budget")
            continue
        results.append(bench_parallel(args.model, args.num_prompts, nw, args.max_tokens, args.repeats))

    seq_tp = next(r["throughput"] for r in results if r["mode"] == "sequential")
    print(f"\n{'Mode':<12}{'Param':<7}{'Tok/s':<9}{'Speedup':<9}{'TTFT ms':<9}{'Peak GB':<8}")
    print("-" * 54)
    for r in results:
        print(f"{r['mode']:<12}{r['param']:<7}{r['throughput']:<9.1f}"
              f"{r['throughput'] / seq_tp:<9.2f}{r['ttft_ms']:<9.0f}{r['peak_gb']:<8.2f}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
