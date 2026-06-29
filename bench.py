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
import queue
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
        r = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, check=True)
        return int(r.stdout.strip()) / (1024**3)
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0  # non-macOS or sysctl missing; caller treats 0 as "unknown"


def make_prompts(n: int, text: str = MEDIUM_PROMPT) -> list[str]:
    return [text] * n


# ---------------------------------------------------------------------------
# Core decode primitive: one greedy batch with KV cache.
# Returns (tokens_generated, ttft_seconds). cache is mutated in place by model().
# ---------------------------------------------------------------------------
def gen_batch(model, tokenizer, ids: list[list[int]], max_tokens: int) -> tuple[int, float]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None) is not None:
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
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
    for step in range(max_tokens):
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
        if not active or step == max_tokens - 1:
            break  # last token counted; don't decode an extra unused one (skews timing)
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


def _parallel_worker(model_path, ps, max_tokens, repeats, out_q):
    """Load the model once, then run `repeats` measured passes (load excluded).
    Returns a list of per-repeat samples, or an {'error': ...} dict on failure."""
    try:
        model, tokenizer = load(model_path)
        ids = [tokenizer.encode(p) for p in ps]
        gen_batch(model, tokenizer, [ids[0]], max_tokens)  # per-shape warmup, discarded
        samples = []
        for _ in range(repeats):
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            total, ttfts = 0, []
            for t in ids:
                tok, ttft = gen_batch(model, tokenizer, [t], max_tokens)
                total += tok
                ttfts.append(ttft)
            samples.append({"gen_s": time.perf_counter() - t0, "tokens": total,
                            "ttft_ms": ttfts[0] * 1000, "peak_gb": mx.get_peak_memory() / 1e9})
        out_q.put(samples)
    except Exception as e:  # OOM, load failure, etc. — report instead of hanging the parent
        out_q.put({"error": repr(e)})


def bench_parallel(model_path: str, n: int, workers: int, max_tokens: int, repeats: int) -> dict:
    # Drop empty chunks (workers > prompts) so no worker exits without producing output.
    chunks = [c for i in range(workers) if (c := make_prompts(n)[i::workers])]
    out_q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_parallel_worker, args=(model_path, c, max_tokens, repeats, out_q))
             for c in chunks]
    for p in procs:
        p.start()

    collected = []
    while len(collected) < len(procs):
        try:
            collected.append(out_q.get(timeout=1.0))
        except queue.Empty:
            if any(p.exitcode not in (None, 0) for p in procs):  # a worker died (likely OOM)
                for p in procs:
                    p.terminate()
                raise RuntimeError("a parallel worker exited unexpectedly (likely OOM)")
    for p in procs:
        p.join()

    errors = [c for c in collected if isinstance(c, dict)]
    if errors:
        raise RuntimeError(f"parallel worker failed: {errors[0]['error']}")

    # collected: per-worker list of per-repeat samples. Aggregate across workers per repeat.
    tps, ttfts, peaks = [], [], []
    for r in range(repeats):
        per = [worker[r] for worker in collected]
        gen_s = max(s["gen_s"] for s in per)  # load excluded, matching other modes
        tps.append(sum(s["tokens"] for s in per) / gen_s)
        ttfts.append(max(s["ttft_ms"] for s in per))
        peaks.append(sum(s["peak_gb"] for s in per))  # aggregate memory across processes
    return {"mode": "parallel", "param": workers, "throughput": median(tps),
            "ttft_ms": median(ttfts), "peak_gb": median(peaks), "repeats": repeats}


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

    ram = get_ram_gb()
    ram_budget = ram - 6 if ram > 0 else float("inf")  # unknown RAM (non-mac) → don't gate
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
