#!/usr/bin/env python3
"""Benchmark: parallel model instances vs batched inference on MLX.

Compares PURE GENERATION throughput (excludes model loading time).

  A) Sequential: generate() one prompt at a time (baseline, uses KV cache)
  B) Parallel: N model instances, each handles 1/N prompts via generate()
  C) Batched: manually batches prompt KV prefill, then generates in batch

For B & C, only pure generation wall time is measured (load excluded).
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


def get_ram_gb():
    try:
        r = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(r.stdout.strip()) / (1024**3)
    except Exception:
        return 0


MEDIUM_PROMPT = (
    "Write a comprehensive analysis of the impact of artificial intelligence "
    "on modern healthcare, covering diagnostic imaging, drug discovery, "
    "personalized medicine, and ethical considerations."
)


def make_prompts(n: int, text: str = MEDIUM_PROMPT):
    return [text] * n


# ---------------------------------------------------------------------------
# Helper: batched prefill + KV-cache-aware generation using MLX low-level API
# ---------------------------------------------------------------------------
def batch_generate(model, tokenizer, prompts: list[str], max_tokens: int):
    """Batched generation with KV cache.

    Steps:
      1. Prefill: run all prompts through model once (populates KV cache)
      2. Generate: sample one token per batch element per step (append to KV cache)
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    # Tokenize
    all_ids = [tokenizer.encode(p) for p in prompts]
    max_len = max(len(t) for t in all_ids)
    B = len(prompts)

    # Left-pad to max_len
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    padded = [([pad_id] * (max_len - len(t))) + t for t in all_ids]
    x = mx.array(padded)  # (B, max_len)

    # KV caches per batch element
    state = [None for _ in range(B)]

    # Prefill: run full prompt through model to populate KV caches
    for i in range(B):
        # Extract just the actual tokens (not padding)
        actual_len = len(all_ids[i])
        prompt_tokens = padded[i][
            -actual_len:
        ]  # last actual_len tokens are the real ones

        # Run each prompt individually through the model to populate KV cache
        px = mx.array([prompt_tokens])
        logits, cache = model(px, cache=state[i])
        state[i] = cache
        # logits shape: (1, actual_len, vocab_size)

    # Now generate in batch
    # After prefill, next position is at index max_len in the full padded sequence
    # But since we only used actual tokens, we need to track position

    # Simple approach: use generate() for individual, no true batch
    # Instead, let me do batched prefill and then individual generation
    # This at least measures the prefill batching benefit

    # Alternative simpler approach:
    # 1. All prompts are the same length (they are, since we duplicated MEDIUM_PROMPT)
    # 2. We can create a true batch for ALL tokens
    # 3. Prefill all B prompts in one forward pass
    # 4. Then generate one token at a time for the entire batch

    # Since all prompts have the same length (we duplicated), we can batch the prefill
    x = mx.array(all_ids)  # (B, prompt_len) - no padding needed since all same length

    # Prefill: one forward pass for all prompts
    logits, cache = model(x, cache=None)
    # cache is the KV cache for all B elements
    # For the next step, we need to create new tokens

    generated = [[] for _ in range(B)]
    finished = [False] * B
    total_tokens = 0

    for step in range(max_tokens):
        # Get logits for last position of each batch element
        next_logits = logits[:, -1, :]
        next_tokens = mx.argmax(next_logits, axis=-1).reshape(-1).tolist()

        any_active = False
        new_tokens = []
        for i in range(B):
            if finished[i]:
                new_tokens.append(pad_id)
            else:
                tok = next_tokens[i]
                if tok == eos_id:
                    finished[i] = True
                    new_tokens.append(pad_id)
                else:
                    any_active = True
                    generated[i].append(tok)
                    total_tokens += 1
                    new_tokens.append(tok)

        if not any_active:
            break

        nt = mx.array([new_tokens]).reshape(B, 1)
        logits, cache = model(nt, cache=cache)

    texts = [tokenizer.decode(g) for g in generated]
    return texts, total_tokens


# ---------------------------------------------------------------------------
# Benchmarks (all measured WITHOUT model loading time)
# ---------------------------------------------------------------------------


class ModelHandle:
    """Keep model loaded across benchmark runs."""

    def __init__(self, path):
        from mlx_lm import load

        self.model, self.tokenizer = load(path)

    def sequential(self, prompts: list[str], max_tokens: int):
        from mlx_lm import generate

        total_tokens = 0
        for p in prompts:
            resp = generate(
                self.model,
                self.tokenizer,
                prompt=p,
                max_tokens=max_tokens,
                verbose=False,
            )
            total_tokens += len(self.tokenizer.encode(resp))
        return total_tokens

    def batched(self, prompts: list[str], max_tokens: int):
        _, total_tokens = batch_generate(
            self.model, self.tokenizer, prompts, max_tokens
        )
        return total_tokens


def bench_sequential(handle: ModelHandle, num_prompts: int, max_tokens: int):
    prompts = make_prompts(num_prompts)
    t0 = time.perf_counter()
    total_tokens = handle.sequential(prompts, max_tokens)
    elapsed = time.perf_counter() - t0
    return {
        "total_tokens": total_tokens,
        "gen_s": elapsed,
        "throughput": total_tokens / elapsed,
    }


def bench_batched(
    handle: ModelHandle, num_prompts: int, batch_size: int, max_tokens: int
):
    prompts = make_prompts(num_prompts)
    t0 = time.perf_counter()
    total_tokens = 0
    for i in range(0, num_prompts, batch_size):
        chunk = prompts[i : i + batch_size]
        _, tok = batch_generate(handle.model, handle.tokenizer, chunk, max_tokens)
        total_tokens += tok
    elapsed = time.perf_counter() - t0
    return {
        "total_tokens": total_tokens,
        "gen_s": elapsed,
        "throughput": total_tokens / elapsed,
        "batch_size": batch_size,
    }


def bench_parallel(
    model_path: str, num_prompts: int, num_workers: int, max_tokens: int
):
    """Each worker loads its own model, then generates on its subset.
    Times reported: load (per worker max), gen (per worker max), wall.
    """
    prompts = make_prompts(num_prompts)
    chunks = [prompts[i::num_workers] for i in range(num_workers)]

    def _worker(ps, out_q):
        from mlx_lm import load, generate

        t0 = time.perf_counter()
        model, tokenizer = load(model_path)
        load_s = time.perf_counter() - t0
        t_gen = time.perf_counter()
        total_tok = 0
        for p in ps:
            resp = generate(
                model, tokenizer, prompt=p, max_tokens=max_tokens, verbose=False
            )
            total_tok += len(tokenizer.encode(resp))
        gen_s = time.perf_counter() - t_gen
        out_q.put({"load_s": load_s, "gen_s": gen_s, "tokens": total_tok})

    out_q = mp.Queue()
    procs = [
        mp.Process(target=_worker, args=(chunks[w], out_q)) for w in range(num_workers)
    ]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall_s = time.perf_counter() - t0

    results = [out_q.get() for _ in range(num_workers)]
    total_tokens = sum(r["tokens"] for r in results)
    max_gen = max(r["gen_s"] for r in results)
    max_load = max(r["load_s"] for r in results)

    return {
        "mode": "parallel",
        "num_prompts": num_prompts,
        "num_workers": num_workers,
        "wall_s": wall_s,
        "load_s": max_load,
        "gen_s": max_gen,
        "total_tokens": total_tokens,
        "throughput_wall": total_tokens / wall_s,
        "throughput_pure_gen": total_tokens / (max_gen * num_workers),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(
            Path.home() / ".lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ram = get_ram_gb()
    model_size_gb = 20
    print(f"RAM: {ram:.0f} GB  CPUs: {os.cpu_count()}")
    print(f"Model: {Path(args.model).name}")
    print(f"All prompts: {args.num_prompts} x {args.max_tokens} tokens each")
    print()

    results = []

    # --- Sequential baseline ---
    print("=" * 55)
    print("PHASE 1: Loading model (shared across sequential + batched)")
    print("=" * 55)
    t0 = time.perf_counter()
    handle = ModelHandle(args.model)
    print(f"  Load: {time.perf_counter() - t0:.1f}s")
    gc.collect()

    print("\n" + "-" * 55)
    print("Sequential (baseline, generate() 1 prompt at a time)")
    print("-" * 55)
    r = bench_sequential(handle, args.num_prompts, args.max_tokens)
    results.append({"mode": "sequential", "num_prompts": args.num_prompts, **r})
    print(
        f"  {r['gen_s']:.2f}s  {r['throughput']:.1f} tok/s  {r['gen_s'] / args.num_prompts * 1000:.0f} ms/prompt"
    )

    # --- Batched ---
    print("\n" + "-" * 55)
    print("Batched (single model, varying batch size)")
    print("-" * 55)
    for bs in args.batch_sizes:
        if bs > args.num_prompts:
            continue
        r = bench_batched(handle, args.num_prompts, bs, args.max_tokens)
        r["mode"] = "batched"
        results.append(r)
        print(
            f"  batch_size={bs:2d}: {r['gen_s']:.2f}s  {r['throughput']:.1f} tok/s  {r['gen_s'] / args.num_prompts * 1000:.0f} ms/prompt"
        )

    del handle
    gc.collect()

    # --- Parallel ---
    print("\n" + "-" * 55)
    print("Parallel (multi-process, each loads own model instance)")
    print("-" * 55)
    for nw in args.worker_counts:
        needed = nw * model_size_gb
        available = ram - 10
        if needed > available:
            print(f"  SKIP {nw} workers: need ~{needed}GB, ~{available:.0f}GB free")
            continue
        r = bench_parallel(args.model, args.num_prompts, nw, args.max_tokens)
        results.append(r)
        print(
            f"  workers={nw}: wall={r['wall_s']:.2f}s load={r['load_s']:.1f}s gen={r['gen_s']:.2f}s"
        )
        print(f"    throughput (wall):  {r['throughput_wall']:.1f} tok/s")
        print(f"    throughput (pure gen): {r['throughput_pure_gen']:.1f} tok/s")

    # --- Comparison ---
    print("\n" + "=" * 65)
    print(f"{'Mode':<14} {'Param':<8} {'Tok/s':<10} {'Speedup':<10} {'Lat/p':<10}")
    print("-" * 65)
    seq_tp = next(r for r in results if r["mode"] == "sequential")["throughput"]
    for r in results:
        if r["mode"] == "sequential":
            param = 1
            tp = r["throughput"]
            lat = r["gen_s"] / args.num_prompts * 1000
            print(
                f"{'sequential':<14} {param:<8} {tp:<10.1f} {'1.00x':<10} {lat:<10.0f}ms"
            )
        elif r["mode"] == "batched":
            tp = r["throughput"]
            lat = r["gen_s"] / args.num_prompts * 1000
            print(
                f"{'batched':<14} {r['batch_size']:<8} {tp:<10.1f} {tp / seq_tp:<10.2f}x {lat:<10.0f}ms"
            )
        elif r["mode"] == "parallel":
            tp = r["throughput_wall"]
            lat = r["wall_s"] / args.num_prompts * 1000
            print(
                f"{'parallel':<14} {r['num_workers']:<8} {tp:<10.1f} {tp / seq_tp:<10.2f}x {lat:<10.0f}ms"
            )
            tp_pure = r["throughput_pure_gen"]
            print(
                f"{'parallel(pure)':<14} {r['num_workers']:<8} {tp_pure:<10.1f} {tp_pure / seq_tp:<10.2f}x {'':<10}"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
