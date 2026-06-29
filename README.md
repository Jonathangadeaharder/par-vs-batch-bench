# par-vs-batch-bench

Micro-benchmark of LLM inference strategies on Apple Silicon (MLX), measuring the
three axes a serving system trades off: **throughput, latency (TTFT), and memory.**

> **Result (8B model, M5 Pro):** batching reaches **4.3× throughput at +8% memory**;
> running 4 parallel processes reaches only **1.34× at +300% memory** (weights are
> replicated per process). On unified memory, **batch — don't fork.**
> Full writeup with tables and the model-size effect: [docs/findings.md](docs/findings.md).

| mode (8B) | tok/s | speedup | TTFT ms | peak GB |
|-----------|-------|---------|---------|---------|
| sequential | 19.9 | 1.00× | 127 | 8.57 |
| batched ×12 | 86.1 | **4.32×** | 469 | 9.29 |
| parallel ×4 | 26.6 | 1.34× | 455 | 34.29 |

## Strategies

All use **greedy decode** so the comparison is apples-to-apples; model load is excluded.

- **Sequential** — one prompt at a time, fresh KV cache (latency-optimal baseline)
- **Batched** — N prompts share one forward pass per step; weights streamed once (throughput-optimal)
- **Parallel** — N processes, each its own model copy (memory-bound; only for fault isolation)

## Method (why the numbers are honest)

- **Per-shape warmup discarded** — MLX JIT-compiles per tensor shape; the first run of each config is thrown away.
- **`mx.eval()` forces compute** before timing — MLX is lazy, so naive timers measure graph build, not execution.
- **Median of N repeats** (`--repeats`, default 3).
- **Peak memory** via `mx.get_peak_memory()`; parallel sums per-process peaks.

## Usage

```bash
uv run bench.py --help
# self-contained (PEP 723 inline deps) — uv installs mlx + mlx-lm automatically

uv run bench.py --model mlx-community/Qwen3-0.6B-4bit          # quick (small, overhead-bound)
uv run bench.py --model khanh2023/Goedel-Code-Prover-8B-mlx-mode_mxfp8 \
  --num-prompts 12 --max-tokens 48 --batch-sizes 2 4 8 12 --worker-counts 1 2 4
```

## Requirements

- macOS, Apple Silicon
- [`uv`](https://docs.astral.sh/uv/) (handles the Python + MLX deps; nothing else to install)
