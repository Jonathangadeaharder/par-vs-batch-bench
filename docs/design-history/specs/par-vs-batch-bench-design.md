# par-vs-batch-bench — MLX Inference Strategy Benchmark

## Overview

Benchmarks ML model inference throughput comparing three strategies on Apple Silicon: sequential (baseline), parallel (multi-process model instances), and batched (manual KV cache prefill). Uses MLX framework for native Apple Silicon performance.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Framework | MLX (mlx-lm) |
| Language | Python 3, single-file |
| Benchmark modes | sequential, parallel, batched |
| Measurement | Pure generation time (excludes load) |
| Output | Terminal table + optional JSON |

## Architecture

```
bench.py
├── Constants & Helpers
│   ├── MEDIUM_PROMPT       # Standard prompt text
│   ├── make_prompts()      # Generate N copies of prompt
│   ├── batch_generate()    # Manual batched KV prefill + generation
│   └── ModelHandle         # Cached model wrapper
├── Benchmark Functions
│   ├── bench_sequential()  # Baseline: generate() one at a time
│   ├── bench_batched()     # Batched KV prefill + generation
│   └── bench_parallel()    # Multi-process, each loads own model
└── Main Entry
    ├── CLI argument parsing
    ├── Phase 1: Load model (shared for seq + batch)
    ├── Phase 2: Sequential baseline
    ├── Phase 3: Batched (varying batch sizes)
    ├── Phase 4: Parallel (varying worker counts)
    └── Comparison table
```

## Measurement Flow

```
                 ┌─────────────────────────────┐
                 │ Phase 1: Load Model          │
                 │   shared handle for seq+batch│
                 └─────────────┬───────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Sequential  │   │ Batched      │   │ Parallel     │
    │ generate()  │   │ batch_generate│  │ mp.Process   │
    │ 1 at a time │   │ varying bs   │   │ own model ea.│
    │ baseline    │   │ KV cache     │   │ gen per worker│
    └─────────────┘   └──────────────┘   └──────────────┘
            │               │                   │
            └───────────────┼───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ Comparison Table │
                  │ tok/s, speedup   │
                  │ latency/prompt   │
                  └──────────────────┘
```

## CLI Usage

```bash
uv run bench.py --model ~/models/qwen.mlx \
  --num-prompts 8 --max-tokens 64 \
  --batch-sizes 2 4 8 --worker-counts 1 2 \
  --output results.json
```
