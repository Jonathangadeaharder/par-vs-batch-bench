# par-vs-batch-bench

Benchmark comparing parallel model instances vs batched inference on MLX.

Compares pure generation throughput (excludes model loading time):

- **A) Sequential**: `generate()` one prompt at a time (baseline, KV cache)
- **B) Parallel**: N model instances, each handles 1/N prompts
- **C) Batched**: Manual prompt KV prefill, batch generation

## Usage

```bash
uv run bench.py --help
# Example:
uv run bench.py --model <path_to_mlx_model>
```

## Requirements

- macOS with Apple Silicon
- `uv` package manager
