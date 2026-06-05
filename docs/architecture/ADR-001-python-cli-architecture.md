# ADR-001: Python CLI Benchmarking Tool

**Status:** Accepted

**Context:** The project needs to benchmark ML model inference throughput comparing sequential, parallel, and batched approaches. It must run on macOS with Apple Silicon and use the MLX framework.

**Decision:** Implement as a single-file Python CLI script using `argparse` for argument parsing and `mlx-lm` for model loading and inference. The script is executed via `uv run` with no formal package structure.

**Consequences:**
- Positive: Zero package overhead — single file to understand and modify
- Positive: Fast iteration cycle (edit and run, no build step)
- Negative: Poor discoverability for reusable modules
- Negative: No formal dependency declaration (no pyproject.toml or requirements.txt)

**Alternatives:**
- Rust CLI: Faster execution but harder to iterate on ML experiments
- Jupyter notebook: Poor for reproducible benchmarking
- Formally structured Python package: Overhead for a benchmarking script
