# ADR-003: Single-File Modular Architecture

**Status:** Accepted

**Context:** The benchmark compares three strategies: sequential (baseline), batched (manual KV prefill), and parallel (multi-process). Each strategy has distinct implementation requirements but shares model loading and prompt generation.

**Decision:** Organize the single file as clearly separated sections: imports, constants, helper functions (`batch_generate`, `ModelHandle`), benchmark functions (`bench_sequential`, `bench_batched`, `bench_parallel`), and main CLI entry point. Class-based model handle caches the loaded model across runs.

**Consequences:**
- Positive: All code visible in one file — easy to compare strategies side by side
- Positive: `ModelHandle` class prevents reloading model for sequential/batched runs
- Positive: Each bench function is independently testable
- Negative: Single file grows large if strategies are added
- Negative: No module-level imports for dependent scripts

**Alternatives:**
- Multi-module package: Better organization but harder to run as a one-off script
- Shell script wrapping MLX CLI: Inflexible for custom batch logic
