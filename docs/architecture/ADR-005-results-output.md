# ADR-005: Results Output and Comparison

**Status:** Accepted

**Context:** Benchmark results must be human-readable in the terminal and optionally machine-readable for tracking over time. The comparison table should clearly show speedup factors.

**Decision:** Output a formatted table with columns: Mode, Parameter (batch size / worker count), Tokens/sec, Speedup, and Latency. Results are also written to a JSON file (`results.json`) when `--output` is specified. Speedup is always relative to sequential baseline.

**Consequences:**
- Positive: Terminal output is immediately readable
- Positive: JSON output enables CI tracking and trend analysis
- Positive: Speedup column directly answers the primary research question
- Negative: Single sequential run is a noisy baseline (no standard deviation)
- Negative: JSON schema is ad-hoc, not versioned

**Alternatives:**
- CSV output: More portable but less readable
- Always write JSON: Users may not want file output on every run
