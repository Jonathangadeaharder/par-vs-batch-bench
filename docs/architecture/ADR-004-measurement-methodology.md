# ADR-004: Measurement Methodology

**Status:** Accepted

**Context:** Accurate benchmarking must exclude model loading time (which varies by strategy) and measure only pure generation throughput. Parallel and sequential strategies measure different wall-clock times.

**Decision:** Exclude model loading time from all generation measurements. Sequential and batched share a loaded model handle. Parallel reports both load time and generation time per worker. Throughput is reported as tokens/second for comparison. Speedup is computed relative to sequential baseline.

**Consequences:**
- Positive: Fair comparison between strategies that load models differently
- Positive: Per-worker timing in parallel mode identifies stragglers
- Negative: Shared model for sequential/batched may not represent cold-start scenarios
- Negative: System memory pressure during parallel runs affects timing

**Alternatives:**
- Include load time: Penalizes parallel, inflates sequential advantage
- Average multiple runs: More statistically sound but significantly more total time
