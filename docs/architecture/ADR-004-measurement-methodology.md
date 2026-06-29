# ADR-004: Measurement Methodology

**Status:** Accepted

**Context:** Accurate benchmarking must exclude model loading time (which varies by strategy) and measure only generation. MLX adds two traps: it is **lazy** (timers wrap graph construction, not execution, unless forced) and it **JIT-compiles per tensor shape** (the first run of a new shape pays compilation). Both inflate whichever config runs first if left unhandled.

**Decision:**
- **Exclude load time** everywhere. Sequential and batched share one loaded handle; parallel uses each worker's generation time (not wall, which would include the worker's own load).
- **Force compute** with `mx.eval()` on the sampled token each step so timings measure execution.
- **Discard a per-shape warmup run** for every config before measuring (kills JIT bias).
- **Report the median of N repeats** (`--repeats`, default 3) to absorb thermal/scheduler noise.
- **Greedy (argmax) decode in all modes** — removes sampling variance and keeps the comparison apples-to-apples.
- **Memory axis:** `mx.get_peak_memory()` per config; parallel sums per-process peaks.
- Throughput in tokens/second; speedup relative to the sequential baseline.

**Consequences:**
- Positive: Fair across strategies that load and compile differently; numbers are reproducible.
- Positive: Memory axis exposes the weights-replicated cost of parallelism that throughput alone hides.
- Negative: Warmup + repeats multiply wall time (~`(repeats+1)×` per config).
- Negative: Shared model for sequential/batched does not represent cold-start latency.
- Negative: Small/overhead-bound models still show noisy batch scaling (documented in findings).

**Alternatives:**
- Include load time: penalizes parallel, inflates sequential advantage. Rejected.
- Single run, no warmup: cheaper but JIT + laziness make the first config unfair. Rejected (this was the original approach; it produced non-monotonic, unreproducible curves).
