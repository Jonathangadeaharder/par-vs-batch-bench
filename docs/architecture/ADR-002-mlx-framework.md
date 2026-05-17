# ADR-002: MLX Framework for ML Benchmarking

**Status:** Accepted

**Context:** The benchmark compares parallel and batched inference strategies. The framework must run efficiently on Apple Silicon (M-series chips) with unified memory architecture. GPU acceleration is essential for meaningful throughput measurements.

**Decision:** Use Apple's MLX framework via the `mlx-lm` Python package. MLX provides NumPy-like API with automatic GPU acceleration on Apple Silicon. The `mlx_lm.generate()` function handles autoregressive text generation with KV cache management.

**Consequences:**
- Positive: Native Apple Silicon performance (no Rosetta, no CUDA overhead)
- Positive: Unified memory allows large models without PCIe transfers
- Positive: MLX's functional API enables manual batch prefill manipulation
- Negative: MLX-only compatibility (no NVIDIA GPU support)
- Negative: Smaller ecosystem than PyTorch or TensorFlow

**Alternatives:**
- PyTorch MPS backend: Less mature than MLX for Apple Silicon
- llama.cpp: C++ native but harder to manipulate KV cache programmatically
