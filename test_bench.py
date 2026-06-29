# pyright: reportMissingImports=false
"""Correctness checks for the benchmark primitives.

Run: uv run --with mlx --with mlx-lm --with pytest pytest test_bench.py
"""
import json
import os
from pathlib import Path

import pytest

import bench


def test_median_odd():
    assert bench.median([3, 1, 2]) == 2


def test_median_even_averages_middle():
    assert bench.median([4, 1, 3, 2]) == pytest.approx(2.5)


def test_gen_batch_rejects_ragged_prompts():
    # The batched path has no attention mask, so unequal lengths must be rejected
    # rather than silently producing wrong results.
    class Tok:
        eos_token_id = 0

    with pytest.raises(AssertionError):
        bench.gen_batch(None, Tok(), [[1, 2, 3], [1, 2]], max_tokens=4)


def test_gen_batch_generates_within_budget():
    mlx_lm = pytest.importorskip("mlx_lm")  # skip if MLX stack absent
    model, tok = mlx_lm.load("mlx-community/Qwen3-0.6B-4bit")

    ids = [tok.encode("Explain entropy briefly."), tok.encode("Explain entropy briefly.")]
    total, ttft = bench.gen_batch(model, tok, ids, max_tokens=8)
    assert total > 0
    assert total <= 8 * len(ids)  # never exceeds max_tokens per sequence
    assert ttft > 0


def test_write_results_writes_bare_filename(tmp_path):
    out = bench.write_results([{"a": 1}], "out.json", base=str(tmp_path))
    assert out == Path(os.path.realpath(str(tmp_path))) / "out.json"
    assert json.loads(out.read_text()) == [{"a": 1}]


def test_write_results_rejects_subdir(tmp_path):
    with pytest.raises(ValueError):
        bench.write_results([{"a": 1}], "sub/out.json", base=str(tmp_path))


def test_write_results_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        bench.write_results([{"a": 1}], "../escape.json", base=str(tmp_path))


def test_write_results_rejects_absolute(tmp_path):
    with pytest.raises(ValueError):
        bench.write_results([{"a": 1}], "/etc/passwd", base=str(tmp_path))
