---
id: TDD-BNCH
kind: tdd
title: par-vs-batch-bench
description: 'MLX inference benchmark comparing sequential, batched, and parallel strategies'
status: draft
date: 2026-05-17T00:00:00.000Z
authors: []
reviewers: []
risk_level: low
scope_type: project
tags:
  - python
  - benchmark
  - mlx
related: []
checksum: 769ff7d75d94c1a8f4c27501b9257562545eb3e915fe1ed170216802087656b8
---

## Executive Summary

A single-purpose 383-line Python benchmarking script comparing sequential vs batched vs parallel MLX inference on Apple Silicon. Key finding: sequential generate() at 43.8 tok/s decisively outperforms both batched and parallel (~22 tok/s) on a 64GB machine. No tests, no packaging, no CI/CD, no documentation beyond the script docstring. Acceptable as a research artifact -- would need full rework for production reuse. Recommendation: add hardware context to results.json and archive unless reuse is planned.

## Scope

Assessed: single bench.py script (383 lines), results.json (64 lines), empty docs/ directories. Excluded: dependency management, CI/CD, testing, packaging, cross-platform support, cross-validation of results.

## Architecture

Single-file experimental benchmark with three independent functions: bench_sequential() (single-model single-process generate), bench_batched() (single-model with manual batch KV prefill using MLX low-level API), bench_parallel() (multi-process with per-process model instances and RAM-aware spawning). Argument parsing via argparse for model path, prompt count, max tokens, batch sizes, worker counts. RAM detection via sysctl for parallel feasibility check. Results output as formatted table + JSON serialization.

## Tech Stack

Python 3.11+, mlx, mlx-lm, uv (environment only, no pyproject.toml). Standard library: multiprocessing, argparse, json, gc, time, pathlib, subprocess, os. No formal dependency declaration -- deps loaded at import time in subprocess workers. No pinned versions -- behavior depends on whatever mlx version is installed.

## Code Quality

Clean structure for a research script. Each strategy is a separate function with clear contracts. Model loading time correctly excluded from generation timing. Memory-aware worker spawning prevents OOM. Commented-out code in batch_generate() suggests iterative development process. No error handling around MLX model loading. Hardcoded model path and 20 GB model size estimate. Dead code from abandoned approaches (lines 91-98).

## Security

Minimal attack surface -- single script with no network access. subprocess call for sysctl is macOS-specific. Hardcoded model path reveals local filesystem structure. No security review possible without formal project structure.

## Scalability & Performance

Script answers its research question definitively: sequential wins at 43.8 tok/s. Results specific to one hardware config (64GB Apple Silicon) and one model (Qwen3.6-35B-A3B-UD-MLX-4bit). Parallel benchmark throughput calculation methodology may not generalize. No cross-validation with different models or hardware. No statistical significance testing on results.

## Operations & DevOps

No CI/CD. No packaging. No release process. No version control discipline beyond the .git directory. No environment reproducibility. Must be run from repo root with mlx pre-installed.

## Dependencies & Third-Party Risk

mlx and mlx-lm loaded at runtime without pinned versions -- results not reproducible with different versions. Apple Silicon only. No supply chain controls. No pyproject.toml or requirements.txt.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Results not reproducible (unpinned deps) | High | Medium | Add pyproject.toml with pinned versions if reuse planned |
| No hardware context in results.json | High | Low | Add machine specs (chip, RAM, OS, mlx version) to results |
| No documentation or README | High | Low | Add docstring or brief README with methodology |
| Dead code in batch_generate() | Medium | Low | Clean up commented-out approaches if reusing |
| No cross-validation of results | Medium | Low | Document as single-run findings |

## Recommendations

1. Add machine specs (Apple Silicon model, RAM, OS version, mlx version) to results.json immediately (P1).
2. If reuse is planned: add pyproject.toml with pinned deps, README with methodology, clean up dead code (P2).
3. If not reusable: archive the repository with results.json as permanent research record (P2).
