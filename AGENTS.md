# AGENTS.md — par-vs-batch-bench

## Build & Test Commands

```bash
# No pyproject.toml — skip uv sync. Dependencies loaded at runtime by mlx/mlx-lm.
uvx ruff check .
uvx ruff format .
uvx pyright .
```

## PR Instructions

- Branch: feature/*, fix/*, chore/*
- Title: `<type>(<scope>): <description>`
- Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore
