# Project Synapse Backend

Synapse is an evidence-first toolkit for diagnosing and, only when measurements justify it, improving local LLM inference workflows.

The first milestone is a reproducible Ollama benchmark harness. The streaming proxy remains behind a benchmark gate: it will not be presented as an optimization until it demonstrates value over correctly configured native Ollama.

## Repository boundary

This repository contains the Python backend, benchmark harness, API contracts, tests, and operational documentation. The dashboard is maintained in the separate `project-synapse-ui` repository.

## Current status

- Phase: benchmark foundation
- Supported environment: Ubuntu on WSL2, with Ollama running on the Windows host
- Locally available models: `qwen2.5:3b` and `qwen2.5-coder:7b`
- Proxy implementation: gated on completed baseline evidence

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for boundaries and [OVERNIGHT_LOG.md](OVERNIGHT_LOG.md) for the current run.

## Benchmark commands

Install the locked development environment and run local quality checks:

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

From Ubuntu/WSL2, run a bounded smoke profile before the full Phase 0 suite:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/synapse-bench-venv"
uv sync --locked --dev
uv run synapse-bench run --profile quick
uv run synapse-bench run --profile full
```

Keep the WSL environment outside the repository. Windows and Linux virtual environments are
not binary-compatible and must not share `.venv`.

Use `--scenario b0` through `--scenario b4` to select individual scenarios. The harness
discovers the WSL default gateway automatically. Override it with `OLLAMA_HOST` only when
needed, for example `OLLAMA_HOST=http://172.27.224.1:11434`. Public hosts are rejected unless
`SYNAPSE_ALLOW_REMOTE=true` is explicitly set.

Results are written atomically under `benchmarks/results/runs/`. B0 is *runtime-cold*: it
unloads Ollama residency but does not clear the operating-system page cache or emulate a reboot.

## Privacy

Synapse must not persist prompt bodies or make undeclared remote calls. Benchmark artifacts contain timings and environment metadata, not prompt content.
