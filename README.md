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

## Privacy

Synapse must not persist prompt bodies or make undeclared remote calls. Benchmark artifacts contain timings and environment metadata, not prompt content.
