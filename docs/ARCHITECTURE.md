# Synapse Architecture

Status: accepted for the benchmark milestone; proxy and cache optimizer remain gated.

## Scope and assumptions

- The backend and UI are separate repositories and communicate only through a versioned HTTP contract.
- Ollama runs on the Windows host. Benchmarks run inside Ubuntu/WSL2.
- The WSL host address is discovered from the default route unless `OLLAMA_HOST` is explicitly set. No bridge IP is hardcoded as a universal default.
- `keep_alive` controls model residency. It does not prove or manage KV-cache reuse.
- Prompt bodies are generated deterministically in memory and are never written to result artifacts.
- No database is justified for the benchmark milestone. Results are immutable JSON artifacts.

## Data flow and trust boundaries

```text
Developer / CI
      |
      v
Benchmark CLI (Ubuntu/WSL2)
  |   - deterministic prompt generator
  |   - residency polling
  |   - timing/statistics
  |
  | HTTP across WSL host boundary
  v
Ollama API (Windows host) ----> Local model files / CPU / accelerator

Future UI repository
      |
      | versioned read-only HTTP API
      v
Future Synapse backend -----> Ollama API
```

The WSL-to-Windows HTTP connection is a trust boundary. The benchmark rejects non-loopback/non-private Ollama hosts unless the operator explicitly opts in.

## Milestone boundaries

### Milestone 0: benchmark harness

Produces machine-readable results for runtime-cold latency, same-prefix warm behavior, prefix-length sensitivity, model-residency stress, and concurrency. A failed or infeasible scenario is recorded as structured evidence instead of hanging or inventing a number.

### Milestone 1: transparent proxy (gated)

May begin only after a usable native baseline exists. It must stream without buffering and initially perform no speculative batching or prompt reordering.

### Milestone 2: optimization experiments (gated)

Any scheduler or prefix policy must be isolated behind configuration, compared against tuned native Ollama, and removable without changing API compatibility.

## Planned endpoint inventory

| Method | Path | Input | Output | Auth | Errors |
|---|---|---|---|---|---|
| GET | `/health` | none | service and Ollama reachability | local-only default | `503` when Ollama is unavailable |
| GET | `/api/v1/status` | none | versioned UI status summary | local-only default | `503` degraded upstream |
| GET | `/api/v1/metrics/summary` | query time window | aggregate metadata only | local-only default | `400`, `503` |
| POST | `/api/generate` | Ollama payload | streamed Ollama response | passthrough initially | upstream status preserved |
| POST | `/api/chat` | Ollama payload | streamed Ollama response | passthrough initially | upstream status preserved |
| GET | `/api/ps` | none | upstream residency state | passthrough initially | upstream status preserved |
| POST | `/v1/chat/completions` | OpenAI-compatible payload | SSE or JSON response | passthrough initially | structured OpenAI-compatible error |
| GET | `/metrics` | none | Prometheus exposition | local-only default | `503` if registry unavailable |

No endpoint persists prompt content. Pluggable authentication is a later boundary, not a reason to add credentials to the benchmark.

## Failure and edge cases

1. Ollama is unreachable before a scenario starts.
2. Ollama accepts a connection but never returns headers.
3. A streaming response emits an error object before a token.
4. Model unload is requested but `/api/ps` still reports residency.
5. A model disappears during a warm sequence.
6. The requested model is not installed.
7. Available memory is insufficient for a model or two-model test.
8. WSL route discovery returns no Windows host address.
9. A result includes too few successful trials for statistics.
10. Concurrent requests are queued, rejected, cancelled, or partially complete.
11. Wall-clock timing moves or a process is suspended during a run.
12. Token approximation differs from the Ollama model tokenizer.
13. OS page cache makes a runtime-cold load warmer than a machine reboot.
14. An interrupted run leaves a model resident.
15. Result writing fails after measurements complete.

Each case must terminate within configured timeouts and appear as a structured status or error.

## Security review

- **Authentication/authorization:** benchmark has no server surface. Future API binds to loopback by default; remote access requires explicit configuration and later authentication.
- **Validation:** URLs, model names, numeric limits, concurrency, and output paths are validated before requests start.
- **Injection:** subprocess invocation is avoided in the Python harness. Model names are data, never shell fragments.
- **Secrets:** no credentials are required or logged. Environment variables containing secrets are not copied into artifacts.
- **Prompt privacy:** only prompt token targets and hashes may be recorded; generated prompt text is excluded.
- **Denial of service:** concurrency and timeouts are bounded. The harness refuses unbounded runs.
- **Supply chain:** dependencies are minimal, version constrained, and checked by CI.

## Capacity expectations

At roughly 100 users, a single local process is not a shared service; each workstation runs its own instance. Team aggregation is out of scope. At 1,000 users, release integrity, compatibility matrices, opt-in telemetry, and signed artifacts matter more than central request throughput. A future enterprise control plane requires a separate architecture review.

## Test plan

- Unit: configuration parsing, route discovery, prompt generation, statistics, status serialization, comparison thresholds.
- Contract: mocked Ollama responses, streaming first-token detection, errors, timeouts, and `/api/ps` transitions.
- Integration: real Ollama smoke tests marked separately and never required for ordinary unit CI.
- Regression: compare only like-for-like environments and scenario schema versions; do not fail CI on machine-dependent absolute timing.
- UI contract: OpenAPI fixture consumed by the separate UI repository.

## Dependencies and fallback behavior

- `httpx`: bounded async HTTP. On timeout, record the failed trial and continue only when the scenario remains valid.
- `pydantic-settings`: typed environment and CLI configuration.
- `typer`: explicit CLI surface.
- `pytest`/`pytest-asyncio`: deterministic tests.
- `ruff` and `mypy`: CI quality gates.

Plot generation is optional. If plotting dependencies are unavailable, JSON and CSV results remain authoritative and the plot is marked unavailable.
