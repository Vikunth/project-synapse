# Synapse Overnight Log

## Morning summary

| Work item | Status | Notes |
|---|---|---|
| Repository split | Done | Backend and UI are separate public repositories with protected `main` branches. |
| Phase 0 harness | Done | B0-B4 quick/full CLI, hard timeouts, atomic artifacts, 43 tests, and CI. |
| Smoke results | Done | Live WSL quick run completed; prompt bodies were not persisted. |
| Full baseline | Done | Prompt-free B0-B4 native artifact committed as `benchmarks/results/baseline.json`. |
| Streaming proxy | Gated | Requires usable native baseline first. |
| Dashboard UI | Done | Accessible mock-only control room in its own repository; live adapter remains gated. |

### Pull requests

- Backend: [PR #1 - resilient Ollama benchmark harness](https://github.com/Vikunth/project-synapse/pull/1) (draft, `feat/benchmark-harness` -> `dev`)
- UI: [PR #1 - Phase 0 benchmark dashboard](https://github.com/Vikunth/project-synapse-ui/pull/1) (draft, `feat/dashboard` -> `dev`)

### Benchmark summary

- B0 quick: 16.28 s median runtime-cold TTFT (3 trials; not reboot/OS-cache cold).
- B1 quick: 0.81 s median stable-prefix TTFT, 95.0% below B0; high variance, not a full statistical claim.
- B2 quick: 0.84 s median TTFT at 128 target tokens; 4.11 s at 512.
- B3 quick: no eviction observed; 3B TTFT 2.21 s and 7B TTFT 52.21 s.
- B4 quick: 4.66 tok/s at concurrency 1; 4.46 tok/s at concurrency 4.

### Blockers for Vikunth

- No credential blocker. A proxy experiment still needs a specific hypothesis and native comparison threshold.

### First three morning actions

1. Review both draft PRs and the smoke methodology.
2. Review and accept the full native `baseline.json` methodology.
3. Define a narrow proxy hypothesis only if it can beat the native baseline without conflating residency and prefix reuse.

## Timeline

### 2026-07-21 03:00 IST — architecture

- Read the overnight instructions and private operating context.
- Chose an evidence-first two-repository architecture.
- Kept runtime agents, speculative scheduling, auth, and paper generation out of the benchmark milestone.
- Recorded failure handling, security boundaries, capacity assumptions, dependencies, and test strategy.

### 2026-07-21 03:56 IST - implementation and QA

- Published separate backend and UI repositories; protected both `main` branches.
- Built the B0-B4 harness and a truthful mock-only dashboard on feature branches.
- Independent QA found and closed hard-timeout, residency, eviction, privacy, error-state,
  progress, and accessibility defects. Final review reported no remaining P0-P3 findings.
- Windows verification: Ruff, formatting, strict MyPy, and 43/43 backend tests passed.
- UI verification: Oxlint, 4/4 Vitest tests, and the production build passed.

### 2026-07-21 04:00 IST - live WSL evidence

- Discarded one interrupted smoke artifact after discovering a cross-platform `.venv` collision;
  no benchmark had completed under the repaired harness at that point.
- Moved the WSL environment to Ubuntu's home directory and reran successfully.
- Completed B0/B1/B2/B4 smoke artifact `20260720T222147Z-78b81ec5.json`.
- Completed isolated B3 artifact `20260720T222441Z-a2b374f1.json` with a 90-second bound.
- Verified both artifacts contain no prompt field or URL userinfo. Unloaded both models after B3.

### 2026-07-21 04:17 IST - full native baseline

- Completed the full B0-B4 profile in 11m58s: 71/71 trials succeeded.
- B0 runtime-cold median TTFT: 15.594 s (trimmed standard deviation 1.072 s).
- B1 stable-prefix median TTFT: 0.635 s; 95.93% below B0, but the comparison includes
  residency and must not be presented as isolated KV-prefix-cache improvement.
- B2 warm TTFT medians: 0.656 s (128), 0.711 s (256), 0.682 s (512), 0.762 s (1024).
- B3 observed no model eviction across ten alternations; no cold-recovery value was applicable.
- B4 throughput: 1.48 tok/s (c1), 4.17 (c4), 4.67 (c8), 4.51 (c16); p95 latency
  increased from 16.21 s at c1 to 81.98 s at c16.
- Verified the artifact contains no prompt body or URL userinfo and copied it byte-for-byte to
  `benchmarks/results/baseline.json` (SHA-256 starts `2F9DEFA94B82CFF`).
- Kept proxy implementation gated: the native-only baseline does not prove that an external
  scheduler or prefix structure can add value beyond Ollama's existing residency/cache behavior.
