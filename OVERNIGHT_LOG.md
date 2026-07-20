# Synapse Overnight Log

## Morning summary

| Work item | Status | Notes |
|---|---|---|
| Repository split | Done | Backend and UI are separate public repositories with protected `main` branches. |
| Phase 0 harness | Done | B0-B4 quick/full CLI, hard timeouts, atomic artifacts, 43 tests, and CI. |
| Smoke results | Done | Live WSL quick run completed; prompt bodies were not persisted. |
| Full baseline | Pending | Smoke observations are not the committed full baseline. |
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

- No credential blocker. The full B0-B4 profile remains required before proxy work.

### First three morning actions

1. Review both draft PRs and the smoke methodology.
2. Run/approve the full WSL profile before accepting `baseline.json`.
3. Keep the proxy gated until the full result shows incremental value beyond native Ollama.

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
