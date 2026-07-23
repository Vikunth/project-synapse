# Randomized Native Probe Protocol

Status: implemented and verification-gated; native-only, no proxy  
Artifact: `NativeProbeArtifact` schema 1.0

## Purpose

The original baseline found cold-load, repeated-prefill, dual-model pressure, and concurrency
signals, but fixed ordering, repeated prompt hashes, and carried runtime state prevent causal
claims. This probe randomizes matched native-Ollama comparisons so Doctor rules can be validated
without introducing a proxy.

It accepts no user prompts, stores no synthetic prompt bodies or model responses, mutates no
server settings, and calls only the configured private/local Ollama endpoint. The probe is a
single-machine experiment, not a service.

## Profiles and Safety

| Setting | Smoke | Full |
|---|---:|---:|
| Prefix pairs | 2 | 30 |
| Dual-residency pairs | 1 | 30 |
| Concurrency rounds | 1 at c1/c4 | 10 at c1/c4/c8/c16 |
| Overall bound | 20 minutes | 4 hours |
| Statistical finding | Never | Threshold-gated |

Full mode requires `--acknowledge-full-load`; that acknowledgement is persisted. The runner
checkpoints before and after each state transition and owns one per-user machine lock across all
output directories. Resume validates canonical configuration and schedule hashes and accepts no
seed, model, profile, probe, or timeout overrides.

The global lock stores the runner PID and fails closed. After a hard-killed process, inspect that
PID before manually removing the stale temporary lock; the runner never guesses that a lock is
stale.

The full dual-residency probe requires validated Windows-host telemetry because Ollama runs on
Windows. A hardcoded, non-interactive PowerShell/CIM collector records total/free host memory and
pagefile usage. A `/proc/meminfo` parser is tested but WSL caller-VM snapshots are not persisted in
schema 1.0; they cannot support host-safety or pressure claims. If Windows telemetry is unavailable,
the full dual probe is infeasible—not negative evidence.

Abort dual testing and unload owned models if host available memory falls below
`max(10% of total, 768 MiB)`, pagefile use grows by at least 512 MiB, Ollama returns OOM/5xx, two
consecutive requests time out, or expected residency disappears. The runner never kills Ollama or
deletes model files.

## Deterministic Scheduling

Schedules derive independent seeds from SHA-256 domains:

```text
native-probe-1.0:<seed>:prefix
native-probe-1.0:<seed>:dual
native-probe-1.0:<seed>:concurrency
native-probe-1.0:<seed>:bootstrap
```

Thirty-pair probes contain exactly 15 AB and 15 BA orders before deterministic shuffling.
Concurrency rounds independently permute c1/c4/c8/c16. The complete schedule and its SHA-256 are
stored before the first runtime mutation.

Every case has a unique synthetic namespace. Prompt text remains memory-only; artifacts store
opaque case IDs and Ollama's actual `prompt_eval_count`. A pair is invalid when matched counts
differ by more than two tokens or 1%, whichever is larger. Failed or unmatched units are retained
but never imputed or silently discarded.

## Prefix-Reuse Crossover

Each arm unloads the primary model, confirms absence, empty-preloads with fixed context, confirms
residency, warms once, measures once, captures state, and unloads.

```text
reuse:   warm P_i + S0; measure P_i + S1
control: warm Q_i + S0; measure R_i + S1
```

`P_i`, `Q_i`, and `R_i` are matched-length unique namespaces. Numeric warm-up token counts are
stored so all matched lengths can be verified without persisting prompt text. Full confirmation
requires:

- at least 27/30 valid pairs;
- median relative TTFT improvement at least 20%;
- median absolute improvement at least 0.150 seconds;
- deterministic 10,000-resample paired-bootstrap 95% lower bound above zero; and
- at least 24/30 pairs favoring reuse.

Order-stratified effects differing by at least 0.150 seconds or 20% of control median mark the
finding limited for carryover. Unload clears runtime residency, not the OS page cache.

## Dual-Residency Crossover

Each pair randomizes:

- `solo`: unload both, preload 3B, measure a unique matched primary prompt;
- `dual`: unload both, preload 3B then idle 7B, verify both via `/api/ps`, then measure primary.

Capture Windows host memory/pagefile telemetry plus full `/api/ps` snapshots before and after
measurement. A full pressure finding requires at least 27 pairs, at least 24 favoring
worse dual TTFT, median dual/solo ratio at least 2×, absolute regression at least 1.0 second, a
positive bootstrap lower bound, and either at least 15% host available-memory decline or pagefile
growth. Eviction is a separate observation.

## Repeated Concurrency

Run ten rounds, each with a randomized c1/c4/c8/c16 order. Preload only 3B and use unique prompts.
Persist batch elapsed time, generated-token total, individual TTFT/total latency, residency, and
telemetry so throughput is recomputable.

Saturation is confirmed only with ten complete batches per level, median c16/c8 throughput no
greater than 1.05, and median c16/c8 p95 latency at least 1.5. High confidence additionally
requires the throughput-ratio bootstrap upper bound at most 1.05 and latency-ratio lower bound at
least 1.5.

## Artifact and Recovery

The versioned artifact stores run identity, seed, schedule and configuration hashes,
environment/configuration, explicit per-section execution status, preflight, blocks and attempts,
analysis, cleanup, owned models, and timestamps. Writes use a same-directory temporary file,
flush/fsync, and `os.replace`.

```text
planned -> preflight -> running -> cleanup -> completed
                           |          |
                           v          v
                       interrupted  cleanup_failed
```

On resume, a running unit becomes interrupted and is retained. The runner resets owned native
state and reruns the entire pair/block as a new attempt. Only the latest complete attempt enters
analysis. Normal completion returns 0, incomplete/infeasible/aborted work returns 3, and cleanup
failure returns 4. Partial execution is explicit; it is never reported as a completed run.

## Interpretation Boundary

Smoke proves mechanics only. A failed full threshold means “not confirmed,” never “no effect.”
No result opens the proxy gate by itself; external-user reproduction and a paired tuned-native
versus intervention comparison remain required.
