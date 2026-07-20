# Synapse Doctor Contract

`synapse doctor` is an offline analyzer for one existing Synapse benchmark artifact. It turns
prompt-free measurements into a deterministic report; it does not run Ollama, inspect live
settings, mutate configuration, intercept requests, or contact a network.

## CLI

```bash
synapse doctor benchmarks/results/baseline.json
synapse doctor benchmarks/results/baseline.json --format json
synapse doctor benchmarks/results/baseline.json --output-dir reports/
```

The default writes Markdown to stdout and creates no files. `--output-dir` stages both
`<run_id>.doctor.v1.json` and `<run_id>.doctor.v1.md`, then publishes each file atomically and
exclusively. The directory must exist, and either existing target causes the operation to fail
without overwriting anything. A caught second-publish failure rolls back the first; a process or
power failure between the two publishes can leave a partial pair and should be rerun after the
operator preserves or removes that partial output.

Exit codes:

- `0`: valid report, including ordinary findings.
- `1`: unexpected analysis or output failure.
- `2`: invalid path, JSON, or artifact schema.
- `3`: `--strict` report is limited or contains a warning/critical diagnosis.

## Trust and Privacy Boundary

```text
Untrusted local JSON
  -> bounded file/type/UTF-8/schema validation
  -> metric recomputation and safe normalization
  -> pure deterministic rules
  -> versioned DoctorReport
       -> escaped Markdown stdout
       -> optional staged JSON + Markdown; each file is atomic/exclusive

  X no Ollama, network, subprocess, environment read, or setting mutation
```

Inputs are limited to regular non-symlink files of at most 10 MiB, 100 scenarios, and 10,000
trials. Reports include the source basename, run ID, schema version, and file SHA-256. They never
copy prompt hashes, prompt bodies, absolute paths, host/IP values, raw error messages, or
arbitrary environment/configuration fields.

## Interpretation Rules

- Residency opportunity requires B0/B1 sufficiency, at least 30% relative TTFT reduction, and
  at least 1.0 second absolute difference. It is always labelled confounded—not KV-cache proof.
- B2 tail instability is directional when p95/median is at least 3.0 with four successes.
- The concurrency knee is the smallest measured level reaching 90% of peak throughput. A higher
  level is dominated when throughput improves less than 10% while p95 grows at least 50%.
  RunArtifact v1 does not preserve enough batch data to recompute throughput, so this diagnosis
  is always limited and explicitly labels the producer summary as unverified.
- Eviction recommendations require an observed B3 eviction; no eviction produces an explicit
  do-not-change finding.
- Invalid, missing, contradictory, or insufficient evidence lowers confidence and marks the
  report limited rather than inventing a metric.

Every recommendation states its evidence, confidence, limitation, risk, reversible operator
action, and verification step. Current setting values remain `unknown`; Doctor never pretends to
have observed them.

The current benchmark producer does not record validated RAM/GPU details, allocated context, or
active Ollama server controls. Rerunning the same producer cannot remove that limitation; a future
or extended artifact producer is required before Doctor may offer memory tuning.

RunArtifact/DoctorReport v1 also lacks the batch elapsed/token totals needed to recompute B4
throughput. A v1 report containing B4 therefore remains deliberately `limited`, and `--strict`
returns 3. A future artifact schema and analyzer—not a simple rerun—are required for a fully
recomputable concurrency diagnosis.

## Product Gate

A single native artifact can never open the proxy gate. Evaluation requires paired tuned-native
and intervention evidence showing at least 30% p95 TTFT improvement, under 10 ms median proxy
overhead, no streaming/schema regression, no unsafe memory behavior, and reproduction by at
least three external users.
