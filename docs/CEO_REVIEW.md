# CEO Review: Post-Baseline Direction

Date: 2026-07-21  
Verdict: **RESHAPE IT**

## Decision

Do not build the transparent proxy, scheduler, or KV-cache optimizer next. Build only an
offline, read-only `synapse doctor` that converts an existing benchmark artifact into a clear
diagnosis and reversible native-Ollama experiments.

The full baseline proves material local latency pain, but it does not prove that an external
proxy is the solution. Runtime-cold median TTFT was 15.594 seconds while resident/stable-prefix
median TTFT was 0.635 seconds. That difference combines residency, OS-cache state, and exact
prior-prompt reuse; it is not isolated evidence of Synapse or KV-cache value.

## Real Product and First Users

The current product is a privacy-safe local inference doctor for developers and maintainers
running Ollama-backed coding agents on 8–32 GB workstations. The first ten users should already
experience model reloads, slow switching, or queue stalls. Their urgency and willingness to pay
remain assumptions, not evidence.

The do-nothing alternative is credible: configure Ollama `keep_alive` and concurrency, use one
model, inspect `ollama ps`, or adopt an existing gateway or observability tool.

## Evidence and Key Risk

- Native residency provides the largest observed latency difference.
- B2 shows strong first-prefill cost followed by cheap native repeats.
- Both models remained API-visible in B3, yet 3B TTFT worsened under dual-model pressure.
- Throughput plateaued near concurrency 8 while c16 p95 latency doubled.
- Fixed ordering, reused prompts, and state carryover limit causal conclusions.

The riskiest assumption is demand. Users may correctly configure Ollama and reject another tool.
Synapse has no unfair advantage today. A privacy-safe, cross-machine evidence corpus could become
one after real adoption.

## Minimum Scope

`synapse doctor` must:

1. Read one local `RunArtifact` without contacting Ollama or any remote service.
2. Produce deterministic versioned JSON and human-readable Markdown.
3. Explain cold loading, prefix evidence limits, model switching, and concurrency saturation.
4. Give evidence, confidence, risk, a reversible operator action, and a verification step.
5. Never persist prompts, mutate settings, intercept traffic, or open the proxy gate.

## 90-Day Success Criteria

- Observe 10 target users and retain 5 weekly testers.
- Reproduce at least three independent ≥30% p95 TTFT improvements over each user's tuned-native
  starting point.
- Obtain one paid pilot or explicit paid commitment.
- Complete the standard diagnostic within 15 minutes and preserve useful partial evidence after
  interruption.

## Explicitly Deferred

Proxying, OpenAI translation, request scheduling, KV-cache/PagedEviction claims, RAM admission,
authentication, databases, cloud telemetry, multi-runtime support, an expanded dashboard,
enterprise control planes, and runtime extensions remain out of scope.
