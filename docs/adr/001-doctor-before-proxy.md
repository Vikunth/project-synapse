# ADR 001: Build Doctor Before a Proxy

Status: accepted  
Date: 2026-07-21

## Context

The full native baseline found a large runtime-cold versus resident difference, cheap repeated
native prefills, no API-visible model eviction, and a concurrency plateau. Fixed order, repeated
prompts, and carried state prevent attribution to a Synapse intervention. A proxy would add a
critical-path failure surface before incremental value is proven.

## Decision

Build a CLI-only, offline Doctor that analyzes immutable `RunArtifact` v1 files. It produces
versioned JSON and Markdown and recommends only controlled, reversible native experiments.

No database or HTTP endpoint is introduced. Doctor has no runtime network capability and no
automatic setting mutation. The UI remains separate and unconnected. Proxy, scheduling, cache,
auth, telemetry, and enterprise features remain gated.

## Consequences

- Evidence becomes explainable and reusable without adding inference-path latency.
- Reports can be tested deterministically and shared without prompt content.
- Native configuration remains the comparison baseline.
- Doctor cannot observe current settings or prove improvement by itself; operators must run a
  controlled follow-up probe.
- A future UI may consume DoctorReport only after a separate schema/integration review.

## Rejected Alternatives

- **Transparent proxy now:** rejected because no paired evidence shows value over tuned native
  Ollama.
- **Automatic configuration mutation:** rejected because current settings and memory headroom
  are absent from the artifact.
- **Cloud report service:** rejected because it adds privacy, auth, retention, and availability
  concerns before demand.
- **Configurable rule policies:** deferred until real users establish which thresholds need
  customization.
