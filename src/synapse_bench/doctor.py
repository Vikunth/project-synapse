"""Pure, offline rules for diagnosing a V1 benchmark artifact."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from synapse_bench.doctor_models import (
    Acceptance,
    Diagnosis,
    DoctorReport,
    Evidence,
    OllamaControl,
    Recommendation,
    ReportSummary,
    Severity,
    Source,
)
from synapse_bench.models import RunArtifact, ScenarioResult, Status, TrialResult
from synapse_bench.stats import percentile

ANALYZER_VERSION = "1.0.1"
RESIDENCY_REDUCTION_PERCENT = 30.0
RESIDENCY_INFO_PERCENT = 10.0
RESIDENCY_REDUCTION_SECONDS = 1.0
TAIL_RATIO = 3.0
PEAK_FRACTION = 0.9
DOMINATED_THROUGHPUT_IMPROVEMENT = 0.1
DOMINATED_P95_GROWTH = 0.5
SUMMARY_RELATIVE_TOLERANCE = 0.001
SUMMARY_ABSOLUTE_TOLERANCE = 0.000001


@dataclass
class Analysis:
    diagnoses: list[Diagnosis] = field(default_factory=list)
    recommendations: dict[str, Recommendation] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    complete: bool = True

    def limit(self, message: str) -> None:
        self.complete = False
        if message not in self.limitations:
            self.limitations.append(message)


def _scenario_map(artifact: RunArtifact, analysis: Analysis) -> dict[str, ScenarioResult]:
    counts = Counter(scenario.name for scenario in artifact.scenarios)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        analysis.limit("Duplicate scenario names make scenario selection ambiguous.")
    result: dict[str, ScenarioResult] = {}
    for scenario in artifact.scenarios:
        identities = Counter(
            (
                trial.scenario,
                trial.trial,
                trial.model,
                trial.prefix_tokens_target,
                trial.concurrency,
            )
            for trial in scenario.trials
        )
        valid_trials = [
            trial
            for trial in scenario.trials
            if trial.scenario == scenario.name
            and identities[
                (
                    trial.scenario,
                    trial.trial,
                    trial.model,
                    trial.prefix_tokens_target,
                    trial.concurrency,
                )
            ]
            == 1
        ]
        if len(valid_trials) != len(scenario.trials):
            analysis.limit("Mismatched or duplicate trial identities were excluded from analysis.")
        result.setdefault(scenario.name, scenario.model_copy(update={"trials": valid_trials}))
    return result


def _successful(scenario: ScenarioResult) -> list[TrialResult]:
    return [trial for trial in scenario.trials if trial.status == Status.SUCCESS]


def _ttfts(scenario: ScenarioResult, *, trim: bool = False) -> list[float]:
    values = sorted(
        float(trial.ttft_s) for trial in _successful(scenario) if trial.ttft_s is not None
    )
    return values[1:-1] if trim and len(values) >= 5 else values


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=SUMMARY_RELATIVE_TOLERANCE,
        abs_tol=SUMMARY_ABSOLUTE_TOLERANCE,
    )


def _check_summary(
    analysis: Analysis, scenario: ScenarioResult, expected: dict[str, float | int]
) -> None:
    for key, computed in expected.items():
        claimed = scenario.summary.get(key)
        if (
            isinstance(claimed, bool)
            or not isinstance(claimed, (int, float))
            or not _close(float(claimed), float(computed))
        ):
            analysis.limit("Stored summaries conflict with recomputed trial metrics.")
            return


def _recommend_residency(diagnosis_id: str) -> Recommendation:
    return Recommendation(
        id="SYN-REC-RESIDENCY-EXPERIMENT-001",
        priority="now",
        type="experiment",
        title="Isolate model residency from prefix reuse",
        rationale=(
            "Warm-versus-cold evidence combines residency, loading, and possible prefix effects."
        ),
        ollama_controls=[OllamaControl(name="keep_alive", scope="request or server default")],
        action=(
            "Run a controlled native Ollama comparison with fixed prompts and explicit keep_alive."
        ),
        verification=(
            "Require repeated paired p95 measurements with identical model and context settings."
        ),
        risk=(
            "Changing residency can increase memory pressure; monitor memory and preserve a "
            "rollback value."
        ),
        diagnosis_ids=[diagnosis_id],
    )


def _diagnose_residency(scenarios: dict[str, ScenarioResult], analysis: Analysis) -> None:
    cold = scenarios.get("B0_runtime_cold")
    warm = scenarios.get("B1_stable_prefix")
    if cold is None or warm is None:
        analysis.limit("B0 and B1 are required to compare runtime-cold and warm latency.")
        return
    cold_values = _ttfts(cold, trim=True)
    warm_values = _ttfts(warm)
    if (
        len(_successful(cold)) < 5
        or len(_successful(warm)) < 4
        or not cold_values
        or not warm_values
    ):
        analysis.limit("B0 requires five successes and B1 requires four successes.")
        return
    cold_median = statistics.median(cold_values)
    warm_median = statistics.median(warm_values)
    reduction_s = cold_median - warm_median
    reduction_percent = reduction_s / cold_median * 100 if cold_median else 0.0
    _check_summary(
        analysis,
        cold,
        {"ttft_median_s": cold_median, "ttft_successful_trials": len(_successful(cold))},
    )
    _check_summary(
        analysis,
        warm,
        {"ttft_median_s": warm_median, "ttft_successful_trials": len(_successful(warm))},
    )
    diagnosis_id = "SYN-DOC-RESIDENCY-001"
    if (
        reduction_percent >= RESIDENCY_REDUCTION_PERCENT
        and reduction_s >= RESIDENCY_REDUCTION_SECONDS
    ):
        finding = "Warm TTFT is materially lower, but the result is confounded by model residency."
        title = "Controlled residency experiment is warranted"
    elif reduction_percent >= RESIDENCY_INFO_PERCENT:
        finding = "Warm TTFT is moderately lower, but this does not isolate prefix reuse."
        title = "Warm-start benefit is moderate"
    else:
        finding = "Warm TTFT shows no material reduction over runtime-cold TTFT."
        title = "No material warm-start benefit"
    analysis.diagnoses.append(
        Diagnosis(
            id=diagnosis_id,
            severity="info",
            category="residency",
            title=title,
            finding=finding,
            confidence="high",
            evidence=Evidence(
                scenario="B0_runtime_cold+B1_stable_prefix",
                metric="ttft_reduction",
                value=round(reduction_percent, 6),
                unit="percent",
                sample_count=len(_successful(cold)) + len(_successful(warm)),
            ),
            limitations=["This comparison cannot prove a KV-prefix-cache effect."],
            recommendation_ids=["SYN-REC-RESIDENCY-EXPERIMENT-001"],
        )
    )
    analysis.recommendations["SYN-REC-RESIDENCY-EXPERIMENT-001"] = _recommend_residency(
        diagnosis_id
    )


def _diagnose_prefix_tail(scenarios: dict[str, ScenarioResult], analysis: Analysis) -> None:
    scaling = scenarios.get("B2_prefix_scaling")
    if scaling is None:
        analysis.limit("B2 is required to evaluate prefix latency tails.")
        return
    grouped: dict[int, list[float]] = {}
    for trial in _successful(scaling):
        if trial.prefix_tokens_target is not None and trial.ttft_s is not None:
            grouped.setdefault(trial.prefix_tokens_target, []).append(trial.ttft_s)
    expected_prefixes = sorted(
        {
            trial.prefix_tokens_target
            for trial in scaling.trials
            if trial.prefix_tokens_target is not None
        }
    )
    if not expected_prefixes:
        analysis.limit("B2 requires at least one measured prefix bucket.")
        return
    for prefix in expected_prefixes:
        values = grouped.get(prefix, [])
        if len(values) < 4:
            analysis.limit("Each requested B2 prefix bucket requires four successful trials.")
            continue
        median = statistics.median(values)
        p95 = percentile(values, 95)
        assert p95 is not None
        _check_summary(
            analysis,
            scaling,
            {
                f"{prefix}_ttft_median_s": median,
                f"{prefix}_ttft_p95_s": p95,
                f"{prefix}_ttft_successful_trials": len(values),
            },
        )
        ratio = p95 / median if median else math.inf
        if ratio < TAIL_RATIO:
            continue
        diagnosis_id = f"SYN-DOC-PREFIX-TAIL-{prefix:04d}"
        analysis.diagnoses.append(
            Diagnosis(
                id=diagnosis_id,
                severity="warning",
                category="latency_tail",
                title=f"Heavy TTFT tail at the {prefix}-token prefix target",
                finding="The recomputed p95-to-median TTFT ratio meets the instability threshold.",
                confidence="high",
                evidence=Evidence(
                    scenario="B2_prefix_scaling",
                    metric="ttft_p95_to_median_ratio",
                    value=round(ratio, 6),
                    unit="ratio",
                    sample_count=len(values),
                ),
                limitations=["Five-trial buckets provide only a coarse tail estimate."],
                recommendation_ids=["SYN-REC-PREFIX-TAIL-OBSERVE-001"],
            )
        )
    if any(item.category == "latency_tail" for item in analysis.diagnoses):
        ids = [item.id for item in analysis.diagnoses if item.category == "latency_tail"]
        analysis.recommendations["SYN-REC-PREFIX-TAIL-OBSERVE-001"] = Recommendation(
            id="SYN-REC-PREFIX-TAIL-OBSERVE-001",
            priority="next",
            type="observe",
            title="Repeat unstable prefix buckets",
            rationale="Tail instability should be reproduced before selecting an intervention.",
            action="Repeat affected B2 buckets while recording queue depth and runtime state.",
            verification=(
                "Confirm the p95-to-median ratio remains at least 3.0 across independent runs."
            ),
            risk="Small samples can overstate p95; do not tune from this run alone.",
            diagnosis_ids=ids,
        )


def _diagnose_residency_eviction(scenarios: dict[str, ScenarioResult], analysis: Analysis) -> None:
    scenario = scenarios.get("B3_model_residency")
    if scenario is None:
        analysis.limit("B3 is required to evaluate model eviction.")
        return
    if scenario.status == Status.INFEASIBLE:
        analysis.limit("B3 was explicitly infeasible, so model eviction is not evaluable.")
        return
    successful = _successful(scenario)
    if len(successful) < 4:
        analysis.limit("B3 requires four successes or an explicit infeasible status.")
        return
    evictions = sum(len(trial.evicted_models or []) for trial in successful)
    warning = evictions > 0
    diagnosis_id = "SYN-DOC-EVICTION-001"
    recommendation_id = (
        "SYN-REC-EVICTION-EXPERIMENT-001" if warning else "SYN-REC-EVICTION-NO-CHANGE-001"
    )
    analysis.diagnoses.append(
        Diagnosis(
            id=diagnosis_id,
            severity="warning" if warning else "info",
            category="model_eviction",
            title="Model evictions observed" if warning else "No model evictions observed",
            finding=(
                "Alternating-model trials recorded an eviction."
                if warning
                else "Alternating-model trials did not record an eviction."
            ),
            confidence="high",
            evidence=Evidence(
                scenario="B3_model_residency",
                metric="observed_evicted_model_count",
                value=evictions,
                unit="models",
                sample_count=len(successful),
            ),
            recommendation_ids=[recommendation_id],
        )
    )
    analysis.recommendations[recommendation_id] = Recommendation(
        id=recommendation_id,
        priority="next" if warning else "defer",
        type="experiment" if warning else "do_not_change",
        title="Reproduce model eviction" if warning else "Keep residency settings unchanged",
        rationale=(
            "Eviction needs a controlled memory experiment."
            if warning
            else "This run provides no evidence that residency tuning is needed."
        ),
        action=(
            "Repeat B3 with memory telemetry and fixed residency controls."
            if warning
            else "Do not change residency settings based on B3."
        ),
        verification="Compare eviction counts across at least three independent runs.",
        risk="Residency changes can cause unsafe memory pressure.",
        diagnosis_ids=[diagnosis_id],
    )


def _diagnose_concurrency(scenarios: dict[str, ScenarioResult], analysis: Analysis) -> None:
    scenario = scenarios.get("B4_concurrency")
    if scenario is None:
        analysis.limit("B4 is required to identify a concurrency knee.")
        return
    analysis.limit(
        "B4 aggregate throughput comes from an unverified producer summary because "
        "RunArtifact 1.0 does not persist batch elapsed time."
    )
    requested: dict[int, list[TrialResult]] = {}
    for trial in scenario.trials:
        if trial.concurrency is not None:
            requested.setdefault(trial.concurrency, []).append(trial)
    if 1 not in requested or len(requested) < 2:
        analysis.limit("B4 requires concurrency 1 and at least one higher level.")
        return
    if any(
        any(trial.status != Status.SUCCESS for trial in trials) for trials in requested.values()
    ):
        analysis.limit("Every requested B4 trial must succeed for knee selection.")
        return
    points: list[tuple[int, float, float, int]] = []
    for level, trials in sorted(requested.items()):
        totals = [trial.total_s for trial in trials]
        claimed_throughput = scenario.summary.get(f"c{level}_tokens_per_second")
        if (
            any(value is None for value in totals)
            or isinstance(claimed_throughput, bool)
            or not isinstance(claimed_throughput, (int, float))
        ):
            analysis.limit("B4 requires aggregate throughput and total-latency metrics.")
            return
        clean_totals = [float(value) for value in totals if value is not None]
        # Aggregate batch elapsed time is not persisted per trial, so aggregate
        # throughput is the one consumed summary metric that cannot be recomputed.
        throughput = float(claimed_throughput)
        p95 = percentile(clean_totals, 95)
        assert p95 is not None
        _check_summary(
            analysis,
            scenario,
            {
                f"c{level}_p95_s": p95,
                f"c{level}_successful": len(trials),
            },
        )
        points.append((level, throughput, p95, len(trials)))
    peak = max(point[1] for point in points)
    knee = next(point for point in points if point[1] >= PEAK_FRACTION * peak)
    knee_id = "SYN-DOC-CONCURRENCY-KNEE-001"
    analysis.diagnoses.append(
        Diagnosis(
            id=knee_id,
            severity="info",
            category="concurrency",
            title=f"Concurrency {knee[0]} is the measured throughput knee",
            finding="This is the smallest level reaching at least 90% of measured peak throughput.",
            confidence="medium",
            evidence=Evidence(
                scenario="B4_concurrency",
                metric="concurrency_knee",
                value=knee[0],
                unit="requests",
                sample_count=knee[3],
            ),
            limitations=[
                "Aggregate throughput is an unverified producer summary in RunArtifact 1.0.",
                "The selected level is descriptive, not a recommended production value.",
            ],
            recommendation_ids=["SYN-REC-CONCURRENCY-EXPERIMENT-001"],
        )
    )
    diagnosis_ids = [knee_id]
    for level, throughput, p95, count in points:
        improvement = (throughput - knee[1]) / knee[1] if knee[1] else math.inf
        p95_growth = (p95 - knee[2]) / knee[2] if knee[2] else math.inf
        if (
            level > knee[0]
            and improvement < DOMINATED_THROUGHPUT_IMPROVEMENT
            and p95_growth >= DOMINATED_P95_GROWTH
        ):
            diagnosis_id = f"SYN-DOC-CONCURRENCY-DOMINATED-{level:04d}"
            diagnosis_ids.append(diagnosis_id)
            analysis.diagnoses.append(
                Diagnosis(
                    id=diagnosis_id,
                    severity="warning",
                    category="concurrency",
                    title=f"Concurrency {level} is dominated by the measured knee",
                    finding=(
                        "Throughput improves less than 10% while p95 latency grows at least 50%."
                    ),
                    confidence="medium",
                    evidence=Evidence(
                        scenario="B4_concurrency",
                        metric="p95_growth_vs_knee",
                        value=round(p95_growth * 100, 6),
                        unit="percent",
                        sample_count=count,
                    ),
                    limitations=[
                        "Aggregate throughput is an unverified producer summary in RunArtifact 1.0."
                    ],
                    recommendation_ids=["SYN-REC-CONCURRENCY-EXPERIMENT-001"],
                )
            )
    analysis.recommendations["SYN-REC-CONCURRENCY-EXPERIMENT-001"] = Recommendation(
        id="SYN-REC-CONCURRENCY-EXPERIMENT-001",
        priority="next",
        type="experiment",
        title="Control native parallelism and queueing",
        rationale="The knee and dominated levels need a paired native-runtime experiment.",
        ollama_controls=[OllamaControl(name="OLLAMA_NUM_PARALLEL", scope="server process")],
        action="Repeat B4 with explicit native parallelism and observed queue depth.",
        verification="Compare paired p95 latency, throughput, failures, and memory behavior.",
        risk="More parallelism increases memory use and may worsen tail latency.",
        diagnosis_ids=diagnosis_ids,
    )


def _diagnose_failures(scenarios: dict[str, ScenarioResult], analysis: Analysis) -> None:
    trials = [trial for scenario in scenarios.values() for trial in scenario.trials]
    failed = [trial for trial in trials if trial.status in {Status.FAILED, Status.PARTIAL}]
    partial_scenarios = sum(
        scenario.status in {Status.FAILED, Status.PARTIAL} for scenario in scenarios.values()
    )
    if not failed and not partial_scenarios:
        return
    rate = len(failed) / len(trials) if trials else 1.0
    severity: Severity = "critical" if rate >= 0.2 else "warning"
    categories = Counter(_safe_error_category(trial.error_type) for trial in failed)
    safe_category_counts = ", ".join(
        f"{name}={count}" for name, count in sorted(categories.items())
    )
    diagnosis_id = "SYN-DOC-FAILURES-001"
    analysis.diagnoses.append(
        Diagnosis(
            id=diagnosis_id,
            severity=severity,
            category="reliability",
            title="Benchmark failures limit the evidence",
            finding="The artifact contains failed or partial measurements.",
            confidence="high",
            evidence=Evidence(
                scenario="all",
                metric="failed_trial_rate",
                value=round(rate * 100, 6),
                unit="percent",
                sample_count=len(trials),
            ),
            limitations=[
                "Raw error messages are intentionally excluded from this report.",
                f"Safe failure categories: {safe_category_counts or 'none'}.",
            ],
            recommendation_ids=["SYN-REC-FAILURES-001"],
        )
    )
    analysis.recommendations["SYN-REC-FAILURES-001"] = Recommendation(
        id="SYN-REC-FAILURES-001",
        priority="now",
        type="observe",
        title="Resolve benchmark failures",
        rationale="Incomplete measurements cannot support tuning or product decisions.",
        action="Inspect the local artifact privately, fix the failure class, and rerun.",
        verification="Require the intended scenario success counts with no partial status.",
        risk="Proceeding can turn an infrastructure failure into a false product conclusion.",
        diagnosis_ids=[diagnosis_id],
    )
    analysis.complete = False


def _safe_error_category(error_type: str | None) -> str:
    normalized = (error_type or "").casefold()
    if "timeout" in normalized:
        return "timeout"
    if "connection" in normalized or "network" in normalized:
        return "connection"
    if "memory" in normalized or "oom" in normalized:
        return "memory"
    if "validation" in normalized or "schema" in normalized:
        return "validation"
    if "http" in normalized or "status" in normalized:
        return "upstream_http"
    return "other"


def _environment_limitations(artifact: RunArtifact, analysis: Analysis) -> None:
    configuration = artifact.configuration
    environment = artifact.environment

    def meaningful(value: Any) -> bool:
        if value is None or value is False:
            return False
        if isinstance(value, str):
            normalized = value.strip().casefold()
            return bool(normalized) and normalized not in {"unknown", "null", "none"}
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    required_groups = {
        "RAM capacity": any(
            ("ram" in key.casefold() or "memory" in key.casefold()) and meaningful(value)
            for key, value in environment.items()
        ),
        "GPU details": any(
            "gpu" in key.casefold() and meaningful(value) for key, value in environment.items()
        ),
        "context setting": any(
            ("ctx" in key.casefold() or "context" in key.casefold()) and meaningful(value)
            for key, value in configuration.items()
        ),
        "current Ollama controls": any(
            ("keep_alive" in key.casefold() or "parallel" in key.casefold()) and meaningful(value)
            for key, value in configuration.items()
        ),
    }
    missing = [name for name, present in required_groups.items() if not present]
    if missing:
        analysis.limit(
            "Memory tuning is unavailable because the artifact omits: " + ", ".join(missing) + "."
        )


def analyze_artifact(artifact: RunArtifact, source: Source) -> DoctorReport:
    """Apply deterministic rules without reading the host or making external calls."""
    analysis = Analysis()
    scenarios = _scenario_map(artifact, analysis)
    _diagnose_residency(scenarios, analysis)
    _diagnose_prefix_tail(scenarios, analysis)
    _diagnose_residency_eviction(scenarios, analysis)
    _diagnose_concurrency(scenarios, analysis)
    _diagnose_failures(scenarios, analysis)
    _environment_limitations(artifact, analysis)
    if analysis.limitations:
        diagnosis_id = "SYN-DOC-ARTIFACT-LIMITED-001"
        analysis.diagnoses.append(
            Diagnosis(
                id=diagnosis_id,
                severity="warning",
                category="artifact_sufficiency",
                title="Artifact limitations constrain diagnosis",
                finding="One or more evidence or metadata requirements are incomplete.",
                confidence="high",
                evidence=Evidence(
                    scenario="all",
                    metric="limitation_count",
                    value=len(analysis.limitations),
                    unit="limitations",
                    sample_count=sum(len(item.trials) for item in artifact.scenarios),
                ),
                limitations=sorted(analysis.limitations),
                recommendation_ids=["SYN-REC-ARTIFACT-RERUN-001"],
            )
        )
        analysis.recommendations["SYN-REC-ARTIFACT-RERUN-001"] = Recommendation(
            id="SYN-REC-ARTIFACT-RERUN-001",
            priority="now",
            type="observe",
            title="Capture richer benchmark metadata",
            rationale="Complete measurements and runtime metadata are required for safe diagnosis.",
            action=(
                "Use a future or extended producer to capture RAM, GPU, context, and current "
                "Ollama settings; rerunning the current runner alone cannot resolve those gaps."
            ),
            verification=(
                "Verify metadata limitations are resolved or explicitly scoped. Doctor and "
                "RunArtifact v1 remain limited for B4 producer-summary throughput; complete B4 "
                "analysis requires a future schema and analyzer with recomputable batch elapsed "
                "time and token totals."
            ),
            risk="Incomplete evidence can produce misleading configuration changes.",
            diagnosis_ids=[diagnosis_id],
        )
    analysis.diagnoses.sort(key=lambda item: item.id)
    recommendations = sorted(analysis.recommendations.values(), key=lambda item: item.id)
    limitations = sorted(analysis.limitations)
    counts = Counter(item.severity for item in analysis.diagnoses)
    return DoctorReport(
        analyzer_version=ANALYZER_VERSION,
        source=source,
        status="complete" if analysis.complete else "limited",
        summary=ReportSummary(
            headline="Native benchmark findings require controlled experiments before proxy work.",
            diagnosis_counts={
                "info": counts["info"],
                "warning": counts["warning"],
                "critical": counts["critical"],
            },
            proxy_gate="not_evaluable",
        ),
        diagnoses=analysis.diagnoses,
        recommendations=recommendations,
        limitations=limitations,
        acceptance=Acceptance(
            artifact_complete=analysis.complete,
            proxy_gate_requirements=[
                "Paired comparison demonstrates at least 30% p95 latency improvement.",
                "Added proxy overhead is below 10 milliseconds median.",
                "No reliability regression or unsafe memory behavior is observed.",
                "At least three external users reproduce the problem and value the intervention.",
            ],
        ),
    )


def contains_private_payload(value: Any) -> bool:
    """Reject prompt-bearing extensions before schema parsing or error echoing."""
    prohibited = {"prompt", "prompts", "messages", "response", "responses", "content"}
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and key.casefold() in prohibited)
            or contains_private_payload(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(contains_private_payload(item) for item in value)
    return False
