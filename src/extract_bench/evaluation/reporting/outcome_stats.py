"""Evaluation outcome statistics collector."""

import re
from collections import defaultdict
from statistics import median
from typing import Dict, List, Optional, Set, Tuple

from ...infra.nodes import RootSchema
from ..metrics.base_metric import MetricResult
from .models import (
    ArrayBreakdown,
    ConfusionCounts,
    ErrorBreakdown,
    ErrorGroup,
    FieldOutcome,
    LlmStats,
    LowScoringField,
    OutcomeStats,
)

LLM_METRIC_IDS = {"string_semantic", "string_llm", "array_llm"}

GOLD_ABSENT_REASONS = {"gold_missing", "gold_null", "gold_empty_array", "gold_absent"}
EXTRACTED_ABSENT_REASONS = {
    "extracted_missing",
    "extracted_null",
    "extracted_empty_array",
    "extracted_absent",
}
BOTH_ABSENT_REASONS = {"both_missing", "both_null", "both_absent", "both_empty_array"}


def _infer_schema_type(metric_id: str) -> str:
    """Infer schema type from metric_id prefix. Fallback to 'unknown' for custom metrics."""
    for prefix, schema_type in (
        ("string_", "string"),
        ("number_", "number"),
        ("integer_", "integer"),
        ("boolean_", "boolean"),
        ("array_", "array"),
    ):
        if metric_id.startswith(prefix):
            return schema_type
    return "unknown"


def normalize_path(path: str) -> str:
    """Normalize JSONPath to human-readable dot notation.

    $.properties.foo.properties.bar -> foo.bar
    """
    normalized = path
    normalized = re.sub(r"^\$\.?", "", normalized)
    normalized = re.sub(r"\.?properties\.?", ".", normalized)
    normalized = re.sub(r"^\.+|\.+$", "", normalized)
    normalized = re.sub(r"\.+", ".", normalized)
    return normalized or "$"


def is_internal_path(path: str) -> bool:
    """Check if path is an internal anyOf subpath that should be hidden."""
    return "anyOf[" in path


def _get_path_prefix(path: str) -> str:
    """Extract top-level prefix from normalized path."""
    normalized = normalize_path(path)
    parts = normalized.split(".")
    return parts[0] if parts else normalized


def _get_reason_category(result: MetricResult) -> str:
    """Categorize the failure reason."""
    details = result.details or {}
    reason = details.get("reason", "")

    if reason == "error" or "error" in details:
        return "error"
    if reason in ("both_missing", "both_absent", "both_null"):
        return "both_absent"
    if reason in ("gold_null", "gold_missing"):
        return "gold_absent"
    if reason in ("extracted_null", "extracted_missing"):
        return "extracted_absent"
    if "mismatch" in str(reason).lower():
        return "mismatch"
    if result.metric_id in LLM_METRIC_IDS:
        return "semantic_mismatch"
    if "number" in result.metric_id or "integer" in result.metric_id:
        return "numeric_mismatch"
    return "value_mismatch"


def _bucket_score(score: float) -> str:
    """Bucket a score into ranges."""
    if score < 0.2:
        return "0.0-0.2"
    if score < 0.4:
        return "0.2-0.4"
    if score < 0.6:
        return "0.4-0.6"
    if score < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def _extract_reasoning(result: MetricResult, max_length: int = 200) -> Optional[str]:
    """Extract reasoning from LLM metric result."""
    details = result.details or {}
    reasoning = details.get("reasoning")
    if reasoning is None:
        structured = details.get("structured_output", {})
        if isinstance(structured, dict):
            reasoning = structured.get("reasoning")

    if reasoning and len(reasoning) > max_length:
        return reasoning[:max_length] + "..."
    return reasoning


def _extract_array_breakdown(details: dict) -> Optional[ArrayBreakdown]:
    """Extract array breakdown from array_llm structured_output."""
    structured = details.get("structured_output")
    if not isinstance(structured, dict):
        return None
    summary = structured.get("matches_summary")
    agg = structured.get("aggregate_metrics")
    if not isinstance(summary, dict) or not isinstance(agg, dict):
        return None
    return ArrayBreakdown(
        matched=summary.get("matched", 0),
        missed_gold=summary.get("missed_gold", 0),
        spurious_pred=summary.get("spurious_pred", 0),
        precision=agg.get("precision", 0.0),
        recall=agg.get("recall", 0.0),
        f1=agg.get("f1", 0.0),
        matched_items=structured.get("matched_items"),
        missed_gold_items=structured.get("missed_gold_items"),
        spurious_pred_items=structured.get("spurious_pred_items"),
    )


def collect_outcome_stats(
    results: Dict[str, Dict[str, MetricResult]],
    schema: RootSchema,
    required_paths: Optional[Set[str]] = None,
    max_reasoning_length: int = 200,
    top_n_lowest: int = 5,
) -> Tuple[OutcomeStats, List[FieldOutcome]]:
    """Collect evaluation outcome statistics.

    Args:
        results: Evaluation results dict from StructuredEvaluator.
        schema: The evaluated schema AST.
        required_paths: Set of required field paths (normalized).
        max_reasoning_length: Max chars for reasoning in summaries.
        top_n_lowest: Number of lowest-scoring fields to track.

    Returns:
        Tuple of (OutcomeStats, list of FieldOutcome).
    """
    required_paths = required_paths or set()

    # Accumulators
    pass_by_metric: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0}
    )
    pass_by_type: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0}
    )
    pass_by_required: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0}
    )
    error_by_prefix: Dict[str, int] = defaultdict(int)
    error_by_metric: Dict[str, int] = defaultdict(int)
    error_by_reason: Dict[str, int] = defaultdict(int)

    # LLM stats accumulators
    llm_scores: List[float] = []
    llm_score_buckets: Dict[str, int] = defaultdict(int)
    low_scoring_fields: List[Tuple[float, LowScoringField]] = []

    # Per-field outcomes
    field_outcomes: List[FieldOutcome] = []

    # Confusion matrix accumulators
    tp, fp, fn, tn = 0, 0, 0, 0

    total_evaluated = 0
    total_passed = 0
    total_failed = 0

    for path, metric_results in results.items():
        if is_internal_path(path):
            continue

        normalized = normalize_path(path)
        is_required = normalized in required_paths or any(
            normalized.startswith(rp + ".") for rp in required_paths
        )
        req_key = "required" if is_required else "optional"

        for metric_id, result in metric_results.items():
            total_evaluated += 1
            passed = result.passed is True
            details = result.details or {}
            reason = details.get("reason", "")
            schema_type = _infer_schema_type(metric_id)

            if passed:
                total_passed += 1
                pass_by_metric[metric_id]["passed"] += 1
                # Confusion matrix: distinguish "both absent" (TN) from real match (TP)
                if reason in BOTH_ABSENT_REASONS:
                    tn += 1
                else:
                    tp += 1
            else:
                total_failed += 1
                pass_by_metric[metric_id]["failed"] += 1

                prefix = _get_path_prefix(path)
                error_by_prefix[prefix] += 1
                error_by_metric[metric_id] += 1
                error_by_reason[_get_reason_category(result)] += 1

                # Confusion matrix: gold absent -> FP (spurious), otherwise -> FN
                if reason in GOLD_ABSENT_REASONS:
                    fp += 1
                else:
                    fn += 1

            pass_by_type[schema_type]["passed" if passed else "failed"] += 1
            pass_by_required[req_key]["passed" if passed else "failed"] += 1

            # LLM stats
            if metric_id in LLM_METRIC_IDS and result.score is not None:
                llm_scores.append(result.score)
                llm_score_buckets[_bucket_score(result.score)] += 1

                reasoning = _extract_reasoning(result, max_reasoning_length)
                low_field = LowScoringField(
                    path=normalized,
                    metric_id=metric_id,
                    score=result.score,
                    reasoning=reasoning or "",
                )
                low_scoring_fields.append((result.score, low_field))

            # Build field outcome
            array_breakdown = (
                _extract_array_breakdown(details)
                if metric_id == "array_llm"
                else None
            )
            field_outcome = FieldOutcome(
                path=path,
                normalized_path=normalized,
                metric_id=metric_id,
                score=result.score,
                passed=passed,
                reason=reason or None,
                gold_value=details.get("gold"),
                extracted_value=details.get("extracted"),
                reasoning=_extract_reasoning(result, max_reasoning_length),
                array_breakdown=array_breakdown,
            )
            field_outcomes.append(field_outcome)

    # Build LLM stats
    llm_stats: Optional[LlmStats] = None
    if llm_scores:
        low_scoring_fields.sort(key=lambda x: x[0])
        top_lowest = [f for _, f in low_scoring_fields[:top_n_lowest]]

        llm_stats = LlmStats(
            call_count=len(llm_scores),
            avg_score=sum(llm_scores) / len(llm_scores),
            median_score=median(llm_scores),
            score_distribution=dict(llm_score_buckets),
            top_lowest_fields=top_lowest,
        )

    # Build error breakdown
    error_breakdown: Optional[ErrorBreakdown] = None
    if total_failed > 0:
        error_groups = [
            ErrorGroup(category=reason, count=count)
            for reason, count in sorted(error_by_reason.items(), key=lambda x: -x[1])
        ]
        error_breakdown = ErrorBreakdown(
            by_path_prefix=dict(error_by_prefix),
            by_metric_type=dict(error_by_metric),
            by_reason=dict(error_by_reason),
            details=error_groups,
        )

    # Build confusion counts
    confusion = ConfusionCounts(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
    )

    pass_rate = total_passed / total_evaluated if total_evaluated > 0 else 0.0

    outcome_stats = OutcomeStats(
        total_evaluated=total_evaluated,
        total_passed=total_passed,
        total_failed=total_failed,
        pass_rate=pass_rate,
        pass_by_metric={k: dict(v) for k, v in pass_by_metric.items()},
        pass_by_type={k: dict(v) for k, v in pass_by_type.items()},
        pass_by_required={k: dict(v) for k, v in pass_by_required.items()},
        confusion=confusion,
        llm_stats=llm_stats,
        error_breakdown=error_breakdown,
    )

    return outcome_stats, field_outcomes
