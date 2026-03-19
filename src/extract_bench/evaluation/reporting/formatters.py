"""Output formatters for evaluation reports."""

import csv
import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any, List

from .models import EvaluationReport, FieldOutcome


class DataclassEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses."""

    def default(self, o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def format_json(report: EvaluationReport, indent: int = 2) -> str:
    """Format report as JSON."""
    return json.dumps(report, cls=DataclassEncoder, indent=indent, ensure_ascii=False)


def format_text_summary(report: EvaluationReport, max_width: int = 80) -> str:
    """Format report as human-readable text summary."""
    lines: List[str] = []
    sep = "=" * max_width

    lines.append(sep)
    lines.append(f"EVALUATION REPORT: {report.output_name}")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append(sep)

    # Overall scores
    lines.append("")
    lines.append("OVERALL RESULTS")
    lines.append("-" * 40)
    lines.append(f"  Overall Score:  {report.overall_score:.3f}  (item-weighted)")
    lines.append(f"  Field Score:    {report.field_score:.3f}  (flat average)")
    lines.append(f"  Pass Rate: {report.overall_pass_rate:.1%}")
    lines.append(f"  Evaluated: {report.outcomes.total_evaluated} fields "
                 f"({report.outcomes.total_passed} passed, "
                 f"{report.outcomes.total_failed} failed)")

    # Schema stats
    lines.append("")
    lines.append("SCHEMA STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Total Nodes: {report.schema_stats.total_nodes}")
    lines.append(f"  Required Keys: {report.schema_stats.required_keys_count}")
    lines.append(f"  Optional Keys: {report.schema_stats.optional_keys_count}")
    type_str = ", ".join(
        f"{t}: {c}" for t, c in sorted(report.schema_stats.counts_by_type.items())
    )
    lines.append(f"  By Type: {type_str}")

    # Coverage
    lines.append("")
    lines.append("COVERAGE")
    lines.append("-" * 40)
    lines.append(f"  Present in Both: {report.coverage.present_in_both}")
    lines.append(f"  Missing in Extracted: {report.coverage.missing_in_extracted}")
    lines.append(f"  Spurious in Extracted: {report.coverage.spurious_in_extracted}")
    lines.append(f"  Required Missing: {report.coverage.required_missing}")

    # Pass rate by metric
    lines.append("")
    lines.append("PASS RATE BY METRIC")
    lines.append("-" * 40)
    for metric_id, counts in sorted(report.outcomes.pass_by_metric.items()):
        passed = counts.get("passed", 0)
        failed = counts.get("failed", 0)
        total = passed + failed
        rate = passed / total if total > 0 else 0.0
        lines.append(f"  {metric_id}: {rate:.1%} ({passed}/{total})")

    # LLM stats
    if report.outcomes.llm_stats:
        llm = report.outcomes.llm_stats
        lines.append("")
        lines.append("LLM METRICS")
        lines.append("-" * 40)
        lines.append(f"  Calls: {llm.call_count}")
        lines.append(f"  Avg Score: {llm.avg_score:.3f}")
        lines.append(f"  Median Score: {llm.median_score:.3f}")

        if llm.top_lowest_fields:
            lines.append("")
            lines.append("  Top Lowest-Scoring Fields:")
            for field in llm.top_lowest_fields:
                lines.append(
                    f"    - {field.path} ({field.metric_id}): {field.score:.3f}"
                )
                if field.reasoning:
                    reasoning_preview = (
                        field.reasoning[:100] + "..."
                        if len(field.reasoning) > 100
                        else field.reasoning
                    )
                    lines.append(f"      Reason: {reasoning_preview}")

    # Error breakdown
    if report.outcomes.error_breakdown:
        eb = report.outcomes.error_breakdown
        lines.append("")
        lines.append("ERROR BREAKDOWN")
        lines.append("-" * 40)

        if eb.by_reason:
            lines.append("  By Reason:")
            for reason, count in sorted(eb.by_reason.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")

        if eb.by_path_prefix:
            lines.append("  By Path Prefix:")
            for prefix, count in sorted(eb.by_path_prefix.items(), key=lambda x: -x[1])[
                :5
            ]:
                lines.append(f"    {prefix}: {count}")

    # Array breakdown
    array_outcomes = [f for f in report.field_outcomes if f.array_breakdown is not None]
    if array_outcomes:
        lines.append("")
        lines.append("ARRAY BREAKDOWN")
        lines.append("-" * 40)
        for outcome in array_outcomes:
            ab = outcome.array_breakdown
            status = "PASS" if outcome.passed else "FAIL"
            lines.append(
                f"  {outcome.normalized_path} [{status}] score={outcome.score:.3f}"
            )
            lines.append(
                f"    Items: {ab.matched} matched, {ab.missed_gold} missed, {ab.spurious_pred} spurious"
            )
            lines.append(
                f"    P={ab.precision:.3f}  R={ab.recall:.3f}  F1={ab.f1:.3f}"
            )

    # Failed fields summary
    failed_outcomes = [f for f in report.field_outcomes if not f.passed]
    if failed_outcomes:
        lines.append("")
        lines.append("FAILED FIELDS (first 10)")
        lines.append("-" * 40)
        for outcome in failed_outcomes[:10]:
            lines.append(f"  {outcome.normalized_path}")
            lines.append(f"    Metric: {outcome.metric_id}, Score: {outcome.score:.3f}")
            if outcome.reason:
                lines.append(f"    Reason: {outcome.reason}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def format_markdown_table(
    field_outcomes: List[FieldOutcome], max_reason_length: int = 50
) -> str:
    """Format field outcomes as markdown table."""
    lines: List[str] = []

    lines.append(
        "| Path | Metric | Score | Passed | Reason | Matched | Missed | Spurious | P | R | F1 |"
    )
    lines.append(
        "|------|--------|-------|--------|--------|---------|--------|----------|---|---|----| "
    )

    for outcome in field_outcomes:
        reason = outcome.reason or ""
        if len(reason) > max_reason_length:
            reason = reason[:max_reason_length] + "..."
        passed_str = "Yes" if outcome.passed else "No"
        ab = outcome.array_breakdown
        if ab:
            lines.append(
                f"| {outcome.normalized_path} | {outcome.metric_id} | {outcome.score:.3f} "
                f"| {passed_str} | {reason} "
                f"| {ab.matched} | {ab.missed_gold} | {ab.spurious_pred} "
                f"| {ab.precision:.2f} | {ab.recall:.2f} | {ab.f1:.2f} |"
            )
        else:
            lines.append(
                f"| {outcome.normalized_path} | {outcome.metric_id} | {outcome.score:.3f} "
                f"| {passed_str} | {reason} "
                f"| | | | | | |"
            )

    return "\n".join(lines)


def format_csv(field_outcomes: List[FieldOutcome]) -> str:
    """Format field outcomes as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "path",
            "normalized_path",
            "metric_id",
            "score",
            "passed",
            "gold_value",
            "extracted_value",
            "reason",
            "reasoning",
            "matched",
            "missed_gold",
            "spurious_pred",
            "precision",
            "recall",
            "f1",
        ]
    )

    for outcome in field_outcomes:
        ab = outcome.array_breakdown
        writer.writerow(
            [
                outcome.path,
                outcome.normalized_path,
                outcome.metric_id,
                f"{outcome.score:.6f}",
                "true" if outcome.passed else "false",
                outcome.gold_value if outcome.gold_value is not None else "",
                outcome.extracted_value if outcome.extracted_value is not None else "",
                outcome.reason or "",
                outcome.reasoning or "",
                ab.matched if ab else "",
                ab.missed_gold if ab else "",
                ab.spurious_pred if ab else "",
                f"{ab.precision:.6f}" if ab else "",
                f"{ab.recall:.6f}" if ab else "",
                f"{ab.f1:.6f}" if ab else "",
            ]
        )

    return output.getvalue()
