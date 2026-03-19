"""Main report builder orchestrator."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..structured_evaluator import StructuredEvaluator, StructuredEvaluatorConfig
from .content_stats import collect_content_stats, compute_coverage
from .formatters import (
    format_csv,
    format_json,
    format_markdown_table,
    format_text_summary,
)
from .models import EvaluationReport
from .outcome_stats import collect_outcome_stats
from .schema_stats import collect_schema_stats


def _compute_hash(data: dict) -> str:
    """Compute SHA256 hash of JSON data (first 8 chars)."""
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:8]


def _generate_output_name(
    json_schema: dict,
    gold_json: dict,
    extracted_json: dict,
) -> str:
    """Generate output name from timestamp and input hashes."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    combined = json.dumps(
        {"schema": json_schema, "gold": gold_json, "extracted": extracted_json},
        sort_keys=True,
    )
    combined_hash = hashlib.sha256(combined.encode()).hexdigest()[:8]
    return f"{timestamp}-{combined_hash}"


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    output_name: Optional[str] = None
    max_reasoning_length: int = 200
    top_n_lowest_fields: int = 5
    save_json: bool = True
    save_text: bool = True
    save_csv: bool = True
    save_markdown: bool = True


class ReportBuilder:
    """Builds evaluation reports from schema, gold, and extracted JSON."""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()

    def build(
        self,
        json_schema: dict,
        gold_json: dict,
        extracted_json: dict,
    ) -> EvaluationReport:
        """Synchronous wrapper for build_async."""
        return asyncio.run(self.build_async(json_schema, gold_json, extracted_json))

    async def build_async(
        self,
        json_schema: dict,
        gold_json: dict,
        extracted_json: dict,
    ) -> EvaluationReport:
        """Build evaluation report asynchronously."""
        # Run evaluation
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        eval_result = await evaluator.evaluate_async(
            json_schema, gold_json, extracted_json
        )

        schema_tree = eval_result["schema"]
        results = eval_result["results"]

        # Collect statistics
        schema_stats = collect_schema_stats(schema_tree)
        gold_stats = collect_content_stats(gold_json, "gold")
        extracted_stats = collect_content_stats(extracted_json, "extracted")
        coverage = compute_coverage(gold_json, extracted_json, schema_tree)

        # Collect outcome stats
        outcome_stats, field_outcomes = collect_outcome_stats(
            results,
            schema_tree,
            max_reasoning_length=self.config.max_reasoning_length,
            top_n_lowest=self.config.top_n_lowest_fields,
        )

        # Generate output name
        output_name = self.config.output_name or _generate_output_name(
            json_schema, gold_json, extracted_json
        )

        # Compute scores
        # Field score: flat average (each field counts equally)
        scores = [f.score for f in field_outcomes if f.score is not None]
        field_score = sum(scores) / len(scores) if scores else 0.0

        # Overall score: item-weighted (arrays weighted by gold item count)
        weighted_sum = 0.0
        total_weight = 0.0
        for f in field_outcomes:
            if f.score is None:
                continue
            if f.array_breakdown:
                weight = max(1, f.array_breakdown.matched + f.array_breakdown.missed_gold)
            else:
                weight = 1
            weighted_sum += f.score * weight
            total_weight += weight
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Build report
        report = EvaluationReport(
            output_name=output_name,
            timestamp=datetime.now().isoformat(),
            schema_hash=_compute_hash(json_schema),
            gold_hash=_compute_hash(gold_json),
            extracted_hash=_compute_hash(extracted_json),
            schema_stats=schema_stats,
            gold_stats=gold_stats,
            extracted_stats=extracted_stats,
            coverage=coverage,
            outcomes=outcome_stats,
            field_outcomes=field_outcomes,
            overall_score=overall_score,
            field_score=field_score,
            overall_pass_rate=outcome_stats.pass_rate,
        )

        return report

    def save(self, report: EvaluationReport) -> Path:
        """Save report to output directory.

        Creates a subdirectory with the output name containing:
        - report.json: Full machine-readable report
        - summary.txt: Human-readable summary
        - fields.csv: Per-field outcomes as CSV
        - fields.md: Per-field outcomes as markdown table

        Returns:
            Path to the output directory.
        """
        output_dir = self.config.output_dir / report.output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_json:
            json_path = output_dir / "report.json"
            json_path.write_text(format_json(report), encoding="utf-8")

        if self.config.save_text:
            text_path = output_dir / "summary.txt"
            text_path.write_text(format_text_summary(report), encoding="utf-8")

        if self.config.save_csv:
            csv_path = output_dir / "fields.csv"
            csv_path.write_text(format_csv(report.field_outcomes), encoding="utf-8")

        if self.config.save_markdown:
            md_path = output_dir / "fields.md"
            md_path.write_text(
                format_markdown_table(
                    report.field_outcomes,
                    max_reason_length=self.config.max_reasoning_length,
                ),
                encoding="utf-8",
            )

        return output_dir
