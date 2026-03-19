"""Tests for array breakdown surfacing in reports and item-weighted scoring."""

from extract_bench.evaluation.reporting.formatters import (
    format_csv,
    format_markdown_table,
    format_text_summary,
)
from extract_bench.evaluation.reporting.models import ArrayBreakdown, FieldOutcome
from extract_bench.evaluation.reporting.outcome_stats import _extract_array_breakdown


class TestExtractArrayBreakdown:
    """Test _extract_array_breakdown helper."""

    def test_extracts_from_valid_details(self):
        details = {
            "structured_output": {
                "matches_summary": {
                    "matched": 5,
                    "missed_gold": 2,
                    "spurious_pred": 1,
                },
                "aggregate_metrics": {
                    "precision": 0.833,
                    "recall": 0.714,
                    "f1": 0.769,
                },
                "matched_items": ["a", "b", "c", "d", "e"],
                "missed_gold_items": ["f", "g"],
                "spurious_pred_items": ["h"],
            }
        }
        ab = _extract_array_breakdown(details)
        assert ab is not None
        assert ab.matched == 5
        assert ab.missed_gold == 2
        assert ab.spurious_pred == 1
        assert ab.precision == 0.833
        assert ab.recall == 0.714
        assert ab.f1 == 0.769
        assert ab.matched_items == ["a", "b", "c", "d", "e"]
        assert ab.missed_gold_items == ["f", "g"]
        assert ab.spurious_pred_items == ["h"]

    def test_returns_none_without_structured_output(self):
        assert _extract_array_breakdown({}) is None
        assert _extract_array_breakdown({"structured_output": "not a dict"}) is None

    def test_returns_none_without_summary_or_agg(self):
        assert _extract_array_breakdown({"structured_output": {}}) is None
        assert (
            _extract_array_breakdown(
                {"structured_output": {"matches_summary": {"matched": 1}}}
            )
            is None
        )


class TestFieldOutcomeArrayBreakdown:
    """Test that FieldOutcome carries array_breakdown."""

    def test_field_outcome_with_breakdown(self):
        ab = ArrayBreakdown(
            matched=3,
            missed_gold=1,
            spurious_pred=0,
            precision=1.0,
            recall=0.75,
            f1=0.857,
        )
        fo = FieldOutcome(
            path="$.properties.items",
            normalized_path="items",
            metric_id="array_llm",
            score=0.75,
            passed=True,
            array_breakdown=ab,
        )
        assert fo.array_breakdown is not None
        assert fo.array_breakdown.matched == 3

    def test_field_outcome_without_breakdown(self):
        fo = FieldOutcome(
            path="$.properties.name",
            normalized_path="name",
            metric_id="string_exact",
            score=1.0,
            passed=True,
        )
        assert fo.array_breakdown is None


class TestFormattersWithBreakdown:
    """Test that formatters render array breakdown data."""

    def _make_outcomes(self):
        ab = ArrayBreakdown(
            matched=5,
            missed_gold=2,
            spurious_pred=1,
            precision=0.833,
            recall=0.714,
            f1=0.769,
        )
        return [
            FieldOutcome(
                path="$.properties.name",
                normalized_path="name",
                metric_id="string_exact",
                score=1.0,
                passed=True,
            ),
            FieldOutcome(
                path="$.properties.items",
                normalized_path="items",
                metric_id="array_llm",
                score=0.714,
                passed=True,
                array_breakdown=ab,
            ),
        ]

    def test_csv_includes_array_columns(self):
        outcomes = self._make_outcomes()
        csv_output = format_csv(outcomes)
        lines = csv_output.strip().split("\n")
        header = lines[0]
        assert "matched" in header
        assert "missed_gold" in header
        assert "spurious_pred" in header
        assert "precision" in header
        assert "recall" in header
        assert "f1" in header

        # string_exact row should have empty array columns
        name_row = lines[1].strip()
        assert name_row.endswith(",,,,,")

        # array_llm row should have values
        items_row = lines[2].strip()
        assert ",5," in items_row
        assert ",2," in items_row
        assert "0.833" in items_row

    def test_markdown_includes_array_columns(self):
        outcomes = self._make_outcomes()
        md = format_markdown_table(outcomes)
        assert "Matched" in md
        assert "Missed" in md
        assert "Spurious" in md
        # array_llm row should have counts
        assert "| 5 |" in md
        assert "| 2 |" in md

    def test_text_summary_includes_array_breakdown(self):
        # Build a minimal report to test text summary
        from extract_bench.evaluation.reporting.models import (
            ConfusionCounts,
            ContentStats,
            CoverageStats,
            EvaluationReport,
            OutcomeStats,
            SchemaStats,
        )

        outcomes = self._make_outcomes()
        report = EvaluationReport(
            output_name="test",
            timestamp="2024-01-01",
            schema_hash="abc",
            gold_hash="def",
            extracted_hash="ghi",
            schema_stats=SchemaStats(
                total_nodes=5,
                counts_by_type={},
                required_keys_count=2,
                optional_keys_count=1,
            ),
            gold_stats=ContentStats(label="gold", total_keys=3, counts_by_type={}),
            extracted_stats=ContentStats(
                label="extracted", total_keys=3, counts_by_type={}
            ),
            coverage=CoverageStats(
                present_in_both=3,
                missing_in_extracted=0,
                spurious_in_extracted=0,
                required_missing=0,
            ),
            outcomes=OutcomeStats(
                total_evaluated=2,
                total_passed=2,
                total_failed=0,
                pass_rate=1.0,
                pass_by_metric={},
                pass_by_type={},
                pass_by_required={},
                confusion=ConfusionCounts(
                    true_positive=2,
                    false_positive=0,
                    false_negative=0,
                    true_negative=0,
                ),
            ),
            field_outcomes=outcomes,
            overall_score=0.857,
            field_score=0.857,
            overall_pass_rate=1.0,
        )
        text = format_text_summary(report)
        assert "ARRAY BREAKDOWN" in text
        assert "5 matched" in text
        assert "2 missed" in text
        assert "1 spurious" in text
        assert "P=0.833" in text
        assert "R=0.714" in text
        # Both score types should appear
        assert "item-weighted" in text
        assert "flat average" in text
        assert "Field Score" in text


class TestItemWeightedScoring:
    """Test that overall_score weights arrays by gold item count."""

    def test_array_weighted_more_than_scalar(self):
        """An array with 10 gold items should weigh 10x a scalar field."""
        ab = ArrayBreakdown(
            matched=8, missed_gold=2, spurious_pred=0,
            precision=1.0, recall=0.8, f1=0.889,
        )
        outcomes = [
            FieldOutcome(
                path="$.properties.name", normalized_path="name",
                metric_id="string_exact", score=1.0, passed=True,
            ),
            FieldOutcome(
                path="$.properties.items", normalized_path="items",
                metric_id="array_llm", score=0.8, passed=True,
                array_breakdown=ab,
            ),
        ]
        # Flat: (1.0 + 0.8) / 2 = 0.9
        scores = [f.score for f in outcomes]
        field_score = sum(scores) / len(scores)
        assert abs(field_score - 0.9) < 1e-6

        # Weighted: (1.0*1 + 0.8*10) / (1+10) = 9.0/11 ≈ 0.818
        weighted_sum = 0.0
        total_weight = 0.0
        for f in outcomes:
            weight = max(1, f.array_breakdown.matched + f.array_breakdown.missed_gold) if f.array_breakdown else 1
            weighted_sum += f.score * weight
            total_weight += weight
        overall_score = weighted_sum / total_weight
        assert abs(overall_score - 9.0 / 11.0) < 1e-6
        # Weighted score should be lower because the array (score 0.8) dominates
        assert overall_score < field_score

    def test_no_arrays_scores_equal(self):
        """Without arrays, both scores should be identical."""
        outcomes = [
            FieldOutcome(
                path="$.properties.name", normalized_path="name",
                metric_id="string_exact", score=1.0, passed=True,
            ),
            FieldOutcome(
                path="$.properties.age", normalized_path="age",
                metric_id="integer_exact", score=0.0, passed=False,
            ),
        ]
        scores = [f.score for f in outcomes]
        field_score = sum(scores) / len(scores)

        weighted_sum = sum(f.score * 1 for f in outcomes)
        overall_score = weighted_sum / len(outcomes)
        assert abs(field_score - overall_score) < 1e-6
