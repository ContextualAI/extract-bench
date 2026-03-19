"""Tests for error handling: metric errors should not recurse into children (Phase 4)."""

import pytest

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig
from extract_bench.evaluation.structured_evaluator import AsyncEvaluationConfig


class TestErrorNoRecurse:
    """Test that metric errors don't recurse into children."""

    @pytest.mark.asyncio
    async def test_array_metric_error_does_not_produce_child_results(
        self, monkeypatch
    ):
        """When array_llm errors, child item fields should NOT be evaluated."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "evaluation_config": "array_llm",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "evaluation_config": "string_exact",
                            },
                            "value": {
                                "type": "integer",
                                "evaluation_config": "integer_exact",
                            },
                        },
                    },
                }
            },
        }
        gold = {"items": [{"name": "a", "value": 1}]}
        extracted = {"items": [{"name": "a", "value": 1}]}

        import extract_bench.evaluation.metrics.llm_metrics as llm_module

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Simulated LLM failure")

        monkeypatch.setattr(llm_module.litellm, "acompletion", always_fail)

        evaluator = StructuredEvaluator(
            StructuredEvaluatorConfig(
                metrics=[],
                async_config=AsyncEvaluationConfig(
                    n_max_retries=1, metric_timeout_seconds=5
                ),
            )
        )
        result = await evaluator.evaluate_async(schema, gold, extracted)
        results = result["results"]

        # The array node should have an error result
        assert "$.properties.items" in results
        array_results = results["$.properties.items"]
        assert len(array_results) == 1
        metric_result = list(array_results.values())[0]
        assert metric_result.passed is False
        assert metric_result.details.get("reason") == "error"
        assert "error" in metric_result.details

        # No child item results should exist
        child_paths = [p for p in results if ".items." in p]
        assert child_paths == [], (
            f"Error should prevent recursion, but found child results: {child_paths}"
        )

    @pytest.mark.asyncio
    async def test_error_result_carries_metric_id(self, monkeypatch):
        """Error results should carry the intended metric_id, not 'unknown'."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "evaluation_config": "array_llm",
                    "items": {"type": "string"},
                }
            },
        }

        import extract_bench.evaluation.metrics.llm_metrics as llm_module

        async def always_fail(*args, **kwargs):
            raise RuntimeError("Simulated LLM failure")

        monkeypatch.setattr(llm_module.litellm, "acompletion", always_fail)

        evaluator = StructuredEvaluator(
            StructuredEvaluatorConfig(
                metrics=[],
                async_config=AsyncEvaluationConfig(
                    n_max_retries=1, metric_timeout_seconds=5
                ),
            )
        )
        result = await evaluator.evaluate_async(
            schema, {"data": ["x"]}, {"data": ["x"]}
        )
        array_results = result["results"]["$.properties.data"]
        metric_result = list(array_results.values())[0]

        assert metric_result.metric_id == "array_llm"
        assert "unknown" not in array_results
