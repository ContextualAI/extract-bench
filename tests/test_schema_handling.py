"""Tests for schema handling: anyOf inference, missing/null policy, complex schemas."""

import pytest

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig


class TestMissingNullPolicy:
    """Test the missing/null handling policy."""

    @pytest.mark.asyncio
    async def test_optional_field_both_missing(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"},
                "nickname": {"type": "string", "evaluation_config": "string_exact"},
            },
        }
        gold = {"name": "John"}
        predicted = {"name": "John"}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = await evaluator.evaluate_async(schema, gold, predicted)

        name_result = result["results"]["$.properties.name"]["string_exact"]
        assert name_result.passed is True

        nickname_result = result["results"]["$.properties.nickname"]["string_exact"]
        assert nickname_result.score == 1.0
        assert nickname_result.passed is True
        assert nickname_result.details["reason"] == "both_missing"

    @pytest.mark.asyncio
    async def test_optional_array_both_missing_no_llm_call(self, monkeypatch):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"},
                "skills": {
                    "type": "array",
                    "evaluation_config": "array_llm",
                    "items": {"type": "string"},
                },
            },
        }
        gold = {"name": "John Doe"}
        predicted = {"name": "John Doe"}

        async def should_not_be_called(*args, **kwargs):
            raise AssertionError(
                "litellm.acompletion should not be called for optional array omitted/omitted"
            )

        import extract_bench.evaluation.metrics.llm_metrics as llm_module

        monkeypatch.setattr(llm_module.litellm, "acompletion", should_not_be_called)

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = await evaluator.evaluate_async(schema, gold, predicted)

        skills_result = result["results"]["$.properties.skills"]["array_llm"]
        assert skills_result.score == 1.0
        assert skills_result.passed is True
        assert skills_result.details["reason"] in {"both_missing", "both_absent"}


class TestAnyOfSchema:
    """Test anyOf schema handling."""

    def test_nullable_string(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "evaluation_config": "number_exact",
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"score": None}, {"score": None})

        assert "$.properties.score" in result["results"]
        metric_result = result["results"]["$.properties.score"]["number_exact"]
        assert metric_result.passed is True


class TestAnyOfEvaluationInference:
    """Test that anyOf nodes infer evaluation config from children."""

    def test_anyof_infers_config_from_uniform_children(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string", "evaluation_config": "string_exact"},
                        {"type": "integer", "evaluation_config": "string_exact"},
                    ]
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"value": "hello"}, {"value": "hello"})
        assert "$.properties.value" in result["results"]
        metric = result["results"]["$.properties.value"]["string_exact"]
        assert metric.passed is True
        assert metric.score == 1.0

    def test_anyof_infers_config_ignoring_null_branch(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string", "evaluation_config": "string_exact"},
                        {"type": "null"},
                    ]
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"value": "test"}, {"value": "test"})
        assert "$.properties.value" in result["results"]
        metric = result["results"]["$.properties.value"]["string_exact"]
        assert metric.passed is True

    def test_anyof_no_inference_when_children_disagree(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string", "evaluation_config": "string_exact"},
                        {"type": "integer", "evaluation_config": "integer_exact"},
                    ]
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"value": "test"}, {"value": "test"})
        assert "$.properties.value" not in result["results"]

    def test_anyof_explicit_config_takes_precedence(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "evaluation_config": "string_fuzzy",
                    "anyOf": [
                        {"type": "string", "evaluation_config": "string_exact"},
                        {"type": "null"},
                    ],
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"value": "hello"}, {"value": "helo"})
        assert "$.properties.value" in result["results"]
        assert "string_fuzzy" in result["results"]["$.properties.value"]


class TestComplexSchema:
    """Test complex schema scenarios."""

    def test_mixed_types_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"},
                "age": {"type": "integer", "evaluation_config": "integer_exact"},
                "gpa": {
                    "type": "number",
                    "evaluation_config": {
                        "metrics": [
                            {
                                "metric_id": "number_tolerance",
                                "params": {"tolerance": 0.1},
                            }
                        ]
                    },
                },
                "is_student": {"type": "boolean", "evaluation_config": "boolean_exact"},
            },
        }
        gold = {"name": "John", "age": 20, "gpa": 3.5, "is_student": True}
        predicted = {"name": "John", "age": 20, "gpa": 3.55, "is_student": True}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        assert result["results"]["$.properties.name"]["string_exact"].passed is True
        assert result["results"]["$.properties.age"]["integer_exact"].passed is True
        assert result["results"]["$.properties.gpa"]["number_tolerance"].passed is True
        assert (
            result["results"]["$.properties.is_student"]["boolean_exact"].passed is True
        )

    def test_nested_object_evaluation(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "zip": {"type": "string", "evaluation_config": "string_exact"},
                        "city": {"type": "string", "evaluation_config": "string_exact"},
                    },
                }
            },
        }
        gold = {"address": {"zip": "12345", "city": "Boston"}}
        predicted = {"address": {"zip": "12345", "city": "Boston"}}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        zip_result = result["results"]["$.properties.address.properties.zip"][
            "string_exact"
        ]
        assert zip_result.passed is True

        city_result = result["results"]["$.properties.address.properties.city"][
            "string_exact"
        ]
        assert city_result.passed is True
