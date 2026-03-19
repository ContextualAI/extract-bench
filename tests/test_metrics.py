"""Tests for individual metric evaluations (no LLM calls)."""

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig


class TestStringMetrics:
    """Tests for string matching metrics."""

    def test_exact_string_match(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"}
            },
        }
        gold = {"name": "John Doe"}
        predicted = {"name": "John Doe"}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        assert "$.properties.name" in result["results"]
        metric_result = result["results"]["$.properties.name"]["string_exact"]
        assert metric_result.passed is True
        assert metric_result.score == 1.0

    def test_exact_string_mismatch(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"}
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"name": "John Doe"}, {"name": "Jane Doe"})

        metric_result = result["results"]["$.properties.name"]["string_exact"]
        assert metric_result.passed is False
        assert metric_result.score == 0.0

    def test_string_fuzzy_matching(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": {
                        "metrics": [
                            {"metric_id": "string_fuzzy", "params": {"threshold": 0.8}}
                        ]
                    },
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema, {"name": "John Doe"}, {"name": "John Do"}
        )

        metric_result = result["results"]["$.properties.name"]["string_fuzzy"]
        assert metric_result.score > 0.8
        assert metric_result.passed is True

    def test_case_insensitive_string_match(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": "string_case_insensitive",
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema, {"name": "John Doe"}, {"name": "JOHN DOE"}
        )

        metric_result = result["results"]["$.properties.name"][
            "string_case_insensitive"
        ]
        assert metric_result.passed is True
        assert metric_result.score == 1.0


class TestNumberMetrics:
    """Tests for number matching metrics."""

    def test_integer_exact_match(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "evaluation_config": "integer_exact"}
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"age": 30}, {"age": 30})

        metric_result = result["results"]["$.properties.age"]["integer_exact"]
        assert metric_result.passed is True
        assert metric_result.score == 1.0

    def test_integer_mismatch(self):
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "evaluation_config": "integer_exact"}
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"age": 30}, {"age": 31})

        metric_result = result["results"]["$.properties.age"]["integer_exact"]
        assert metric_result.passed is False

    def test_number_tolerance_within_range(self):
        schema = {
            "type": "object",
            "properties": {
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
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"gpa": 3.5}, {"gpa": 3.55})

        metric_result = result["results"]["$.properties.gpa"]["number_tolerance"]
        assert metric_result.passed is True

    def test_number_tolerance_outside_range(self):
        schema = {
            "type": "object",
            "properties": {
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
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"gpa": 3.5}, {"gpa": 3.7})

        metric_result = result["results"]["$.properties.gpa"]["number_tolerance"]
        assert metric_result.passed is False


class TestBooleanMetrics:
    """Tests for boolean matching metrics."""

    def test_boolean_exact_match(self):
        schema = {
            "type": "object",
            "properties": {
                "is_active": {"type": "boolean", "evaluation_config": "boolean_exact"}
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"is_active": True}, {"is_active": True})

        metric_result = result["results"]["$.properties.is_active"]["boolean_exact"]
        assert metric_result.passed is True
        assert metric_result.score == 1.0

    def test_boolean_mismatch(self):
        schema = {
            "type": "object",
            "properties": {
                "is_active": {"type": "boolean", "evaluation_config": "boolean_exact"}
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, {"is_active": True}, {"is_active": False})

        metric_result = result["results"]["$.properties.is_active"]["boolean_exact"]
        assert metric_result.passed is False
        assert metric_result.score == 0.0
