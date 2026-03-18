import pytest

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig


class TestBasicEvaluation:
    """Basic evaluation tests without LLM calls."""

    def test_exact_string_match(self):
        """Test exact string matching metric."""
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
        """Test exact string matching with different values."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"}
            },
        }
        gold = {"name": "John Doe"}
        predicted = {"name": "Jane Doe"}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.name"]["string_exact"]
        assert metric_result.passed is False
        assert metric_result.score == 0.0

    def test_integer_exact_match(self):
        """Test exact integer matching."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "evaluation_config": "integer_exact"}
            },
        }
        gold = {"age": 30}
        predicted = {"age": 30}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.age"]["integer_exact"]
        assert metric_result.passed is True
        assert metric_result.score == 1.0

    def test_integer_mismatch(self):
        """Test integer matching with different values."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "evaluation_config": "integer_exact"}
            },
        }
        gold = {"age": 30}
        predicted = {"age": 31}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.age"]["integer_exact"]
        assert metric_result.passed is False

    def test_number_tolerance_within_range(self):
        """Test number tolerance metric within acceptable range."""
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
        gold = {"gpa": 3.5}
        predicted = {"gpa": 3.55}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.gpa"]["number_tolerance"]
        assert metric_result.passed is True

    def test_number_tolerance_outside_range(self):
        """Test number tolerance metric outside acceptable range."""
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
        gold = {"gpa": 3.5}
        predicted = {"gpa": 3.7}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.gpa"]["number_tolerance"]
        assert metric_result.passed is False

    def test_boolean_exact_match(self):
        """Test exact boolean matching."""
        schema = {
            "type": "object",
            "properties": {
                "is_active": {"type": "boolean", "evaluation_config": "boolean_exact"}
            },
        }
        gold = {"is_active": True}
        predicted = {"is_active": True}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.is_active"]["boolean_exact"]
        assert metric_result.passed is True
        assert metric_result.score == 1.0

    def test_boolean_mismatch(self):
        """Test boolean matching with different values."""
        schema = {
            "type": "object",
            "properties": {
                "is_active": {"type": "boolean", "evaluation_config": "boolean_exact"}
            },
        }
        gold = {"is_active": True}
        predicted = {"is_active": False}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.is_active"]["boolean_exact"]
        assert metric_result.passed is False
        assert metric_result.score == 0.0

    def test_nested_object_evaluation(self):
        """Test evaluation of nested objects."""
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

        assert "$.properties.address.properties.zip" in result["results"]
        assert "$.properties.address.properties.city" in result["results"]

        zip_result = result["results"]["$.properties.address.properties.zip"][
            "string_exact"
        ]
        assert zip_result.passed is True

        city_result = result["results"]["$.properties.address.properties.city"][
            "string_exact"
        ]
        assert city_result.passed is True

    def test_string_fuzzy_matching(self):
        """Test fuzzy string matching with threshold."""
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
        gold = {"name": "John Doe"}
        predicted = {"name": "John Do"}  # Small typo

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.name"]["string_fuzzy"]
        assert metric_result.score > 0.8
        assert metric_result.passed is True

    def test_case_insensitive_string_match(self):
        """Test case insensitive string matching."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": "string_case_insensitive",
                }
            },
        }
        gold = {"name": "John Doe"}
        predicted = {"name": "JOHN DOE"}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        metric_result = result["results"]["$.properties.name"][
            "string_case_insensitive"
        ]
        assert metric_result.passed is True
        assert metric_result.score == 1.0


class TestMissingNullPolicy:
    """Test the missing/null handling policy."""

    @pytest.mark.asyncio
    async def test_optional_field_both_missing(self, monkeypatch):
        """Test that optional field omitted in both gold and predicted scores 1.0."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "evaluation_config": "string_exact"},
                "nickname": {"type": "string", "evaluation_config": "string_exact"},
            },
        }
        gold = {"name": "John"}  # nickname omitted
        predicted = {"name": "John"}  # nickname omitted

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = await evaluator.evaluate_async(schema, gold, predicted)

        # name should match
        name_result = result["results"]["$.properties.name"]["string_exact"]
        assert name_result.passed is True

        # nickname should be treated as both_missing and score 1.0
        nickname_result = result["results"]["$.properties.nickname"]["string_exact"]
        assert nickname_result.score == 1.0
        assert nickname_result.passed is True
        assert nickname_result.details["reason"] == "both_missing"

    @pytest.mark.asyncio
    async def test_optional_array_both_missing_no_llm_call(self, monkeypatch):
        """Test that optional array omitted in both should short-circuit without LLM call."""
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
        gold = {"name": "John Doe"}  # skills omitted
        predicted = {"name": "John Doe"}  # skills omitted

        async def should_not_be_called(*args, **kwargs):
            raise AssertionError(
                "litellm.acompletion should not be called for optional array omitted/omitted"
            )

        import extract_bench.evaluation.metrics.llm_metrics as llm_module

        monkeypatch.setattr(llm_module.litellm, "acompletion", should_not_be_called)

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = await evaluator.evaluate_async(schema, gold, predicted)

        assert "$.properties.name" in result["results"]
        assert "$.properties.skills" in result["results"]

        name_result = result["results"]["$.properties.name"]["string_exact"]
        assert name_result.score == 1.0
        assert name_result.passed is True

        skills_result = result["results"]["$.properties.skills"]["array_llm"]
        assert skills_result.score == 1.0
        assert skills_result.passed is True
        assert skills_result.details["reason"] in {"both_missing", "both_absent"}


class TestAnyOfSchema:
    """Test anyOf schema handling."""

    def test_nullable_string(self):
        """Test nullable string (anyOf with null)."""
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "evaluation_config": "number_exact",
                }
            },
        }
        gold = {"score": None}
        predicted = {"score": None}

        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(schema, gold, predicted)

        # anyOf nodes are treated as terminal nodes
        assert "$.properties.score" in result["results"]
        metric_result = result["results"]["$.properties.score"]["number_exact"]
        assert metric_result.passed is True


class TestUrlNormalizedStringMatch:
    """Test URL-normalized string matching metric."""

    def _make_schema(self):
        return {
            "type": "object",
            "properties": {
                "link": {
                    "type": "string",
                    "format": "uri",
                    "evaluation_config": "string_url",
                }
            },
        }

    def test_strips_http_protocol(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "http://linkedin.com/in/user"},
            {"link": "linkedin.com/in/user"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is True
        assert metric.score == 1.0

    def test_strips_https_protocol(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "https://github.com/user"},
            {"link": "github.com/user"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is True

    def test_strips_www(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "https://www.example.com/page"},
            {"link": "example.com/page"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is True

    def test_strips_trailing_slash(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "https://example.com/page/"},
            {"link": "example.com/page"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is True

    def test_case_insensitive(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "HTTPS://GitHub.Com/User"},
            {"link": "github.com/user"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is True

    def test_different_paths_fail(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "https://github.com/user1"},
            {"link": "https://github.com/user2"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.passed is False
        assert metric.score == 0.0

    def test_normalized_values_in_details(self):
        schema = self._make_schema()
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "http://www.example.com/page/"},
            {"link": "example.com/page"},
        )
        metric = result["results"]["$.properties.link"]["string_url"]
        assert metric.details["gold_normalized"] == "example.com/page"
        assert metric.details["extracted_normalized"] == "example.com/page"

    def test_string_exact_with_format_uri_infers_string_url(self):
        """string_exact + format: uri should automatically use string_url."""
        schema = {
            "type": "object",
            "properties": {
                "link": {
                    "type": "string",
                    "format": "uri",
                    "evaluation_config": "string_exact",
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"link": "https://github.com/user"},
            {"link": "github.com/user"},
        )
        metric_results = result["results"]["$.properties.link"]
        assert "string_url" in metric_results
        assert "string_exact" not in metric_results
        assert metric_results["string_url"].passed is True

    def test_string_exact_without_format_stays_exact(self):
        """string_exact without format: uri should stay as string_exact."""
        schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "evaluation_config": "string_exact",
                }
            },
        }
        evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
        result = evaluator.evaluate(
            schema,
            {"name": "John"},
            {"name": "John"},
        )
        metric_results = result["results"]["$.properties.name"]
        assert "string_exact" in metric_results
        assert "string_url" not in metric_results


class TestAnyOfEvaluationInference:
    """Test that anyOf nodes infer evaluation config from children."""

    def test_anyof_infers_config_from_uniform_children(self):
        """anyOf where all non-null children share the same config should be evaluated."""
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
        """Nullable anyOf should infer from non-null child."""
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
        """anyOf with different configs on children should not infer (skip)."""
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
        # anyOf node should be skipped since children disagree
        assert "$.properties.value" not in result["results"]

    def test_anyof_explicit_config_takes_precedence(self):
        """Explicit evaluation_config on anyOf should override child inference."""
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
        # Should use string_fuzzy (the explicit config), not string_exact
        assert "string_fuzzy" in result["results"]["$.properties.value"]


class TestComplexSchema:
    """Test complex schema scenarios."""

    def test_mixed_types_schema(self):
        """Test schema with multiple different types."""
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

        # All should pass
        assert result["results"]["$.properties.name"]["string_exact"].passed is True
        assert result["results"]["$.properties.age"]["integer_exact"].passed is True
        assert result["results"]["$.properties.gpa"]["number_tolerance"].passed is True
        assert (
            result["results"]["$.properties.is_student"]["boolean_exact"].passed is True
        )
