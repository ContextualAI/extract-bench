"""Tests for URL-normalized string matching and format inference."""

from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig


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


class TestFormatUriInference:
    """Test that string_exact + format: uri auto-upgrades to string_url."""

    def test_string_exact_with_format_uri_infers_string_url(self):
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
        result = evaluator.evaluate(schema, {"name": "John"}, {"name": "John"})
        metric_results = result["results"]["$.properties.name"]
        assert "string_exact" in metric_results
        assert "string_url" not in metric_results
