"""Tests for schema-aware coverage computation."""

from extract_bench.evaluation.reporting.content_stats import compute_coverage
from extract_bench.infra.construct_ast import construct_ast


class TestSchemaAwareCoverage:
    """Test that coverage only considers schema-declared paths."""

    def _coverage(self, schema_dict, gold, extracted):
        schema = construct_ast(schema_dict)
        return compute_coverage(gold, extracted, schema)

    def test_ignores_fields_not_in_schema(self):
        """Extra fields in gold (like array_index) should not appear."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                }
            },
        }
        gold = {"items": [{"name": "a", "array_index": 0}]}
        extracted = {"items": [{"name": "a"}]}
        cov = self._coverage(schema, gold, extracted)

        assert cov.missing_in_extracted == 0
        assert "array_index" not in str(cov.missing_paths)

    def test_collapses_array_indices(self):
        """Multiple array items should collapse to one path per field."""
        schema = {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                    },
                }
            },
        }
        gold = {
            "people": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ]
        }
        extracted = {
            "people": [
                {"name": "Alice", "age": 30},
            ]
        }
        cov = self._coverage(schema, gold, extracted)

        assert cov.present_in_both == 2
        assert cov.missing_in_extracted == 0

    def test_detects_missing_schema_field(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        gold = {"name": "Alice", "email": "alice@example.com"}
        extracted = {"name": "Alice"}
        cov = self._coverage(schema, gold, extracted)

        assert cov.missing_in_extracted == 1
        assert "email" in cov.missing_paths

    def test_detects_spurious_schema_field(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        gold = {"name": "Alice"}
        extracted = {"name": "Alice", "email": "alice@example.com"}
        cov = self._coverage(schema, gold, extracted)

        assert cov.spurious_in_extracted == 1
        assert "email" in cov.spurious_paths

    def test_ignores_spurious_field_not_in_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        gold = {"name": "Alice"}
        extracted = {"name": "Alice", "debug_info": "internal"}
        cov = self._coverage(schema, gold, extracted)

        assert cov.spurious_in_extracted == 0
        assert cov.present_in_both == 1

    def test_additional_properties_wildcard(self):
        schema = {
            "type": "object",
            "properties": {
                "skills": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                }
            },
        }
        gold = {"skills": {"Python": ["flask"], "Go": ["gin"]}}
        extracted = {"skills": {"Python": ["flask"], "Go": ["gin"], "Rust": ["axum"]}}
        cov = self._coverage(schema, gold, extracted)

        assert "skills.Python" not in cov.missing_paths
        assert "skills.Rust" in cov.spurious_paths

    def test_nested_object_paths(self):
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                            },
                        }
                    },
                }
            },
        }
        gold = {"person": {"contact": {"email": "a@b.com"}}}
        extracted = {}
        cov = self._coverage(schema, gold, extracted)

        assert "person.contact.email" in cov.missing_paths

    def test_required_missing_tracked(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "bio": {"type": "string"},
            },
        }
        gold = {"name": "Alice", "bio": "engineer"}
        extracted = {"bio": "engineer"}
        cov = self._coverage(schema, gold, extracted)

        assert cov.required_missing == 1
        assert cov.missing_in_extracted == 1
        assert "name" in cov.missing_paths

    def test_anyof_paths_collected(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"},
                    ]
                }
            },
        }
        gold = {"value": "hello"}
        extracted = {"value": "hello"}
        cov = self._coverage(schema, gold, extracted)

        assert cov.present_in_both == 1
        assert cov.missing_in_extracted == 0

    def test_perfect_match(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        data = {"name": "Alice", "age": 30}
        cov = self._coverage(schema, data, data)

        assert cov.present_in_both == 2
        assert cov.missing_in_extracted == 0
        assert cov.spurious_in_extracted == 0
