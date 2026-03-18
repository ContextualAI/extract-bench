"""Helper functions for working with evaluation configs on schema nodes."""

from typing import Any, Dict

from ..infra.nodes import AnyOfSchema, NullSchema, Schema, StringSchema
from .evaluation_config import EvaluationConfig
from .presets import (
    ARRAY_DEFAULT_PRESET,
    BOOLEAN_DEFAULT_PRESET,
    INTEGER_DEFAULT_PRESET,
    NUMBER_DEFAULT_PRESET,
    SKIP_PRESET,
    STRING_DEFAULT_PRESET,
    get_preset_config,
    get_preset_configs,
)

# Format-based preset overrides: when a string field has one of these formats
# and uses `string_exact`, automatically upgrade to a format-aware metric.
_FORMAT_PRESET_OVERRIDES: Dict[str, str] = {
    "uri": "string_url",
    "url": "string_url",
    "iri": "string_url",
}


def _apply_format_override(schema: Schema, config: EvaluationConfig) -> EvaluationConfig:
    """Upgrade string_exact to a format-aware metric when the schema declares a known format."""
    if not isinstance(schema, StringSchema) or not schema.format:
        return config
    override_preset = _FORMAT_PRESET_OVERRIDES.get(schema.format)
    if override_preset is None:
        return config
    if (
        len(config.metrics) == 1
        and config.metrics[0].metric_id == "string_exact"
    ):
        return get_preset_config(override_preset)
    return config


def get_evaluation_config(schema: Schema) -> EvaluationConfig:
    """Get evaluation config for a schema node.

    Returns type-specific defaults if not explicitly set.
    Validates and converts the config on access.
    """
    if schema.evaluation_config is not None:
        if isinstance(schema.evaluation_config, str):
            config = get_preset_config(schema.evaluation_config)
            return _apply_format_override(schema, config)

        if isinstance(schema.evaluation_config, dict):
            try:
                config = EvaluationConfig.model_validate(schema.evaluation_config)
                return _apply_format_override(schema, config)
            except Exception as e:
                raise ValueError(
                    f"Invalid evaluation_config for schema node: {e}. "
                    f"Expected format: {{'metrics': [...]}} with valid MetricConfig objects."
                ) from e

        return schema.evaluation_config

    return get_default_evaluation_config(schema)


def _infer_anyof_evaluation_config(schema: AnyOfSchema) -> EvaluationConfig | None:
    """Infer evaluation config for an anyOf node from its children.

    If all non-null children share the same evaluation_config, use it for
    the anyOf node itself. This handles cases like `skills` which is
    anyOf[array, object] where both branches use array_llm.
    """
    non_null_children = [
        child for child in schema.any_of if not isinstance(child, NullSchema)
    ]
    if not non_null_children:
        return None

    configs = []
    for child in non_null_children:
        raw = child.evaluation_config
        if raw is None:
            continue
        if isinstance(raw, str):
            configs.append(raw)
        elif isinstance(raw, EvaluationConfig) and len(raw.metrics) == 1:
            configs.append(raw.metrics[0].metric_id)
        elif isinstance(raw, dict):
            metrics = raw.get("metrics", [])
            if len(metrics) == 1:
                configs.append(metrics[0].get("metric_id", ""))

    if not configs:
        return None

    # All children agree on the same metric — use it
    if len(set(configs)) == 1:
        return get_preset_config(configs[0])

    return None


def get_default_evaluation_config(schema: Schema) -> EvaluationConfig:
    """Get type-specific default evaluation config for a schema."""
    schema_type = schema.get_type()
    preset_configs = get_preset_configs()

    match schema_type:
        case "string":
            return preset_configs[STRING_DEFAULT_PRESET]
        case "integer":
            return preset_configs[INTEGER_DEFAULT_PRESET]
        case "number":
            return preset_configs[NUMBER_DEFAULT_PRESET]
        case "boolean":
            return preset_configs[BOOLEAN_DEFAULT_PRESET]
        case "array":
            return preset_configs[ARRAY_DEFAULT_PRESET]
        case "object":
            return preset_configs[SKIP_PRESET]
        case "null":
            return preset_configs[SKIP_PRESET]
        case _:
            if isinstance(schema, AnyOfSchema):
                inferred = _infer_anyof_evaluation_config(schema)
                if inferred is not None:
                    return inferred
            return preset_configs[SKIP_PRESET]


def should_evaluate(schema: Schema) -> bool:
    """Check if a schema node should be evaluated."""
    config = get_evaluation_config(schema)
    return len(config.metrics) > 0


def add_evaluation_configs_to_export(
    schema: Schema, include_defaults: bool = True
) -> Dict[str, Any]:
    """Export schema to dict and include evaluation configs."""
    result = schema.export_to_dict()

    if include_defaults:
        _add_config_recursive(result, schema)

    return result


def _add_config_recursive(result_dict: Dict[str, Any], schema: Schema) -> None:
    """Recursively add evaluation configs to exported dict."""
    config = get_evaluation_config(schema)
    if config.metrics:
        result_dict["evaluation_config"] = config.model_dump(exclude_none=True)

    if hasattr(schema, "properties") and schema.properties:
        for key, child_schema in schema.properties.items():
            if "properties" in result_dict and key in result_dict["properties"]:
                _add_config_recursive(result_dict["properties"][key], child_schema)

    if hasattr(schema, "items") and "items" in result_dict:
        _add_config_recursive(result_dict["items"], schema.items)

    if hasattr(schema, "any_of"):
        for i, child_schema in enumerate(schema.any_of):
            if "anyOf" in result_dict and i < len(result_dict["anyOf"]):
                _add_config_recursive(result_dict["anyOf"][i], child_schema)

    if hasattr(schema, "defs") and schema.defs:
        for key, def_schema in schema.defs.items():
            if "$defs" in result_dict and key in result_dict["$defs"]:
                _add_config_recursive(result_dict["$defs"][key], def_schema)
