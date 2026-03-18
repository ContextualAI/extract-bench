"""Preset evaluation configurations for common use cases."""

from typing import Dict

from .evaluation_config import EvaluationConfig, MetricConfig

VALID_PRESETS = {
    "string_exact": "Exact string match (case-sensitive)",
    "string_fuzzy": "Levenshtein distance with threshold",
    "string_case_insensitive": "Case-insensitive match",
    "string_url": "URL-normalized match (strips protocol, www, trailing slash)",
    "string_llm": "LLM-based semantic similarity",
    "string_semantic": "LLM-based semantic string evaluation",
    "number_exact": "Exact numeric match",
    "number_tolerance": "Match within tolerance margin",
    "integer_exact": "Exact integer match",
    "boolean_exact": "Exact boolean match",
    "array_llm": "LLM evaluation of entire array",
    "skip": "Skip evaluation for this node",
}


STRING_DEFAULT_PRESET = "string_semantic"
INTEGER_DEFAULT_PRESET = "integer_exact"
NUMBER_DEFAULT_PRESET = "number_tolerance"
BOOLEAN_DEFAULT_PRESET = "boolean_exact"
ARRAY_DEFAULT_PRESET = "array_llm"
SKIP_PRESET = "skip"


def _build_preset_configs() -> Dict[str, EvaluationConfig]:
    """Build preset configurations mapped to metric IDs."""

    return {
        "string_exact": EvaluationConfig(
            metrics=[MetricConfig(metric_id="string_exact")]
        ),
        "string_fuzzy": EvaluationConfig(
            metrics=[
                MetricConfig(
                    metric_id="string_fuzzy",
                    params={"threshold": 0.8},
                )
            ]
        ),
        "string_case_insensitive": EvaluationConfig(
            metrics=[MetricConfig(metric_id="string_case_insensitive")]
        ),
        "string_url": EvaluationConfig(
            metrics=[MetricConfig(metric_id="string_url")]
        ),
        "string_llm": EvaluationConfig(metrics=[MetricConfig(metric_id="string_llm")]),
        "string_semantic": EvaluationConfig(
            metrics=[MetricConfig(metric_id="string_semantic")]
        ),
        "number_exact": EvaluationConfig(
            metrics=[MetricConfig(metric_id="number_exact")]
        ),
        "number_tolerance": EvaluationConfig(
            metrics=[
                MetricConfig(
                    metric_id="number_tolerance",
                    params={"tolerance": 1e-3},
                )
            ]
        ),
        "integer_exact": EvaluationConfig(
            metrics=[MetricConfig(metric_id="integer_exact")]
        ),
        "boolean_exact": EvaluationConfig(
            metrics=[MetricConfig(metric_id="boolean_exact")]
        ),
        "array_llm": EvaluationConfig(metrics=[MetricConfig(metric_id="array_llm")]),
        SKIP_PRESET: EvaluationConfig(metrics=[]),
    }


_PRESET_TO_CONFIG: Dict[str, EvaluationConfig] | None = None


def get_preset_configs() -> Dict[str, EvaluationConfig]:
    """Get all preset configurations, lazily loading if needed."""
    global _PRESET_TO_CONFIG
    if _PRESET_TO_CONFIG is None:
        _PRESET_TO_CONFIG = _build_preset_configs()
    return _PRESET_TO_CONFIG


def get_preset_config(preset: str) -> EvaluationConfig:
    """Get evaluation config for a preset name."""
    configs = get_preset_configs()
    if preset not in configs:
        raise ValueError(get_preset_error_message(preset))
    return configs[preset]


def validate_preset(preset: str) -> None:
    """Validate that a preset name is valid."""
    if preset not in VALID_PRESETS:
        raise ValueError(get_preset_error_message(preset))


def get_preset_error_message(invalid_preset: str) -> str:
    """Generate a helpful error message for invalid presets."""
    available = "\n".join(
        f"  - {name}: {desc}" for name, desc in sorted(VALID_PRESETS.items())
    )
    return (
        f"Invalid evaluation preset '{invalid_preset}'.\n\n"
        f"Available presets:\n{available}\n\n"
        f"Use one of these presets, or provide a full evaluation config object:\n"
        f'  {{"metrics": [{{"metric_id": "...", "weight": 1.0, "params": {{...}}}}]}}'
    )
