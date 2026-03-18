"""Array evaluation metrics."""

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from ...infra.nodes import BaseSchema
from ..evaluation_config import MetricConfig
from .base_metric import MetricContext
from .llm_metrics import LlmJudgeMetric
from .metric_descriptors import MetricPromptDescriptor
from .metric_prompts import ARRAY_LLM_PROMPT


@dataclass(frozen=True, slots=True)
class ArrayLlmJudgeMetric(LlmJudgeMetric):
    metric_id: ClassVar[str] = "array_llm"
    default_prompt_template: ClassVar[str] = ARRAY_LLM_PROMPT
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="array_llm",
        summary="LLM-based array evaluation with item matching and confusion summary.",
        pass_rule="pass iff score >= pass_threshold (default threshold is metric-specific)",
        score_rule="matched/(matched+missed_gold) after postprocessing",
        params={
            "pass_threshold": {"type": "number", "meaning": "minimum score to pass"},
            "additional_instructions": {
                "type": "string",
                "meaning": "extra domain rules",
            },
        },
    )
    default_structured_output_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "matched_items": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
            "missed_gold_items": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
            "spurious_pred_items": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object"}]}},
            "matches_summary": {
                "type": "object",
                "properties": {
                    "matched": {"type": "integer", "minimum": 0},
                    "missed_gold": {"type": "integer", "minimum": 0},
                    "spurious_pred": {"type": "integer", "minimum": 0},
                },
                "required": ["matched", "missed_gold", "spurious_pred"],
            },
        },
        "required": ["reasoning", "matches_summary"],
        "additionalProperties": True,
    }

    def _get_additional_prompt_context(self, node: MetricContext) -> str | None:
        """Append metric definitions for child fields so the LLM understands per-field evaluation rules."""
        from ..metric_id_collector import collect_metric_ids_with_params
        from ..metric_registry import global_metric_registry

        if not isinstance(node, BaseSchema):
            return None
        metric_params = collect_metric_ids_with_params(schema=node)
        if not metric_params:
            return None
        definitions: dict[str, Any] = {}
        for metric_id, params_examples in sorted(metric_params.items()):
            descriptor = global_metric_registry.get_metric_descriptor(metric_id)
            if descriptor is None:
                definitions[metric_id] = {
                    "metric_id": metric_id,
                    "summary": "No descriptor registered for this metric_id.",
                    "params_examples": params_examples,
                }
            else:
                definitions[metric_id] = descriptor.describe(
                    params_examples=params_examples
                )
        return (
            f"Metric definitions (JSON):\n"
            f"{json.dumps(definitions, ensure_ascii=True, indent=2)}"
        )

    def postprocess_parsed_result(
        self,
        node: MetricContext,
        config: MetricConfig | None,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        parsed = dict(parsed)
        summary = parsed["matches_summary"]
        matched, missed, spurious = (
            summary["matched"],
            summary["missed_gold"],
            summary["spurious_pred"],
        )
        denominator = max(1, matched + missed)
        parsed["score"] = matched / denominator

        total = matched + missed + spurious
        gold_total = matched + missed
        pred_total = matched + spurious

        accuracy = matched / total if total > 0 else 1.0
        precision = matched / pred_total if pred_total > 0 else 1.0
        recall = matched / gold_total if gold_total > 0 else 1.0
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        parsed["aggregate_metrics"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return parsed
