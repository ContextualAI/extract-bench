"""LLM-judge based evaluation metrics."""

import copy
import json
import os
import re
from typing import Any, ClassVar, Dict, List

import litellm
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from ..evaluation_config import MetricConfig
from ..schema_value_instantiator import MISSING
from .base_metric import BaseMetric, MetricContext, MetricResult
from .metric_prompts import LLM_JUDGE_PROMPT
from .policy_metric import PolicyAwareMetric

DEFAULT_LLM_MODEL_ID = os.getenv("DEFAULT_LLM_MODEL", "vertex_ai/gemini-2.5-flash")
DEFAULT_PASS_THRESHOLD = 0.7
_DEFAULT_SYSTEM_PROMPT = "You are an impartial and detail-oriented judge that evaluates the quality of a predicted value against a gold reference. "


def _format_value(value: Any) -> str:
    if value is MISSING:
        return "<MISSING>"
    try:
        return json.dumps(value, ensure_ascii=True, indent=2)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True, indent=2)


class LlmJudgeMetric(PolicyAwareMetric, BaseMetric):
    """Base metric that defers judgement to an LLM model."""

    metric_id: ClassVar[str] = "string_llm"
    recurse_into_children: ClassVar[bool] = False
    default_model_id: ClassVar[str] = DEFAULT_LLM_MODEL_ID
    default_pass_threshold: ClassVar[float] = DEFAULT_PASS_THRESHOLD
    system_prompt: ClassVar[str] = _DEFAULT_SYSTEM_PROMPT
    default_prompt_template: ClassVar[str] = LLM_JUDGE_PROMPT
    default_structured_output_schema: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["score", "reasoning"],
        "additionalProperties": True,
    }

    @staticmethod
    def _validate_prompt_template(template: str) -> None:
        required_placeholders = {"gold", "predicted", "structured_output_schema"}
        found_placeholders = set(re.findall(r"\{(\w+)\}", template))
        missing = required_placeholders - found_placeholders

        if missing:
            raise ValueError(
                f"Custom prompt_template missing required placeholders: {missing}. "
                f"Required placeholders are: {required_placeholders}"
            )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        params = config.params if config and config.params else {}
        model_id = params.get("model") or self.default_model_id
        pass_threshold = float(
            params.get("pass_threshold", self.default_pass_threshold)
        )

        prompt_template = params.get("prompt_template") or self.default_prompt_template
        self._validate_prompt_template(prompt_template)

        structured_schema = copy.deepcopy(
            params.get(
                "structured_output_schema", self.default_structured_output_schema
            )
        )

        additional_instructions = params.get("additional_instructions")

        messages = self._build_messages(
            node,
            gold,
            extracted,
            prompt_template,
            structured_schema,
            additional_instructions=additional_instructions,
        )

        try:
            response = await litellm.acompletion(
                model=model_id,
                messages=messages,
                temperature=params.get("temperature", 0.0),
                max_tokens=params.get("max_tokens"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"LLM judge metric '{self.metric_id}' failed for model '{model_id}': {exc}"
            ) from exc

        content = self._extract_message_content(response)
        parsed = self._parse_structured_output(content, structured_schema)
        parsed = self.postprocess_parsed_result(node, config, parsed)

        score = parsed.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError(
                f"LLM judge metric '{self.metric_id}' expected 'score' in output. Got: {parsed!r}"
            )
        score = float(score)
        passed = score >= pass_threshold

        details = {
            "model": model_id,
            "pass_threshold": pass_threshold,
            "structured_output": parsed,
            "raw_response": content,
            "gold": gold,
            "extracted": extracted,
        }

        reasoning = parsed.get("reasoning")
        if reasoning is not None:
            details["reasoning"] = reasoning

        result = MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details=details,
        )
        return result

    def _build_messages(
        self,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        prompt_template: str,
        structured_schema: Dict[str, Any],
        additional_instructions: str | None = None,
    ) -> List[Dict[str, str]]:
        metadata = json.dumps(node.get_metadata_summary(), ensure_ascii=True, indent=2)
        prompt = prompt_template.format(
            metadata=metadata,
            gold=_format_value(gold),
            predicted=_format_value(extracted),
            structured_output_schema=json.dumps(
                structured_schema, ensure_ascii=True, indent=2
            ),
        )

        if additional_instructions:
            prompt = f"{prompt}\n\nAdditional Instructions:\n{additional_instructions}"

        extra_context = self._get_additional_prompt_context(node)
        if extra_context:
            prompt = f"{prompt}\n\n{extra_context}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _get_additional_prompt_context(self, node: MetricContext) -> str | None:
        """Hook for subclasses to append extra context to the prompt. Returns None by default."""
        return None

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(
                "LLM judge metric received unexpected response format from litellm."
            ) from exc

    @staticmethod
    def _parse_structured_output(
        content: str, structured_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        candidate = content.strip()

        code_block = _extract_code_block(candidate)
        raw_json = code_block if code_block is not None else candidate

        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM judge metric expected JSON output but failed to parse."
            ) from exc

        try:
            validate_json_schema(parsed, structured_schema)
        except JsonSchemaValidationError as exc:
            raise ValueError(
                "LLM judge metric output did not match the expected schema."
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("LLM judge metric output must be a JSON object.")

        return parsed

    def postprocess_parsed_result(
        self,
        node: MetricContext,
        config: MetricConfig | None,
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        return parsed


def _extract_code_block(content: str) -> str | None:
    parts = content.split("```")
    for part in parts[1::2]:
        cleaned = part.lstrip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.lstrip()
        if cleaned:
            return cleaned.rstrip()
    return None
