"""String evaluation metrics."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlparse

from ..evaluation_config import MetricConfig
from .base_metric import BaseMetric, MetricContext, MetricResult
from .llm_metrics import LlmJudgeMetric
from .metric_descriptors import MetricPromptDescriptor
from .metric_prompts import STRING_SEMANTIC_PROMPT
from .metric_utils import _is_missing, get_config_param
from .policy_metric import PolicyAwareMetric


def _normalize_optional_string(value: Any) -> tuple[bool, str | None]:
    if _is_missing(value):
        return True, None
    if value is None:
        return False, None
    return False, str(value)


@dataclass(frozen=True, slots=True)
class ExactStringMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "string_exact"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="string_exact",
        summary="Exact, case-sensitive string equality.",
        pass_rule="pass iff normalized(gold) == normalized(predicted) as strings",
        score_rule="1.0 if pass else 0.0",
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        _, gold_str = _normalize_optional_string(gold)
        _, extracted_str = _normalize_optional_string(extracted)

        if gold_str is None or extracted_str is None:
            passed = gold_str is extracted_str
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={"gold": gold, "extracted": extracted},
            )

        passed = gold_str == extracted_str
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={"gold": gold_str, "extracted": extracted_str},
        )


@dataclass(frozen=True, slots=True)
class CaseInsensitiveStringMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "string_case_insensitive"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="string_case_insensitive",
        summary="Case-insensitive string equality.",
        pass_rule="pass iff casefold(gold) == casefold(predicted)",
        score_rule="1.0 if pass else 0.0",
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        nullable_gold, gold_str = _normalize_optional_string(gold)
        nullable_extracted, extracted_str = _normalize_optional_string(extracted)

        if gold_str is None or extracted_str is None:
            passed = gold_str is extracted_str and not (
                nullable_gold ^ nullable_extracted
            )
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={"gold": gold, "extracted": extracted},
            )

        normalized_gold = gold_str.casefold()
        normalized_extracted = extracted_str.casefold()
        passed = normalized_gold == normalized_extracted
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={
                "gold": gold_str,
                "extracted": extracted_str,
                "gold_normalized": normalized_gold,
                "extracted_normalized": normalized_extracted,
            },
        )


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison by stripping protocol, www, trailing slashes, and fragments."""
    s = url.strip().lower()
    # Strip protocol
    s = re.sub(r"^https?://", "", s)
    # Strip www.
    s = re.sub(r"^www\.", "", s)
    # Strip trailing slashes
    s = s.rstrip("/")
    # Strip fragment
    s = s.split("#")[0]
    return s


@dataclass(frozen=True, slots=True)
class UrlNormalizedStringMatchMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "string_url"
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="string_url",
        summary="URL-normalized string equality (strips protocol, www, trailing slash).",
        pass_rule="pass iff normalize_url(gold) == normalize_url(predicted)",
        score_rule="1.0 if pass else 0.0",
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        _, gold_str = _normalize_optional_string(gold)
        _, extracted_str = _normalize_optional_string(extracted)

        if gold_str is None or extracted_str is None:
            passed = gold_str is extracted_str
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={"gold": gold, "extracted": extracted},
            )

        gold_normalized = _normalize_url(gold_str)
        extracted_normalized = _normalize_url(extracted_str)
        passed = gold_normalized == extracted_normalized
        score = 1.0 if passed else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            passed=passed,
            details={
                "gold": gold_str,
                "extracted": extracted_str,
                "gold_normalized": gold_normalized,
                "extracted_normalized": extracted_normalized,
            },
        )


@dataclass(frozen=True, slots=True)
class NormalizedLevenshteinSimilarityMetric(PolicyAwareMetric, BaseMetric):
    metric_id: str = "string_fuzzy"
    default_threshold: float = 0.8
    default_case_sensitive: bool = False
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="string_fuzzy",
        summary="Fuzzy string match using normalized similarity with a threshold.",
        pass_rule="pass iff similarity(gold, predicted) >= threshold",
        score_rule="similarity in [0.0, 1.0] (pass determined by threshold)",
        params={
            "threshold": {"type": "number", "meaning": "minimum similarity to pass"},
            "case_sensitive": {
                "type": "boolean",
                "meaning": "whether to treat casing as significant",
            },
        },
    )

    async def _evaluate_values(
        self,
        *,
        node: MetricContext,
        gold: Any,
        extracted: Any,
        config: MetricConfig | None,
    ) -> MetricResult:
        _, gold_str = _normalize_optional_string(gold)
        _, extracted_str = _normalize_optional_string(extracted)

        if gold_str is None or extracted_str is None:
            passed = gold_str is extracted_str
            score = 1.0 if passed else 0.0
            return MetricResult(
                metric_id=self.metric_id,
                score=score,
                passed=passed,
                details={"gold": gold, "extracted": extracted},
            )

        threshold = get_config_param(config, "threshold", self.default_threshold, float)
        case_sensitive = get_config_param(
            config, "case_sensitive", self.default_case_sensitive, bool
        )
        if not case_sensitive:
            gold_str = gold_str.lower()
            extracted_str = extracted_str.lower()
        similarity = SequenceMatcher(None, gold_str, extracted_str).ratio()
        passed = similarity >= threshold
        return MetricResult(
            metric_id=self.metric_id,
            score=similarity,
            passed=passed,
            details={
                "gold": gold_str,
                "extracted": extracted_str,
                "threshold": threshold,
                "similarity": similarity,
                "case_sensitive": case_sensitive,
            },
        )


@dataclass(frozen=True, slots=True)
class StringSemanticMetric(LlmJudgeMetric):
    metric_id: str = "string_semantic"
    default_prompt_template: str = STRING_SEMANTIC_PROMPT
    prompt_descriptor = MetricPromptDescriptor(
        metric_id="string_semantic",
        summary="LLM-based semantic string evaluation (meaning-level match).",
        pass_rule="pass iff score >= pass_threshold",
        score_rule="LLM-assigned score in [0.0, 1.0]",
        params={
            "pass_threshold": {"type": "number", "meaning": "minimum score to pass"},
            "additional_instructions": {
                "type": "string",
                "meaning": "extra domain rules",
            },
        },
        notes=("Use schema metadata and additional instructions when provided.",),
    )
