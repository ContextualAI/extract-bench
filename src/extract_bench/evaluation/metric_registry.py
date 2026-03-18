"""Metric registry for structured extraction evaluation."""

from collections.abc import Callable
from typing import Dict, Mapping

from .metrics.array_metrics import ArrayLlmJudgeMetric
from .metrics.base_metric import BaseMetric
from .metrics.boolean_metrics import BooleanExactMatchMetric
from .metrics.llm_metrics import LlmJudgeMetric
from .metrics.metric_descriptors import MetricPromptDescriptor
from .metrics.number_metrics import (
    ExactNumberMatchMetric,
    IntegerExactMatchMetric,
    NumberToleranceMetric,
)
from .metrics.string_metrics import (
    CaseInsensitiveStringMatchMetric,
    ExactStringMatchMetric,
    NormalizedLevenshteinSimilarityMetric,
    StringSemanticMetric,
    UrlNormalizedStringMatchMetric,
)

MetricFactory = Callable[[], BaseMetric]


class MetricRegistry:
    """Registry storing factories for metric instances."""

    def __init__(self, *, aliases: Mapping[str, str] | None = None) -> None:
        self._factories: Dict[str, MetricFactory] = {}
        self._aliases: Dict[str, str] = dict(aliases or {})
        self._descriptors: Dict[str, MetricPromptDescriptor] = {}

    def register_metric_factory(
        self, metric_id: str, factory: MetricFactory, *, override: bool = False
    ) -> None:
        if metric_id in self._factories and not override:
            raise ValueError(f"Metric '{metric_id}' is already registered")
        self._factories[metric_id] = factory

    def register_metric(
        self,
        factory: MetricFactory,
        *,
        override: bool = False,
        aliases: tuple[str, ...] = (),
    ) -> str:
        sample = factory()
        metric_id = sample.metric_id
        self.register_metric_factory(metric_id, factory, override=override)
        descriptor = getattr(sample, "prompt_descriptor", None)
        if descriptor is not None:
            self.register_metric_descriptor(metric_id, descriptor, override=override)
        for alias in aliases:
            self._aliases[alias] = metric_id
        return metric_id

    def has_metric(self, metric_id: str) -> bool:
        resolved = self._aliases.get(metric_id, metric_id)
        return resolved in self._factories

    def create_metric(self, metric_id: str) -> BaseMetric:
        resolved = self._aliases.get(metric_id, metric_id)
        if resolved not in self._factories:
            raise KeyError(f"Metric '{metric_id}' is not registered")
        metric = self._factories[resolved]()
        return metric

    def register_metric_descriptor(
        self,
        metric_id: str,
        descriptor: MetricPromptDescriptor,
        *,
        override: bool = False,
    ) -> None:
        if metric_id in self._descriptors and not override:
            raise ValueError(f"Metric descriptor '{metric_id}' is already registered")
        if descriptor.metric_id != metric_id:
            raise ValueError(
                f"Metric descriptor id mismatch: expected '{metric_id}', got '{descriptor.metric_id}'"
            )
        self._descriptors[metric_id] = descriptor

    def get_metric_descriptor(self, metric_id: str) -> MetricPromptDescriptor | None:
        resolved = self._aliases.get(metric_id, metric_id)
        return self._descriptors.get(resolved)

    def available_metrics(self) -> tuple[str, ...]:
        return tuple(self._factories.keys())

    def unregister_metric(self, metric_id: str) -> None:
        resolved = self._aliases.get(metric_id, metric_id)
        self._factories.pop(resolved, None)
        self._descriptors.pop(resolved, None)
        self._aliases = {
            alias: target
            for alias, target in self._aliases.items()
            if alias != metric_id and target != resolved
        }


def _register_default_metrics(registry: MetricRegistry) -> None:
    default_metrics = [
        ExactStringMatchMetric,
        CaseInsensitiveStringMatchMetric,
        UrlNormalizedStringMatchMetric,
        NormalizedLevenshteinSimilarityMetric,
        StringSemanticMetric,
        ExactNumberMatchMetric,
        NumberToleranceMetric,
        IntegerExactMatchMetric,
        BooleanExactMatchMetric,
        LlmJudgeMetric,
        ArrayLlmJudgeMetric,
    ]
    for metric in default_metrics:
        registry.register_metric(metric)


global_metric_registry = MetricRegistry()
_register_default_metrics(global_metric_registry)
