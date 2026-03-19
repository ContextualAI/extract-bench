"""Main structured evaluator for comparing predicted vs gold JSON."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List

from ..infra.asyncio_utils import run_asyncio_task_with_retry
from ..infra.construct_ast import construct_ast
from ..infra.nodes import (
    AnyOfSchema,
    ArraySchema,
    ObjectSchema,
    RootSchema,
    Schema,
)
from ..infra.visitors import AsyncPathAnalyzerVisitor, TransformerVisitor
from .evaluation_config import MetricConfig
from .metric_registry import global_metric_registry
from .metrics.base_metric import BaseMetric, MetricResult
from .schema_config_helpers import get_evaluation_config
from .schema_value_instantiator import SchemaValueInstantiator


@dataclass
class AsyncEvaluationConfig:
    """Configuration for parallelizing evaluation calls."""

    metric_timeout_seconds: int = 60
    n_max_retries: int = 3
    sleep_base_seconds_for_retry: int = 3
    max_sleep_time_seconds: int = 30
    parallel_traversal: bool = True


@dataclass
class StructuredEvaluatorConfig:
    metrics: List[BaseMetric]
    async_config: AsyncEvaluationConfig = None

    def __post_init__(self):
        if self.async_config is None:
            self.async_config = AsyncEvaluationConfig()


class StructuredEvaluator:
    """Evaluates structured extraction quality by comparing predicted vs gold JSON."""

    def __init__(self, config: StructuredEvaluatorConfig):
        self.config = config

    def evaluate(
        self, json_schema: dict, gold_json: dict, extracted_json: dict
    ) -> dict:
        """Synchronous wrapper for backward compatibility."""
        return asyncio.run(self.evaluate_async(json_schema, gold_json, extracted_json))

    async def evaluate_async(
        self, json_schema: dict, gold_json: dict, extracted_json: dict
    ) -> dict:
        """Primary async evaluation method with parallel traversal."""
        schema_tree = construct_ast(json_schema)

        schema_tree = _resolve_evaluation_configs(schema_tree)

        instantiated_schema = SchemaValueInstantiator(
            schema_tree, gold_json, extracted_json
        ).instantiate()

        visitor = _AsyncMetricsEvaluatorVisitor(self.config.async_config)
        await instantiated_schema.accept_async(visitor)

        return {"schema": instantiated_schema, "results": visitor.get_results()}


async def execute_metric_async(
    metric_config: MetricConfig, node: Schema
) -> tuple[MetricResult, BaseMetric]:
    """Execute a single metric asynchronously."""
    metric = global_metric_registry.create_metric(metric_config.metric_id)
    result = await metric.evaluate(node, metric_config)
    return result, metric


class _AsyncMetricsEvaluatorVisitor(AsyncPathAnalyzerVisitor):
    """Async visitor that evaluates metrics at each node."""

    def __init__(self, async_config: AsyncEvaluationConfig):
        super().__init__(parallel_traversal=async_config.parallel_traversal)
        self.results: Dict[str, Dict[str, MetricResult]] = {}
        self.async_config = async_config

    def get_results(self) -> Dict[str, Dict[str, MetricResult]]:
        return self.results

    async def visit_rootschema(self, node: RootSchema) -> None:
        await self._evaluate_node(node)
        await super().visit_rootschema(node)

    async def visit_objectschema(self, node: ObjectSchema) -> None:
        recurse = await self._evaluate_node(node)
        if recurse:
            await super().visit_objectschema(node)

    async def visit_arrayschema(self, node: ArraySchema) -> None:
        recurse = await self._evaluate_node(node)
        if recurse:
            await super().visit_arrayschema(node)

    async def visit_anyofschema(self, node: AnyOfSchema) -> None:
        recurse = await self._evaluate_node(node)
        if recurse:
            await super().visit_anyofschema(node)

    async def generic_leaf_visit(self, node: Schema) -> Schema:
        await self._evaluate_node(node)
        return node

    async def _evaluate_node(self, node: Schema) -> bool:
        """Evaluate all metrics for a node."""
        config = node.evaluation_config
        if config is None or not config.metrics:
            return True

        metric_configs = config.metrics
        results = await asyncio.gather(
            *(
                self._execute_metric_with_retry(mc, node)
                for mc in metric_configs
            ),
            return_exceptions=True,
        )

        metrics_results: Dict[str, MetricResult] = {}
        should_recurse = True

        for mc, raw_result in zip(metric_configs, results):
            should_recurse = should_recurse and self._accumulate_metric_result(
                raw_result, mc, metrics_results
            )

        if metrics_results:
            path = self._path()
            node.set_evaluation_result(metrics_results)
            self.results[path] = metrics_results

        return should_recurse

    async def _execute_metric_with_retry(
        self, metric_config: MetricConfig, node: Schema
    ) -> tuple[MetricResult, BaseMetric]:
        return await run_asyncio_task_with_retry(
            task_factory=lambda: execute_metric_async(metric_config, node),
            n_max_retries=self.async_config.n_max_retries,
            sleep_base_seconds_for_retry=self.async_config.sleep_base_seconds_for_retry,
            max_sleep_time_seconds=self.async_config.max_sleep_time_seconds,
            timeout_seconds=self.async_config.metric_timeout_seconds,
        )

    def _accumulate_metric_result(
        self,
        raw_result: Exception | tuple[MetricResult, BaseMetric],
        metric_config: MetricConfig,
        metrics_results: Dict[str, MetricResult],
    ) -> bool:
        if isinstance(raw_result, Exception):
            error_result = self._build_error_metric_result(
                raw_result, metric_config.metric_id
            )
            metrics_results[error_result.metric_id] = error_result
            return False

        metric_result, metric = raw_result
        metrics_results[metric_result.metric_id] = metric_result
        return getattr(metric, "recurse_into_children", True)

    @staticmethod
    def _build_error_metric_result(
        error: Exception, metric_id: str = "unknown"
    ) -> MetricResult:
        return MetricResult(
            metric_id=metric_id,
            score=0.0,
            passed=False,
            details={
                "error": str(error),
                "error_type": type(error).__name__,
                "reason": "error",
            },
        )


def _resolve_evaluation_configs(schema: Schema) -> Schema:
    resolver = _EvaluationConfigResolver()
    resolved = schema.accept(resolver)
    if isinstance(resolved, RootSchema):
        resolved.set_root_schema_for_children()
    return resolved


class _EvaluationConfigResolver(TransformerVisitor):
    def generic_leaf_visit(self, node: Schema) -> Schema:
        transformed = super().generic_leaf_visit(node)
        return self._apply(transformed)

    def visit_arrayschema(self, node: ArraySchema) -> ArraySchema:
        transformed = super().visit_arrayschema(node)
        return self._apply(transformed)

    def visit_anyofschema(self, node: AnyOfSchema) -> AnyOfSchema:
        transformed = super().visit_anyofschema(node)
        return self._apply(transformed)

    def visit_objectschema(self, node: ObjectSchema) -> ObjectSchema:
        transformed = super().visit_objectschema(node)
        return self._apply(transformed)

    def visit_rootschema(
        self, node: RootSchema, set_root_schema_for_children: bool = True
    ) -> RootSchema:
        transformed = super().visit_rootschema(node, set_root_schema_for_children)
        return self._apply(transformed)

    def _apply(self, node: Schema) -> Schema:
        resolved = get_evaluation_config(node)
        current = node.evaluation_config

        if resolved and resolved.metrics:
            return node.create_copy_with_updated_fields({"evaluation_config": resolved})

        if current is None:
            return node

        return node.create_copy_with_updated_fields({"evaluation_config": None})
