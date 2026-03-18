"""Content statistics collector for gold/extracted JSON."""

from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

from ...infra.nodes import (
    AnyOfSchema,
    ArraySchema,
    NullSchema,
    ObjectSchema,
    RootSchema,
)
from .models import ArrayStats, ContentStats, CoverageStats


def _get_value_type(value: Any) -> str:
    """Map Python type to schema type name."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _walk_json(
    data: Any,
    path: str = "",
) -> Tuple[int, Dict[str, int], List[int]]:
    """Recursively walk JSON and collect stats.

    Returns:
        Tuple of (key_count, type_counts, array_lengths)
    """
    key_count = 0
    type_counts: Dict[str, int] = defaultdict(int)
    array_lengths: List[int] = []

    if isinstance(data, dict):
        for key, value in data.items():
            key_count += 1
            value_type = _get_value_type(value)
            type_counts[value_type] += 1

            child_path = f"{path}.{key}" if path else key
            child_keys, child_types, child_arrays = _walk_json(value, child_path)
            key_count += child_keys
            for t, c in child_types.items():
                type_counts[t] += c
            array_lengths.extend(child_arrays)

    elif isinstance(data, list):
        array_lengths.append(len(data))
        for i, item in enumerate(data):
            child_path = f"{path}[{i}]"
            child_keys, child_types, child_arrays = _walk_json(item, child_path)
            key_count += child_keys
            for t, c in child_types.items():
                type_counts[t] += c
            array_lengths.extend(child_arrays)

    return key_count, dict(type_counts), array_lengths


def collect_content_stats(data: dict, label: str) -> ContentStats:
    """Collect content statistics from a JSON dict."""
    key_count, type_counts, array_lengths = _walk_json(data)

    array_stats: Optional[ArrayStats] = None
    if array_lengths:
        array_stats = ArrayStats(
            array_field_count=len(array_lengths),
            total_items=sum(array_lengths),
            min_length=min(array_lengths),
            median_length=median(array_lengths),
            max_length=max(array_lengths),
        )

    return ContentStats(
        label=label,
        total_keys=key_count,
        counts_by_type=type_counts,
        array_stats=array_stats,
        null_count=type_counts.get("null", 0),
        missing_count=0,
    )


def _collect_schema_paths(
    schema: Any,
    prefix: str = "",
) -> Tuple[Set[str], Set[str]]:
    """Walk the schema tree to collect all valid property paths and required paths.

    Returns (all_paths, required_paths). Array item fields are collected
    without index (e.g. ``education.description``) so they can be matched
    against indexed data paths.
    """
    all_paths: Set[str] = set()
    required_paths: Set[str] = set()

    if isinstance(schema, (ObjectSchema, RootSchema)):
        required_set = set(schema.required or []) if hasattr(schema, "required") else set()
        if schema.properties:
            for key, child in schema.properties.items():
                path = f"{prefix}.{key}" if prefix else key
                all_paths.add(path)
                if key in required_set:
                    required_paths.add(path)
                child_all, child_req = _collect_schema_paths(child, path)
                all_paths.update(child_all)
                required_paths.update(child_req)
        # additionalProperties are dynamic keys — we can't enumerate them
        # but we record the prefix so the caller knows this object allows extras
        if schema.additional_properties and not isinstance(schema.additional_properties, bool):
            all_paths.add(f"{prefix}.*")

    elif isinstance(schema, ArraySchema):
        # Array items share one schema. Collect paths without index.
        child_all, child_req = _collect_schema_paths(schema.items, prefix)
        all_paths.update(child_all)
        required_paths.update(child_req)

    elif isinstance(schema, AnyOfSchema):
        # Collect paths from all non-null branches
        for child in schema.any_of:
            if isinstance(child, NullSchema):
                continue
            child_all, child_req = _collect_schema_paths(child, prefix)
            all_paths.update(child_all)
            required_paths.update(child_req)

    return all_paths, required_paths


def _collect_data_paths(data: Any, prefix: str = "") -> Set[str]:
    """Collect leaf paths from JSON data, stripping array indices.

    ``education[0].description`` becomes ``education.description`` so it
    can be matched against schema paths.
    """
    paths: Set[str] = set()

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)) and value:
                paths.update(_collect_data_paths(value, path))
            else:
                paths.add(path)
    elif isinstance(data, list):
        for item in data:
            # Recurse without adding an index — collapse all items into
            # the same set of paths so we get one entry per field name.
            paths.update(_collect_data_paths(item, prefix))
    else:
        if prefix:
            paths.add(prefix)

    return paths


def _is_schema_known(path: str, schema_paths: Set[str]) -> bool:
    """Check if a data path matches any schema-declared path.

    Handles additionalProperties wildcards: if the schema has ``skills.*``
    then ``skills.Python`` is considered known.
    """
    if path in schema_paths:
        return True
    # Check wildcard: skills.Programming Languages → skills.*
    parts = path.rsplit(".", 1)
    if len(parts) == 2:
        parent_wildcard = f"{parts[0]}.*"
        if parent_wildcard in schema_paths:
            return True
    return False


def compute_coverage(
    gold: dict,
    extracted: dict,
    schema: RootSchema,
) -> CoverageStats:
    """Compute schema-aware coverage statistics comparing gold vs extracted.

    Only considers paths declared in the schema. Fields not in the schema
    (e.g. ``array_index`` metadata) are ignored.
    """
    schema_paths, required_paths = _collect_schema_paths(schema)
    gold_paths = _collect_data_paths(gold)
    extracted_paths = _collect_data_paths(extracted)

    # Filter to only schema-known paths
    gold_known = {p for p in gold_paths if _is_schema_known(p, schema_paths)}
    extracted_known = {p for p in extracted_paths if _is_schema_known(p, schema_paths)}

    present_in_both = gold_known & extracted_known
    missing_in_extracted = gold_known - extracted_known
    spurious_in_extracted = extracted_known - gold_known

    required_missing = missing_in_extracted & required_paths

    return CoverageStats(
        present_in_both=len(present_in_both),
        missing_in_extracted=len(missing_in_extracted),
        spurious_in_extracted=len(spurious_in_extracted),
        required_missing=len(required_missing),
        missing_paths=sorted(missing_in_extracted),
        spurious_paths=sorted(spurious_in_extracted),
    )
