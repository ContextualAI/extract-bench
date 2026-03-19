"""Data models for evaluation reports."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SchemaStats:
    """Schema shape statistics."""

    total_nodes: int
    counts_by_type: Dict[str, int]
    required_keys_count: int
    optional_keys_count: int
    required_by_type: Dict[str, int] = field(default_factory=dict)
    optional_by_type: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ArrayStats:
    """Array field statistics."""

    array_field_count: int
    total_items: int
    min_length: int
    median_length: float
    max_length: int


@dataclass(frozen=True)
class ContentStats:
    """Content statistics for gold or extracted JSON."""

    label: str
    total_keys: int
    counts_by_type: Dict[str, int]
    array_stats: Optional[ArrayStats] = None
    null_count: int = 0
    missing_count: int = 0


@dataclass(frozen=True)
class CoverageStats:
    """Coverage statistics comparing gold vs extracted."""

    present_in_both: int
    missing_in_extracted: int
    spurious_in_extracted: int
    required_missing: int
    missing_paths: List[str] = field(default_factory=list)
    spurious_paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConfusionCounts:
    """Confusion matrix counts for field-level evaluation.

    TP: passed, both values present (correct extraction).
    TN: passed, both values absent (correct omission).
    FP: failed, gold absent but extracted present (spurious extraction).
    FN: failed, gold present but extracted absent or value mismatch.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass(frozen=True)
class LowScoringField:
    """A low-scoring field with reasoning snippet."""

    path: str
    metric_id: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class LlmStats:
    """Statistics for LLM-based metrics."""

    call_count: int
    avg_score: float
    median_score: float
    score_distribution: Dict[str, int]  # bucketed: 0-0.2, 0.2-0.4, etc.
    top_lowest_fields: List[LowScoringField] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorGroup:
    """A group of errors by category."""

    category: str
    count: int
    paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorBreakdown:
    """Grouped failure analysis."""

    by_path_prefix: Dict[str, int]
    by_metric_type: Dict[str, int]
    by_reason: Dict[str, int]
    details: List[ErrorGroup] = field(default_factory=list)


@dataclass(frozen=True)
class ArrayBreakdown:
    """Rich breakdown for array_llm evaluations."""

    matched: int
    missed_gold: int
    spurious_pred: int
    precision: float
    recall: float
    f1: float
    matched_items: Optional[List[Any]] = None
    missed_gold_items: Optional[List[Any]] = None
    spurious_pred_items: Optional[List[Any]] = None


@dataclass(frozen=True)
class FieldOutcome:
    """Per-field evaluation outcome."""

    path: str
    normalized_path: str
    metric_id: str
    score: float
    passed: bool
    reason: Optional[str] = None
    gold_value: Any = None
    extracted_value: Any = None
    reasoning: Optional[str] = None
    array_breakdown: Optional[ArrayBreakdown] = None


@dataclass(frozen=True)
class OutcomeStats:
    """Evaluation outcome statistics."""

    total_evaluated: int
    total_passed: int
    total_failed: int
    pass_rate: float
    pass_by_metric: Dict[str, Dict[str, int]]  # metric_id -> {passed, failed}
    pass_by_type: Dict[str, Dict[str, int]]  # schema_type -> {passed, failed}
    pass_by_required: Dict[str, Dict[str, int]]  # required/optional -> {passed, failed}
    confusion: ConfusionCounts
    llm_stats: Optional[LlmStats] = None
    error_breakdown: Optional[ErrorBreakdown] = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    # Metadata
    output_name: str
    timestamp: str

    # Input references (not serialized fully)
    schema_hash: str
    gold_hash: str
    extracted_hash: str

    # Statistics
    schema_stats: SchemaStats
    gold_stats: ContentStats
    extracted_stats: ContentStats
    coverage: CoverageStats
    outcomes: OutcomeStats

    # Per-field details
    field_outcomes: List[FieldOutcome]

    # Summary scores
    overall_score: float  # Item-weighted: arrays weighted by gold item count
    field_score: float  # Flat average: each field counts equally
    overall_pass_rate: float
