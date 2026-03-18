# Extract Bench

[![arXiv](https://img.shields.io/badge/arXiv-2602.12247-b31b1b.svg)](https://arxiv.org/abs/2602.12247)

A benchmark for structured extraction from PDF documents, comprising:

1. **[Dataset](dataset/)** -- 35 human-validated PDF-to-JSON extraction tasks across 5 schemas and 4 domains (finance, academia, hiring, sports).
2. **Evaluation Suite** -- A Python package that scores predicted JSON against gold JSON with per-field metrics (exact match, fuzzy, semantic/LLM-based, numeric tolerance, and more).

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Evaluation API](#evaluation-api)
  - [ReportConfig Options](#reportconfig-options)
  - [Batch Evaluation](#batch-evaluation)
  - [Report Output Format](#report-output-format)
  - [Low-Level API](#low-level-api)
- [Metrics](#metrics)
  - [Available Metrics](#available-metrics)
  - [Evaluation Presets](#evaluation-presets)
  - [Custom Metrics](#custom-metrics)
- [Configuration](#configuration)
  - [Environment Setup](#environment-setup)
  - [LLM Model Configuration](#llm-model-configuration)
- [Architecture](#architecture)
- [Citation](#citation)
- [Development](#development)

## Getting Started

### Installation

```bash
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Quick Start

```python
import json
from pathlib import Path
from extract_bench import ReportBuilder, ReportConfig

# Load your data
schema = json.load(open("schema.json"))
gold = json.load(open("gold.json"))
extracted = json.load(open("model_output.json"))

# Configure and build report
config = ReportConfig(
    output_dir=Path("./eval_results"),
    output_name="nvidia-10k-extract-gemini-flash",  # Identifies this experiment
)
builder = ReportBuilder(config)
report = builder.build(schema, gold, extracted)

# Save all outputs
output_path = builder.save(report)
print(f"Results saved to: {output_path}")
```

This creates `eval_results/nvidia-10k-extract-gemini-flash/` containing:

| File          | Purpose                                                  |
| ------------- | -------------------------------------------------------- |
| `report.json` | Machine-readable full report (for programmatic analysis) |
| `summary.txt` | Human-readable one-page summary (for quick inspection)   |
| `fields.csv`  | Per-field outcomes (for csv analysis)                    |
| `fields.md`   | Markdown table (for documentation/sharing)               |

Key metrics on the report object:

```python
print(f"Overall pass rate: {report.overall_pass_rate:.1%}")
print(f"Overall score: {report.overall_score:.3f}")
print(f"Fields evaluated: {report.outcomes.total_evaluated}")
print(f"Passed: {report.outcomes.total_passed}")
print(f"Failed: {report.outcomes.total_failed}")
```

## Evaluation API

### ReportConfig Options

```python
config = ReportConfig(
    output_dir=Path("./outputs"),      # Where to save reports
    output_name="my-experiment",       # Subdirectory name (auto-generated if None)
    max_reasoning_length=200,          # Truncate LLM reasoning in outputs
    top_n_lowest_fields=5,             # Track N lowest-scoring fields
    save_json=True,                    # Generate report.json
    save_text=True,                    # Generate summary.txt
    save_csv=True,                     # Generate fields.csv
    save_markdown=True,                # Generate fields.md
)
```

### Batch Evaluation

For running many experiments:

```python
import asyncio
import json
from pathlib import Path
from extract_bench import ReportBuilder, ReportConfig

async def evaluate_model_outputs(
    schema_path: Path,
    gold_path: Path,
    outputs_dir: Path,
    results_dir: Path,
):
    """Evaluate all model outputs in a directory."""
    schema = json.load(schema_path.open())
    gold = json.load(gold_path.open())

    results = []
    for output_file in outputs_dir.glob("*.json"):
        extracted = json.load(output_file.open())

        config = ReportConfig(
            output_dir=results_dir,
            output_name=output_file.stem,  # Use filename as experiment ID
        )
        builder = ReportBuilder(config)
        report = await builder.build_async(schema, gold, extracted)
        builder.save(report)

        results.append({
            "model": output_file.stem,
            "pass_rate": report.overall_pass_rate,
            "score": report.overall_score,
        })

    return results

# Run batch evaluation
results = asyncio.run(evaluate_model_outputs(
    schema_path=Path("schema.json"),
    gold_path=Path("gold.json"),
    outputs_dir=Path("./model_outputs"),
    results_dir=Path("./eval_results"),
))

# Print comparison
for r in sorted(results, key=lambda x: -x["score"]):
    print(f"{r['model']}: {r['pass_rate']:.1%} pass, {r['score']:.3f} avg score")
```

### Report Output Format

#### summary.txt

```
================================================================================
                        EVALUATION REPORT: my-experiment
================================================================================

OVERALL RESULTS
---------------
Pass Rate: 85.2% (23/27 fields)
Average Score: 0.891

SCHEMA SHAPE
------------
Total nodes: 45
By type: object=12, string=18, number=8, array=5, boolean=2

COVERAGE
--------
Present in both: 25
Missing in extracted: 2
Spurious in extracted: 0

PASS/FAIL BY METRIC
-------------------
string_semantic: 15/18 passed (83.3%)
number_tolerance: 6/6 passed (100.0%)
integer_exact: 2/3 passed (66.7%)

LOWEST SCORING FIELDS
---------------------
1. borrower.address (0.45) - Partial match, missing suite number
2. terms.rate_type (0.60) - Semantic mismatch
...
```

#### fields.csv

| Column            | Description                                 |
| ----------------- | ------------------------------------------- |
| `path`            | Full JSONPath to the field                  |
| `normalized_path` | Human-readable path (e.g., `borrower.name`) |
| `metric_id`       | Metric used for evaluation                  |
| `score`           | Numeric score (0.0-1.0)                     |
| `passed`          | Boolean pass/fail                           |
| `gold_value`      | Expected value                              |
| `extracted_value` | Model's output value                        |
| `reasoning`       | LLM reasoning (for semantic metrics)        |

### Low-Level API

For direct access to evaluation results without reporting:

```python
from extract_bench import StructuredEvaluator, StructuredEvaluatorConfig

evaluator = StructuredEvaluator(StructuredEvaluatorConfig(metrics=[]))
result = evaluator.evaluate(schema, gold, predicted)

# Raw results dict: path -> metric_id -> MetricResult
for path, metrics in result["results"].items():
    for metric_id, metric_result in metrics.items():
        print(f"{path} [{metric_id}]: passed={metric_result.passed}, score={metric_result.score}")
```

Use `evaluate_async()` for better performance with LLM-based metrics.

## Metrics

### Available Metrics

| Category | Metric                    | Description                             |
| -------- | ------------------------- | --------------------------------------- |
| String   | `string_exact`            | Case-sensitive exact match              |
|          | `string_case_insensitive` | Case-insensitive match                  |
|          | `string_fuzzy`            | Levenshtein similarity                  |
|          | `string_semantic`         | LLM-based semantic comparison (default) |
|          | `string_url`              | URL-normalized match (strips protocol, www, trailing slash) |
| Number   | `number_exact`            | Exact numeric equality                  |
|          | `number_tolerance`        | Match within tolerance (default)        |
|          | `integer_exact`           | Exact integer equality                  |
| Boolean  | `boolean_exact`           | Exact boolean equality                  |
| Array    | `array_llm`               | LLM-based array comparison              |
| General  | `string_llm`              | LLM judge for any comparison            |

> **Automatic inference:**
> - Fields with `"format": "uri"` and `"evaluation_config": "string_exact"` are automatically evaluated with `string_url` instead, so `http://example.com` and `example.com` are treated as equal.
> - `anyOf` nodes without an explicit `evaluation_config` inherit the config from their children when all non-null branches agree on the same metric. For example, `anyOf: [{type: array, evaluation_config: "array_llm"}, {type: object, ...}]` will automatically use `array_llm` on the anyOf node.

### Evaluation Presets

Specify `evaluation_config` in schema fields to control which metric is used:

```python
schema = {
    "type": "object",
    "properties": {
        "price": {
            "type": "number",
            "evaluation_config": {
                "metrics": [{"metric_id": "number_tolerance", "params": {"tolerance": 0.01}}]
            }
        },
        "description": {
            "type": "string",
            "evaluation_config": "string_fuzzy"  # Use preset shorthand
        }
    }
}
```

| Preset                    | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `string_exact`            | Case-sensitive exact match                           |
| `string_fuzzy`            | Levenshtein similarity (case-insensitive by default) |
| `string_case_insensitive` | Case-insensitive match                               |
| `string_semantic`         | LLM-based semantic similarity (default for strings)  |
| `string_url`              | URL-normalized match (auto-applied for `format: "uri"` fields) |
| `number_exact`            | Exact numeric equality                               |
| `number_tolerance`        | Match within tolerance (default for numbers)         |
| `integer_exact`           | Exact integer equality (default for integers)        |
| `boolean_exact`           | Exact boolean equality (default for booleans)        |
| `array_llm`               | LLM evaluation of entire array (default for arrays)  |
| `skip`                    | Skip evaluation for this node                        |

### Custom Metrics

```python
from extract_bench import global_metric_registry
from extract_bench.evaluation.metrics import BaseMetric, MetricResult

class MyCustomMetric(BaseMetric):
    metric_id = "my_custom"

    async def evaluate(self, node, config=None):
        gold = node.get_gold_value()
        extracted = node.get_extracted_value()
        return MetricResult(
            metric_id=self.metric_id,
            score=1.0,
            passed=True,
            details={"custom": "data"}
        )

global_metric_registry.register_metric(MyCustomMetric)
```

## Configuration

### Environment Setup

LLM-based metrics use LiteLLM. Configure your provider:

```bash
# Vertex AI (Google Cloud)
gcloud auth application-default login

# OpenAI
export OPENAI_API_KEY=sk-...

# Or copy .env.example to .env
```

### LLM Model Configuration

Default model: `vertex_ai/gemini-2.5-flash` (or set `DEFAULT_LLM_MODEL` in `.env`).

Override per-field in schema:

```python
schema = {
    "type": "object",
    "properties": {
        "company": {
            "type": "string",
            "evaluation_config": {
                "metrics": [{"metric_id": "string_semantic", "params": {"model": "openai/gpt-4o-mini"}}]
            },
        }
    },
}
```

## Architecture

```
extract-bench/
├── dataset/                   # Benchmark dataset (see dataset/README.md)
│   ├── {domain}/{schema}/     #   e.g. finance/10k/, academic/research/
│   │   ├── *-schema.json      #   JSON Schema with evaluation_config per field
│   │   └── pdf+gold/          #   Source PDFs + human-validated gold JSONs
├── src/extract_bench/         # Evaluation suite
│   ├── infra/                 #   Schema AST (nodes, visitors)
│   └── evaluation/
│       ├── metrics/           #   Metric implementations
│       └── reporting/         #   Report generation (see reporting/README.md)
```

Schema → AST → Values instantiated → Metrics evaluated async in parallel → Report generated.

## Citation

If you use ExtractBench in your research, please cite:

```bibtex
@article{ferguson2026extractbench,
  title={ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction},
  author={Ferguson, Nick and Pennington, Josh and Beghian, Narek and Mohan, Aravind and Kiela, Douwe and Agrawal, Sheshansh and Nguyen, Thien Hang},
  journal={arXiv preprint arXiv:2602.12247},
  year={2026}
}
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
