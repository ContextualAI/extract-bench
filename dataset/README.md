# Extract Bench Dataset

A multi-domain benchmark dataset for evaluating structured extraction from PDF documents. Each task consists of a source PDF, a target JSON Schema defining what to extract, and a human-validated gold JSON with the expected output.

## Dataset Organization

```
dataset/
├── {domain}/
│   └── {schema}/
│       ├── {schema}-schema.json          # JSON Schema defining extraction target
│       └── pdf+gold/
│           ├── {document}.pdf            # Source PDF
│           └── {document}.gold.json      # Human-validated extraction output
```

Each **domain** (e.g., `finance`, `academic`) contains one or more **schemas** (e.g., `10k`, `credit_agreement`). Each schema defines a structured extraction task. Under `pdf+gold/`, source PDFs are paired with their corresponding `.gold.json` files.

## Domains and Schemas

| Domain     | Schema             | Documents        | Description                                                                           |
| ---------- | ------------------ | ---------------- | ------------------------------------------------------------------------------------- |
| `finance`  | `10k`              | 7                | Financial metrics from SEC 10-K/10-Q filings (EPS, revenue, net income, segment data) |
| `finance`  | `credit_agreement` | 10               | Parties and key terms from corporate credit agreements                                |
| `academic` | `research`         | 6                | Metadata from research papers (authors, affiliations, abstract, citations)            |
| `hiring`   | `resume`           | 7                | Structured fields from resumes (experience, education, skills, certifications)        |
| `sport`    | `swimming`         | 5                | Championship results tables (rankings, times, records, athlete details)               |
| **Total**  | **5 schemas**      | **35 documents** |                                                                                       |

## Schema Format

Schemas are [JSON Schema](https://json-schema.org/) documents extended with an `evaluation_config` field that specifies how each leaf node should be scored during evaluation. For example:

```json
{
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
      "description": "The full title of the research paper.",
      "evaluation_config": "string_semantic"
    },
    "ids": {
      "type": "string",
      "description": "Unique identifier (DOI, arXiv ID, etc.).",
      "evaluation_config": "string_exact"
    },
    "page_count": {
      "type": "integer",
      "evaluation_config": "integer_exact"
    }
  }
}
```

Available evaluation metrics include `string_exact`, `string_fuzzy`, `string_semantic` (LLM-based), `number_tolerance`, `integer_exact`, `boolean_exact`, `array_llm`, and others. See the [evaluation suite README](../README.md) for the full list.

## Gold JSON Format

Each `.gold.json` file contains a JSON object conforming to its corresponding schema. Values are human-validated extractions from the paired PDF. Example (credit agreement):

```json
{
  "parties": {
    "borrower": "Amazon.com, Inc.",
    "administrative_agent": "JPMorgan Chase Bank, N.A.",
    "lenders": ["JPMorgan Chase Bank, N.A.", "Bank of America, N.A.", "..."]
  },
  "key_terms": {
    "agreement_date": "September 5, 2014",
    "governing_law": "State of New York",
    "total_loan_commitment": "$2,000,000,000"
  }
}
```

## Design Principles

- **Domain diversity**: Finance, academia, hiring, sports -- covering prose, tables, and mixed layouts.
- **Schema complexity variety**: From flat key-value extraction (swimming results) to deeply nested structures with arrays of objects (credit agreements, resumes).
- **Per-field evaluation control**: Each schema field declares its own evaluation metric, enabling precise scoring (e.g., exact match for IDs, semantic similarity for descriptions, tolerance for financial figures).

## Usage with Extract Bench

This dataset is designed to be used with the [Extract Bench evaluation suite](../README.md):

```python
import json
from pathlib import Path
from extract_bench import ReportBuilder, ReportConfig

schema = json.load(open("dataset/finance/10k/10k-schema.json"))
gold = json.load(open("dataset/finance/10k/pdf+gold/dell_10q_fy2025q2.gold.json"))
predicted = your_extraction_model(pdf_path, schema)

config = ReportConfig(output_dir=Path("./results"), output_name="dell-10q-gpt4o")
report = ReportBuilder(config).build(schema, gold, predicted)
```

## License

See the repository [LICENSE](../LICENSE) for terms.
