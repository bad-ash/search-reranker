# Failure Analysis

This directory contains exploratory analysis utilities for understanding per-query reranking failures from `training.evaluate`.

The main script is:

- `analysis/lexical_failure_analysis.py`

Supporting utility:

- `analysis/idf_lookup.py`

## Current Scope

`lexical_failure_analysis.py` currently:

- loads a diagnostics report produced by `training.evaluate`
- filters to failure cases where the first positive is not ranked first
- extracts the top-ranked negative and first positive passage
- computes simple overlap and score-difference features
- optionally enriches matched query terms with BM25 IDF values
- writes:
  - a summary JSON
  - a detailed JSON for failed queries

The script can be used with both BM25 and SBERT diagnostics reports.

BM25-specific enrichment is optional and only happens when `--artifact-path` is provided.

## Inputs

Required input:

- a diagnostics report produced by `training.evaluate`, for example:
  - `artifacts/eval/bm25_dev_query_diagnostics.json`
  - `artifacts/eval/sbert_dev_query_diagnostics.json`

Optional input:

- a BM25 artifact for IDF enrichment:
  - `artifacts/bm25_artifact.json`

## Output Shape

The summary output currently includes:

- `total_queries`
- `failed_queries`
- `source_model_type`
- `source_model_version`

The detailed output currently includes:

- `source_diagnostics_path`
- `source_model_type`
- `source_model_version`
- `failure_records`

Each failure record currently includes:

- `query`
- `top_negative`
- `first_positive`
- `score_differential`
- `pos_doc_ratio`
- `neg_doc_ratio`

For BM25-enriched runs, `matched_terms` entries are tuples of:

- query term
- IDF score
- occurrence count in the candidate text

## Run

BM25 diagnostics with IDF enrichment:

```bash
python analysis/lexical_failure_analysis.py \
  --artifact-path artifacts/bm25_artifact.json \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --summary-output artifacts/eval/bm25_failure_summary.json \
  --details-output artifacts/eval/bm25_failure_details.json
```

SBERT diagnostics without BM25 artifact enrichment:

```bash
python analysis/lexical_failure_analysis.py \
  --diagnostics-path artifacts/eval/sbert_dev_query_diagnostics.json \
  --summary-output artifacts/eval/sbert_failure_summary.json \
  --details-output artifacts/eval/sbert_failure_details.json
```

Limit the run to the first `N` failures while iterating:

```bash
python analysis/lexical_failure_analysis.py \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --summary-output artifacts/eval/tmp_summary.json \
  --details-output artifacts/eval/tmp_details.json \
  --max-failures 100
```

## Notes

- This is exploratory analysis code, not serving-path code.
- `assign_failure_bucket()` is still a placeholder and currently returns `"other"` for every record.
- The current script is best used to generate structured failure details for notebook analysis, not final categorical reporting.
