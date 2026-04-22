# Failure Analysis Notebooks

Use this directory for exploratory notebook work over the generated failure-detail JSON files.

Recommended inputs:

- `artifacts/eval/bm25_failure_details.json`
- `artifacts/eval/sbert_failure_details.json`

Install notebook dependencies:

```bash
pip install -e '.[dev]'
```

Launch JupyterLab from the repo root:

```bash
jupyter lab
```

Suggested first analyses:

- summary statistics for `score_differential`
- BM25 vs SBERT failure-count comparison
- overlap of failed queries between models
- distributions of `pos_doc_ratio` and `neg_doc_ratio`
- matched-term counts and IDF comparisons
- joined analysis on shared failed queries across both models

The starter notebook is:

- `analysis/notebooks/failure_analysis.ipynb`

Notes:

- when merging BM25 and SBERT dataframes in pandas, duplicate column names will default to suffixes like `_x` and `_y` unless you set `suffixes=...`
- BM25 `matched_terms` may include IDF scores when the failure-analysis script was run with `--artifact-path`
- SBERT failure details will not contain BM25-specific IDF enrichment unless you add a separate enrichment step
