# Lexical Failure Analysis

## Goal

Analyze the BM25 per-query diagnostics report and identify common failure patterns in the worst-ranked queries.

Use this to answer:

- Are BM25 failures mostly lexical or semantic?
- Are top-ranked negatives beating positives because of stronger token overlap?
- Are there many cases where the positive has little or no lexical overlap with the query?
- Are there likely annotation / ambiguity cases?

## Inputs

Expected input:

- a diagnostics report produced by `training.evaluate`, for example:
  - `artifacts/eval/bm25_dev_query_diagnostics.json`

## Deliverables

Implement the analysis so that it produces:

1. A summary report with:
   - number of total queries analyzed
   - number of failed queries analyzed
   - counts per failure bucket

2. A detailed output file for failed queries with:
   - query id
   - query text
   - first positive rank
   - first negative rank
   - top negative score
   - first positive score
   - lexical feature values
   - assigned failure bucket

3. A short written interpretation of the results:
   - what are the most common failure types?
   - do the failures suggest embeddings are likely to help?

## Suggested Buckets

These are suggested failure categories. You can change them if your analysis supports a better taxonomy.

- `semantic_miss`
- `lexical_distractor`
- `close_lexical_competition`
- `possible_label_or_ambiguity_issue`
- `other`

## Suggested Process

1. Load the diagnostics JSON.
2. Filter to failed queries.
   - Suggested rule: `first_positive_rank > 1`
3. Compute lexical features using the same tokenizer as BM25.
4. Assign a bucket to each failed query using simple heuristics.
5. Aggregate counts and inspect representative examples.
6. Write out summary and detailed reports.

## Suggested Features

- query token count
- positive token count
- top negative token count
- query-positive token overlap count
- query-top-negative token overlap count
- overlap ratio for positive
- overlap ratio for top negative
- score gap between top negative and first positive
- whether positive contains all query terms
- whether top negative contains all query terms

## Run

Example:

```bash
python analysis/lexical_failure_analysis.py \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --summary-output artifacts/eval/bm25_failure_summary.json \
  --details-output artifacts/eval/bm25_failure_details.json
```

## Notes

- This is exploratory analysis, not production code.
- Keep the heuristics simple and explicit.
- Prefer clarity over sophistication.
