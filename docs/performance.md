# Performance Notes

This document records the current benchmark and evaluation baselines for the two deployed reranking paths:

- `POST /rerank_bm25`
- `POST /rerank_sbert`

The goal is not to claim a final production SLO. The goal is to make the quality/latency tradeoff explicit and reproducible.

## Benchmark Context

Environment:

- deployment target: Azure Container Apps
- container registry: Azure Container Registry
- image build includes a baked SBERT cache
- BM25 artifact built from `collection.tsv`
- SBERT model: `sentence-transformers/all-MiniLM-L6-v2`

Evaluation dataset:

- `data/processed_dev_top1000/msmarco_rerank_subset.jsonl`
- split: `dev`

Load test methodology:

- warm steady-state tests
- endpoint warmed and readiness checked before testing
- k6 used as the load generator
- current scripts:
  - `loadtest_bm25.js`
  - `loadtest_sbert.js`

## Offline Quality

BM25 dev evaluation:

- `MRR`: `0.8961`
- `Recall@1`: `0.8394`
- `Recall@3`: `0.9141`
- `Recall@10`: `0.9420`

SBERT dev evaluation:

- `MRR`: `0.9566`
- `Recall@1`: `0.9080`
- `Recall@3`: `0.9694`
- `Recall@10`: `0.9906`

Interpretation:

- SBERT is materially better on the shared dev reranking benchmark.
- The `Recall@1` gain is especially important, because it means SBERT places the relevant passage first more often.
- On 5000 dev queries, the `Recall@1` lift corresponds to hundreds of additional queries with the relevant passage at rank 1.

## Online Latency

### BM25

Warm ACA baseline:

- `25 VUs / 120s`
- `p50`: `37.74 ms`
- `p95`: `62.68 ms`
- failures: `0.00%`
- throughput: `611.81 req/s`

Interpretation:

- BM25 is operationally cheap.
- It supports high throughput with low latency on the current ACA deployment.

### SBERT

Stable ACA run after increasing the SBERT timeout to `3.0s`:

- `10 VUs / 120s`
- `p50`: `1.79 s`
- `p95`: `2.34 s`
- failures: `0.00%`
- throughput: `5.83 req/s`
- `minReplicas`: `1`

Interpretation:

- SBERT is viable on the current deployment, but it is much slower than BM25.
- Earlier SBERT tests with a shared `2.0s` timeout produced errors, which indicates the timeout budget was too tight for this model.
- After splitting timeouts by model and increasing SBERT to `3.0s`, the endpoint became stable at `10 VUs`.

## Current Conclusion

The two endpoints demonstrate a real production tradeoff:

- BM25:
  - lower quality
  - very fast
  - high throughput
  - operationally simple

- SBERT:
  - higher quality
  - much slower
  - far lower throughput
  - requires a more careful serving configuration

This is the intended result for the project: the system does not pretend that the better model is free.

## Reproduction

Rebuild and deploy:

```bash
python -m training.train \
  --raw-dir data/raw \
  --output-artifact artifacts/bm25_artifact.json

./scripts/deploy_aca.sh <tag>
```

Run BM25 load test:

```bash
k6 run -e BASE_URL=$BASE_URL loadtest_bm25.js
```

Run SBERT load test:

```bash
k6 run -e BASE_URL=$BASE_URL --vus 10 --duration 120s loadtest_sbert.js
```

Structured benchmark summaries are stored under:

- `reports/benchmarks/`
