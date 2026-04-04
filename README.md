# Real-time Document Reranking API

Production-shaped ML engineering project focused on online reranking service design

## Project Focus

This project is intentionally scoped around:

- online inference
- clean service boundaries
- artifact loading at startup
- latency measurement
- structured logging
- containerization
- testing
- deployment readiness

## Status

Current implementation includes:

- data preparation for a small MS MARCO passage-ranking subset
- a BM25 baseline scorer and artifact builder
- a service-side model interface
- FastAPI health, readiness, and rerank endpoints
- request timing and JSON-formatted service logs

## Local Setup

Create and activate a virtual environment, then install the project with development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Data Preparation

Place the raw MS MARCO files under `data/raw/` with these names:

- `collection.tsv`
- `queries.train.tsv`
- `qrels.train.tsv`
- optional: `top1000.train.tsv`, `top1000.tsv`, or `candidates.tsv`

Prepare a grouped reranking dataset:

```bash
python -m training.prepare_data \
  --raw-dir data/raw \
  --output-dir data/processed \
  --max-queries 500 \
  --negatives-per-query 10 \
  --seed 42 \
  --max-positives-per-query 3
```

This writes:

- `data/processed/msmarco_rerank_subset.jsonl`
- `data/processed/msmarco_rerank_subset_metadata.json`

## Build BM25 Artifact

Build the BM25 artifact from the raw passage collection:

```bash
python -m training.train \
  --raw-dir data/raw \
  --output-artifact artifacts/bm25_artifact.json
```

Evaluate the artifact against the processed dataset:

```bash
python -m training.evaluate \
  --artifact-path artifacts/bm25_artifact.json \
  --dataset-path data/processed/msmarco_rerank_subset.jsonl
```

## Run the API

Start the FastAPI service locally:

```bash
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

Example requests:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
curl -X POST http://127.0.0.1:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "p1", "text": "python list comprehension tutorial"},
      {"id": "p2", "text": "weather forecast for tomorrow"}
    ]
  }'
```

The service expects `artifacts/bm25_artifact.json` to exist at startup.

## Run Tests

Run the full test suite:

```bash
pytest -q
```

## Docker

Build the image:

```bash
docker build -t search-reranker .
```

Run the container:

```bash
docker run --rm -p 8000:8000 search-reranker
```

The container is inference-only and expects a valid `artifacts/bm25_artifact.json` to be present in the build context.

## Baseline Performance

Warm steady-state load tests against the deployed BM25 service on Azure Container Apps produced the following baseline results for the current small rerank payload:

- `10 VUs / 2 min`: `p50=31.11ms`, `p95=38.14ms`, `0.00%` failed requests
- `25 VUs / 2 min`: `p50=30.73ms`, `p95=37.27ms`, `0.00%` failed requests
- `25 VUs / 5 min`: `p50=30.73ms`, `p95=37.62ms`, `0.00%` failed requests

These numbers should be treated as the current BM25 reference point for warm service behavior. They do not represent cold-start latency, and they will vary with payload shape and candidate count.
