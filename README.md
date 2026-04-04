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

## Deployment and Operational Checks

The current deployment target is Azure Container Apps backed by Azure Container Registry.

Build and push an updated image:

```bash
az acr build \
  --registry <acr-name> \
  --image search-reranker:<tag> \
  .
```

Update the Container App to the new image:

```bash
az containerapp update \
  --name <app-name> \
  --resource-group <resource-group> \
  --image <acr-name>.azurecr.io/search-reranker:<tag>
```

For steady-state testing, keep at least one warm replica:

```bash
az containerapp update \
  --name <app-name> \
  --resource-group <resource-group> \
  --min-replicas 1
```

Get the deployed app hostname:

```bash
az containerapp show \
  --name <app-name> \
  --resource-group <resource-group> \
  --query properties.configuration.ingress.fqdn \
  -o tsv
```

Operational checks after each deployment:

1. Verify liveness:

```bash
curl https://<fqdn>/healthz
```

2. Verify readiness:

```bash
curl https://<fqdn>/readyz
```

3. Verify reranking behavior:

```bash
curl -X POST https://<fqdn>/rerank \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: manual-check-1" \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "c1", "text": "weather forecast for tomorrow in chicago"},
      {"id": "c2", "text": "python list comprehension tutorial and examples"},
      {"id": "c3", "text": "paris is the capital city of france"}
    ]
  }'
```

4. Inspect service logs:

```bash
az containerapp logs show \
  --name <app-name> \
  --resource-group <resource-group> \
  --follow
```

Useful things to verify in logs and responses:

- `/readyz` returns `200` and `model_loaded=true`
- `/rerank` returns ranked results with `200`
- response headers include `X-Request-ID` and `X-Process-Time-Ms`
- logs include `request_id`, `num_candidates`, `model_version`, `status_code`, and `duration_ms`

If latency looks unexpectedly high, distinguish between:

- cold-start or scale-up latency
- network/platform latency
- application processing time

The `X-Process-Time-Ms` header is useful for separating application time from total end-to-end request duration.

## Baseline Performance

Warm steady-state load tests against the deployed BM25 service on Azure Container Apps produced the following baseline results for the current small rerank payload:

- `10 VUs / 2 min`: `p50=31.11ms`, `p95=38.14ms`, `0.00%` failed requests
- `25 VUs / 2 min`: `p50=30.73ms`, `p95=37.27ms`, `0.00%` failed requests
- `25 VUs / 5 min`: `p50=30.73ms`, `p95=37.62ms`, `0.00%` failed requests

These numbers should be treated as the current BM25 reference point for warm service behavior. They do not represent cold-start latency, and they will vary with payload shape and candidate count.
