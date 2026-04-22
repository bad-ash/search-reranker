# Real-time Document Reranking API

Production-shaped ML engineering project focused on:

- offline evaluation on MS MARCO-style reranking data
- online serving behind FastAPI
- model/version-aware deployment
- logging, timing, and load testing
- containerization and Azure Container Apps deployment

## Current Scope

The repository currently includes:

- MS MARCO subset preparation for `train` and `dev`
- BM25 artifact building and evaluation
- SBERT evaluation using `sentence-transformers/all-MiniLM-L6-v2`
- failure-analysis tooling and notebooks
- FastAPI endpoints for:
  - `POST /rerank_bm25`
  - `POST /rerank_sbert`
- JSON request logging and request timing
- Docker packaging for deployment to ACA

## Local Setup

Create and activate a virtual environment, then install the project with development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the test suite:

```bash
pytest -q
```

## Data Preparation

Place the raw MS MARCO files under `data/raw/`.

Required files:

- `collection.tsv`
- `queries.train.tsv`
- `qrels.train.tsv`

Optional dev/evaluation files:

- `queries.dev.tsv`
- `qrels.dev.tsv`
- candidate files matching the selected split, for example:
  - `top1000.train.tsv`
  - `top1000.dev.tsv`
  - `top1000.tsv`
  - `candidates.train.tsv`
  - `candidates.dev.tsv`
  - `candidates.tsv`

Prepare a grouped training subset from raw train data:

```bash
python -m training.prepare_data \
  --raw-dir data/raw \
  --output-dir data/processed \
  --raw-split train \
  --max-queries 500 \
  --negatives-per-query 10 \
  --seed 42 \
  --max-positives-per-query 3
```

Prepare a dev reranking dataset from the official dev split and candidate file:

```bash
python -m training.prepare_data \
  --raw-dir data/raw \
  --output-dir data/processed_dev_top1000 \
  --raw-split dev \
  --candidate-file top1000.dev \
  --max-queries 5000 \
  --negatives-per-query 100 \
  --seed 42 \
  --max-positives-per-query 3
```

Behavior by raw split:

- `--raw-split train`: retained queries are reshuffled into local `train` / `val` / `test`
- `--raw-split dev`: retained queries keep `split="dev"` so the original evaluation semantics are preserved

## Build the BM25 Artifact

`artifacts/` is generated local workspace output and is not committed to Git.

Build the BM25 artifact from the raw passage collection:

```bash
python -m training.train \
  --raw-dir data/raw \
  --output-artifact artifacts/bm25_artifact.json
```

## Evaluate Models

### BM25

Evaluate BM25 on the prepared dataset:

```bash
python -m training.evaluate \
  --artifact-path artifacts/bm25_artifact.json \
  --dataset-path data/processed_dev_top1000/msmarco_rerank_subset.jsonl \
  --split dev \
  --output-path artifacts/eval/bm25_dev_report.json \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --model-type bm25
```

### SBERT

Evaluate SBERT on the same dataset:

```bash
python -m training.evaluate \
  --dataset-path data/processed_dev_top1000/msmarco_rerank_subset.jsonl \
  --split dev \
  --output-path artifacts/eval/sbert_dev_report.json \
  --diagnostics-path artifacts/eval/sbert_dev_query_diagnostics.json \
  --model-type sbert
```

These commands write:

- an aggregate JSON report with:
  - model metadata
  - dataset metadata
  - summary counts
  - `MRR`
  - `Recall@1`
  - `Recall@3`
  - `Recall@10`
- a per-query diagnostics report suitable for failure analysis

## Failure Analysis

Generate failure summaries/details from a diagnostics report:

```bash
python analysis/lexical_failure_analysis.py \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --summary-output artifacts/eval/bm25_failure_summary.json \
  --details-output artifacts/eval/bm25_failure_details.json
```

For BM25-specific IDF enrichment, include the artifact:

```bash
python analysis/lexical_failure_analysis.py \
  --artifact-path artifacts/bm25_artifact.json \
  --diagnostics-path artifacts/eval/bm25_dev_query_diagnostics.json \
  --summary-output artifacts/eval/bm25_failure_summary.json \
  --details-output artifacts/eval/bm25_failure_details.json
```

For interactive comparison of BM25 and SBERT failures:

```bash
jupyter lab
```

Starter notebook:

- `analysis/notebooks/failure_analysis.ipynb`

## Run the API Locally

Start the FastAPI service:

```bash
uvicorn service.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /healthz`
- `GET /readyz`
- `POST /rerank_bm25`
- `POST /rerank_sbert`

Example requests:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz

curl -X POST http://127.0.0.1:8000/rerank_bm25 \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "p1", "text": "python list comprehension tutorial"},
      {"id": "p2", "text": "weather forecast for tomorrow"}
    ]
  }'

curl -X POST http://127.0.0.1:8000/rerank_sbert \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "p1", "text": "python list comprehension tutorial"},
      {"id": "p2", "text": "weather forecast for tomorrow"}
    ]
  }'
```

Important note:

- BM25 requires `artifacts/bm25_artifact.json`
- SBERT service startup is currently designed around a baked local cache in `/app/.hf/...`
- because of that, the most reliable way to exercise the dual-model service end to end is the Docker image path below

## Docker

The Docker image is the intended runtime path for the dual-model service.

The image:

- installs the app
- copies the BM25 artifact
- preloads `sentence-transformers/all-MiniLM-L6-v2` into the image cache
- runs SBERT with `local_files_only=True` at runtime

Build the image:

```bash
python -m training.train \
  --raw-dir data/raw \
  --output-artifact artifacts/bm25_artifact.json

docker build -t search-reranker .
```

Run the container:

```bash
docker run --rm -p 8000:8000 search-reranker
```

The build-time preload step is implemented in:

- `scripts/preload_models.py`

## Azure Deployment

The current deployment target is Azure Container Apps backed by Azure Container Registry.

Reusable deployment script:

```bash
export ACR_NAME=<acr-name>
export APP_NAME=<app-name>
export RESOURCE_GROUP=<resource-group>

./scripts/deploy_aca.sh <tag>
```

The script performs:

- a local Docker build
- a local `/healthz` and `/readyz` smoke test
- `az acr build`
- `az containerapp update`

Build and push a new image:

```bash
python -m training.train \
  --raw-dir data/raw \
  --output-artifact artifacts/bm25_artifact.json

az acr build \
  --registry <acr-name> \
  --image search-reranker:<tag> \
  .
```

Update the Container App:

```bash
az containerapp update \
  --name <app-name> \
  --resource-group <resource-group> \
  --image <acr-name>.azurecr.io/search-reranker:<tag>
```

For warm steady-state testing, keep one replica warm:

```bash
az containerapp update \
  --name <app-name> \
  --resource-group <resource-group> \
  --min-replicas 1
```

Get the deployed hostname:

```bash
az containerapp show \
  --name <app-name> \
  --resource-group <resource-group> \
  --query properties.configuration.ingress.fqdn \
  -o tsv
```

## Operational Checks

After each deployment:

1. Verify liveness:

```bash
curl https://<fqdn>/healthz
```

2. Verify readiness:

```bash
curl https://<fqdn>/readyz
```

3. Verify both rerank endpoints:

```bash
curl -X POST https://<fqdn>/rerank_bm25 \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: bm25-manual-check" \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "c1", "text": "weather forecast for tomorrow in chicago"},
      {"id": "c2", "text": "python list comprehension tutorial and examples"},
      {"id": "c3", "text": "paris is the capital city of france"}
    ]
  }'

curl -X POST https://<fqdn>/rerank_sbert \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: sbert-manual-check" \
  -d '{
    "query": "python list comprehension",
    "candidates": [
      {"id": "c1", "text": "weather forecast for tomorrow in chicago"},
      {"id": "c2", "text": "python list comprehension tutorial and examples"},
      {"id": "c3", "text": "paris is the capital city of france"}
    ]
  }'
```

4. Inspect logs:

```bash
az containerapp logs show \
  --name <app-name> \
  --resource-group <resource-group> \
  --follow
```

Useful checks:

- `/readyz` returns `200` with `bm25_loaded=true` and `sbert_loaded=true`
- both rerank endpoints return `200`
- response headers include:
  - `X-Request-ID`
  - `X-Process-Time-Ms`
- logs include:
  - `request_id`
  - `num_candidates`
  - `model_version`
  - `status_code`
  - `duration_ms`

If latency looks high, separate:

- cold-start / scale-up time
- network or platform overhead
- application processing time

`X-Process-Time-Ms` is the key app-side timing signal.

## Baseline Performance

Warm steady-state load tests against the deployed BM25 service on ACA produced:

- `10 VUs / 2 min`: `p50=31.11ms`, `p95=38.14ms`, `0.00%` failed requests
- `25 VUs / 2 min`: `p50=30.73ms`, `p95=37.27ms`, `0.00%` failed requests
- `25 VUs / 5 min`: `p50=30.73ms`, `p95=37.62ms`, `0.00%` failed requests

These are warm-service BM25 reference numbers, not cold-start numbers. They depend on request shape and candidate count.

For current benchmark summaries and model tradeoff notes, see:

- `docs/performance.md`
- `reports/benchmarks/`
