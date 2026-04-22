#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <tag>" >&2
  echo "Required env vars: ACR_NAME, APP_NAME, RESOURCE_GROUP" >&2
  exit 1
fi

TAG="$1"
LOCAL_IMAGE="search-reranker:test"
LOCAL_CONTAINER="search-reranker-local-smoke"
ACR_NAME="${ACR_NAME:-}"
APP_NAME="${APP_NAME:-}"
RESOURCE_GROUP="${RESOURCE_GROUP:-}"

if [[ -z "${ACR_NAME}" || -z "${APP_NAME}" || -z "${RESOURCE_GROUP}" ]]; then
  echo "Missing required env vars." >&2
  echo "Set ACR_NAME, APP_NAME, and RESOURCE_GROUP before running this script." >&2
  exit 1
fi

REMOTE_IMAGE="${ACR_NAME}.azurecr.io/search-reranker:${TAG}"

cleanup() {
  docker rm -f "${LOCAL_CONTAINER}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Building local smoke-test image..."
docker build -t "${LOCAL_IMAGE}" .

echo "Starting local smoke-test container..."
docker rm -f "${LOCAL_CONTAINER}" >/dev/null 2>&1 || true
docker run -d --rm --name "${LOCAL_CONTAINER}" -p 8000:8000 "${LOCAL_IMAGE}" >/dev/null

echo "Waiting for local health endpoint..."
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8000/healthz >/dev/null; then
    break
  fi
  sleep 1
done

echo "Checking local readiness..."
curl -sf http://127.0.0.1:8000/readyz >/dev/null

echo "Building and pushing image to ACR..."
az acr build --registry "${ACR_NAME}" --image "search-reranker:${TAG}" .

echo "Updating Azure Container App..."
az containerapp update \
  --name "${APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --image "${REMOTE_IMAGE}"

echo "Deployment submitted for image ${REMOTE_IMAGE}"
