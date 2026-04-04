import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.api import (
    RerankExecutionError,
    RerankTimeoutError,
    _execute_rerank,
    build_app,
)
from service.model import CandidateDocument, RerankerModel
from training.bm25 import BM25Artifact


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    BM25Artifact.from_corpus(["python list comprehension tutorial"]).save(artifact_path)

    with TestClient(build_app(artifact_path=artifact_path)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Process-Time-Ms" in response.headers
    assert "X-Request-ID" in response.headers


def test_readiness_endpoint_returns_503_when_model_fails_to_load(tmp_path: Path) -> None:
    missing_artifact = tmp_path / "artifacts" / "missing.json"

    with TestClient(build_app(artifact_path=missing_artifact)) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert "Artifact file not found" in response.json()["detail"]


def test_readiness_endpoint_returns_ready_when_model_loads(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    BM25Artifact.from_corpus(["python list comprehension tutorial"]).save(artifact_path)

    with TestClient(build_app(artifact_path=artifact_path)) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True}


def test_rerank_endpoint_returns_scored_results_in_rank_order(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial",
            "weather forecast for tomorrow",
            "capital city of france paris",
        ]
    ).save(artifact_path)

    with TestClient(build_app(artifact_path=artifact_path)) as client:
        response = client.post(
            "/rerank",
            headers={"X-Request-ID": "req-123"},
            json={
                "query": "python list comprehension",
                "candidates": [
                    {"id": "p2", "text": "weather forecast for tomorrow"},
                    {"id": "p1", "text": "python list comprehension tutorial"},
                    {"id": "p3", "text": "capital city of france paris"},
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert [result["id"] for result in payload["results"]] == ["p1", "p2", "p3"]
    assert payload["results"][0]["score"] >= payload["results"][1]["score"]
    assert float(response.headers["X-Process-Time-Ms"]) >= 0.0
    assert response.headers["X-Request-ID"] == "req-123"


def test_rerank_endpoint_returns_503_when_model_not_loaded(tmp_path: Path) -> None:
    missing_artifact = tmp_path / "artifacts" / "missing.json"

    with TestClient(build_app(artifact_path=missing_artifact)) as client:
        response = client.post(
            "/rerank",
            json={
                "query": "python list comprehension",
                "candidates": [{"id": "p1", "text": "python list comprehension tutorial"}],
            },
        )

    assert response.status_code == 503
    assert "Artifact file not found" in response.json()["detail"]


def test_end_to_end_app_serves_readyz_and_rerank_from_real_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial and examples",
            "generator expressions are similar to list comprehensions",
            "weather forecast for tomorrow in chicago",
        ]
    ).save(artifact_path)

    with TestClient(build_app(artifact_path=artifact_path)) as client:
        ready_response = client.get("/readyz")
        rerank_response = client.post(
            "/rerank",
            json={
                "query": "python list comprehension",
                "candidates": [
                    {"id": "c1", "text": "weather forecast for tomorrow in chicago"},
                    {"id": "c2", "text": "python list comprehension tutorial and examples"},
                    {"id": "c3", "text": "generator expressions are similar to list comprehensions"},
                ],
            },
        )

    assert ready_response.status_code == 200
    assert ready_response.json() == {"status": "ready", "model_loaded": True}

    assert rerank_response.status_code == 200
    payload = rerank_response.json()
    assert [result["id"] for result in payload["results"]] == ["c2", "c3", "c1"]


class _SlowReranker(RerankerModel):
    @property
    def model_version(self) -> str:
        return "test:slow"

    def score(self, query: str, document: str) -> float:
        return 0.0

    def rerank(self, query: str, candidates: list[CandidateDocument]):
        time.sleep(0.05)
        return []


class _FailingReranker(RerankerModel):
    @property
    def model_version(self) -> str:
        return "test:failing"

    def score(self, query: str, document: str) -> float:
        return 0.0

    def rerank(self, query: str, candidates: list[CandidateDocument]):
        raise RuntimeError("boom")


def test_execute_rerank_times_out() -> None:
    with pytest.raises(RerankTimeoutError, match="exceeded timeout"):
        asyncio.run(
            _execute_rerank(
                model=_SlowReranker(),
                query="python list comprehension",
                candidates=[CandidateDocument(id="p1", text="python list comprehension tutorial")],
                timeout_seconds=0.01,
            )
        )


def test_execute_rerank_wraps_internal_failure() -> None:
    with pytest.raises(RerankExecutionError, match="Rerank execution failed"):
        asyncio.run(
            _execute_rerank(
                model=_FailingReranker(),
                query="python list comprehension",
                candidates=[CandidateDocument(id="p1", text="python list comprehension tutorial")],
                timeout_seconds=1.0,
            )
        )
