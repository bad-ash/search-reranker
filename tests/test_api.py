from pathlib import Path

from fastapi.testclient import TestClient

from service.api import build_app
from training.bm25 import BM25Artifact


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    BM25Artifact.from_corpus(["python list comprehension tutorial"]).save(artifact_path)

    with TestClient(build_app(artifact_path=artifact_path)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Process-Time-Ms" in response.headers


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
