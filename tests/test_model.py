import json
from pathlib import Path

import pytest

from service.model import BM25Reranker, CandidateDocument, ModelLoadError, RankedDocument, load_model
from training.bm25 import BM25Artifact


def test_load_model_returns_bm25_reranker(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    artifact = BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial",
            "weather forecast for tomorrow",
        ]
    )
    artifact.save(artifact_path)

    model = load_model(artifact_path)

    assert isinstance(model, BM25Reranker)


def test_rerank_orders_candidates_by_descending_score(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    artifact = BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial",
            "weather forecast for tomorrow",
            "capital city of france paris",
        ]
    )
    artifact.save(artifact_path)
    model = load_model(artifact_path)

    ranked = model.rerank(
        "python list comprehension",
        [
            CandidateDocument(id="p2", text="weather forecast for tomorrow"),
            CandidateDocument(id="p1", text="python list comprehension tutorial"),
            CandidateDocument(id="p3", text="capital city of france paris"),
        ],
    )

    assert ranked == sorted(ranked, key=lambda candidate: candidate.score, reverse=True)
    assert ranked[0].id == "p1"
    assert all(isinstance(candidate, RankedDocument) for candidate in ranked)


def test_model_load_raises_clear_error_for_missing_artifact(tmp_path: Path) -> None:
    missing_artifact = tmp_path / "artifacts" / "missing.json"

    with pytest.raises(ModelLoadError, match="Artifact file not found"):
        load_model(missing_artifact)


def test_model_load_raises_clear_error_for_invalid_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps({"document_count": 10}) + "\n", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="missing required field"):
        load_model(artifact_path)
