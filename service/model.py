from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from training.bm25 import BM25Artifact, BM25Scorer


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")


@dataclass(frozen=True)
class CandidateDocument:
    id: str
    text: str


@dataclass(frozen=True)
class RankedDocument:
    id: str
    text: str
    score: float


class ModelLoadError(Exception):
    """Raised when a reranker artifact cannot be loaded."""


class RerankerModel(ABC):
    """Interface for loading an artifact-backed reranker and scoring candidates."""

    @abstractmethod
    def score(self, query: str, document: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[CandidateDocument]) -> list[RankedDocument]:
        raise NotImplementedError


class BM25Reranker(RerankerModel):
    """Service-side adapter around the BM25 scorer and serialized artifact."""

    def __init__(self, artifact: BM25Artifact) -> None:
        self.artifact = artifact
        self.scorer = BM25Scorer(artifact)

    @classmethod
    def load(cls, artifact_path: Path) -> BM25Reranker:
        try:
            artifact = BM25Artifact.load(artifact_path.resolve())
        except FileNotFoundError as exc:
            raise ModelLoadError(f"Artifact file not found: {artifact_path}") from exc
        except KeyError as exc:
            raise ModelLoadError(f"Artifact file is missing required field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(f"Artifact file is invalid: {artifact_path}") from exc
        return cls(artifact)

    def score(self, query: str, document: str) -> float:
        return self.scorer.score(query, document)

    def rerank(self, query: str, candidates: Sequence[CandidateDocument]) -> list[RankedDocument]:
        ranked_candidates = [
            RankedDocument(
                id=candidate.id,
                text=candidate.text,
                score=self.score(query, candidate.text),
            )
            for candidate in candidates
        ]
        return sorted(ranked_candidates, key=lambda candidate: candidate.score, reverse=True)


def load_model(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> RerankerModel:
    """Load the default service reranker implementation from an artifact path."""

    return BM25Reranker.load(artifact_path)
