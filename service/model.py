from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from training.bm25 import BM25Artifact, BM25Scorer


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")
DEFAULT_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SBERT_CACHE_FOLDER = "/app/.hf/sentence_transformers"


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

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: str, document: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[CandidateDocument]) -> list[RankedDocument]:
        raise NotImplementedError


class BM25Reranker(RerankerModel):
    """Service-side adapter around the BM25 scorer and serialized artifact."""

    def __init__(self, artifact: BM25Artifact, artifact_path: Path) -> None:
        self.artifact = artifact
        self.artifact_path = artifact_path
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
        return cls(artifact, artifact_path.resolve())

    @property
    def model_version(self) -> str:
        return f"bm25:{self.artifact_path.name}"

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


class SBERTReranker(RerankerModel):
    """Service-side adapter around a sentence-transformers bi-encoder."""

    def __init__(self, embedder: SentenceTransformer, model_name: str) -> None:
        self.embedder = embedder
        self.model_name = model_name

    @classmethod
    def load(cls, model_name: str = DEFAULT_SBERT_MODEL_NAME) -> SBERTReranker:
        try:
            embedder = SentenceTransformer(
                model_name,
                cache_folder=DEFAULT_SBERT_CACHE_FOLDER,
                local_files_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(f"SBERT model failed to load: {model_name}") from exc
        return cls(embedder, model_name)

    @property
    def model_version(self) -> str:
        return self.model_name

    def score(self, query: str, document: str) -> float:
        document_embeddings = self.embedder.encode_document([document], convert_to_tensor=True)
        query_embedding = self.embedder.encode_query(query, convert_to_tensor=True)
        if not isinstance(document_embeddings, torch.Tensor) or not isinstance(query_embedding, torch.Tensor):
            raise TypeError("Expected sentence-transformers encoder to return torch.Tensor embeddings.")
        similarity = self.embedder.similarity(query_embedding, document_embeddings)
        return float(similarity[0][0])

    def rerank(self, query: str, candidates: Sequence[CandidateDocument]) -> list[RankedDocument]:
        candidate_texts = [candidate.text for candidate in candidates]
        candidate_embeddings = self.embedder.encode_document(candidate_texts, convert_to_tensor=True)
        query_embedding = self.embedder.encode_query(query, convert_to_tensor=True)
        if not isinstance(candidate_embeddings, torch.Tensor) or not isinstance(query_embedding, torch.Tensor):
            raise TypeError("Expected sentence-transformers encoder to return torch.Tensor embeddings.")
        similarity_scores = self.embedder.similarity(query_embedding, candidate_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, len(candidates))

        ranked_candidates: list[RankedDocument] = []
        for score, idx in zip(scores, indices):
            candidate = candidates[int(idx)]
            ranked_candidates.append(
                RankedDocument(
                    id=candidate.id,
                    text=candidate.text,
                    score=float(score),
                )
            )
        return ranked_candidates


def load_bm25_model(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> RerankerModel:
    return BM25Reranker.load(artifact_path)


def load_sbert_model(model_name: str = DEFAULT_SBERT_MODEL_NAME) -> RerankerModel:
    return SBERTReranker.load(model_name)


def load_model(artifact_path: Path = DEFAULT_ARTIFACT_PATH) -> RerankerModel:
    """Backward-compatible default loader for the BM25 service model."""

    return load_bm25_model(artifact_path)
