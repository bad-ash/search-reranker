from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into word-like terms."""

    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Artifact:
    document_count: int
    average_document_length: float
    document_frequencies: dict[str, int]
    k1: float
    b: float
    @classmethod
    def from_corpus(
        cls,
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Artifact:
        """Build a BM25 artifact from a corpus of documents."""

        if not documents:
            raise ValueError("Cannot build a BM25 artifact from an empty corpus.")

        document_frequencies: Counter[str] = Counter()
        total_length = 0

        for document in documents:
            tokens = tokenize(document)
            total_length += len(tokens)
            document_frequencies.update(set(tokens))

        return cls(
            document_count=len(documents),
            average_document_length=total_length / len(documents),
            document_frequencies=dict(document_frequencies),
            k1=k1,
            b=b,
        )
    @classmethod
    def load(cls, path: Path) -> BM25Artifact:
        """Load a BM25 artifact from a JSON file."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            document_count=payload["document_count"],
            average_document_length=payload["average_document_length"],
            document_frequencies=payload["document_frequencies"],
            k1=payload["k1"],
            b=payload["b"],
        )
    def save(self, path: Path) -> None:
        """Serialize this BM25 artifact to JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

class BM25Scorer:
    """Score query-document pairs using a BM25 artifact."""

    def __init__(self, artifact: BM25Artifact) -> None:
        self.artifact = artifact

    def score(self, query: str, document: str) -> float:
        """Compute a BM25 relevance score for one query-document pair."""

        query_terms = tokenize(query)
        document_terms = tokenize(document)
        if not query_terms or not document_terms:
            return 0.0

        term_frequencies = Counter(document_terms)
        document_length = len(document_terms)
        score = 0.0

        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue

            document_frequency = self.artifact.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1.0
                + (
                    (self.artifact.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )
            denominator = frequency + self.artifact.k1 * (
                1.0
                - self.artifact.b
                + self.artifact.b * document_length / self.artifact.average_document_length
            )
            score += inverse_document_frequency * (
                frequency * (self.artifact.k1 + 1.0) / denominator
            )

        return score

    def score_documents(self, query: str, documents: list[str]) -> list[float]:
        """Score a batch of documents for the same query."""

        return [self.score(query, document) for document in documents]
