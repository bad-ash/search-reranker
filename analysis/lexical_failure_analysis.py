from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.bm25 import BM25Artifact, BM25Scorer, tokenize

DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")
DEFAULT_DIAGNOSTICS_PATH = Path("artifacts/eval/query_diagnostics.json")
DEFAULT_SUMMARY_OUTPUT = Path("artifacts/eval/failure_summary.json")
DEFAULT_DETAILS_OUTPUT = Path("artifacts/eval/failure_details.json")


@dataclass(frozen=True)
class FailureFeatures:
    positive_avg_length_ratio: float # maps to |D| / avgdl
    negative_avg_length_ratio: float # maps to |D| / avgdl
    query_positive_overlap_count: int # maps to sum of f(q_i, D) over all query terms
    query_top_negative_overlap_count: int # maps to sum of f(q_i, D) over all query terms
    positive_contains_all_query_terms: bool
    top_negative_contains_all_query_terms: bool
    score_gap_top_negative_minus_positive: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--diagnostics-path", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS_OUTPUT)
    parser.add_argument("--max-failures", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def is_failed_query(query_diagnostic: dict[str, Any]) -> bool:
    """Return whether this query should be treated as a failure case."""
    first_positive_rank = query_diagnostic["metrics"]["first_positive_rank"]
    return first_positive_rank is None or first_positive_rank > 1


def get_first_positive(ranked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in ranked_candidates:
        if candidate["label"] == 1:
            return candidate
    raise ValueError("No positive candidate found in ranked_candidates.")


def get_top_negative(ranked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in ranked_candidates:
        if candidate["label"] == 0:
            return candidate
    raise ValueError("No negative candidate found in ranked_candidates.")


def overlap_count(a_tokens: set[str], b_tokens: set[str]) -> int:
    return len(a_tokens & b_tokens)


def term_idf(term: str, artifact: BM25Artifact) -> float:
    return BM25Scorer(artifact).IDF(term)


def matched_terms(
    query_tokens: set[str],
    candidate_tokens: list[str],
    artifact: BM25Artifact | None,
) -> list[tuple[str, float | None, int]]:
    candidate_term_frequencies = Counter(candidate_tokens)
    shared_terms = sorted(query_tokens & set(candidate_tokens))
    return [
        (
            term,
            term_idf(term, artifact) if artifact is not None else None,
            candidate_term_frequencies[term],
        )
        for term in shared_terms
    ]


def overlap_ratio(shared: int, total_query_tokens: int) -> float:
    if total_query_tokens == 0:
        return 0.0
    return shared / total_query_tokens


def compute_failure_features(query_diagnostic: dict[str, Any], avg_document_length: float) -> FailureFeatures:
    """
    TODO:
    Compute lexical features for one failed query.

    Suggested approach:
    - tokenize the query
    - find the first positive candidate
    - find the top-ranked negative candidate
    - compute overlap counts and ratios
    - compute the score gap
    """
    query = query_diagnostic["query"]
    ranked_candidates = query_diagnostic["ranked_candidates"]

    query_tokens = set(tokenize(query))
    positive = get_first_positive(ranked_candidates)
    top_negative = get_top_negative(ranked_candidates)

    positive_tokens = set(tokenize(positive["text"]))
    top_negative_tokens = set(tokenize(top_negative["text"]))

    query_positive_overlap = overlap_count(query_tokens, positive_tokens)
    query_top_negative_overlap = overlap_count(query_tokens, top_negative_tokens)

    return FailureFeatures(
        positive_avg_length_ratio=len(positive_tokens) / avg_document_length,
        negative_avg_length_ratio=len(top_negative_tokens) / avg_document_length,
        query_positive_overlap_count=query_positive_overlap,
        query_top_negative_overlap_count=query_top_negative_overlap,
        positive_contains_all_query_terms=query_tokens.issubset(positive_tokens),
        top_negative_contains_all_query_terms=query_tokens.issubset(top_negative_tokens),
        score_gap_top_negative_minus_positive=top_negative["score"] - positive["score"],
    )


def assign_failure_bucket(query_diagnostic: dict[str, Any], features: FailureFeatures) -> str:
    """
    TODO:
    Replace this placeholder with your own heuristic bucketing logic.

    Suggested buckets:
    - semantic_miss
    - lexical_distractor
    - close_lexical_competition
    - possible_label_or_ambiguity_issue
    - other
    """
    _ = query_diagnostic
    _ = features
    return "other"


def build_failure_record(query_diagnostic: dict[str, Any], artifact: BM25Artifact | None) -> dict[str, Any]:
    average_document_length = artifact.average_document_length if artifact is not None else 1.0
    features = compute_failure_features(query_diagnostic, average_document_length)
    bucket = assign_failure_bucket(query_diagnostic, features)
    query_tokens = set(tokenize(query_diagnostic["query"]))
    positive = get_first_positive(query_diagnostic["ranked_candidates"])
    top_negative = get_top_negative(query_diagnostic["ranked_candidates"])
    positive_matched_terms = matched_terms(query_tokens, tokenize(positive["text"]), artifact)
    top_negative_matched_terms = matched_terms(query_tokens, tokenize(top_negative["text"]), artifact)

    return {
        "query": query_diagnostic["query"],
        "top_negative": {
            "id": top_negative["id"],
            "score": top_negative["score"],
            "matched_terms": top_negative_matched_terms,
            "text": top_negative["text"],
        },
        "first_positive": {
            "id": positive["id"],
            "score": positive["score"],
            "matched_terms": positive_matched_terms,
            "text": positive["text"],
        },
        "score_differential": features.score_gap_top_negative_minus_positive,
        "pos_doc_ratio": features.positive_avg_length_ratio,
        "neg_doc_ratio": features.negative_avg_length_ratio,
    }


def build_summary(failure_records: list[dict[str, Any]], total_queries: int) -> dict[str, Any]:
    return {
        "total_queries": total_queries,
        "failed_queries": len(failure_records),
    }


def main() -> None:
    args = parse_args()
    diagnostics_report = load_json(args.diagnostics_path.resolve())
    query_diagnostics = diagnostics_report["query_diagnostics"]

    failed_queries = [query for query in query_diagnostics if is_failed_query(query)]
    if args.max_failures is not None:
        failed_queries = failed_queries[: args.max_failures]

    artifact = None
    if args.artifact_path is not None:
        artifact = BM25Artifact.load(args.artifact_path.resolve())
    failure_records = [build_failure_record(query, artifact) for query in failed_queries]
    failure_records.sort(
        key=lambda record: record["score_differential"],
        reverse=True,
    )
    summary = build_summary(failure_records, total_queries=len(query_diagnostics))
    summary["source_model_type"] = diagnostics_report.get("model_type")
    summary["source_model_version"] = diagnostics_report.get("model_version")

    write_json(args.summary_output.resolve(), summary)
    write_json(
        args.details_output.resolve(),
        {
            "source_diagnostics_path": str(args.diagnostics_path.resolve()),
            "source_model_type": diagnostics_report.get("model_type"),
            "source_model_version": diagnostics_report.get("model_version"),
            "failure_records": failure_records,
        },
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
