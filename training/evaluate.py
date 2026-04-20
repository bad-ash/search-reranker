from __future__ import annotations

import argparse
import json
from typing import cast
import torch


from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Any

from training.bm25 import BM25Artifact, BM25Scorer


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")
DEFAULT_DATASET_PATH = Path("data/processed/msmarco_rerank_subset.jsonl")
DEFAULT_REPORT_PATH = Path("artifacts/eval/bm25_eval_report.json")
DEFAULT_DIAGNOSTICS_PATH = Path("artifacts/eval/bm25_query_diagnostics.json")
DEFAULT_K_VALUES = (1, 3, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--diagnostics-path", type=Path, default=None)
    parser.add_argument("--model-type",choices=("bm25", "sbert"),default="bm25")

    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file into a list of dictionaries."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

def reciprocal_rank(labels: list[int]) -> float:
    """Return the reciprocal rank of the first relevant result in a ranked label list."""

    for index, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / index
    return 0.0

def recall_at_k(labels: list[int], total_positives: int, k: int) -> float:
    """Return the fraction of relevant results retrieved within the top k positions."""

    if total_positives == 0:
        return 0.0
    return sum(labels[:k]) / total_positives


def first_positive_rank(labels: list[int]) -> int | None:
    for index, label in enumerate(labels, start=1):
        if label == 1:
            return index
    return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_bm25_evaluation_report(
    *,
    artifact_path: Path,
    dataset_path: Path,
    split: str = "test",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate BM25 on one split of the grouped reranking dataset and return a report."""

    resolved_artifact_path = artifact_path.resolve()
    resolved_dataset_path = dataset_path.resolve()
    artifact = BM25Artifact.load(resolved_artifact_path)
    scorer = BM25Scorer(artifact)
    records = load_jsonl(resolved_dataset_path)
    split_records = [record for record in records if record["split"] == split]
    if not split_records:
        raise ValueError(f"No records found for split '{split}'.")

    reciprocal_ranks: list[float] = []
    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}
    candidate_counts: list[int] = []
    query_diagnostics: list[dict[str, Any]] = []

    for record in split_records:
        candidate_counts.append(len(record["candidates"]))
        scored_candidates = sorted(
            (
                {
                    "id": candidate["id"],
                    "text": candidate["text"],
                    "label": candidate["label"],
                    "score": scorer.score(record["query"], candidate["text"]),
                }
                for candidate in record["candidates"]
            ),
            key=lambda candidate: candidate["score"],
            reverse=True,
        )
        labels = [candidate["label"] for candidate in scored_candidates]
        total_positives = sum(labels)
        rr = reciprocal_rank(labels)
        first_rank = first_positive_rank(labels)
        reciprocal_ranks.append(rr)
        for k in k_values:
            recall_scores[k].append(recall_at_k(labels, total_positives, k))
        per_query_metrics = {
            "reciprocal_rank": rr,
            "first_positive_rank": first_rank,
        }
        for k in k_values:
            per_query_metrics[f"recall@{k}"] = recall_at_k(labels, total_positives, k)
        query_diagnostics.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "candidate_count": len(record["candidates"]),
                "positive_count": total_positives,
                "metrics": per_query_metrics,
                "ranked_candidates": scored_candidates,
            }
        )

    metrics: dict[str, float] = {
        "query_count": float(len(split_records)),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = sum(recall_scores[k]) / len(recall_scores[k])

    return {
        "artifact_path": str(resolved_artifact_path),
        "dataset_path": str(resolved_dataset_path),
        "model_type": "bm25",
        "model_version": f"bm25:{resolved_artifact_path.name}",
        "split": split,
        "k_values": list(k_values),
        "summary": {
            "query_count": len(split_records),
            "candidate_count_total": sum(candidate_counts),
            "candidate_count_avg": sum(candidate_counts) / len(candidate_counts),
        },
        "metrics": metrics,
        "query_diagnostics": query_diagnostics,
    }
    
    
def build_sbert_evaluation_report(
    *,
    dataset_path: Path,
    split: str = "test",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """Evaluate SBERT on one split of the grouped reranking dataset and return a report."""

    resolved_dataset_path = dataset_path.resolve()
    records = load_jsonl(resolved_dataset_path)
    split_records = [record for record in records if record["split"] == split]
    if not split_records:
        raise ValueError(f"No records found for split '{split}'.")

    reciprocal_ranks: list[float] = []
    recall_scores: dict[int, list[float]] = {k: [] for k in k_values}
    candidate_counts: list[int] = []
    query_diagnostics: list[dict[str, Any]] = []

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    for record in split_records:
        candidates: list[dict[str, Any]] = record["candidates"]
        candidate_texts = [candidate["text"] for candidate in candidates]

        candidate_counts.append(len(candidates))
        scored_candidates: list[dict[str, Any]] = []
        
        candidate_embeddings = cast(
            torch.Tensor,
            embedder.encode_document(candidate_texts, convert_to_tensor=True)
        )
        query_embedding = cast(
            torch.Tensor,
            embedder.encode_query(record["query"], convert_to_tensor=True)
        )
        similarity_scores: torch.Tensor = embedder.similarity(query_embedding, candidate_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, len(candidates))
        
        for score, idx in zip(scores, indices):
            candidate = candidates[int(idx)]
            scored_candidates.append(
                {
                    "id": candidate["id"],
                    "text": candidate["text"],
                    "label": candidate["label"],
                    "score": float(score)
                }
            )
        labels = [candidate["label"] for candidate in scored_candidates]
        total_positives = sum(labels)
        rr = reciprocal_rank(labels)
        first_rank = first_positive_rank(labels)
        reciprocal_ranks.append(rr)
        for k in k_values:
            recall_scores[k].append(recall_at_k(labels, total_positives, k))
        per_query_metrics = {
            "reciprocal_rank": rr,
            "first_positive_rank": first_rank,
        }
        for k in k_values:
            per_query_metrics[f"recall@{k}"] = recall_at_k(labels, total_positives, k)
        query_diagnostics.append(
            {
                "query_id": record["query_id"],
                "query": record["query"],
                "candidate_count": len(record["candidates"]),
                "positive_count": total_positives,
                "metrics": per_query_metrics,
                "ranked_candidates": scored_candidates,
            }
        )

    metrics: dict[str, float] = {
        "query_count": float(len(split_records)),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = sum(recall_scores[k]) / len(recall_scores[k])

    return {
        "dataset_path": str(resolved_dataset_path),
        "model_type": "sbert",
        "model_version": "all-MiniLM-L6-v2",
        "split": split,
        "k_values": list(k_values),
        "summary": {
            "query_count": len(split_records),
            "candidate_count_total": sum(candidate_counts),
            "candidate_count_avg": sum(candidate_counts) / len(candidate_counts),
        },
        "metrics": metrics,
        "query_diagnostics": query_diagnostics,
    }


def evaluate_bm25(
    *,
    artifact_path: Path,
    dataset_path: Path,
    split: str = "test",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    output_path: Path | None = None,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate BM25, optionally write a JSON report, and return the report."""

    report = build_bm25_evaluation_report(
        artifact_path=artifact_path,
        dataset_path=dataset_path,
        split=split,
        k_values=k_values,
    )
    if output_path is not None:
        aggregate_report = {key: value for key, value in report.items() if key != "query_diagnostics"}
        write_json(output_path.resolve(), aggregate_report)
    if diagnostics_path is not None:
        diagnostics_report = {
            "artifact_path": report["artifact_path"],
            "dataset_path": report["dataset_path"],
            "model_type": report["model_type"],
            "model_version": report["model_version"],
            "split": report["split"],
            "query_diagnostics": sorted(
                report["query_diagnostics"],
                key=lambda item: (
                    item["metrics"]["reciprocal_rank"],
                    item["metrics"]["first_positive_rank"] or float("inf"),
                ),
            ),
        }
        write_json(diagnostics_path.resolve(), diagnostics_report)
    return {key: value for key, value in report.items() if key != "query_diagnostics"}

def evaluate_sbert(
    dataset_path: Path,
    split: str = "test",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    output_path: Path | None = None,
    diagnostics_path: Path | None = None,
) -> dict[str, Any]:
    
    report = build_sbert_evaluation_report(
        dataset_path=dataset_path,
        split=split,
        k_values=k_values,
    )
    if output_path is not None:
        aggregate_report = {key: value for key, value in report.items() if key != "query_diagnostics"}
        write_json(output_path.resolve(), aggregate_report)
    if diagnostics_path is not None:
        diagnostics_report = {
            "dataset_path": report["dataset_path"],
            "model_type": report["model_type"],
            "model_version": report["model_version"],
            "split": report["split"],
            "query_diagnostics": sorted(
                report["query_diagnostics"],
                key=lambda item: (
                    item["metrics"]["reciprocal_rank"],
                    item["metrics"]["first_positive_rank"] or float("inf"),
                ),
            ),
        }
        write_json(diagnostics_path.resolve(), diagnostics_report)
    return {key: value for key, value in report.items() if key != "query_diagnostics"}


def main() -> None:
    args = parse_args()
    if args.model_type == "bm25":
        report = evaluate_bm25(
            artifact_path=args.artifact_path,
            dataset_path=args.dataset_path,
            split=args.split,
            output_path=args.output_path,
            diagnostics_path=args.diagnostics_path,
        )
    elif args.model_type == "sbert":
        report = evaluate_sbert(
            dataset_path=args.dataset_path,
            split=args.split,
            output_path=args.output_path,
            diagnostics_path=args.diagnostics_path,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
