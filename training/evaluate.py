from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.bm25 import BM25Artifact, BM25Scorer


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")
DEFAULT_DATASET_PATH = Path("data/processed/msmarco_rerank_subset.jsonl")
DEFAULT_REPORT_PATH = Path("artifacts/eval/bm25_eval_report.json")
DEFAULT_K_VALUES = (1, 3, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_REPORT_PATH)
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

    for record in split_records:
        candidate_counts.append(len(record["candidates"]))
        scored_candidates = sorted(
            (
                {
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
        reciprocal_ranks.append(reciprocal_rank(labels))
        for k in k_values:
            recall_scores[k].append(recall_at_k(labels, total_positives, k))

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
    }


def evaluate_bm25(
    *,
    artifact_path: Path,
    dataset_path: Path,
    split: str = "test",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate BM25, optionally write a JSON report, and return the report."""

    report = build_bm25_evaluation_report(
        artifact_path=artifact_path,
        dataset_path=dataset_path,
        split=split,
        k_values=k_values,
    )
    if output_path is not None:
        write_json(output_path.resolve(), report)
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_bm25(
        artifact_path=args.artifact_path,
        dataset_path=args.dataset_path,
        split=args.split,
        output_path=args.output_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
