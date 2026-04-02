import json
from pathlib import Path

from training.bm25 import BM25Artifact, BM25Scorer
from training.evaluate import evaluate_bm25
from training.train import train_bm25_artifact


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_bm25_scores_more_relevant_document_higher() -> None:
    artifact = BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial",
            "weather forecast for tomorrow",
        ]
    )
    scorer = BM25Scorer(artifact)

    relevant_score = scorer.score("python list comprehension", "python list comprehension tutorial")
    irrelevant_score = scorer.score("python list comprehension", "weather forecast for tomorrow")

    assert relevant_score > irrelevant_score


def test_train_bm25_artifact_writes_artifact_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_artifact = tmp_path / "artifacts" / "bm25_artifact.json"

    write_text(
        raw_dir / "collection.tsv",
        "\n".join(
            [
                "p1\tpython list comprehension tutorial",
                "p2\tweather forecast for tomorrow",
            ]
        )
        + "\n",
    )

    artifact = train_bm25_artifact(
        raw_dir=raw_dir,
        output_artifact=output_artifact,
        k1=1.2,
        b=0.7,
    )

    assert output_artifact.exists()
    payload = json.loads(output_artifact.read_text(encoding="utf-8"))
    assert payload["document_count"] == 2
    assert payload["k1"] == 1.2
    assert payload["b"] == 0.7
    assert artifact.document_frequencies["python"] == 1


def test_evaluate_bm25_reports_perfect_metrics_on_easy_dataset(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifacts" / "bm25_artifact.json"
    dataset_path = tmp_path / "processed" / "dataset.jsonl"

    artifact = BM25Artifact.from_corpus(
        [
            "python list comprehension tutorial",
            "weather forecast for tomorrow",
            "capital city of france paris",
            "deep sea fish habitat",
        ]
    )
    artifact.save(artifact_path)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "query_id": "q1",
            "query": "python list comprehension",
            "split": "test",
            "candidates": [
                {"id": "p1", "text": "python list comprehension tutorial", "label": 1},
                {"id": "p2", "text": "deep sea fish habitat", "label": 0},
            ],
        },
        {
            "query_id": "q2",
            "query": "capital of france",
            "split": "test",
            "candidates": [
                {"id": "p3", "text": "capital city of france paris", "label": 1},
                {"id": "p4", "text": "weather forecast for tomorrow", "label": 0},
            ],
        },
    ]
    write_text(dataset_path, "\n".join(json.dumps(record) for record in records) + "\n")

    metrics = evaluate_bm25(
        artifact_path=artifact_path,
        dataset_path=dataset_path,
    )

    assert metrics["query_count"] == 2.0
    assert metrics["mrr"] == 1.0
    assert metrics["recall@1"] == 1.0
