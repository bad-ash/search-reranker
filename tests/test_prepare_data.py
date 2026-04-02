import json
from pathlib import Path

import pytest

from training.prepare_data import DataPreparationError, prepare_dataset


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_dataset_creates_grouped_records_and_query_level_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_text(
        raw_dir / "queries.train.tsv",
        "\n".join(
            [
                "q1\thow long does cooked chicken last",
                "q2\tweather tomorrow",
                "q3\tpython list comprehension",
                "q4\tcapital of france",
            ]
        )
        + "\n",
    )
    write_text(
        raw_dir / "collection.tsv",
        "\n".join(
            [
                "p1\t cooked chicken lasts 3 to 4 days in the fridge ",
                "p2\tfreezing can extend storage time",
                "p3\tforecast data predicts tomorrow weather",
                "p4\tweather can change quickly",
                "p5\tlist comprehensions create lists concisely",
                "p6\tgenerator expressions are similar",
                "p7\tParis is the capital city of France",
                "p8\tLyon is a major city in France",
            ]
        )
        + "\n",
    )
    write_text(
        raw_dir / "qrels.train.tsv",
        "\n".join(
            [
                "q1\t0\tp1\t1",
                "q2\t0\tp3\t1",
                "q3\t0\tp5\t1",
                "q4\t0\tp7\t1",
            ]
        )
        + "\n",
    )

    stats = prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_queries=4,
        negatives_per_query=2,
        seed=7,
        max_positives_per_query=2,
        candidate_file=None,
    )

    records = read_jsonl(output_dir / "msmarco_rerank_subset.jsonl")
    metadata = json.loads((output_dir / "msmarco_rerank_subset_metadata.json").read_text("utf-8"))

    assert len(records) == 4
    assert stats.total_queries_retained == 4
    assert metadata["candidate_source_mode"] == "random_sampling"
    assert {record["split"] for record in records} == {"train", "test"}

    seen_query_ids = set()
    for record in records:
        assert set(record) == {"query_id", "query", "candidates", "split"}
        assert record["query_id"] not in seen_query_ids
        seen_query_ids.add(record["query_id"])
        assert record["split"] in {"train", "val", "test"}
        assert len(record["candidates"]) == 3

        positives = [candidate for candidate in record["candidates"] if candidate["label"] == 1]
        negatives = [candidate for candidate in record["candidates"] if candidate["label"] == 0]
        assert len(positives) == 1
        assert len(negatives) == 2
        positive_ids = {candidate["id"] for candidate in positives}
        negative_ids = {candidate["id"] for candidate in negatives}
        assert positive_ids.isdisjoint(negative_ids)
        assert all(candidate["text"] == candidate["text"].strip() for candidate in record["candidates"])


def test_prepare_dataset_is_deterministic_with_fixed_seed(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output_dir_one = tmp_path / "processed_one"
    output_dir_two = tmp_path / "processed_two"

    write_text(raw_dir / "queries.train.tsv", "q1\tquery one\nq2\tquery two\n")
    write_text(raw_dir / "collection.tsv", "p1\tpositive one\np2\tnegative one\np3\tpositive two\np4\tnegative two\n")
    write_text(raw_dir / "qrels.train.tsv", "q1\t0\tp1\t1\nq2\t0\tp3\t1\n")

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir_one,
        max_queries=2,
        negatives_per_query=1,
        seed=11,
        max_positives_per_query=1,
        candidate_file=None,
    )
    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir_two,
        max_queries=2,
        negatives_per_query=1,
        seed=11,
        max_positives_per_query=1,
        candidate_file=None,
    )

    assert (
        (output_dir_one / "msmarco_rerank_subset.jsonl").read_text("utf-8")
        == (output_dir_two / "msmarco_rerank_subset.jsonl").read_text("utf-8")
    )
    assert (
        (output_dir_one / "msmarco_rerank_subset_metadata.json").read_text("utf-8")
        == (output_dir_two / "msmarco_rerank_subset_metadata.json").read_text("utf-8")
    )


def test_prepare_dataset_uses_candidate_file_for_negatives_when_present(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_text(raw_dir / "queries.train.tsv", "q1\tquery one\n")
    write_text(raw_dir / "collection.tsv", "p1\tpositive one\np2\tcandidate negative\np3\trandom negative\n")
    write_text(raw_dir / "qrels.train.tsv", "q1\t0\tp1\t1\n")
    write_text(raw_dir / "top1000.tsv", "q1\tp2\t1\t9.1\nq1\tp1\t2\t8.0\n")

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        max_queries=1,
        negatives_per_query=1,
        seed=3,
        max_positives_per_query=1,
        candidate_file=None,
    )

    records = read_jsonl(output_dir / "msmarco_rerank_subset.jsonl")
    negatives = [candidate for candidate in records[0]["candidates"] if candidate["label"] == 0]

    assert len(negatives) == 1
    assert negatives[0]["id"] == "p2"

    metadata = json.loads((output_dir / "msmarco_rerank_subset_metadata.json").read_text("utf-8"))
    assert metadata["candidate_source_mode"] == "candidate_file"


def test_prepare_dataset_raises_clear_error_when_required_file_is_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_text(raw_dir / "queries.train.tsv", "q1\tquery one\n")
    write_text(raw_dir / "collection.tsv", "p1\tpositive one\n")

    with pytest.raises(DataPreparationError, match="Required raw file not found"):
        prepare_dataset(
            raw_dir=raw_dir,
            output_dir=output_dir,
            max_queries=1,
            negatives_per_query=1,
            seed=1,
            max_positives_per_query=1,
            candidate_file=None,
        )
