from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_NAME = "msmarco_rerank_subset"
DEFAULT_QUERY_LIMIT = 500
DEFAULT_NEGATIVES_PER_QUERY = 10
DEFAULT_MAX_POSITIVES_PER_QUERY = 3
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_CANDIDATE_FILENAMES = ("top1000.train.tsv", "top1000.tsv", "candidates.tsv")

"""Custom Exception class for errors while preparing IR data"""
class DataPreparationError(Exception):
    pass


@dataclass(frozen=True)
class PreparationStats:
    source_files: dict[str, str | None]
    random_seed: int
    max_queries: int
    negatives_per_query: int
    max_positives_per_query: int
    candidate_source_mode: str
    total_queries_retained: int
    total_candidates: int
    split_counts: dict[str, int]
    malformed_counts: dict[str, int]
    skipped_counts: dict[str, int]

""" Define CLI arguments using argparse """
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a small grouped JSONL reranking dataset from MS MARCO raw files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing pre-downloaded MS MARCO raw TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed JSONL and metadata will be written.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=DEFAULT_QUERY_LIMIT,
        help="Maximum number of queries to retain in the prepared subset.",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=DEFAULT_NEGATIVES_PER_QUERY,
        help="Target number of negative passages per query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling and splitting.",
    )
    parser.add_argument(
        "--max-positives-per-query",
        type=int,
        default=DEFAULT_MAX_POSITIVES_PER_QUERY,
        help="Maximum number of relevant passages to retain per query.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="Optional candidate TSV file. If omitted, common filenames in raw-dir are checked.",
    )
    return parser.parse_args()

""" 
    Replace runs of whitespace with a single space,
    and remove leading and trailing whitespace
"""
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def resolve_required_file(raw_dir: Path, filename: str) -> Path:
    path = raw_dir / filename
    if not path.exists():
        raise DataPreparationError(
            f"Required raw file not found: {path}. Expected '{filename}' under {raw_dir}."
        )
    return path

""" 
    Try to resolve candidate file path given candidate_file.
    If candidate_file not given, try to resolve the DEFAULT_CANDIDATE_FILENAMES.
    If default names don't resolve, returns None.
"""
def resolve_optional_candidate_file(raw_dir: Path, candidate_file: Path | None) -> Path | None:
    if candidate_file is not None:
        candidate_path = candidate_file if candidate_file.is_absolute() else raw_dir / candidate_file
        if not candidate_path.exists():
            raise DataPreparationError(
                f"Candidate file not found: {candidate_path}. Pass a valid --candidate-file path "
                "or omit the flag to use random negative sampling."
            )
        return candidate_path

    for filename in DEFAULT_CANDIDATE_FILENAMES:
        path = raw_dir / filename
        if path.exists():
            return path
    return None

"""
    Parse the queries file located at "path" for valid queries and return as a dict[str, str] where:
        key: query id
        value: query text
        
    Queries file is expected to have a format
        query_id\tquery_value
    for each line when rstripped of newlines.
    
    If a line does not follow that format, it is considered malformed and the malformed counter is incremented.
    If a line has either an empty id or value, it is skipped and the skipped counter is incremented.
"""
def load_queries(path: Path, malformed_counts: dict[str, int], skipped_counts: dict[str, int]) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            parts = stripped.split("\t", maxsplit=1)
            if len(parts) != 2:
                malformed_counts["queries"] += 1
                continue

            query_id, query_text = parts
            query_text = clean_text(query_text)
            if not query_id or not query_text:
                skipped_counts["empty_queries"] += 1
                continue

            queries[query_id] = query_text
    return queries

"""
    Parses collection file at path line by line and returns collection dict[str, str], where
        key: passage_id
        value: passage_text_1\t...\tpassage_text_n
    Assumes format of passage_id\tpassage_text_1\t...\tpassage_text_n for each line
    If not at least one passage_text in line, adds to malformed counts of collections
    If id or text empty in line, adds to skipped counts of collection    
"""

def load_collection(
    path: Path,
    malformed_counts: dict[str, int],
    skipped_counts: dict[str, int],
) -> dict[str, str]:
    collection: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            parts = stripped.split("\t")
            if len(parts) < 2:
                malformed_counts["collection"] += 1
                continue

            passage_id = parts[0]
            passage_text = clean_text("\t".join(parts[1:]))
            if not passage_id or not passage_text:
                skipped_counts["empty_passages"] += 1
                continue

            collection[passage_id] = passage_text
    return collection

"""
    Parses qrels file at path line by line and returns qrels dict[str, set[str]], where
        key: query_id
        value: set of passage_ids
    If less than 4 elements on a line, add to malformed count of qrels
    If query_id or passage_id is empty, add to malformed count of qrels
    If relevance score is not one, add to skipped count of qrels
"""

def load_qrels(
    path: Path,
    malformed_counts: dict[str, int],
    skipped_counts: dict[str, int],
) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            parts = stripped.split("\t")
            if len(parts) < 4:
                malformed_counts["qrels"] += 1
                continue

            query_id = parts[0].strip()
            passage_id = parts[2].strip()
            relevance = parts[3].strip()
            if not query_id or not passage_id:
                malformed_counts["qrels"] += 1
                continue
            if relevance != "1":
                skipped_counts["non_relevant_qrels"] += 1
                continue

            qrels[query_id].add(passage_id)
    return dict(qrels)

"""
    Parses candidates file at path line by line, and returns candidates dict[str, list[str]], where:
        key: query_id
        value: list of passage_ids
    If less than two elements in line, or query_id/passage_id is empty, add to malformed counts of candidates
"""

def load_candidates(
    path: Path,
    malformed_counts: dict[str, int],
) -> dict[str, list[str]]:
    candidates: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            parts = stripped.split("\t")
            if len(parts) < 2:
                malformed_counts["candidate_file"] += 1
                continue

            query_id = parts[0].strip()
            passage_id = parts[1].strip()
            if not query_id or not passage_id:
                malformed_counts["candidate_file"] += 1
                continue

            candidates[query_id].append(passage_id)
    return dict(candidates)

""" Calculate totals for each split based on total_queries and the default split ratios"""
def choose_split_counts(total_queries: int) -> dict[str, int]:
    train_count = int(total_queries * DEFAULT_SPLIT_RATIOS[0])
    val_count = int(total_queries * DEFAULT_SPLIT_RATIOS[1])
    test_count = total_queries - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}

""" 
    Shuffles list of query_ids, and returns assignments: dict[str, str], where
        key: query_id
        value: "train"/"val"/"test"
    Basically a dict that identifies which split the query belongs to.
"""
def assign_splits(query_ids: list[str], rng: random.Random) -> dict[str, str]:
    shuffled = list(query_ids)
    rng.shuffle(shuffled)
    split_counts = choose_split_counts(len(shuffled))
    assignments: dict[str, str] = {}

    train_end = split_counts["train"]
    val_end = train_end + split_counts["val"]
    for query_id in shuffled[:train_end]:
        assignments[query_id] = "train"
    for query_id in shuffled[train_end:val_end]:
        assignments[query_id] = "val"
    for query_id in shuffled[val_end:]:
        assignments[query_id] = "test"
    return assignments

"""
    For a given query_id, construct a random sample of negative passages as a list[str]
        - If candidates_by_query exists, use it to construct the list
        - If it doesn't, use collection_ids
"""

def select_negative_passages(
    query_id: str,
    positive_ids: set[str],
    collection_ids: list[str],
    candidates_by_query: dict[str, list[str]] | None,
    negatives_per_query: int,
    rng: random.Random,
    skipped_counts: dict[str, int],
) -> list[str]:
    if candidates_by_query is not None:
        candidate_pool = [
            passage_id
            for passage_id in candidates_by_query.get(query_id, [])
            if passage_id not in positive_ids
        ]
        if candidate_pool:
            unique_pool = list(dict.fromkeys(candidate_pool))
            rng.shuffle(unique_pool)
            return unique_pool[:negatives_per_query]
        skipped_counts["queries_without_candidate_negatives"] += 1

    pool = [passage_id for passage_id in collection_ids if passage_id not in positive_ids]
    if not pool:
        return []
    sample_size = min(negatives_per_query, len(pool))
    return rng.sample(pool, k=sample_size)


"""
    Generate a list of records which contains:
        query_id
        the query text
        a list of positive and negative candidate passages
        what split this record belongs to.
    
"""
def build_grouped_records(
    queries: dict[str, str],
    collection: dict[str, str],
    qrels: dict[str, set[str]],
    max_queries: int,
    negatives_per_query: int,
    max_positives_per_query: int,
    rng: random.Random,
    malformed_counts: dict[str, int],
    skipped_counts: dict[str, int],
    candidates_by_query: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    retained_query_ids: list[str] = []
    collection_ids = sorted(collection)
    # TODO: these for loops feel inefficient. look into possible optimization
    for query_id in sorted(qrels):
        query_text = queries.get(query_id)
        if not query_text:
            skipped_counts["missing_query_text"] += 1
            continue

        valid_positive_ids = sorted(
            passage_id for passage_id in qrels[query_id] if passage_id in collection and collection[passage_id]
        )
        if not valid_positive_ids:
            skipped_counts["queries_without_valid_positives"] += 1
            continue

        retained_query_ids.append(query_id) # query_ids that resolve and their positive passages resolve

    rng.shuffle(retained_query_ids)
    limited_query_ids = retained_query_ids[:max_queries]
    split_assignments = assign_splits(limited_query_ids, rng)

    records: list[dict[str, Any]] = []
    for query_id in limited_query_ids:
        positive_ids = sorted(qrels[query_id])
        valid_positive_ids = [
            passage_id for passage_id in positive_ids if passage_id in collection and collection[passage_id]
        ][:max_positives_per_query]
        if not valid_positive_ids:
            skipped_counts["queries_without_valid_positives"] += 1
            continue

        negative_ids = select_negative_passages(
            query_id=query_id,
            positive_ids=set(valid_positive_ids),
            collection_ids=collection_ids,
            candidates_by_query=candidates_by_query,
            negatives_per_query=negatives_per_query,
            rng=rng,
            skipped_counts=skipped_counts,
        )
        if not negative_ids:
            skipped_counts["queries_without_negatives"] += 1
            continue

        candidate_items = [
            {"id": passage_id, "text": collection[passage_id], "label": 1}
            for passage_id in valid_positive_ids
        ]
        candidate_items.extend(
            {"id": passage_id, "text": collection[passage_id], "label": 0}
            for passage_id in negative_ids
            if passage_id in collection and collection[passage_id]
        )
        if len(candidate_items) <= len(valid_positive_ids):
            skipped_counts["queries_without_negatives"] += 1
            continue

        records.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "candidates": candidate_items,
                "split": split_assignments[query_id],
            }
        )

    malformed_counts.setdefault("candidate_file", 0)
    return records

#TODO: Consider making these grouped records an actual class to enforce structure
def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")

"""
    Writes out grouped records file, metadata file, and returns metadata stats
"""
def prepare_dataset(
    *,
    raw_dir: Path,
    output_dir: Path,
    max_queries: int,
    negatives_per_query: int,
    seed: int,
    max_positives_per_query: int,
    candidate_file: Path | None,
) -> PreparationStats:
    if max_queries <= 0:
        raise DataPreparationError("--max-queries must be greater than 0.")
    if negatives_per_query <= 0:
        raise DataPreparationError("--negatives-per-query must be greater than 0.")
    if max_positives_per_query <= 0:
        raise DataPreparationError("--max-positives-per-query must be greater than 0.")

    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    malformed_counts: dict[str, int] = defaultdict(int)
    skipped_counts: dict[str, int] = defaultdict(int)

    queries_path = resolve_required_file(raw_dir, "queries.train.tsv")
    collection_path = resolve_required_file(raw_dir, "collection.tsv")
    qrels_path = resolve_required_file(raw_dir, "qrels.train.tsv")
    candidate_path = resolve_optional_candidate_file(raw_dir, candidate_file)

    queries = load_queries(queries_path, malformed_counts, skipped_counts)
    collection = load_collection(collection_path, malformed_counts, skipped_counts)
    qrels = load_qrels(qrels_path, malformed_counts, skipped_counts)
    candidates_by_query = (
        load_candidates(candidate_path, malformed_counts) if candidate_path is not None else None
    )

    rng = random.Random(seed)
    records = build_grouped_records(
        queries=queries,
        collection=collection,
        qrels=qrels,
        max_queries=max_queries,
        negatives_per_query=negatives_per_query,
        max_positives_per_query=max_positives_per_query,
        rng=rng,
        malformed_counts=malformed_counts,
        skipped_counts=skipped_counts,
        candidates_by_query=candidates_by_query,
    )

    output_jsonl = output_dir / f"{DEFAULT_OUTPUT_NAME}.jsonl"
    output_metadata = output_dir / f"{DEFAULT_OUTPUT_NAME}_metadata.json"
    write_jsonl(output_jsonl, records)
    #TODO: maybe make train, val and test here enums or something
    split_counts = {
        split: sum(1 for record in records if record["split"] == split)
        for split in ("train", "val", "test")
    }
    total_candidates = sum(len(record["candidates"]) for record in records)
    candidate_source_mode = "candidate_file" if candidate_path is not None else "random_sampling"
    stats = PreparationStats(
        source_files={
            "queries": str(queries_path),
            "collection": str(collection_path),
            "qrels": str(qrels_path),
            "candidate_file": str(candidate_path) if candidate_path is not None else None,
        },
        random_seed=seed,
        max_queries=max_queries,
        negatives_per_query=negatives_per_query,
        max_positives_per_query=max_positives_per_query,
        candidate_source_mode=candidate_source_mode,
        total_queries_retained=len(records),
        total_candidates=total_candidates,
        split_counts=split_counts,
        malformed_counts=dict(malformed_counts),
        skipped_counts=dict(skipped_counts),
    )
    write_json(output_metadata, stats.__dict__)
    return stats


def main() -> None:
    args = parse_args()
    stats = prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        negatives_per_query=args.negatives_per_query,
        seed=args.seed,
        max_positives_per_query=args.max_positives_per_query,
        candidate_file=args.candidate_file,
    )
    print(
        "Prepared reranking subset: "
        f"queries={stats.total_queries_retained}, "
        f"candidates={stats.total_candidates}, "
        f"mode={stats.candidate_source_mode}, "
        f"splits={stats.split_counts}"
    )


if __name__ == "__main__":
    main()
