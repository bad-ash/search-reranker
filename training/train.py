from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from training.bm25 import BM25Artifact
from training.prepare_data import load_collection, resolve_required_file


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def train_bm25_artifact(
    *,
    raw_dir: Path,
    output_artifact: Path,
    k1: float,
    b: float,
) -> BM25Artifact:
    collection_path = resolve_required_file(raw_dir.resolve(), "collection.tsv")
    malformed_counts: dict[str, int] = defaultdict(int)
    skipped_counts: dict[str, int] = defaultdict(int)
    collection = load_collection(collection_path, malformed_counts, skipped_counts)
    artifact = BM25Artifact.from_corpus(list(collection.values()), k1=k1, b=b)
    artifact.save(output_artifact.resolve())
    return artifact


def main() -> None:
    args = parse_args()
    artifact = train_bm25_artifact(
        raw_dir=args.raw_dir,
        output_artifact=args.output_artifact,
        k1=args.k1,
        b=args.b,
    )
    print(
        "Built BM25 artifact: "
        f"documents={artifact.document_count}, "
        f"avg_doc_length={artifact.average_document_length:.2f}, "
        f"vocab={len(artifact.document_frequencies)}"
    )


if __name__ == "__main__":
    main()
