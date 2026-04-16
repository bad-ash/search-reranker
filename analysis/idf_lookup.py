from __future__ import annotations

import argparse
from pathlib import Path

from training.bm25 import BM25Artifact, BM25Scorer


DEFAULT_ARTIFACT_PATH = Path("artifacts/bm25_artifact.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("term")
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = BM25Artifact.load(args.artifact_path.resolve())
    scorer = BM25Scorer(artifact)
    print(scorer.IDF(args.term))


if __name__ == "__main__":
    main()
