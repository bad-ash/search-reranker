from __future__ import annotations

from sentence_transformers import SentenceTransformer


DEFAULT_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_FOLDER = "/app/.hf/sentence_transformers"


def main() -> None:
    SentenceTransformer(
        DEFAULT_SBERT_MODEL_NAME,
        cache_folder=DEFAULT_CACHE_FOLDER,
    )
    SentenceTransformer(
        DEFAULT_SBERT_MODEL_NAME,
        cache_folder=DEFAULT_CACHE_FOLDER,
        local_files_only=True,
    )


if __name__ == "__main__":
    main()
