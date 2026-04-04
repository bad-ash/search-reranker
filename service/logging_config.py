from __future__ import annotations

import json
import logging
import sys
from typing import Any


LOGGER_NAME = "search_reranker.service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if hasattr(record, "method"):
            payload["method"] = record.method
        if hasattr(record, "path"):
            payload["path"] = record.path
        if hasattr(record, "status_code"):
            payload["status_code"] = record.status_code
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = record.duration_ms
        if hasattr(record, "num_candidates"):
            payload["num_candidates"] = record.num_candidates
        if hasattr(record, "model_version"):
            payload["model_version"] = record.model_version
        return json.dumps(payload, sort_keys=True)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
