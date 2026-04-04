from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status

from service.logging_config import configure_logging
from service.model import CandidateDocument, ModelLoadError, RerankerModel, load_model
from service.schemas import (
    HealthResponse,
    ReadinessResponse,
    RerankRequest,
    RerankResponse,
)


DEFAULT_RERANK_TIMEOUT_SECONDS = 2.0


class RerankTimeoutError(Exception):
    """Raised when reranking exceeds the configured timeout."""


class RerankExecutionError(Exception):
    """Raised when reranking fails for an internal reason."""


def build_app(*, artifact_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.logger = configure_logging()
        app.state.model = None
        app.state.model_error = None
        try:
            app.state.model = load_model(artifact_path) if artifact_path is not None else load_model()
        except ModelLoadError as exc:
            app.state.model_error = str(exc)
            app.state.logger.exception(
                "Model load failed",
                extra={"event": "model_load_failed", "path": str(artifact_path) if artifact_path else None},
            )
        else:
            app.state.logger.info(
                "Model loaded",
                extra={"event": "model_loaded", "path": str(artifact_path) if artifact_path else "default"},
            )
        yield

    app = FastAPI(title="Real-time Document Reranking API", lifespan=lifespan)

    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        start = perf_counter()
        response = await call_next(request)
        duration_ms = round((perf_counter() - start) * 1000.0, 3)
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        response.headers["X-Request-ID"] = request_id
        model = request.app.state.model
        request.app.state.logger.info(
            "Request completed",
            extra={
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "num_candidates": getattr(request.state, "num_candidates", None),
                "model_version": model.model_version if model is not None else None,
            },
        )
        return response

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadinessResponse)
    async def readyz(request: Request) -> ReadinessResponse:
        model_loaded = request.app.state.model is not None
        if not model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=request.app.state.model_error or "Model not loaded.",
            )
        return ReadinessResponse(status="ready", model_loaded=True)

    @app.post("/rerank", response_model=RerankResponse)
    async def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
        model = _get_loaded_model(request)
        request.state.num_candidates = len(payload.candidates)
        candidates = [
            CandidateDocument(id=candidate.id, text=candidate.text)
            for candidate in payload.candidates
        ]
        try:
            ranked = await _execute_rerank(
                model=model,
                query=payload.query,
                candidates=candidates,
                timeout_seconds=DEFAULT_RERANK_TIMEOUT_SECONDS,
            )
        except RerankTimeoutError as exc:
            request.app.state.logger.warning(
                "Rerank timed out",
                extra={
                    "event": "rerank_timeout",
                    "request_id": request.state.request_id,
                    "num_candidates": len(candidates),
                    "model_version": model.model_version,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=str(exc),
            ) from exc
        except RerankExecutionError as exc:
            request.app.state.logger.exception(
                "Rerank failed",
                extra={
                    "event": "rerank_failed",
                    "request_id": request.state.request_id,
                    "num_candidates": len(candidates),
                    "model_version": model.model_version,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Rerank request failed.",
            ) from exc
        return RerankResponse(results=ranked)

    return app


def _get_loaded_model(request: Request) -> RerankerModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=request.app.state.model_error or "Model not loaded.",
        )
    return model


async def _execute_rerank(
    *,
    model: RerankerModel,
    query: str,
    candidates: list[CandidateDocument],
    timeout_seconds: float,
):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(model.rerank, query, candidates),
            timeout=timeout_seconds,
        )
    except TimeoutError as exc:
        raise RerankTimeoutError(
            f"Rerank request exceeded timeout of {timeout_seconds} seconds."
        ) from exc
    except Exception as exc:
        raise RerankExecutionError("Rerank execution failed.") from exc


app = build_app()
