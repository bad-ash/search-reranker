from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, status

from service.logging_config import configure_logging
from service.model import CandidateDocument, ModelLoadError, RerankerModel, load_model
from service.schemas import (
    HealthResponse,
    ReadinessResponse,
    RerankRequest,
    RerankResponse,
)


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
        start = perf_counter()
        response = await call_next(request)
        duration_ms = round((perf_counter() - start) * 1000.0, 3)
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        request.app.state.logger.info(
            "Request completed",
            extra={
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
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
        ranked = model.rerank(
            payload.query,
            [
                CandidateDocument(id=candidate.id, text=candidate.text)
                for candidate in payload.candidates
            ],
        )
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


app = build_app()
