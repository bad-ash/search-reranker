from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status

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
        app.state.model = None
        app.state.model_error = None
        try:
            app.state.model = load_model(artifact_path) if artifact_path is not None else load_model()
        except ModelLoadError as exc:
            app.state.model_error = str(exc)
        yield

    app = FastAPI(title="Real-time Document Reranking API", lifespan=lifespan)

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
