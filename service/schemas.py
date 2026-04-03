from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateDocumentRequest(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: list[CandidateDocumentRequest] = Field(min_length=1)


class RankedDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    text: str
    score: float


class RerankResponse(BaseModel):
    results: list[RankedDocumentResponse]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
