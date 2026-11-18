from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from job_agent.agent import AgentConfigurationError, AgentResponse, JobAgent
from job_agent.config import DATA, load_settings
from job_agent.db import Store
from job_agent.retrieval import (
    HybridRetriever,
    MiniLMCrossEncoder,
    MiniLMEmbedder,
    PostgresVectorStore,
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_settings()
    dsn = os.getenv("PGVECTOR_DSN")
    retriever = (
        HybridRetriever(
            MiniLMEmbedder(),
            PostgresVectorStore(dsn),
            MiniLMCrossEncoder(),
        )
        if dsn
        else None
    )
    application.state.retriever = retriever
    try:
        yield
    finally:
        if retriever:
            retriever.vector_store.close()


app = FastAPI(
    title="Job Application Prep Agent",
    description="Search saved job matches and create human-reviewed application drafts.",
    version="0.2.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class JobSearchResult(BaseModel):
    uid: str
    title: str
    company: str
    location: str
    url: str
    score: float
    method: str


class SearchResponse(BaseModel):
    jobs: list[JobSearchResult]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=100)


def _require_api_key(
    supplied_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected_key = os.getenv("JOB_AGENT_API_KEY")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Set JOB_AGENT_API_KEY before enabling API access.",
        )
    if not supplied_key or not hmac.compare_digest(supplied_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _store() -> Iterator[Store]:
    store = Store(DATA / "jobs.sqlite")
    try:
        yield store
    finally:
        store.conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/jobs/search", response_model=SearchResponse)
def search_jobs(
    request: SearchRequest,
    http_request: Request,
    store: Annotated[Store, Depends(_store)],
    _: Annotated[None, Depends(_require_api_key)],
) -> SearchResponse:
    return SearchResponse(
        jobs=JobAgent(store, retriever=http_request.app.state.retriever)._search_jobs(
            request.query, request.limit
        )
    )


@app.post("/v1/chat", response_model=AgentResponse)
def chat(
    request: ChatRequest,
    http_request: Request,
    store: Annotated[Store, Depends(_store)],
    _: Annotated[None, Depends(_require_api_key)],
) -> AgentResponse:
    try:
        return JobAgent(store, retriever=http_request.app.state.retriever).chat(
            request.message, request.conversation_id
        )
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/v1/matches/{uid}")
def explain_match(
    uid: str,
    store: Annotated[Store, Depends(_store)],
    _: Annotated[None, Depends(_require_api_key)],
) -> dict[str, Any]:
    result = JobAgent(store)._explain_match(uid)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
