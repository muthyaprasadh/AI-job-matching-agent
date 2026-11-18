from __future__ import annotations

import importlib.util
import logging
import os

from job_agent.config import DATA, load_profile, load_settings
from job_agent.db import Store
from job_agent.drafter import draft_application
from job_agent.matcher import location_ok, score_job, type_ok
from job_agent.models import Match
from job_agent.resumes import (
    load_for_role,
    mapped_paths,
    pick_role,
    resume_query,
)
from job_agent.retrieval import (
    HybridRetriever,
    InMemoryVectorStore,
    MiniLMCrossEncoder,
    MiniLMEmbedder,
    PostgresVectorStore,
    RetrievalResult,
    bm25_scores,
    job_text,
)
from job_agent.sources import collect_jobs

logger = logging.getLogger(__name__)


def _role_queries(profile: dict) -> dict[str, str]:
    roles = list(mapped_paths(profile))
    if not roles:
        roles = [str(profile.get("default_resume_role") or "sde")]
    queries: dict[str, str] = {}
    for role in roles:
        resume_file = load_for_role(role, profile)
        if resume_file.placeholder:
            logger.warning(
                "%s resume is a placeholder or empty; ranking this role from profile skills.",
                role,
            )
        query = resume_query(role, profile, resume_file).strip()
        if query:
            queries[role] = query
    return queries


def _rank_bm25(
    queries: dict[str, str], jobs: list, limit: int
) -> dict[str, list[RetrievalResult]]:
    documents = [job_text(job) for job in jobs]
    rankings: dict[str, list[RetrievalResult]] = {}
    for role, query in queries.items():
        scores = bm25_scores(query, documents)
        ranked = sorted(
            (
                (job, score)
                for job, score in zip(jobs, scores)
                if score > 0
            ),
            key=lambda result: result[1],
            reverse=True,
        )[:limit]
        rankings[role] = [
            RetrievalResult(job=job, score=score, method="resume+BM25")
            for job, score in ranked
        ]
    return rankings


def _resume_rankings(
    queries: dict[str, str], jobs: list, limit: int
) -> dict[str, list[RetrievalResult]]:
    if not jobs or not queries:
        return {role: [] for role in queries}

    dsn = os.getenv("PGVECTOR_DSN")
    if importlib.util.find_spec("sentence_transformers") is None:
        if dsn:
            raise RuntimeError(
                "PGVECTOR_DSN is set; install retrieval dependencies with "
                "`pip install .[retrieval]`."
            )
        logger.info(
            "Sentence Transformers is not installed; using resume-based BM25 retrieval."
        )
        return _rank_bm25(queries, jobs, limit)

    vector_store = (
        PostgresVectorStore(dsn) if dsn else InMemoryVectorStore()
    )
    if not dsn:
        logger.info(
            "PGVECTOR_DSN is unset; resume embeddings will be indexed in memory for this run."
        )
    try:
        retriever = HybridRetriever(
            MiniLMEmbedder(),
            vector_store,
            MiniLMCrossEncoder(),
        )
        return retriever.search_many(queries, jobs, limit=limit)
    finally:
        if isinstance(vector_store, PostgresVectorStore):
            vector_store.close()


def ingest(limit: int | None = None) -> dict[str, int]:
    settings = load_settings()
    profile = load_profile()
    store = Store(DATA / "jobs.sqlite")
    cap = limit or int(settings.get("daily_limit", 50))

    try:
        raw = collect_jobs(settings)
        skipped_existing = 0
        eligible = []
        keyword_matches = {}
        for job in raw:
            if store.exists(job.uid):
                skipped_existing += 1
                continue
            if not location_ok(job, settings) or not type_ok(job, settings):
                continue
            eligible.append(job)
            match = score_job(job, settings)
            if match:
                keyword_matches[job.uid] = match

        rankings = _resume_rankings(_role_queries(profile), eligible, cap)
        evidence: dict[str, dict[str, float]] = {}
        reasons: dict[str, list[str]] = {}
        jobs_by_uid = {job.uid: job for job in eligible}
        minimum_score = float(settings.get("min_score", 0.42))

        for role, results in rankings.items():
            denominator = max(len(results), 1)
            for rank, result in enumerate(results, start=1):
                rank_score = 1 - (rank - 1) / denominator
                score = minimum_score + (1 - minimum_score) * rank_score
                scores = evidence.setdefault(result.job.uid, {})
                scores[role] = max(scores.get(role, 0.0), score)
                reasons.setdefault(result.job.uid, []).append(
                    f"{result.method} ranking for {role} resume (rank {rank})"
                )

        for uid, match in keyword_matches.items():
            scores = evidence.setdefault(uid, {})
            for role in match.matched_roles:
                scores[role] = max(scores.get(role, 0.0), match.score)
            reasons.setdefault(uid, []).extend(match.reasons)

        matches = []
        for uid, role_scores in evidence.items():
            matched_roles = sorted(
                role_scores, key=lambda role: (-role_scores[role], role)
            )
            score = max(role_scores.values())
            if score < minimum_score:
                continue
            matches.append(
                Match(
                    job=jobs_by_uid[uid],
                    score=round(score, 3),
                    matched_roles=matched_roles,
                    reasons=list(dict.fromkeys(reasons[uid])),
                    primary_role=matched_roles[0],
                )
            )

        matches.sort(key=lambda match: match.score, reverse=True)
        selected = matches[:cap]
        for match in selected:
            match.primary_role = pick_role(match.matched_roles, profile)
            resume_file = load_for_role(match.primary_role, profile)
            draft = draft_application(
                match,
                profile,
                resume_file.text,
                resume_file,
            )
            store.insert(match, draft)

        return {
            "fetched": len(raw),
            "matched": len(matches),
            "queued": len(selected),
            "skipped_existing": skipped_existing,
        }
    finally:
        store.conn.close()
