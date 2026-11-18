from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, Sequence

from job_agent.models import Job

TOKEN_RE = re.compile(r"\b[\w+#.]+\b", re.UNICODE)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def job_text(job: Job) -> str:
    return " ".join((job.title, job.company, job.location, *job.tags, job.description))


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Return BM25 scores in document order."""
    query_tokens = _tokens(query)
    tokenized = [_tokens(document) for document in documents]
    if not query_tokens or not tokenized:
        return [0.0] * len(tokenized)

    document_count = len(tokenized)
    average_length = sum(map(len, tokenized)) / document_count or 1.0
    frequencies = [Counter(document) for document in tokenized]
    document_frequency: Counter[str] = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())

    k1, b = 1.5, 0.75
    scores: list[float] = []
    for counts, terms in zip(frequencies, tokenized):
        length = len(terms)
        score = 0.0
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


@dataclass(frozen=True)
class RetrievalResult:
    job: Job
    score: float
    method: str


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, jobs: Sequence[Job], embeddings: Sequence[Sequence[float]]) -> None: ...

    def search(self, embedding: Sequence[float], limit: int) -> list[Job]: ...


class Reranker(Protocol):
    def rerank(self, query: str, jobs: Sequence[Job]) -> list[tuple[Job, float]]: ...


class MiniLMEmbedder:
    """Lazy-loaded all-MiniLM-L6-v2 encoder; vectors are normalized for cosine search."""

    def __init__(
        self, model_name: str = EMBEDDING_MODEL, revision: str | None = None
    ) -> None:
        self.model_name = model_name
        self.revision = revision or (
            EMBEDDING_REVISION if model_name == EMBEDDING_MODEL else None
        )
        self._model = None

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies with `pip install .[retrieval]`."
            ) from exc
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, revision=self.revision)
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


class PostgresVectorStore:
    """Persist job text and MiniLM vectors in PostgreSQL with the pgvector extension."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies with `pip install .[retrieval]`."
            ) from exc
        self._connection = psycopg.connect(dsn, autocommit=True)
        with self._connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS job_vectors (
                    uid TEXT PRIMARY KEY,
                    job_json JSONB NOT NULL,
                    embedding VECTOR(384) NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS job_vectors_embedding_hnsw
                ON job_vectors USING hnsw (embedding vector_cosine_ops)
                """
            )

        register_vector(self._connection)

    def upsert(self, jobs: Sequence[Job], embeddings: Sequence[Sequence[float]]) -> None:
        if len(jobs) != len(embeddings):
            raise ValueError("Each job must have exactly one embedding.")
        import json
        from psycopg.types.json import Jsonb
        job_uids = [job.uid for job in jobs]

        with self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM job_vectors WHERE NOT (uid = ANY(%s))",
                    (job_uids,),
                )
                for job, embedding in zip(jobs, embeddings):
                    cursor.execute(
                        """
                        INSERT INTO job_vectors (uid, job_json, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (uid) DO UPDATE SET
                            job_json = EXCLUDED.job_json,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            job.uid,
                            Jsonb(json.loads(json.dumps(job.__dict__))),
                            list(embedding),
                        ),
                    )

    def search(self, embedding: Sequence[float], limit: int) -> list[Job]:
        import json

        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT job_json
                FROM job_vectors
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (list(embedding), limit),
            )
            rows = cursor.fetchall()
        return [Job(**(row[0] if isinstance(row[0], dict) else json.loads(row[0]))) for row in rows]

    def close(self) -> None:
        self._connection.close()


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, jobs: Sequence[Job], embeddings: Sequence[Sequence[float]]) -> None:
        if len(jobs) != len(embeddings):
            raise ValueError("Each job must have exactly one embedding.")
        self._jobs = {job.uid: job for job in jobs}
        self._vectors = {
            job.uid: list(embedding) for job, embedding in zip(jobs, embeddings)
        }

    def search(self, embedding: Sequence[float], limit: int) -> list[Job]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        query_norm = math.sqrt(sum(value * value for value in embedding))

        def cosine(uid: str) -> float:
            vector = self._vectors[uid]
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if not query_norm or not vector_norm:
                return 0.0
            return sum(a * b for a, b in zip(vector, embedding)) / (
                query_norm * vector_norm
            )

        ranked_uids = sorted(self._vectors, key=cosine, reverse=True)[:limit]
        return [self._jobs[uid] for uid in ranked_uids]


class MiniLMCrossEncoder:
    def __init__(
        self, model_name: str = RERANKER_MODEL, revision: str | None = None
    ) -> None:
        self.model_name = model_name
        self.revision = revision or (
            RERANKER_REVISION if model_name == RERANKER_MODEL else None
        )
        self._model = None

    def rerank(self, query: str, jobs: Sequence[Job]) -> list[tuple[Job, float]]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install retrieval dependencies with `pip install .[retrieval]`."
            ) from exc
        if self._model is None:
            self._model = CrossEncoder(self.model_name, revision=self.revision)
        pairs = [(query, job_text(job)) for job in jobs]
        scores = self._model.predict(pairs)
        return sorted(
            zip(jobs, (float(score) for score in scores)),
            key=lambda result: result[1],
            reverse=True,
        )


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def search(
        self,
        query: str,
        jobs: Sequence[Job],
        limit: int = 10,
        candidate_multiplier: int = 5,
    ) -> list[RetrievalResult]:
        return self.search_many(
            {"query": query},
            jobs,
            limit=limit,
            candidate_multiplier=candidate_multiplier,
        )["query"]

    def search_many(
        self,
        queries: Mapping[str, str],
        jobs: Sequence[Job],
        limit: int = 10,
        candidate_multiplier: int = 5,
    ) -> dict[str, list[RetrievalResult]]:
        if limit < 1:
            raise ValueError("limit must be positive.")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive.")
        if not jobs or not queries:
            return {key: [] for key in queries}

        documents = [job_text(job) for job in jobs]
        embeddings = self.embedder.encode(documents)
        if len(embeddings) != len(jobs):
            raise ValueError("Embedder returned a different number of vectors than input jobs.")
        self.vector_store.upsert(jobs, embeddings)
        query_embeddings = self.embedder.encode(list(queries.values()))
        if len(query_embeddings) != len(queries):
            raise ValueError("Embedder returned a different number of query vectors.")

        output: dict[str, list[RetrievalResult]] = {}
        for (query_id, query), query_embedding in zip(
            queries.items(), query_embeddings
        ):
            lexical_scores = bm25_scores(query, documents)
            lexical_order = sorted(
                (index for index, score in enumerate(lexical_scores) if score > 0),
                key=lambda index: lexical_scores[index],
                reverse=True,
            )
            lexical_candidates = [
                jobs[index]
                for index in lexical_order[:limit * candidate_multiplier]
            ]
            dense_candidates = self.vector_store.search(
                query_embedding, limit * candidate_multiplier
            )

            candidate_by_uid = {job.uid: job for job in lexical_candidates}
            candidate_by_uid.update({job.uid: job for job in dense_candidates})
            lexical_rank = {
                job.uid: rank for rank, job in enumerate(lexical_candidates, 1)
            }
            dense_rank = {
                job.uid: rank for rank, job in enumerate(dense_candidates, 1)
            }
            fused = [
                (
                    job,
                    (1 / (60 + lexical_rank[job.uid]) if job.uid in lexical_rank else 0)
                    + (1 / (60 + dense_rank[job.uid]) if job.uid in dense_rank else 0),
                )
                for job in candidate_by_uid.values()
            ]
            fused.sort(key=lambda result: result[1], reverse=True)
            candidates = [job for job, _ in fused[:limit * candidate_multiplier]]
            if self.reranker:
                reranked = self.reranker.rerank(query, candidates)
                output[query_id] = [
                    RetrievalResult(job=job, score=score, method="hybrid+reranker")
                    for job, score in reranked[:limit]
                ]
            else:
                output[query_id] = [
                    RetrievalResult(job=job, score=score, method="hybrid")
                    for job, score in fused[:limit]
                ]
        return output
