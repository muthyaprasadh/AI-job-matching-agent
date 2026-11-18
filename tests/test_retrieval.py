from job_agent.models import Job
from job_agent.retrieval import HybridRetriever, bm25_scores


def _job(source_id: str, title: str, description: str) -> Job:
    return Job(
        source="test",
        source_id=source_id,
        title=title,
        company="Acme",
        location="Remote",
        url="https://example.com",
        description=description,
    )


def test_bm25_ranks_relevant_document_first():
    scores = bm25_scores(
        "Python machine learning",
        ["Java backend APIs", "Python machine learning with PyTorch"],
    )

    assert scores[1] > scores[0]


def test_hybrid_retriever_fuses_dense_and_bm25_candidates():
    class FakeEmbedder:
        def encode(self, texts):
            return [
                [float("python" in text.lower()), float("java" in text.lower())]
                for text in texts
            ]

    class FakeVectorStore:
        def upsert(self, jobs, embeddings):
            self.jobs = list(jobs)
            self.embeddings = list(embeddings)

        def search(self, embedding, limit):
            ranked = sorted(
                zip(self.jobs, self.embeddings),
                key=lambda item: sum(a * b for a, b in zip(item[1], embedding)),
                reverse=True,
            )
            return [job for job, _ in ranked[:limit]]

    jobs = [
        _job("1", "Java backend developer", "Build APIs with Spring."),
        _job("2", "ML engineer", "Train Python models using PyTorch."),
    ]
    retriever = HybridRetriever(FakeEmbedder(), FakeVectorStore())

    results = retriever.search("Python machine learning", jobs, limit=1)

    assert results[0].job.source_id == "2"
    assert results[0].method == "hybrid"


def test_hybrid_retriever_batches_resume_queries():
    class FakeEmbedder:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, texts):
            self.batch_sizes.append(len(texts))
            return [
                [float("python" in text.lower()), float("java" in text.lower())]
                for text in texts
            ]

    class FakeVectorStore:
        def upsert(self, jobs, embeddings):
            self.jobs = list(jobs)
            self.embeddings = list(embeddings)

        def search(self, embedding, limit):
            ranked = sorted(
                zip(self.jobs, self.embeddings),
                key=lambda item: sum(a * b for a, b in zip(item[1], embedding)),
                reverse=True,
            )
            return [job for job, _ in ranked[:limit]]

    embedder = FakeEmbedder()
    retriever = HybridRetriever(embedder, FakeVectorStore())
    jobs = [
        _job("1", "Java backend developer", "Build APIs with Spring."),
        _job("2", "Python ML engineer", "Train models."),
    ]

    results = retriever.search_many(
        {"backend": "Java", "ai_ml": "Python"}, jobs, limit=1
    )

    assert embedder.batch_sizes == [2, 2]
    assert results["backend"][0].job.source_id == "1"
    assert results["ai_ml"][0].job.source_id == "2"
