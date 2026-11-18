from job_agent import pipeline
from job_agent.models import Draft, Job
from job_agent.resumes import ResumeFile
from job_agent.retrieval import RetrievalResult


def test_ingest_ranks_jobs_using_resume_even_without_keyword_match(
    tmp_path, monkeypatch
):
    resume_path = tmp_path / "backend.md"
    resume_path.write_text(
        "Built payment services using Java and Spring Boot.", encoding="utf-8"
    )
    profile = {
        "full_name": "Candidate",
        "skills": {"backend": ["Spring Boot"], "languages": ["Java"]},
        "resumes": {"backend_java": str(resume_path)},
        "default_resume_role": "backend_java",
    }
    job = Job(
        source="test",
        source_id="resume-ranked",
        title="Platform Engineer",
        company="Acme",
        location="Bengaluru, India",
        url="https://example.com",
        description="Build cloud services for the team.",
    )
    settings = {
        "daily_limit": 1,
        "min_score": 0.3,
        "job_types": ["internship", "full_time"],
        "roles": {"backend_java": ["java"]},
        "locations": {"include": ["india"], "exclude": []},
    }
    captured_queries = {}
    monkeypatch.setattr(pipeline, "DATA", tmp_path)
    monkeypatch.setattr(pipeline, "load_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "load_profile", lambda: profile)
    monkeypatch.setattr(pipeline, "collect_jobs", lambda _: [job])

    def fake_rank(queries, jobs, limit):
        captured_queries.update(queries)
        return {
            "backend_java": [
                RetrievalResult(job=jobs[0], score=0.02, method="hybrid+reranker")
            ]
        }

    monkeypatch.setattr(pipeline, "_resume_rankings", fake_rank)
    monkeypatch.setattr(
        pipeline,
        "draft_application",
        lambda match, profile, resume, resume_file: Draft(
            cover_letter="Draft letter",
            resume_notes="Notes",
            apply_checklist=["Review before applying."],
            model="test",
        ),
    )

    stats = pipeline.ingest()

    assert "Built payment services" in captured_queries["backend_java"]
    assert stats == {"fetched": 1, "matched": 1, "queued": 1, "skipped_existing": 0}
    stored = pipeline.Store(tmp_path / "jobs.sqlite")
    try:
        row = stored.get(job.uid)
        assert row is not None
        assert row["matched_roles"] == '["backend_java"]'
    finally:
        stored.conn.close()
