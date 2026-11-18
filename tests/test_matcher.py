from job_agent.matcher import classify_job_type, score_job
from job_agent.models import Job


def test_classifies_internship():
    assert classify_job_type("Backend Intern - Java") == "internship"


def test_scores_java_backend_in_india():
    job = Job(
        source="test",
        source_id="1",
        title="Java Backend Engineer",
        company="Acme",
        location="Bengaluru, India",
        url="https://example.com",
        description="Build Spring Boot microservices and REST APIs.",
    )
    settings = {
        "min_score": 0.3,
        "job_types": ["internship", "full_time"],
        "roles": {"backend_java": ["java", "spring", "backend"]},
        "locations": {"include": ["india", "bengaluru"], "exclude": []},
    }
    match = score_job(job, settings)
    assert match is not None
    assert "backend_java" in match.matched_roles
    assert match.primary_role == "backend_java"


def test_keyword_matches_observe_word_boundaries():
    job = Job(
        source="test",
        source_id="2",
        title="Platform Engineer",
        company="Acme",
        location="India",
        url="https://example.com",
        description="Build a springboard for internal developer tools.",
    )
    settings = {
        "min_score": 0.3,
        "job_types": ["internship", "full_time"],
        "roles": {"backend_java": ["spring"]},
        "locations": {"include": ["india"], "exclude": []},
    }

    assert score_job(job, settings) is None


def test_classification_does_not_match_substrings():
    assert classify_job_type("Internally maintained platform") == "unknown"
