import pytest

from job_agent import drafter
from job_agent.drafter import (
    DraftResponse,
    GroundednessError,
    validate_groundedness,
)
from job_agent.models import Job, Match


def _response(skill: str) -> DraftResponse:
    return DraftResponse(
        cover_letter=(
            f"I am interested in this role and would welcome a conversation about "
            f"how my {skill} experience can contribute to the team."
        ),
        resume_notes="Highlight relevant experience and projects from the candidate profile.",
        checklist=["Review the application before submitting."],
        skills_used=[skill],
    )


def test_groundedness_accepts_skills_present_in_profile():
    validate_groundedness(_response("Python"), {"skills": {"languages": ["Python"]}})


def test_groundedness_rejects_skills_missing_from_profile():
    with pytest.raises(GroundednessError, match="rust"):
        validate_groundedness(_response("Rust"), {"skills": {"languages": ["Python"]}})


def test_groundedness_rejects_unlisted_skill_mentions():
    response = DraftResponse(
        cover_letter=(
            "I am interested in this role and would welcome a conversation about "
            "how Kotlin can contribute to the team."
        ),
        resume_notes="Highlight relevant experience and projects from the profile.",
        checklist=["Review the application before submitting."],
        skills_used=["Python"],
    )

    with pytest.raises(GroundednessError, match="kotlin"):
        validate_groundedness(response, {"skills": {"languages": ["Python"]}})


def test_draft_retries_after_schema_validation_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    prompts = []
    responses = iter(
        [
            {"cover_letter": "too short"},
            {
                "cover_letter": (
                    "I am interested in this role and would welcome a conversation about "
                    "how my Python experience can contribute to the team."
                ),
                "resume_notes": "Highlight relevant experience and projects from the profile.",
                "checklist": ["Review the application before submitting."],
                "skills_used": ["Python"],
                "_model": "openai:test",
            },
        ]
    )

    def fake_openai(prompt):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(drafter, "_from_openai", fake_openai)
    monkeypatch.setattr(
        drafter, "_from_ollama", lambda prompt: pytest.fail("unexpected Ollama fallback")
    )
    match = Match(
        job=Job(
            source="test",
            source_id="1",
            title="Python Engineer",
            company="Acme",
            location="Remote",
            url="https://example.com",
            description="Build Python services.",
        ),
        score=0.8,
        matched_roles=["sde"],
        reasons=["sde: python"],
        primary_role="sde",
    )

    draft = drafter.draft_application(
        match,
        {"full_name": "Applicant", "skills": {"languages": ["Python"]}},
        "Python experience",
    )

    assert len(prompts) == 2
    assert "failed validation" in prompts[1]
    assert draft.model == "openai:test"
