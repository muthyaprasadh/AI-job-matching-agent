from job_agent.resumes import (
    ResumeFile,
    is_placeholder_resume,
    pick_role,
    resume_query,
)


def test_picks_first_matched_mapped_role():
    profile = {
        "resumes": {
            "backend_java": "resumes/backend_java.md",
            "sde": "resumes/sde.md",
        }
    }
    assert pick_role(["backend_java", "sde"], profile) == "backend_java"
    assert pick_role(["web_mern", "sde"], profile) == "sde"


def test_resume_query_uses_real_resume_text(tmp_path):
    resume_path = tmp_path / "backend.md"
    resume_path.write_text("Built Java services with Spring Boot.", encoding="utf-8")
    profile = {
        "skills": {"backend": ["Spring Boot"], "languages": ["Java"]},
    }
    resume = ResumeFile(
        role="backend_java",
        path=resume_path,
        text=resume_path.read_text(encoding="utf-8"),
    )

    query = resume_query("backend_java", profile, resume)

    assert "Built Java services" in query
    assert "Spring Boot" in query
    assert not is_placeholder_resume(resume.text)


def test_placeholder_resume_query_falls_back_to_profile_skills(tmp_path):
    resume = ResumeFile(
        role="backend_java",
        path=tmp_path / "placeholder.md",
        text="# Replace this file with your resume",
        placeholder=True,
    )

    query = resume_query(
        "backend_java",
        {"skills": {"backend": ["Spring Boot"], "languages": ["Java"]}},
        resume,
    )

    assert "Spring Boot" in query
    assert "Replace this file" not in query
