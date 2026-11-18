from job_agent.resumes import pick_role


def test_picks_first_matched_mapped_role():
    profile = {
        "resumes": {
            "backend_java": "resumes/backend_java.md",
            "sde": "resumes/sde.md",
        }
    }
    assert pick_role(["backend_java", "sde"], profile) == "backend_java"
    assert pick_role(["web_mern", "sde"], profile) == "sde"
