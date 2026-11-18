from __future__ import annotations

from job_agent.config import DATA, load_profile, load_settings
from job_agent.db import Store
from job_agent.drafter import draft_application
from job_agent.matcher import score_job
from job_agent.resumes import load_for_role, pick_role
from job_agent.sources import collect_jobs


def ingest(limit: int | None = None) -> dict[str, int]:
    settings = load_settings()
    profile = load_profile()
    store = Store(DATA / "jobs.sqlite")
    cap = limit or int(settings.get("daily_limit", 50))

    raw = collect_jobs(settings)
    matches = []
    skipped_existing = 0
    for job in raw:
        if store.exists(job.uid):
            skipped_existing += 1
            continue
        match = score_job(job, settings)
        if match:
            matches.append(match)

    matches.sort(key=lambda m: m.score, reverse=True)
    selected = matches[:cap]
    drafted = 0
    for match in selected:
        match.primary_role = pick_role(match.matched_roles, profile)
        resume_file = load_for_role(match.primary_role, profile)
        draft = draft_application(match, profile, resume_file.text, resume_file)
        store.insert(match, draft)
        drafted += 1

    return {
        "fetched": len(raw),
        "matched": len(matches),
        "queued": drafted,
        "skipped_existing": skipped_existing,
    }
