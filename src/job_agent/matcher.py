from __future__ import annotations

import re
from typing import Any

from job_agent.models import Job, Match

INDIA_HINTS = (
    "india",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "pune",
    "mumbai",
    "delhi",
    "ncr",
    "gurgaon",
    "gurugram",
    "noida",
    "chennai",
    "kolkata",
    "ahmedabad",
    "kochi",
    "in-remote",
)

REMOTE_HINTS = (
    "remote",
    "worldwide",
    "anywhere",
    "work from home",
    "distributed",
    "wfh",
)

INTERN_HINTS = ("intern", "internship", "trainee", "apprentice", "campus")
FT_HINTS = (
    "full time",
    "full-time",
    "permanent",
    "new grad",
    "graduate",
    "entry level",
    "junior",
    "sde-1",
    "sde i",
    "associate",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = _norm(phrase).strip()
    if not normalized:
        return False
    pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
    return re.search(pattern, _norm(text)) is not None


def classify_job_type(text: str) -> str:
    blob = _norm(text)
    intern = any(_contains_phrase(blob, hint) for hint in INTERN_HINTS)
    full = any(_contains_phrase(blob, hint) for hint in FT_HINTS)
    if intern and not full:
        return "internship"
    if intern:
        return "internship"
    if full:
        return "full_time"
    return "unknown"


def location_ok(job: Job, settings: dict[str, Any]) -> bool:
    blob = _norm(f"{job.location} {job.title} {job.description[:800]}")
    excludes = [_norm(x) for x in settings.get("locations", {}).get("exclude", [])]
    if any(_contains_phrase(blob, phrase) for phrase in excludes):
        return False
    if job.remote or any(_contains_phrase(blob, hint) for hint in REMOTE_HINTS):
        return True
    includes = [_norm(x) for x in settings.get("locations", {}).get("include", [])]
    return any(_contains_phrase(blob, hint) for hint in includes + list(INDIA_HINTS))


def type_ok(job: Job, settings: dict[str, Any]) -> bool:
    wanted = set(settings.get("job_types", ["internship", "full_time"]))
    job.job_type = classify_job_type(f"{job.title} {job.description}")
    if job.job_type == "unknown":
        return True
    return job.job_type in wanted


def score_job(job: Job, settings: dict[str, Any]) -> Match | None:
    if not location_ok(job, settings):
        return None
    if not type_ok(job, settings):
        return None

    blob = _norm(f"{job.title} {job.company} {job.location} {' '.join(job.tags)} {job.description}")
    roles = settings.get("roles", {})
    ranked: list[tuple[int, int, str, list[str]]] = []
    reasons: list[str] = []
    hits = 0
    title_bonus = 0.0

    for role, keywords in roles.items():
        role_hits = [kw for kw in keywords if _contains_phrase(blob, kw)]
        if not role_hits:
            continue
        title_hit = 1 if any(_contains_phrase(job.title, kw) for kw in keywords) else 0
        ranked.append((len(role_hits), title_hit, role, role_hits))
        hits += min(len(role_hits), 6)
        reasons.append(f"{role}: {', '.join(role_hits[:5])}")
        if title_hit:
            title_bonus += 0.12

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    matched = [item[2] for item in ranked]
    primary_role = matched[0]

    score = min(1.0, 0.25 + hits * 0.08 + title_bonus)
    if job.remote or any(
        _contains_phrase(job.location, hint) for hint in REMOTE_HINTS + INDIA_HINTS
    ):
        score = min(1.0, score + 0.08)
    if job.job_type == "internship":
        score = min(1.0, score + 0.05)
    if score < float(settings.get("min_score", 0.42)):
        return None
    return Match(
        job=job,
        score=round(score, 3),
        matched_roles=matched,
        reasons=reasons,
        primary_role=primary_role,
    )
