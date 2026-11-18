from __future__ import annotations

import html
import re
from typing import Any, Iterable

import httpx

from job_agent.models import Job

UA = {"User-Agent": "job-apply-agent/0.1 (personal application prep; +local)"}
TIMEOUT = 25.0


def _strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _from_remotive() -> list[Job]:
    url = "https://remotive.com/api/remote-jobs"
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        data = client.get(url).json()
    jobs: list[Job] = []
    for item in data.get("jobs", []):
        jobs.append(
            Job(
                source="remotive",
                source_id=str(item.get("id")),
                title=item.get("title") or "",
                company=item.get("company_name") or "",
                location=item.get("candidate_required_location") or "Remote",
                url=item.get("url") or item.get("job_type") or "",
                description=_strip_html(item.get("description") or ""),
                tags=item.get("tags") or [],
                remote=True,
                posted_at=item.get("publication_date"),
            )
        )
    return jobs


def _from_remoteok() -> list[Job]:
    url = "https://remoteok.com/api"
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        data = client.get(url).json()
    jobs: list[Job] = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item or "position" not in item:
            continue
        loc = item.get("location") or "Remote"
        jobs.append(
            Job(
                source="remoteok",
                source_id=str(item["id"]),
                title=item.get("position") or "",
                company=item.get("company") or "",
                location=str(loc),
                url=item.get("url") or item.get("apply_url") or "",
                description=_strip_html(item.get("description") or ""),
                tags=item.get("tags") or [],
                remote=True,
                posted_at=str(item.get("date") or ""),
            )
        )
    return jobs


def _from_arbeitnow() -> list[Job]:
    url = "https://www.arbeitnow.com/api/job-board-api"
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        data = client.get(url).json()
    jobs: list[Job] = []
    for item in data.get("data", []):
        loc = ", ".join(item.get("location") or []) if isinstance(item.get("location"), list) else (
            item.get("location") or ""
        )
        jobs.append(
            Job(
                source="arbeitnow",
                source_id=str(item.get("slug") or item.get("url")),
                title=item.get("title") or "",
                company=item.get("company_name") or "",
                location=str(loc) or "Remote",
                url=item.get("url") or "",
                description=_strip_html(item.get("description") or ""),
                tags=item.get("tags") or [],
                remote=bool(item.get("remote")),
            )
        )
    return jobs


def _from_greenhouse(boards: Iterable[str]) -> list[Job]:
    jobs: list[Job] = []
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        for board in boards:
            url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
            except httpx.HTTPError:
                continue
            for item in payload.get("jobs", []):
                loc_obj = item.get("location") or {}
                loc = loc_obj.get("name") if isinstance(loc_obj, dict) else str(loc_obj or "")
                jobs.append(
                    Job(
                        source="greenhouse",
                        source_id=f"{board}:{item.get('id')}",
                        title=item.get("title") or "",
                        company=board,
                        location=loc or "Unspecified",
                        url=item.get("absolute_url") or "",
                        description=_strip_html(item.get("content") or ""),
                        remote="remote" in (loc or "").lower(),
                        posted_at=item.get("updated_at"),
                    )
                )
    return jobs


def _from_lever(companies: Iterable[str]) -> list[Job]:
    jobs: list[Job] = []
    with httpx.Client(headers=UA, timeout=TIMEOUT, follow_redirects=True) as client:
        for company in companies:
            url = f"https://api.lever.co/v0/postings/{company}?mode=json"
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
            except httpx.HTTPError:
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                cats = item.get("categories") or {}
                loc = cats.get("location") or item.get("country") or ""
                jobs.append(
                    Job(
                        source="lever",
                        source_id=str(item.get("id") or item.get("hostedUrl")),
                        title=item.get("text") or "",
                        company=company,
                        location=str(loc),
                        url=item.get("hostedUrl") or item.get("applyUrl") or "",
                        description=_strip_html(item.get("descriptionPlain") or item.get("description") or ""),
                        tags=[cats.get("team", ""), cats.get("commitment", "")],
                        remote="remote" in str(loc).lower() or "remote" in str(item.get("workplaceType", "")).lower(),
                        posted_at=str(item.get("createdAt") or ""),
                    )
                )
    return jobs


def collect_jobs(settings: dict[str, Any]) -> list[Job]:
    enabled = settings.get("sources", {})
    jobs: list[Job] = []
    errors: list[str] = []

    fetchers = []
    if enabled.get("remotive", True):
        fetchers.append(("remotive", _from_remotive))
    if enabled.get("remoteok", True):
        fetchers.append(("remoteok", _from_remoteok))
    if enabled.get("arbeitnow", True):
        fetchers.append(("arbeitnow", _from_arbeitnow))
    if enabled.get("greenhouse", True):
        boards = settings.get("greenhouse_boards") or []
        fetchers.append(("greenhouse", lambda: _from_greenhouse(boards)))
    if enabled.get("lever", True):
        companies = settings.get("lever_companies") or []
        fetchers.append(("lever", lambda: _from_lever(companies)))

    for name, fn in fetchers:
        try:
            batch = fn()
            jobs.extend(batch)
        except Exception as exc:  # keep the pipeline alive if one board fails
            errors.append(f"{name}: {exc}")

    if errors:
        print("Source warnings:", "; ".join(errors))
    return jobs
