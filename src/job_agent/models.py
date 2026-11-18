from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


JobType = Literal["internship", "full_time", "unknown"]
Status = Literal["pending", "approved", "rejected", "applied"]


@dataclass
class Job:
    source: str
    source_id: str
    title: str
    company: str
    location: str
    url: str
    description: str
    job_type: JobType = "unknown"
    tags: list[str] = field(default_factory=list)
    remote: bool = False
    posted_at: str | None = None

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"


@dataclass
class Match:
    job: Job
    score: float
    matched_roles: list[str]
    reasons: list[str]
    primary_role: str = ""


@dataclass
class Draft:
    cover_letter: str
    resume_notes: str
    apply_checklist: list[str]
    model: str
    resume_role: str = ""
    resume_path: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
