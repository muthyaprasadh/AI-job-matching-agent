from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from job_agent.config import DATA, ROOT

ROLE_LABELS = {
    "backend_java": "Backend / Java",
    "sde": "Software Engineer / SDE",
    "cybersecurity": "Cybersecurity",
    "ai_ml": "AI / ML",
    "data_sql": "SQL / Data Engineer",
    "web_mern": "Web Developer / MERN",
}


@dataclass
class ResumeFile:
    role: str
    path: Path
    text: str
    missing: bool = False


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def read_resume_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path if (ROOT / path).exists() else DATA / path
    return path


def mapped_paths(profile: dict[str, Any]) -> dict[str, Path]:
    mapping = profile.get("resumes") or {}
    return {role: resolve_path(str(rel)) for role, rel in mapping.items()}


def pick_role(matched_roles: list[str], profile: dict[str, Any]) -> str:
    mapping = mapped_paths(profile)
    for role in matched_roles:
        if role in mapping:
            return role
    default = profile.get("default_resume_role")
    if default:
        return str(default)
    if mapping:
        return next(iter(mapping))
    return matched_roles[0] if matched_roles else "sde"


def load_for_role(role: str, profile: dict[str, Any]) -> ResumeFile:
    mapping = mapped_paths(profile)
    fallback = resolve_path(str(profile.get("default_resume") or "resume.md"))
    path = mapping.get(role, fallback)
    if not path.exists():
        if fallback.exists() and fallback != path:
            return ResumeFile(
                role=role,
                path=fallback,
                text=read_resume_file(fallback),
                missing=True,
            )
        return ResumeFile(role=role, path=path, text="", missing=True)
    return ResumeFile(role=role, path=path, text=read_resume_file(path), missing=False)


def inventory(profile: dict[str, Any]) -> list[tuple[str, Path, bool]]:
    rows = []
    for role, path in mapped_paths(profile).items():
        rows.append((role, path, path.exists()))
    return rows
