from __future__ import annotations

import json
import os
from typing import Any

import httpx

from job_agent.models import Draft, Match
from job_agent.resumes import ROLE_LABELS, ResumeFile


def _profile_block(profile: dict[str, Any], resume: str) -> str:
    return (
        f"Name: {profile.get('full_name')}\n"
        f"Email: {profile.get('email')}\n"
        f"Location: {profile.get('location')}\n"
        f"Targets: {', '.join(profile.get('target_roles') or [])}\n"
        f"Skills: {json.dumps(profile.get('skills', {}), ensure_ascii=False)}\n"
        f"Education: {profile.get('education')}\n"
        f"Summary: {profile.get('experience_summary')}\n"
        f"Projects: {json.dumps(profile.get('projects', []), ensure_ascii=False)}\n"
        f"Resume:\n{resume[:5000]}"
    )


def _prompt(match: Match, profile: dict[str, Any], resume: str) -> str:
    job = match.job
    return f"""You prepare job applications for human approval. Do not invent employers, dates, or skills.
Write:
1) A 180-250 word cover letter in first person.
2) 4-6 resume tweak bullets tailored to this posting (truthful, from the profile only).
3) A short apply checklist (account, resume PDF, form fields, questions to expect).

Candidate:
{_profile_block(profile, resume)}

Job:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Type: {job.job_type}
Matched roles: {', '.join(match.matched_roles)}
Use this resume version only: {ROLE_LABELS.get(match.primary_role, match.primary_role)}
URL: {job.url}
Description:
{job.description[:4000]}

Return JSON with keys cover_letter, resume_notes, checklist (array of strings).
"""


def _template_draft(
    match: Match, profile: dict[str, Any], resume_file: ResumeFile | None = None
) -> Draft:
    job = match.job
    name = profile.get("full_name", "Applicant")
    cover = (
        f"Dear {job.company} hiring team,\n\n"
        f"I am {name}, based in {profile.get('location', 'India')}, applying for the "
        f"{job.title} role. I am targeting {', '.join(match.matched_roles)} positions "
        f"and this posting matches my background in {', '.join((profile.get('skills') or {}).get('languages') or [])}.\n\n"
        f"{(profile.get('experience_summary') or '').strip()}\n\n"
        "I would welcome the chance to contribute and can start conversations over email. "
        "Thank you for your time.\n\n"
        f"Sincerely,\n{name}\n{profile.get('email', '')}"
    )
    resume_path = str(resume_file.path) if resume_file else ""
    role_label = ROLE_LABELS.get(match.primary_role, match.primary_role)
    notes = (
        f"Submit the {role_label} resume ({resume_path or 'data/resumes/'}). "
        "Do not mix bullets from other role resumes. "
        f"Matched themes: {', '.join(match.reasons)}."
    )
    checklist = [
        "Read the full JD and confirm you meet must-have requirements.",
        f"Upload this resume file: {resume_path}",
        "Paste the cover letter into the ATS or upload as PDF.",
        f"Open apply URL: {job.url}",
        "Submit only after you personally review every field.",
    ]
    return Draft(
        cover_letter=cover,
        resume_notes=notes,
        apply_checklist=checklist,
        model="template",
        resume_role=match.primary_role,
        resume_path=resume_path,
    )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def _from_openai(prompt: str) -> dict[str, Any] | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": "You write truthful application drafts as JSON."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    data = _parse_json_payload(content)
    if data:
        data["_model"] = f"openai:{model}"
    return data


def _from_ollama(prompt: str) -> dict[str, Any] | None:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                f"{host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            )
            if resp.status_code != 200:
                return None
            content = resp.json().get("response", "")
    except httpx.HTTPError:
        return None
    data = _parse_json_payload(content)
    if data:
        data["_model"] = f"ollama:{model}"
    return data


def draft_application(
    match: Match, profile: dict[str, Any], resume: str, resume_file: ResumeFile | None = None
) -> Draft:
    prompt = _prompt(match, profile, resume)
    data = None
    try:
        data = _from_openai(prompt)
    except httpx.HTTPError:
        data = None
    if data is None:
        data = _from_ollama(prompt)
    if not data:
        return _template_draft(match, profile, resume_file)

    checklist = data.get("checklist") or []
    if isinstance(checklist, str):
        checklist = [line.strip("- ") for line in checklist.splitlines() if line.strip()]
    path = str(resume_file.path) if resume_file else ""
    if path:
        checklist = [f"Upload this resume file: {path}"] + [str(x) for x in checklist]
        if resume_file and resume_file.missing:
            checklist.insert(0, f"WARNING: mapped resume missing; using fallback {path}")
    else:
        checklist = [str(x) for x in checklist]
    cover = str(data.get("cover_letter") or "").strip()
    if not cover:
        cover = _template_draft(match, profile, resume_file).cover_letter
    return Draft(
        cover_letter=cover,
        resume_notes=str(data.get("resume_notes") or "").strip(),
        apply_checklist=checklist,
        model=str(data.get("_model") or "llm"),
        resume_role=match.primary_role,
        resume_path=path,
    )
