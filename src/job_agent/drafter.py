from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from job_agent.models import Draft, Match
from job_agent.resumes import ROLE_LABELS, ResumeFile

logger = logging.getLogger(__name__)

KNOWN_SKILLS = (
    "amazon s3",
    "aws",
    "angular",
    "azure",
    "bash",
    "c++",
    "c#",
    "c",
    "css",
    "computer vision",
    "django",
    "docker",
    "express.js",
    "fastapi",
    "flask",
    "git",
    "graphql",
    "hibernate",
    "html",
    "java",
    "javascript",
    "jenkins",
    "jwt",
    "kafka",
    "kotlin",
    "kubernetes",
    "linux",
    "machine learning",
    "mongodb",
    "mysql",
    "next.js",
    "nlp",
    "node.js",
    "numpy",
    "opencv",
    "pandas",
    "php",
    "postgresql",
    "python",
    "pytorch",
    "rabbitmq",
    "react",
    "redis",
    "ruby",
    "rust",
    "scala",
    "scikit-learn",
    "spark",
    "spring boot",
    "spring security",
    "sql",
    "swift",
    "tensorflow",
    "terraform",
    "typescript",
    "vue",
)


class DraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cover_letter: str = Field(min_length=80, max_length=6000)
    resume_notes: str = Field(min_length=10, max_length=3000)
    checklist: list[str] = Field(min_length=1, max_length=12)
    skills_used: list[str] = Field(
        max_length=30,
        description="Every skill and technology explicitly claimed in the draft.",
    )

    @field_validator("checklist", "skills_used")
    @classmethod
    def strip_list_values(cls, values: list[str]) -> list[str]:
        cleaned = [item.strip() for item in values]
        if any(not item for item in cleaned):
            raise ValueError("List entries must not be empty.")
        return cleaned


class GroundednessError(ValueError):
    pass


class DraftProviderError(RuntimeError):
    pass


def _profile_skills(profile: dict[str, Any]) -> set[str]:
    skills: set[str] = set()
    categories = profile.get("skills") or {}
    if isinstance(categories, dict):
        for values in categories.values():
            if isinstance(values, list):
                skills.update(str(value).strip().casefold() for value in values if str(value).strip())
    for project in profile.get("projects") or []:
        if isinstance(project, dict):
            technologies = project.get("technologies") or []
            skills.update(
                str(value).strip().casefold()
                for value in technologies
                if str(value).strip()
            )
    return skills


def validate_groundedness(
    response: DraftResponse, profile: dict[str, Any]
) -> None:
    allowed = _profile_skills(profile)
    text = f"{response.cover_letter}\n{response.resume_notes}".casefold()
    unsupported = [
        skill for skill in response.skills_used if skill.casefold() not in allowed
    ]
    unsupported.extend(
        skill
        for skill in KNOWN_SKILLS
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text)
        and not any(
            re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", profile_skill)
            for profile_skill in allowed
        )
    )
    unsupported = list(
        {skill.casefold(): skill for skill in unsupported}.values()
    )
    if unsupported:
        raise GroundednessError(
            "Draft cites skills absent from the profile: " + ", ".join(unsupported)
        )
    declared = {skill.casefold() for skill in response.skills_used}
    mentioned = {
        skill
        for skill in allowed
        if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", text)
    }
    if mentioned - declared:
        raise GroundednessError(
            "Draft mentions skills missing from skills_used: "
            + ", ".join(sorted(mentioned - declared))
        )
    if declared - mentioned:
        raise GroundednessError(
            "skills_used contains skills not mentioned in the draft: "
            + ", ".join(sorted(declared - mentioned))
        )

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

Return JSON with keys cover_letter, resume_notes, checklist (array of strings), and
skills_used (array of every technical skill you claim in the cover letter or resume notes).
Every item in skills_used must appear in the candidate profile's skills or project
technologies. Never add unsupported skills.
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
    if resume_file and (resume_file.missing or resume_file.placeholder):
        checklist.insert(
            0,
            f"WARNING: replace this missing or placeholder resume before applying: {resume_path}",
        )
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
                raise DraftProviderError(
                    f"Ollama draft request returned HTTP {resp.status_code}."
                )
            content = resp.json().get("response", "")
    except httpx.HTTPError as exc:
        raise DraftProviderError(f"Ollama draft request failed: {exc}") from exc
    data = _parse_json_payload(content)
    if data:
        data["_model"] = f"ollama:{model}"
    return data


def draft_application(
    match: Match, profile: dict[str, Any], resume: str, resume_file: ResumeFile | None = None
) -> Draft:
    if resume_file and resume_file.placeholder:
        resume = ""
    prompt = _prompt(match, profile, resume)
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", _from_openai))
    providers.append(("ollama", _from_ollama))
    validated: tuple[DraftResponse, str] | None = None
    for provider_name, provider in providers:
        retry_prompt = prompt
        for attempt in range(2):
            try:
                data = provider(retry_prompt)
            except (httpx.HTTPError, DraftProviderError) as exc:
                logger.warning("%s draft request failed: %s", provider_name, exc)
                break
            if not data:
                logger.warning(
                    "%s returned invalid JSON (attempt %d).",
                    provider_name,
                    attempt + 1,
                )
                retry_prompt = prompt + "\nThe previous response was invalid. Return the complete JSON object."
                continue
            model_name = str(data.pop("_model", provider_name))
            try:
                response = DraftResponse.model_validate(data)
                validate_groundedness(response, profile)
            except (ValidationError, GroundednessError) as exc:
                details = (
                    "; ".join(
                        f"{error['loc']}: {error['msg']}"
                        for error in exc.errors(include_input=False)
                    )
                    if isinstance(exc, ValidationError)
                    else str(exc)
                )
                logger.warning(
                    "%s draft failed validation (attempt %d): %s",
                    provider_name,
                    attempt + 1,
                    details,
                )
                retry_prompt = (
                    prompt
                    + "\nThe previous response failed validation. Correct these errors and "
                    + "return the complete JSON object: "
                    + details
                )
                continue
            validated = (response, model_name)
            break
        if validated:
            break
    if not validated:
        return _template_draft(match, profile, resume_file)

    response, model_name = validated
    checklist = response.checklist
    path = str(resume_file.path) if resume_file else ""
    if path:
        checklist = [f"Upload this resume file: {path}"] + [str(x) for x in checklist]
        if resume_file and (resume_file.missing or resume_file.placeholder):
            checklist.insert(
                0,
                f"WARNING: replace this missing or placeholder resume before applying: {path}",
            )
    return Draft(
        cover_letter=response.cover_letter,
        resume_notes=response.resume_notes,
        apply_checklist=checklist,
        model=model_name,
        resume_role=match.primary_role,
        resume_path=path,
    )
