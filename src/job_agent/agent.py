from __future__ import annotations

import json
import os
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from job_agent.config import load_profile
from job_agent.db import Store
from job_agent.drafter import draft_application
from job_agent.models import Job, Match
from job_agent.resumes import load_for_role, pick_role
from job_agent.retrieval import HybridRetriever, RetrievalResult, bm25_scores, job_text


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchJobsArguments(ToolArguments):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


class ExplainMatchArguments(ToolArguments):
    uid: str = Field(min_length=1, max_length=300)


class DraftCoverLetterArguments(ToolArguments):
    uid: str = Field(min_length=1, max_length=300)


class AgentResponse(BaseModel):
    conversation_id: str
    answer: str = Field(min_length=1)
    tools_used: list[str]


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    arguments: str


class ModelToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: Literal["function"]
    function: ToolCallFunction


class ModelAssistantMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"]
    content: str | None = None
    tool_calls: list[ModelToolCall] | None = None

    @model_validator(mode="after")
    def require_content_or_tools(self) -> ModelAssistantMessage:
        if not self.tool_calls and not (self.content or "").strip():
            raise ValueError("Assistant response must include content or tool calls.")
        return self


TOOL_MODELS: dict[str, type[ToolArguments]] = {
    "search_jobs": SearchJobsArguments,
    "explain_match": ExplainMatchArguments,
    "draft_cover_letter": DraftCoverLetterArguments,
}

TOOL_DESCRIPTIONS = {
    "search_jobs": "Find saved job postings relevant to a query or the candidate's skills.",
    "explain_match": "Explain the saved match score and reasons for a job UID.",
    "draft_cover_letter": "Create a truthful cover letter draft for a saved job UID. Never submit it.",
}


class AgentConfigurationError(RuntimeError):
    pass


def _function_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            },
        }
        for name, model in TOOL_MODELS.items()
    ]


class JobAgent:
    def __init__(
        self,
        store: Store,
        retriever: HybridRetriever | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        profile = load_profile()
        if store.get_profile("default") != profile:
            store.save_profile("default", profile)
        self.profile = profile

    def _search_jobs(self, query: str, limit: int) -> list[dict[str, Any]]:
        jobs = self.store.all_jobs()
        if self.retriever:
            results: list[RetrievalResult] = self.retriever.search(query, jobs, limit=limit)
        else:
            scores = bm25_scores(query, [job_text(job) for job in jobs])
            ranked = sorted(
                (
                    (job, score)
                    for job, score in zip(jobs, scores)
                    if score > 0
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            results = [
                RetrievalResult(job=job, score=score, method="bm25")
                for job, score in ranked[:limit]
            ]
        return [
            {
                "uid": result.job.uid,
                "title": result.job.title,
                "company": result.job.company,
                "location": result.job.location,
                "url": result.job.url,
                "score": round(result.score, 4),
                "method": result.method,
            }
            for result in results
        ]

    def _explain_match(self, uid: str) -> dict[str, Any]:
        row = self.store.get(uid)
        if row is None:
            return {"error": f"No saved job with UID {uid}."}
        return {
            "uid": uid,
            "title": row["title"],
            "company": row["company"],
            "score": row["score"],
            "matched_roles": json.loads(row["matched_roles"] or "[]"),
            "reasons": json.loads(row["reasons"] or "[]"),
            "status": row["status"],
        }

    def _draft_cover_letter(self, uid: str) -> dict[str, Any]:
        row = self.store.get(uid)
        if row is None:
            return {"error": f"No saved job with UID {uid}."}
        job = Job(
            source=row["source"],
            source_id=row["source_id"],
            title=row["title"],
            company=row["company"],
            location=row["location"],
            url=row["url"],
            description=row["description"],
            job_type=row["job_type"],
        )
        matched_roles = json.loads(row["matched_roles"] or "[]")
        role = pick_role(matched_roles, self.profile)
        match = Match(
            job=job,
            score=float(row["score"]),
            matched_roles=matched_roles,
            reasons=json.loads(row["reasons"] or "[]"),
            primary_role=role,
        )
        resume_file = load_for_role(role, self.profile)
        draft = draft_application(match, self.profile, resume_file.text, resume_file)
        return {
            "uid": uid,
            "cover_letter": draft.cover_letter,
            "model": draft.model,
            "resume_role": draft.resume_role,
        }

    def _invoke_tool(self, name: str, raw_arguments: str) -> dict[str, Any]:
        if name not in TOOL_MODELS:
            return {"error": f"Unknown tool: {name}"}
        try:
            arguments = TOOL_MODELS[name].model_validate_json(raw_arguments)
        except ValidationError as exc:
            return {"error": f"Invalid tool arguments: {exc}"}
        if isinstance(arguments, SearchJobsArguments):
            args = arguments
            return {"jobs": self._search_jobs(args.query, args.limit)}
        if isinstance(arguments, ExplainMatchArguments):
            args = arguments
            return self._explain_match(args.uid)
        args = arguments
        return self._draft_cover_letter(args.uid)

    def _request_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.api_key:
            raise AgentConfigurationError(
                "Set OPENAI_API_KEY to use the conversational agent."
            )
        with httpx.Client(timeout=60.0) as client:
            for attempt in range(2):
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": _function_tools(),
                        "tool_choice": "auto",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices")
                if (
                    not isinstance(choices, list)
                    or not choices
                    or not isinstance(choices[0], dict)
                ):
                    raise ValueError("Model response did not contain a chat completion choice.")
                message = choices[0].get("message")
                try:
                    return ModelAssistantMessage.model_validate(message).model_dump(
                        exclude_none=True
                    )
                except ValidationError as exc:
                    if attempt:
                        raise ValueError(
                            "Model returned an invalid assistant message twice."
                        ) from exc
                    messages.append(
                        {
                            "role": "system",
                            "content": "Your previous response failed schema validation. "
                            "Return a valid assistant message using the declared tools.",
                        }
                    )
        raise RuntimeError("Model request retry loop ended unexpectedly.")

    def chat(self, message: str, conversation_id: str | None = None) -> AgentResponse:
        if not message.strip():
            raise ValueError("Message must not be empty.")
        identifier = self.store.ensure_conversation(conversation_id)
        self.store.add_conversation_message(
            identifier, {"role": "user", "content": message}
        )
        history = [
            item
            for item in self.store.get_conversation_messages(identifier)
            if item.get("role") in {"user", "assistant"} and not item.get("tool_calls")
        ][-20:]
        profile_memory = {
            "target_roles": self.profile.get("target_roles", []),
            "skills": self.profile.get("skills", {}),
            "experience_summary": self.profile.get("experience_summary", ""),
        }
        system_message = {
            "role": "system",
            "content": (
                "You are a job-search assistant. Use tools for saved job facts; "
                "never invent match scores, skills, or job details. Never apply or "
                "submit anything. Cover letters are drafts for human review.\n"
                "Candidate profile memory:\n"
                + json.dumps(profile_memory, ensure_ascii=False)
            ),
        }
        messages: list[dict[str, Any]] = [system_message, *history]
        tools_used: list[str] = []

        for _ in range(5):
            assistant_message = self._request_model(messages)
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                answer = assistant_message.get("content")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Agent returned an empty assistant response.")
                self.store.add_conversation_message(identifier, assistant_message)
                return AgentResponse(
                    conversation_id=identifier,
                    answer=answer,
                    tools_used=tools_used,
                )

            self.store.add_conversation_message(identifier, assistant_message)
            for call in tool_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = str(function.get("arguments") or "{}")
                tool_result = self._invoke_tool(name, arguments)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                messages.append(tool_message)
                self.store.add_conversation_message(identifier, tool_message)
                if name in TOOL_MODELS:
                    tools_used.append(name)

        raise RuntimeError("Agent reached the tool-call limit without a final answer.")
