from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from job_agent.models import Draft, Job, Match

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT NOT NULL,
    url TEXT NOT NULL,
    job_type TEXT NOT NULL,
    score REAL NOT NULL,
    matched_roles TEXT NOT NULL,
    reasons TEXT NOT NULL,
    description TEXT NOT NULL,
    cover_letter TEXT NOT NULL,
    resume_notes TEXT NOT NULL,
    checklist TEXT NOT NULL,
    model TEXT NOT NULL,
    resume_role TEXT NOT NULL DEFAULT '',
    resume_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    message_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(applications)")}
        if "resume_role" not in cols:
            self.conn.execute(
                "ALTER TABLE applications ADD COLUMN resume_role TEXT NOT NULL DEFAULT ''"
            )
        if "resume_path" not in cols:
            self.conn.execute(
                "ALTER TABLE applications ADD COLUMN resume_path TEXT NOT NULL DEFAULT ''"
            )

    def exists(self, uid: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM applications WHERE uid = ?", (uid,)
        ).fetchone()
        return row is not None

    def insert(self, match: Match, draft: Draft) -> None:
        job = match.job
        self.conn.execute(
            """
            INSERT OR IGNORE INTO applications (
                uid, source, source_id, title, company, location, url, job_type,
                score, matched_roles, reasons, description, cover_letter,
                resume_notes, checklist, model, resume_role, resume_path, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                job.uid,
                job.source,
                job.source_id,
                job.title,
                job.company,
                job.location,
                job.url,
                job.job_type,
                match.score,
                json.dumps(match.matched_roles),
                json.dumps(match.reasons),
                job.description,
                draft.cover_letter,
                draft.resume_notes,
                json.dumps(draft.apply_checklist),
                draft.model,
                draft.resume_role,
                draft.resume_path,
            ),
        )
        self.conn.commit()

    def list_status(self, status: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM applications
            WHERE status = ?
            ORDER BY score DESC, created_at DESC
            """,
            (status,),
        ).fetchall()

    def get(self, uid: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM applications WHERE uid = ?", (uid,)
        ).fetchone()

    def set_status(self, uid: str, status: str) -> None:
        self.conn.execute(
            """
            UPDATE applications
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE uid = ?
            """,
            (status, uid),
        )
        self.conn.commit()

    def save_profile(self, profile_id: str, profile: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO profiles (profile_id, profile_json)
            VALUES (?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (profile_id, json.dumps(profile, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT profile_json FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        return json.loads(row["profile_json"]) if row else None

    def ensure_conversation(self, conversation_id: str | None = None) -> str:
        identifier = conversation_id or str(uuid.uuid4())
        self.conn.execute(
            "INSERT OR IGNORE INTO conversations (conversation_id) VALUES (?)",
            (identifier,),
        )
        self.conn.commit()
        return identifier

    def add_conversation_message(
        self, conversation_id: str, message: dict[str, Any]
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO conversation_messages (conversation_id, message_json)
            VALUES (?, ?)
            """,
            (conversation_id, json.dumps(message, ensure_ascii=False)),
        )
        self.conn.execute(
            """
            UPDATE conversations SET updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )
        self.conn.commit()

    def get_conversation_messages(
        self, conversation_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT message_json FROM conversation_messages
            WHERE conversation_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
        return [json.loads(row["message_json"]) for row in reversed(rows)]

    def all_jobs(self) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM applications ORDER BY created_at DESC"
        ).fetchall()
        return [
            Job(
                source=row["source"],
                source_id=row["source_id"],
                title=row["title"],
                company=row["company"],
                location=row["location"],
                url=row["url"],
                description=row["description"],
                job_type=row["job_type"],
            )
            for row in rows
        ]
