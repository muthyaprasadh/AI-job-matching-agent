from __future__ import annotations

import json
import webbrowser

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from job_agent.config import DATA, OUTBOX, load_profile
from job_agent.db import Store
from job_agent.pipeline import ingest
from job_agent.resumes import ROLE_LABELS, inventory

console = Console()


def _store() -> Store:
    return Store(DATA / "jobs.sqlite")


@click.group()
def main() -> None:
    """Prepare job applications for your approval. Never auto-submits."""


@main.command()
@click.option("--limit", type=int, default=None, help="Max new drafts this run.")
def fetch(limit: int | None) -> None:
    """Pull public job boards, score matches, and queue drafts."""
    stats = ingest(limit)
    console.print(stats)


@main.command("list")
@click.option(
    "--status",
    type=click.Choice(["pending", "approved", "rejected", "applied"]),
    default="pending",
)
def list_cmd(status: str) -> None:
    rows = _store().list_status(status)
    table = Table(title=f"{status.title()} applications ({len(rows)})")
    table.add_column("uid", overflow="fold")
    table.add_column("score")
    table.add_column("type")
    table.add_column("resume")
    table.add_column("company")
    table.add_column("title")
    table.add_column("location")
    for row in rows:
        keys = row.keys()
        role = row["resume_role"] if "resume_role" in keys else ""
        table.add_row(
            row["uid"],
            f"{row['score']:.2f}",
            row["job_type"],
            role or "-",
            row["company"],
            row["title"][:60],
            row["location"][:40],
        )
    console.print(table)


@main.command()
@click.argument("uid")
def show(uid: str) -> None:
    row = _store().get(uid)
    if not row:
        raise click.ClickException("Unknown uid")
    body = f"""# {row['title']} — {row['company']}

- **UID:** `{row['uid']}`
- **Score:** {row['score']}
- **Type:** {row['job_type']}
- **Location:** {row['location']}
- **Roles:** {row['matched_roles']}
- **Why:** {row['reasons']}
- **URL:** {row['url']}
- **Model:** {row['model']}
- **Status:** {row['status']}
- **Resume:** {row['resume_role'] if 'resume_role' in row.keys() else ''} — `{row['resume_path'] if 'resume_path' in row.keys() else ''}`

## Cover letter
{row['cover_letter']}

## Resume notes
{row['resume_notes']}

## Checklist
"""
    for item in json.loads(row["checklist"] or "[]"):
        body += f"- {item}\n"
    console.print(Markdown(body))


@main.command()
@click.argument("uid")
@click.option("--open-url/--no-open-url", default=True)
def approve(uid: str, open_url: bool) -> None:
    """Mark ready for you to submit. Opens the apply page; does not submit."""
    store = _store()
    row = store.get(uid)
    if not row:
        raise click.ClickException("Unknown uid")
    store.set_status(uid, "approved")
    OUTBOX.mkdir(parents=True, exist_ok=True)
    packet = _packet_markdown(row)
    path = OUTBOX / f"{_safe(row['company'])}_{_safe(row['title'])}.md"
    path.write_text(packet, encoding="utf-8")
    console.print(f"Approved. Packet: {path}")
    if open_url and row["url"]:
        webbrowser.open(row["url"])


@main.command()
@click.argument("uid")
def reject(uid: str) -> None:
    _store().set_status(uid, "rejected")
    console.print("Rejected.")


@main.command()
@click.argument("uid")
def applied(uid: str) -> None:
    """Call this after you personally submit the application."""
    _store().set_status(uid, "applied")
    console.print("Marked applied.")


def _safe(text: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)
    return keep[:80]


def _packet_markdown(row) -> str:
    profile = load_profile()
    checklist = "\n".join(f"- [ ] {x}" for x in json.loads(row["checklist"] or "[]"))
    return f"""# Application packet (human submit only)

**Candidate:** {profile.get('full_name')} <{profile.get('email')}>
**Role:** {row['title']}
**Company:** {row['company']}
**Apply URL:** {row['url']}
**Score:** {row['score']} | **Type:** {row['job_type']} | **Location:** {row['location']}
**Resume to upload:** {row['resume_path'] if 'resume_path' in row.keys() else ''} ({row['resume_role'] if 'resume_role' in row.keys() else ''})

## Cover letter

{row['cover_letter']}

## Resume edits

{row['resume_notes']}

## Checklist

{checklist}
- [ ] I reviewed every claim against my resume
- [ ] I submitted this myself (agent does not auto-apply)
"""


@main.command("resumes")
def resumes_cmd() -> None:
    """Show which role resume files are present."""
    profile = load_profile()
    table = Table(title="Role resumes")
    table.add_column("role")
    table.add_column("label")
    table.add_column("path")
    table.add_column("status")
    for role, path, exists in inventory(profile):
        table.add_row(role, ROLE_LABELS.get(role, role), str(path), "ok" if exists else "MISSING")
    console.print(table)


@main.command()
def export_pending() -> None:
    """Write all pending packets to data/outbox for batch review."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    rows = _store().list_status("pending")
    for row in rows:
        path = OUTBOX / f"pending_{_safe(row['company'])}_{_safe(row['title'])}.md"
        path.write_text(_packet_markdown(row), encoding="utf-8")
    console.print(f"Wrote {len(rows)} files to {OUTBOX}")
