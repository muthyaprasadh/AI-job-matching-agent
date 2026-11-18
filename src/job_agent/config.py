from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTBOX = DATA / "outbox"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings() -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    return load_yaml(ROOT / "config.yaml")


def load_profile() -> dict[str, Any]:
    return load_yaml(DATA / "profile.yaml")


def load_resume() -> str:
    return (DATA / "resume.md").read_text(encoding="utf-8")
