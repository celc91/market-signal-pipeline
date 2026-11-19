"""Project settings, driven by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def _path(env_name: str, default: Path) -> Path:
    value = os.getenv(env_name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Settings:
    project: str | None = field(default_factory=lambda: os.getenv("GCP_PROJECT"))
    dataset: str = field(default_factory=lambda: os.getenv("BQ_DATASET", "market_signals"))
    location: str = field(default_factory=lambda: os.getenv("BQ_LOCATION", "US"))
    raw_dir: Path = field(default_factory=lambda: _path("RAW_DIR", DATA_DIR / "raw"))
    processed_dir: Path = field(
        default_factory=lambda: _path("PROCESSED_DIR", DATA_DIR / "processed")
    )


settings = Settings()
