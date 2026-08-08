"""Shared pytest isolation fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import settings


@pytest.fixture
def runtime_database_url(
    tmp_path: Path,
) -> str:
    """Return an isolated SQLite URL for one test."""

    database_path = (
        tmp_path
        / "runtime-predictions.db"
    )

    return (
        f"sqlite:///{database_path}"
    )


@pytest.fixture(autouse=True)
def isolate_runtime_database(
    monkeypatch: pytest.MonkeyPatch,
    runtime_database_url: str,
) -> None:
    """Prevent API tests from writing synthetic events to the real DB."""

    monkeypatch.setattr(
        settings,
        "database_url",
        runtime_database_url,
    )
