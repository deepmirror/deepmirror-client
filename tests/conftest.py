"""Fixtures for testing."""

import os
from pathlib import Path
from typing import Generator

import pytest
from pydantic import SecretStr

from deepmirror import api
from deepmirror.config import settings


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    path = tmp_path / "data.csv"
    return path


@pytest.fixture(scope="session", autouse=True)
def setup_auth() -> Generator[None, None, None]:
    """Log in with credentials and save token for testing."""

    username = os.environ.get("DM_CLIENT_EMAIL")
    password = os.environ.get("DM_CLIENT_PASSWORD")

    if not settings.TOKEN_FILE.exists():
        if not username or not password:
            raise RuntimeError("Missing DM_CLIENT_EMAIL or DM_CLIENT_PASSWORD")
        api.login(username, SecretStr(password))
    yield
    if username and password:
        settings.TOKEN_FILE.unlink(missing_ok=True)
