"""Configuration for the deepmirror CLI.

This module defines the settings for the deepmirror CLI, including the API host,
and the directory where the API token is stored.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


# pylint: disable=too-few-public-methods
class Settings(BaseSettings):
    """Settings for the deepmirror CLI."""

    HOST: str = "https://api.app.deepmirror.ai"

    BASE_DIR: Path = (
        Path(os.getenv("APPDATA", str(Path.home()))) / "Local"
        if os.name == "nt"
        else Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    )

    CONFIG_DIR: Path = BASE_DIR / "deepmirror"
    TOKEN_FILE: Path = CONFIG_DIR / "token"
    HOST_FILE: Path = CONFIG_DIR / "host"
    API_TIMEOUT: int = 29


settings = Settings()
