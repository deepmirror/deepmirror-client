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

    if os.name == "nt":
        BASE_DIR: Path = Path(os.getenv("APPDATA", str(Path.home()))) / "Local"
    else:
        BASE_DIR: Path = Path(
            os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        )

    CONFIG_DIR: Path = BASE_DIR / "deepmirror"
    TOKEN_FILE: Path = CONFIG_DIR / "token"
    HOST_FILE: Path = CONFIG_DIR / "host"


settings = Settings()
