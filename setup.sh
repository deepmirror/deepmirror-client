#!/bin/bash
set -e

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi


# Install python dependencies and activate venv
uv sync
. .venv/bin/activate

# Install the project in development mode
uv pip install -e ".[dev]"

pre-commit install
