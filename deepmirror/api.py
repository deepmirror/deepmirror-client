"""deepmirror API client for interacting with the deepmirror platform.

This module provides functions for authentication, model management, and inference
using the deepmirror API. It handles API token management and provides a clean
interface for making API requests.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests
from pydantic import SecretStr

from .config import settings


def save_token(token: SecretStr) -> None:
    """Save the API token to the config directory."""
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    settings.TOKEN_FILE.write_text(token.get_secret_value())
    settings.TOKEN_FILE.chmod(0o600)
    print(f"API token saved to {settings.TOKEN_FILE}")


def load_token() -> SecretStr:
    """Load the API token from the config directory."""
    if not settings.TOKEN_FILE.exists():
        raise RuntimeError("API token not found; please login or provide one")
    return SecretStr(settings.TOKEN_FILE.read_text().strip())


def authenticate(username: str, password: SecretStr) -> SecretStr:
    """Authenticate with the deepmirror API."""
    host = settings.HOST
    url = f"{host}/api/v3/public/authenticate?token_lifetime_minutes=720"
    data = {
        "grant_type": "password",
        "username": username,
        "password": password.get_secret_value(),
        "scope": "",
        "client_id": "string",
        "client_secret": "string",
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    response = requests.post(
        url, data=data, headers=headers, timeout=settings.API_TIMEOUT
    )
    if response.status_code != 200:
        raise RuntimeError(f"Login failed: {response.text}")
    token = response.json().get("api_token")
    if token is None:
        raise RuntimeError("No API token returned")
    return SecretStr(token)


def login(username: str, password: SecretStr) -> None:
    """Login to the deepmirror API."""
    save_token(authenticate(username, password))


def list_models() -> Any:
    """List all models."""
    token = load_token()
    url = f"{settings.HOST}/api/v3/public/models/"
    response = requests.post(
        url,
        json={"api_token": token.get_secret_value()},
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def train(
    model_name: str,
    csv_file: str,
    smiles_column: str,
    value_column: str,
    classification: bool = False,
) -> Any:
    """Train a model."""
    token = load_token()
    df = pd.read_csv(csv_file)
    if smiles_column not in df.columns or value_column not in df.columns:
        raise ValueError("CSV missing required columns")
    x = df[smiles_column].astype(str).tolist()
    y = df[value_column].astype(float).tolist()
    payload = {
        "model_name": model_name,
        "api_token": token.get_secret_value(),
        "x": x,
        "y": y,
        "is_classification": classification,
    }
    response = requests.post(
        f"{settings.HOST}/api/v3/public/train/",
        json=payload,
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def predict_hlm(
    smiles: list[str],
) -> Any:
    """Predict using HLM."""
    token = load_token()
    payload = {
        "api_token": token.get_secret_value(),
        "x": smiles,
    }

    response = requests.post(
        f"{settings.HOST}/api/v3/public/predict-hlm/",
        json=payload,
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def get_predict_hlm(
    task_id: str,
) -> Any:
    """Get predictions from HLM."""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/predict-hlm/{task_id}",
        json={"api_token": token.get_secret_value()},
        timeout=settings.API_TIMEOUT,
    )

    if response.status_code not in [200, 202]:
        raise RuntimeError(response.text)
    return response.json()


def predict(
    model_name: str,
    csv_file: str | None = None,
    smiles_column: str | None = None,
    smiles: list[str] | None = None,
) -> Any:
    """Predict using a model."""
    token = load_token()
    # Handle direct SMILES input
    if smiles is not None:
        inputs = smiles
    # Handle CSV input with pandas
    elif csv_file and csv_file.endswith(".csv") and smiles_column:
        df = pd.read_csv(csv_file)
        if smiles_column not in df.columns:
            raise ValueError(f"CSV missing required column: {smiles_column}")
        inputs = df[smiles_column].astype(str).tolist()
    else:
        raise ValueError(
            "Either csv_file & smiles_column or smiles must be provided"
        )

    payload = {
        "model_name": model_name,
        "input": inputs,
        "api_token": token.get_secret_value(),
    }
    response = requests.post(
        f"{settings.HOST}/api/v3/public/inference/",
        json=payload,
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)

    predictions = response.json()

    # Add predictions to CSV if using CSV input
    if csv_file and csv_file.endswith(".csv") and smiles_column:
        df["prediction"] = predictions["prediction"]
        df["confidence"] = predictions["confidence"]
        output_file = csv_file.replace(".csv", "_predictions.csv")
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        return {"output_file": output_file}

    # Create predictions DataFrame for text file or direct SMILES input
    results_df = pd.DataFrame(
        {
            "smiles": inputs,
            "prediction": predictions["prediction"],
            "confidence": predictions["confidence"],
        }
    )

    if csv_file:
        output_file = csv_file.rsplit(".", 1)[0] + "_predictions.csv"
        results_df.to_csv(output_file, index=False)
        return {"output_file": output_file}

    return predictions


def structure_prediction(
    chains: list[dict[str, str]],
    constraint_settings: dict[str, str | float | int] | None = None,
    model: Literal["chai", "boltz", "boltz2"] = "chai",
) -> Any:
    """Create structure prediction"""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/structure-prediction/v2/",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "chains": chains,
            "model": model,
            "api_token": token.get_secret_value(),
            "settings": constraint_settings,
        },
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def get_structure_prediction(task_id: str) -> Any:
    """Get structure prediction"""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/structure_prediction/{task_id}",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "api_token": token.get_secret_value(),
        },
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def list_structure_tasks() -> Any:
    """List all structure prediction tasks."""
    token = load_token()
    payload = {"api_token": token.get_secret_value()}
    response = requests.post(
        f"{settings.HOST}/api/v3/public/structure_prediction/get_all_tasks/",
        json=payload,
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def download_structure_prediction(
    task_id: str,
) -> bytes:
    """Download a structure prediction task."""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/structure_prediction/download/{task_id}",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "api_token": token.get_secret_value(),
        },
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return bytes(response.content)


def model_metadata(model_id: str) -> Any:
    """Get metadata for a specific model."""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/models/metadata/{model_id}",
        json={"api_token": token.get_secret_value()},
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def model_info(model_id: str) -> Any:
    """Get detailed information for a specific model."""
    token = load_token()
    response = requests.post(
        f"{settings.HOST}/api/v3/public/models/{model_id}",
        json={"api_token": token.get_secret_value()},
        timeout=settings.API_TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()


def upload_onnx_model(
    onnx_file: str,
) -> Any:
    """Upload an ONNX model to the deepmirror API."""
    token = load_token()
    with open(onnx_file, "rb") as f:
        file_content = f.read()

    file = (
        Path(onnx_file).name,
        file_content,
        "application/octet-stream",
    )

    response = requests.post(
        f"{settings.HOST}/api/v3/public/onnx-model/upload/",
        headers={
            "X-API-Key": token.get_secret_value(),
        },
        files={
            "file": file,
        },
        timeout=settings.API_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)
    return response.json()
