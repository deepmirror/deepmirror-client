"""Tests that the model and batch endpoints match the current public API."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from deepmirror import api
from deepmirror.config import settings

BASE = f"{settings.HOST}/api/v3/public"
FAKE_TOKEN = MagicMock(
    return_value=Mock(get_secret_value=MagicMock(return_value="test_token"))
)


def _ok(mock: MagicMock, payload: object) -> None:
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = payload


@patch("deepmirror.api.load_token", new=FAKE_TOKEN)
@patch("deepmirror.api.httpx.get")
def test_list_models_uses_get(mock_get: MagicMock) -> None:
    """Models are listed with GET /models/."""
    _ok(mock_get, [{"model_id": 1}])
    assert api.list_models() == [{"model_id": 1}]
    assert mock_get.call_args.args[0] == f"{BASE}/models/"
    assert mock_get.call_args.kwargs["headers"] == {"X-API-Key": "test_token"}


@patch("deepmirror.api.load_token", new=FAKE_TOKEN)
@patch("deepmirror.api.httpx.get")
def test_model_info_and_metadata_use_get(mock_get: MagicMock) -> None:
    """Model info and metadata are read from the REST-style GET routes."""
    _ok(mock_get, {"model_id": 7})
    api.model_info(7)
    assert mock_get.call_args.args[0] == f"{BASE}/models/7"
    api.model_metadata(7)
    assert mock_get.call_args.args[0] == f"{BASE}/models/7/metadata"


@patch("deepmirror.api.load_token", new=FAKE_TOKEN)
@patch("deepmirror.api.httpx.put")
def test_rename_model_uses_put_name(mock_put: MagicMock) -> None:
    """Renaming sends the new name as a JSON string to PUT /models/{id}/name."""
    _ok(mock_put, {"model_id": 7, "model_name": "new"})
    assert api.rename_model(7, "new") == {"model_id": 7, "model_name": "new"}
    assert mock_put.call_args.args[0] == f"{BASE}/models/7/name"
    assert mock_put.call_args.kwargs["json"] == "new"


@patch("deepmirror.api.load_token", new=FAKE_TOKEN)
@patch("deepmirror.api.httpx.post")
def test_create_batch_inference_sends_form_fields(
    mock_post: MagicMock, tmp_path: Path
) -> None:
    """Batch jobs post model_id (and optional version) as form fields."""
    parquet = tmp_path / "input.parquet"
    parquet.write_bytes(b"data")
    _ok(mock_post, {"task_id": "abc"})

    api.create_batch_inference(7, str(parquet))
    assert mock_post.call_args.args[0] == f"{BASE}/batch-inference/"
    assert mock_post.call_args.kwargs["data"] == {"model_id": "7"}
    assert "file" in mock_post.call_args.kwargs["files"]

    api.create_batch_inference(7, str(parquet), model_version_id=3)
    assert mock_post.call_args.kwargs["data"] == {
        "model_id": "7",
        "model_version_id": "3",
    }


def test_removed_endpoints_are_gone() -> None:
    """Endpoints dropped from the public API are no longer exposed."""
    for name in ("predict_hlm", "get_predict_hlm", "deregister_model"):
        assert not hasattr(api, name)
