"""Test that the train functionality can correctly load data from a CSV file."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from deepmirror import api


@patch(
    "deepmirror.api.load_token",
    new=MagicMock(
        return_value=Mock(get_secret_value=MagicMock(return_value="test_token"))
    ),
)
@patch("deepmirror.api.httpx.post")
def test_train_valid_columns(mock_post: MagicMock, csv_path: Path) -> None:
    """Test training with valid columns."""
    csv_path.write_text("smiles,value\nCCO,1\n")

    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"ok": True}
    result = api.train(
        "mymodel",
        str(csv_path),
        "smiles",
        "value",
        False,
    )
    assert result == {"ok": True}

    headers = mock_post.call_args.kwargs["headers"]
    assert headers == {"X-API-Key": "test_token"}

    payload = mock_post.call_args.kwargs["json"]
    assert payload == {
        "model_name": "mymodel",
        "x": ["CCO"],
        "y": [1.0],
        "is_classification": False,
    }


def test_train_missing_columns(csv_path: Path) -> None:  # pylint: disable=redefined-outer-name
    """Test training with missing columns."""
    csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError) as exc:
        api.train(
            "mymodel",
            str(csv_path),
            "smiles",
            "value",
            False,
        )

    assert str(exc.value) == "CSV missing required columns"
