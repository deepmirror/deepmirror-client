"""Test that the train functionality can correctly load data from a CSV file."""

from pathlib import Path
from unittest import mock

import pytest

from deepmirror import api


def test_train_valid_columns(csv_path: Path) -> None:  # pylint: disable=redefined-outer-name
    """Test training with valid columns."""
    csv_path.write_text("smiles,value\nCCO,1\n")
    with mock.patch("deepmirror.api.requests.post") as mock_post:
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
        payload = mock_post.call_args.kwargs["json"]
        assert payload["x"] == ["CCO"]
        assert payload["y"] == [1.0]


def test_train_missing_columns(csv_path: Path) -> None:  # pylint: disable=redefined-outer-name
    """Test training with missing columns."""
    csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        api.train(
            "mymodel",
            str(csv_path),
            "smiles",
            "value",
            False,
        )
