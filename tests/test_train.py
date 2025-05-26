"""Test that the train functionality can correcelty load data from a CSV file."""

from pathlib import Path
from unittest import mock

import pytest

from deepmirror import api


@pytest.fixture
def csv_path(tmp_path):
    """Create a temporary CSV file for testing."""
    path = tmp_path / "data.csv"
    return path


def test_train_valid_columns(test_csv_path: Path) -> None:
    """Test training with valid columns."""
    test_csv_path.write_text("smiles,value\nCCO,1\n")
    with mock.patch("deepmirror.api.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}
        result = api.train(
            "mymodel",
            str(test_csv_path),
            "smiles",
            "value",
            False,
        )
        assert result == {"ok": True}
        payload = mock_post.call_args.kwargs["json"]
        assert payload["x"] == ["CCO"]
        assert payload["y"] == [1.0]


def test_train_missing_columns(test_csv_path: Path) -> None:
    """Test training with missing columns."""
    test_csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        api.train(
            "mymodel",
            str(test_csv_path),
            "smiles",
            "value",
            False,
        )
