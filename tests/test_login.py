"""Tests for the login command's two-factor authentication handling."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from deepmirror.cli import cli


@pytest.mark.parametrize("test_status", [401, 403])
@patch("deepmirror.cli.getpass.getpass")
@patch("deepmirror.cli.api")
def test_login_prompts_for_otp(
    mock_api: MagicMock, mock_getpass: MagicMock, test_status: int
) -> None:
    """A rejected fresh token prompts for OTP and saves the verified token."""
    mock_api.authenticate.return_value = SecretStr("token")
    mock_api.test_response_code.return_value = test_status
    mock_api.verify_otp.return_value = SecretStr("otp-token")
    mock_getpass.side_effect = ["password", "123456"]

    result = CliRunner().invoke(cli, ["login", "user@example.com"])

    assert result.exit_code == 0
    mock_api.verify_otp.assert_called_once()
    assert mock_api.save_token.call_args.args[0].get_secret_value() == (
        "otp-token"
    )


@patch("deepmirror.cli.getpass.getpass")
@patch("deepmirror.cli.api")
def test_login_skips_otp_when_logged_in(
    mock_api: MagicMock, mock_getpass: MagicMock
) -> None:
    """A 200 from the test call saves the token without OTP verification."""
    mock_api.authenticate.return_value = SecretStr("token")
    mock_api.test_response_code.return_value = 200
    mock_getpass.return_value = "password"

    result = CliRunner().invoke(cli, ["login", "user@example.com"])

    assert result.exit_code == 0
    mock_api.verify_otp.assert_not_called()
    assert mock_api.save_token.call_args.args[0].get_secret_value() == "token"
