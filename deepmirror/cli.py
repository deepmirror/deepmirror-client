"""Command line interface for interacting with the deepmirror API.

This module provides a CLI for authenticating, managing models, and making
predictions using the deepmirror platform. It wraps the API client functionality
in an easy-to-use command line tool.
"""

import getpass
import json

import click
from pydantic import SecretStr

from . import api


@click.group()
def cli() -> None:
    """Interact with the deepmirror public API."""


@cli.command()
@click.argument("username", required=True)
def login(username: str) -> None:
    """Authenticate and obtain an API token."""
    if not username:
        username = input("Email: ")
    password = getpass.getpass("Password: ")
    try:
        token = api.authenticate(username, SecretStr(password))
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if not token:
        raise click.ClickException("Login failed")
    api.save_token(token)


@cli.group()
def model() -> None:
    """Model operations."""


@model.command("list")
def model_list() -> None:
    """List available models for inference."""
    try:
        data = api.list_models()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@model.command()
@click.argument("model_id")
def metadata(model_id: str) -> None:
    """Get metadata for a specific model."""
    try:
        data = api.model_metadata(model_id)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@model.command()
@click.argument("model_id")
def info(model_id: str) -> None:
    """Get detailed information for a specific model."""
    try:
        data = api.model_info(model_id)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@cli.command()
@click.option("--model-name", required=True)
@click.option("--csv-file", type=click.Path(exists=True), required=True)
@click.option("--smiles-column", default="smiles", show_default=True)
@click.option("--value-column", default="target", show_default=True)
@click.option("--classification", is_flag=True, default=False)
def train(
    model_name: str,
    csv_file: str,
    smiles_column: str,
    value_column: str,
    classification: bool,
) -> None:
    """Train a custom model from a CSV file."""
    try:
        data = api.train(
            model_name,
            csv_file,
            smiles_column,
            value_column,
            classification,
        )
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@cli.command()
@click.option("--model-name", required=True)
@click.option("--input-file", type=click.Path(exists=True))
@click.option(
    "--input-header",
    default="smiles",
    help="Column name for SMILES in CSV input",
)
@click.option("--input-smiles", multiple=True, help="Direct SMILES input")
def predict(
    model_name: str,
    csv_file: str | None,
    smiles_column: str | None,
    smiles: tuple[str, ...] | None,
) -> None:
    """Run prediction using a trained model.

    Takes input as either a CSV file with SMILES column, a text file with SMILES per line,
    or direct SMILES strings via --input-smiles.
    """
    if not csv_file and not smiles:
        raise click.UsageError("Either --csv-file or --smiles must be provided")

    try:
        data = api.predict(
            model_name,
            csv_file=csv_file,
            smiles_column=smiles_column,
            smiles=list(smiles) if smiles else None,
        )
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@cli.group()
def structure() -> None:
    """Structure prediction operations."""


@structure.command("list")
def structure_list() -> None:
    """List submitted structure prediction tasks."""
    try:
        data = api.list_structure_tasks()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(data, indent=2))


@structure.command("download")
@click.argument("task_id")
@click.argument("output_file", type=click.Path())
def structure_download(task_id: str, output_file: str) -> None:
    """Download structure prediction results."""
    try:
        data = api.download_structure_prediction(task_id)
        with open(output_file, "wb") as f:
            f.write(data)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Downloaded structure prediction to {output_file}")


if __name__ == "__main__":
    cli()
