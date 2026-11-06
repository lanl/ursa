"""Preview, apply, resume, or restore retention for a completed Harbor job."""

import json
from pathlib import Path
from typing import Annotated

import typer

from ursa.artifacts.retention import restore_artifacts

app = typer.Typer(help=__doc__)


@app.command()
def plan(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    policy: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Print a read-only inventory and planned actions (no files are changed)."""
    from ursa.integrations.harbor_retention import (
        load_retention_config,
        plan_job_retention,
    )

    typer.echo(
        plan_job_retention(
            run_dir, load_retention_config(policy)
        ).model_dump_json(indent=2)
    )


@app.command()
def apply(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    policy: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Apply a policy, or resume its recorded cleanup after an interruption."""
    from ursa.integrations.harbor_retention import (
        apply_job_retention,
        load_retention_config,
    )

    typer.echo(
        json.dumps(
            apply_job_retention(run_dir, load_retention_config(policy)),
            indent=2,
        )
    )


@app.command()
def restore(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    destination: Annotated[Path, typer.Option()],
) -> None:
    """Extract verified archived payloads into a new directory."""
    deleted = restore_artifacts(manifest, destination)
    typer.echo(
        json.dumps({
            "destination": str(destination),
            "unrecoverable_deleted_files": deleted,
        })
    )


if __name__ == "__main__":
    app()
