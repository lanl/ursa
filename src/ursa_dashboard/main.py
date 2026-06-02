from pathlib import Path

import typer

from ursa.util.http import inject_truststore_into_ssl

app = typer.Typer(help="Ursa Dashboard Runner")


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", help="The interface to bind to."),
    port: int = typer.Option(8080, help="The port to bind to."),
    group: str = typer.Option("default", help="Agent group to use."),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="YAML/JSON URSA config whose llm_model settings initialize the dashboard LLM endpoint.",
    ),
):
    """Launch the Ursa Web Dashboard."""
    inject_truststore_into_ssl()
    try:
        import os

        import uvicorn

        os.environ["URSA_DASHBOARD_GROUP"] = str(group or "default")
        if config is not None:
            os.environ["URSA_DASHBOARD_CONFIG"] = str(config)
        else:
            os.environ.pop("URSA_DASHBOARD_CONFIG", None)
        uvicorn.run(
            "ursa_dashboard.app:create_app", factory=True, host=host, port=port
        )
    except ImportError:
        print("Error: Dashboard dependencies not found.")
        print("Please install them with: pip install 'ursa-ai[dashboard]'")
        return


if __name__ == "__main__":
    app()
