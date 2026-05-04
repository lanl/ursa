import typer

from ursa.util.http import inject_truststore_into_ssl

app = typer.Typer(help="Ursa Dashboard Runner")


@app.command()
def main(
    host: str = typer.Option("127.0.0.1", help="The interface to bind to."),
    port: int = typer.Option(8080, help="The port to bind to."),
    group: str = typer.Option("default", help="Agent group to use."),
):
    """Launch the Ursa Web Dashboard."""
    inject_truststore_into_ssl()
    try:
        import uvicorn

        import os

        os.environ["URSA_DASHBOARD_GROUP"] = str(group or "default")
        uvicorn.run(
            "ursa_dashboard.app:create_app", factory=True, host=host, port=port
        )
    except ImportError:
        print("Error: Dashboard dependencies not found.")
        print("Please install them with: pip install 'ursa-ai[dashboard]'")
        return


if __name__ == "__main__":
    app()
