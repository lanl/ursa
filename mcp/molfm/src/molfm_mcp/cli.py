import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from mcp.server.fastmcp import FastMCP

from ursa.tools.fm_base_tool import fastmcp_add_basetool

from .smiles import SMILES

app = typer.Typer()


@app.command()
def mcp(
    models_directory: Annotated[
        Path, typer.Argument(default_factory=lambda: Path("models/"))
    ],
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "INFO",
    pubchem: Annotated[
        bool,
        typer.Option(
            help="Include tool to look up SMILES encodings on PubChem"
        ),
    ] = True,
    transport: Annotated[
        Literal["stdio", "sse", "streamable-http"],
        typer.Option(
            "--transport",
            "-t",
            case_sensitive=False,
            help="Transport to expose the MCP server on",
        ),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Host to bind for network transports (ignored for stdio)",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Port to bind for network transports (ignored for stdio)",
        ),
    ] = 8000,
    sse_mount_path: Annotated[
        str,
        typer.Option(
            "--sse-mount-path",
            help="Mount path to serve SSE transport under",
        ),
    ] = "/",
):
    """Launch an MCP-server for the MIST models in `models_directory`"""
    from .mist import MistModel

    server = FastMCP(
        "mist",
        log_level=log_level,
        host=host,
        port=port,
    )
    models: dict[str, MistModel] = {}
    assert models_directory.is_dir()
    for model_dir in models_directory.iterdir():
        if not model_dir.is_dir():
            continue
        if not model_dir.joinpath("config.json").is_file():
            logging.warning("Missing config.json in %s -> Skipping", model_dir)

        try:
            model = MistModel.from_pretrained(model_dir)
            if model.name in models:
                logging.error(
                    "Duplicate Molecular Foundation Model: %s, skipping duplicate in %s",
                    model.name,
                    model_dir,
                )
                continue
            fastmcp_add_basetool(server, model)
            models[model.name] = model
            logging.info("Loaded %s", model_dir)
        except Exception as e:
            logging.error(f"Failed to load model {model_dir}: {e}")

    if pubchem:
        from .pubchem import search_pubchem

        server.tool()(search_pubchem)

    run_kwargs = {}
    if transport == "sse":
        run_kwargs["mount_path"] = sse_mount_path

    server.run(transport=transport, **run_kwargs)


@app.command()
def search(
    name: Annotated[
        str, typer.Argument(help="Compound to search for on PubChem")
    ],
):
    from .pubchem import search_pubchem

    out = search_pubchem.invoke(name)
    out = [o.model_dump() for o in out]
    print(json.dumps(out, indent=4))


@app.command()
def predict(
    model_directory: Annotated[
        Path, typer.Argument(help="Path to model to query")
    ],
    molecules: Annotated[
        list[str], typer.Argument(help="Molecules to get properties for")
    ],
):
    """Get predicted properties for an input molecule"""
    from .mist import MistModel

    model = MistModel.from_pretrained(model_directory)
    molecules = [SMILES(mol) for mol in molecules]
    out = model.batch(molecules)
    out = [o.model_dump() for o in out]
    print(json.dumps(out, indent=4))
