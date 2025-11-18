import importlib
import inspect
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from rich.console import Console
from typer import Exit, Option, Typer, colors, secho

from ursa.cli.config import Settings

app = Typer()


def _parameter_defaults(func) -> Dict[str, Any]:
    """Return a mapping of parameter defaults for a callable."""

    signature = inspect.signature(func)
    defaults: Dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.default is not inspect._empty:
            defaults[name] = param.default
    return defaults


def _build_settings(
    config_file: Optional[Path],
    cli_values: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Settings:
    """Load settings from YAML (if provided) and overlay CLI overrides."""

    file_values: Dict[str, Any] = {}
    if config_file is not None:
        config_path = config_file.expanduser()
        if not config_path.exists():
            secho(
                f"Config file '{config_path}' not found.",
                fg=colors.RED,
            )
            raise Exit(code=1)
        try:
            import yaml
        except ImportError as exc:
            secho(
                "PyYAML is required to load configuration files. "
                "Install with: pip install pyyaml",
                fg=colors.RED,
            )
            raise Exit(code=1) from exc

        try:
            loaded = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            secho(
                f"Failed to read config file '{config_path}': {exc}",
                fg=colors.RED,
            )
            raise Exit(code=1) from exc

        if loaded is None:
            file_values = {}
        elif isinstance(loaded, dict):
            file_values = loaded
        else:
            secho(
                "The YAML configuration must contain a top-level mapping.",
                fg=colors.RED,
            )
            raise Exit(code=1)

    settings = Settings(**file_values)
    overrides = {
        key: value
        for key, value in cli_values.items()
        if key not in defaults or value != defaults[key]
    }
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings


@app.command(help="Start ursa REPL")
def run(
    workspace: Annotated[
        Path, Option(help="Directory to store ursa ouput")
    ] = Path("ursa_workspace"),
    config_file: Annotated[
        Optional[Path],
        Option(
            "--config",
            "-c",
            help="Path to YAML settings file.",
            envvar="URSA_CONFIG_FILE",
        ),
    ] = None,
    llm_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and name of LLM to use for agent tasks. "
                "Use format <provider>:<model-name>. "
                "For example 'openai:gpt-5'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
            envvar="URSA_LLM_NAME",
        ),
    ] = "openai:gpt-5",
    llm_base_url: Annotated[
        str, Option(help="Base url for LLM.", envvar="URSA_LLM_BASE_URL")
    ] = None,
    llm_api_key: Annotated[
        Optional[str],
        Option(help="API key for LLM", envvar="URSA_LLM_API_KEY"),
    ] = None,
    max_completion_tokens: Annotated[
        int, Option(help="Maximum tokens for LLM to output")
    ] = 50000,
    emb_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and Embedding model name. "
                "Use format <provider>:<embedding-model-name>. "
                "For example, 'openai:text-embedding-3-small'. "
                "See: https://reference.langchain.com/python/langchain/embeddings/?h=init_embeddings#langchain.embeddings.init_embeddings"
            ),
            envvar="URSA_EMB_NAME",
        ),
    ] = None,
    emb_base_url: Annotated[
        Optional[str],
        Option(help="Base url for embedding model", envvar="URSA_EMB_BASE_URL"),
    ] = None,
    emb_api_key: Annotated[
        Optional[str],
        Option(help="API key for embedding model", envvar="URSA_EMB_API_KEY"),
    ] = None,
    share_key: Annotated[
        bool,
        Option(
            help=(
                "Whether or not the LLM and embedding model share the same "
                "API key. If yes, then you can specify only one of them."
            )
        ),
    ] = False,
    thread_id: Annotated[
        str,
        Option(help="Thread ID for persistance", envvar="URSA_THREAD_ID"),
    ] = "ursa_cli",
    safe_codes: Annotated[
        list[str],
        Option(
            help="Programming languages that the execution agent can trust by default.",
            envvar="URSA_THREAD_ID",
        ),
    ] = ["python", "julia"],
    arxiv_summarize: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to summarize response."
        ),
    ] = True,
    arxiv_process_images: Annotated[
        bool,
        Option(help="Whether or not to allow ArxivAgent to process images."),
    ] = False,
    arxiv_max_results: Annotated[
        int,
        Option(
            help="Maximum number of results for ArxivAgent to retrieve from ArXiv."
        ),
    ] = 10,
    arxiv_database_path: Annotated[
        Optional[Path],
        Option(
            help="Path to download/downloaded ArXiv documents; used by ArxivAgent."
        ),
    ] = None,
    arxiv_summaries_path: Annotated[
        Optional[Path],
        Option(help="Path to store ArXiv paper summaries; used by ArxivAgent."),
    ] = None,
    arxiv_vectorstore_path: Annotated[
        Optional[Path],
        Option(
            help="Path to store ArXiv paper vector store; used by ArxivAgent."
        ),
    ] = None,
    arxiv_download_papers: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to download ArXiv papers."
        ),
    ] = True,
    ssl_verify: Annotated[
        bool, Option(help="Whether or not to verify SSL certificates.")
    ] = True,
) -> None:
    console = Console()
    with console.status("[grey50]Loading ursa ..."):
        from ursa.cli.hitl import HITL, UrsaRepl

    cli_values = {
        "workspace": workspace,
        "llm_model_name": llm_model_name,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "max_completion_tokens": max_completion_tokens,
        "emb_model_name": emb_model_name,
        "emb_base_url": emb_base_url,
        "emb_api_key": emb_api_key,
        "share_key": share_key,
        "safe_codes": safe_codes,
        "thread_id": thread_id,
        "arxiv_summarize": arxiv_summarize,
        "arxiv_process_images": arxiv_process_images,
        "arxiv_max_results": arxiv_max_results,
        "arxiv_database_path": arxiv_database_path,
        "arxiv_summaries_path": arxiv_summaries_path,
        "arxiv_vectorstore_path": arxiv_vectorstore_path,
        "arxiv_download_papers": arxiv_download_papers,
        "ssl_verify": ssl_verify,
    }

    settings = _build_settings(
        config_file=config_file,
        cli_values=cli_values,
        defaults=_parameter_defaults(run),
    )
    hitl = HITL.from_settings(settings)
    UrsaRepl(hitl).run()


@app.command()
def version() -> None:
    from ursa import __version__

    print(__version__)


@app.command(help="Start MCP server to serve ursa agents")
def serve(
    host: Annotated[
        str,
        Option("--host", help="Bind address.", envvar="URSA_HOST"),
    ] = "127.0.0.1",
    config_file: Annotated[
        Optional[Path],
        Option(
            "--config",
            "-c",
            help="Path to YAML settings file.",
            envvar="URSA_CONFIG_FILE",
        ),
    ] = None,
    port: Annotated[
        int,
        Option("--port", "-p", help="Bind port.", envvar="URSA_PORT"),
    ] = 8000,
    reload: Annotated[
        bool,
        Option("--reload/--no-reload", help="Auto-reload on code changes."),
    ] = False,
    log_level: Annotated[
        str,
        Option(
            "--log-level",
            "-l",
            help="Uvicorn log level: critical|error|warning|info|debug|trace",
            envvar="URSA_LOG_LEVEL",
        ),
    ] = "info",
    workspace: Annotated[
        Path, Option(help="Directory to store ursa ouput")
    ] = Path("ursa_mcp"),
    llm_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and name of LLM to use for agent tasks. "
                "Use format <provider>:<model-name>. "
                "For example 'openai:gpt-5'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
        ),
    ] = "openai:gpt-5",
    llm_base_url: Annotated[
        Optional[str],
        Option(help="Base url for LLM.", envvar="URSA_LLM_BASE_URL"),
    ] = None,
    llm_api_key: Annotated[
        Optional[str],
        Option(help="API key for LLM", envvar="URSA_LLM_API_KEY"),
    ] = None,
    max_completion_tokens: Annotated[
        int, Option(help="Maximum tokens for LLM to output")
    ] = 50000,
    emb_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and Embedding model name. "
                "Use format <provider>:<embedding-model-name>. "
                "For example, 'openai:text-embedding-3-small'. "
                "See: https://reference.langchain.com/python/langchain/embeddings/?h=init_embeddings#langchain.embeddings.init_embeddings"
            ),
            envvar="URSA_EMB_NAME",
        ),
    ] = None,
    emb_base_url: Annotated[
        Optional[str],
        Option(help="Base url for embedding model", envvar="URSA_EMB_BASE_URL"),
    ] = None,
    emb_api_key: Annotated[
        Optional[str],
        Option(help="API key for embedding model", envvar="URSA_EMB_API_KEY"),
    ] = None,
    share_key: Annotated[
        bool,
        Option(
            help=(
                "Whether or not the LLM and embedding model share the same "
                "API key. If yes, then you can specify only one of them."
            )
        ),
    ] = False,
    thread_id: Annotated[
        str,
        Option(help="Thread ID for persistance", envvar="URSA_THREAD_ID"),
    ] = "ursa_mcp",
    safe_codes: Annotated[
        list[str],
        Option(
            help="Programming languages that the execution agent can trust by default.",
            envvar="URSA_THREAD_ID",
        ),
    ] = ["python", "julia"],
    arxiv_summarize: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to summarize response."
        ),
    ] = True,
    arxiv_process_images: Annotated[
        bool,
        Option(help="Whether or not to allow ArxivAgent to process images."),
    ] = False,
    arxiv_max_results: Annotated[
        int,
        Option(
            help="Maximum number of results for ArxivAgent to retrieve from ArXiv."
        ),
    ] = 10,
    arxiv_database_path: Annotated[
        Optional[Path],
        Option(
            help="Path to download/downloaded ArXiv documents; used by ArxivAgent."
        ),
    ] = None,
    arxiv_summaries_path: Annotated[
        Optional[Path],
        Option(help="Path to store ArXiv paper summaries; used by ArxivAgent."),
    ] = None,
    arxiv_vectorstore_path: Annotated[
        Optional[Path],
        Option(
            help="Path to store ArXiv paper vector store; used by ArxivAgent."
        ),
    ] = None,
    arxiv_download_papers: Annotated[
        bool,
        Option(
            help="Whether or not to allow ArxivAgent to download ArXiv papers."
        ),
    ] = True,
    ssl_verify: Annotated[
        bool, Option(help="Whether or not to verify SSL certificates.")
    ] = True,
) -> None:
    console = Console()
    with console.status("[grey50]Starting ursa MCP server ..."):
        from ursa.cli.hitl import HITL

    app_path = "ursa.cli.hitl_api:mcp_app"

    try:
        import uvicorn
    except Exception as e:
        secho(
            f"Uvicorn is required for 'ursa serve'. Install with: pip install uvicorn[standard]\n{e}",
            fg=colors.RED,
        )
        raise Exit(code=1)

    cli_values = {
        "workspace": workspace,
        "llm_model_name": llm_model_name,
        "llm_base_url": llm_base_url,
        "llm_api_key": llm_api_key,
        "max_completion_tokens": max_completion_tokens,
        "emb_model_name": emb_model_name,
        "emb_base_url": emb_base_url,
        "emb_api_key": emb_api_key,
        "share_key": share_key,
        "safe_codes": safe_codes,
        "thread_id": thread_id,
        "arxiv_summarize": arxiv_summarize,
        "arxiv_process_images": arxiv_process_images,
        "arxiv_max_results": arxiv_max_results,
        "arxiv_database_path": arxiv_database_path,
        "arxiv_summaries_path": arxiv_summaries_path,
        "arxiv_vectorstore_path": arxiv_vectorstore_path,
        "arxiv_download_papers": arxiv_download_papers,
        "ssl_verify": ssl_verify,
    }

    settings = _build_settings(
        config_file=config_file,
        cli_values=cli_values,
        defaults=_parameter_defaults(serve),
    )
    hitl = HITL.from_settings(settings)
    module_name, var_name = app_path.split(":")
    mod = importlib.import_module(module_name)
    asgi_app = getattr(mod, var_name)
    asgi_app.state.hitl = hitl

    config = uvicorn.Config(
        app=asgi_app,
        host=host,
        port=port,
        reload=reload,
        workers=1,
        log_level=log_level.lower(),
    )

    server = uvicorn.Server(config)
    console.print(
        f"[bold]URSA MCP server[/bold] starting at "
        f"http://{host}:{port} "
        f"(app: {app_path})"
    )
    try:
        server.run()
    except KeyboardInterrupt:
        console.print("[grey50]Shutting down...[/grey50]")


def main():
    app()
