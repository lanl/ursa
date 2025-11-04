import importlib
from pathlib import Path
from typing import Annotated, Optional

from rich.console import Console
from typer import Exit, Option, Typer, colors, secho

app = Typer()


@app.command(help="Start ursa REPL")
def run(
    workspace: Annotated[
        Path, Option(help="Directory to store ursa ouput")
    ] = Path("ursa_workspace"),
    llm_model_name: Annotated[
        str,
        Option(
            help=(
                "Model provider and name of LLM to use for agent tasks. "
                "Use format <provider>:<model-name>. "
                "For example 'openai:gpt-5-mini'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
            envvar="URSA_LLM_NAME",
        ),
    ] = "openai:gpt-5-mini",
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
    ] = "openai:text-embedding-3-small",
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

    hitl = HITL(
        workspace=workspace,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        max_completion_tokens=max_completion_tokens,
        emb_model_name=emb_model_name,
        emb_base_url=emb_base_url,
        emb_api_key=emb_api_key,
        share_key=share_key,
        thread_id=thread_id,
        safe_codes=safe_codes,
        arxiv_summarize=arxiv_summarize,
        arxiv_process_images=arxiv_process_images,
        arxiv_max_results=arxiv_max_results,
        arxiv_database_path=arxiv_database_path,
        arxiv_summaries_path=arxiv_summaries_path,
        arxiv_vectorstore_path=arxiv_vectorstore_path,
        arxiv_download_papers=arxiv_download_papers,
        ssl_verify=ssl_verify,
    )
    UrsaRepl(hitl).run()


@app.command()
def version() -> None:
    from importlib.metadata import version as get_version

    print(get_version("ursa-ai"))


@app.command(help="Start MCP server to serve ursa agents")
def serve(
    host: Annotated[
        str,
        Option("--host", help="Bind address.", envvar="URSA_HOST"),
    ] = "127.0.0.1",
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
                "For example 'openai:gpt-5-mini'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
        ),
    ] = "openai:gpt-5-mini",
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
    ] = "openai:text-embedding-3-small",
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

    hitl = HITL(
        workspace=workspace,
        llm_model_name=llm_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        max_completion_tokens=max_completion_tokens,
        emb_model_name=emb_model_name,
        emb_base_url=emb_base_url,
        emb_api_key=emb_api_key,
        share_key=share_key,
        thread_id=thread_id,
        safe_codes=safe_codes,
        arxiv_summarize=arxiv_summarize,
        arxiv_process_images=arxiv_process_images,
        arxiv_max_results=arxiv_max_results,
        arxiv_database_path=arxiv_database_path,
        arxiv_summaries_path=arxiv_summaries_path,
        arxiv_vectorstore_path=arxiv_vectorstore_path,
        arxiv_download_papers=arxiv_download_papers,
        ssl_verify=ssl_verify,
    )
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
