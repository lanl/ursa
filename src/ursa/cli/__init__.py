import asyncio
from pathlib import Path
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP
from rich.console import Console
from typer import Option, Typer

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
                "For example 'openai:gpt-5'. "
                "See https://reference.langchain.com/python/langchain/models/?h=init_chat_model#langchain.chat_models.init_chat_model"
            ),
            envvar="URSA_LLM_NAME",
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
        Optional[str],
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
    ssl_verify_llm: Annotated[
        bool, Option(help="Whether or not to verify SSL certificates for LLM.")
    ] = True,
    ssl_verify_emb: Annotated[
        bool,
        Option(
            help="Whether or not to verify SSL certificates for embedding model."
        ),
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
        ssl_verify_llm=ssl_verify_llm,
        ssl_verify_emb=ssl_verify_emb,
    )
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
        Optional[str],
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
    ssl_verify_llm: Annotated[
        bool, Option(help="Whether or not to verify SSL certificates for LLM.")
    ] = True,
    ssl_verify_emb: Annotated[
        bool,
        Option(
            help="Whether or not to verify SSL certificates for embedding model."
        ),
    ] = True,
) -> None:
    console = Console()
    with console.status("[grey50]Starting ursa MCP server ..."):
        from ursa.cli.hitl import HITL

    app_path = "ursa.cli.hitl_mcp:mcp_http_app"

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
        ssl_verify_llm=ssl_verify_llm,
        ssl_verify_emb=ssl_verify_emb,
    )
    console.print(
        f"[bold]URSA MCP server[/bold] starting at "
        f"http://{host}:{port} "
        f"(app: {app_path})"
    )

    try:
        mcp = FastMCP(
            name="URSA Server",
            host=host,
            port=port,
            # description="URSA agents exposed as MCP tools (arxiv, plan, execute, web, recall, hypothesize, chat).",
        )

        console.print("[bold]Starting MCP Server[/bold]")

        # Each tool is a thin shim to your HITL methods. The type hints become the tool's JSON Schema.
        @mcp.tool(
            description="Search for papers on arXiv and summarize in the query context."
        )
        def arxiv(query):
            return hitl.run_arxiv(query)

        @mcp.tool(
            description="Build a step-by-step plan to solve the user's problem."
        )
        def plan(query):
            return hitl.run_planner(query)

        @mcp.tool(
            description="Execute a ReAct agent that can write/edit code & run commands."
        )
        def execute(query):
            return hitl.run_executor(query)

        @mcp.tool(
            description="Search the web and summarize results in context."
        )
        def web(query):
            result = hitl.run_websearcher(query)
            return result

        # @mcp.tool(description="Recall prior execution steps from memory (RAG).")
        # def remember(query):
        #     return hitl.run_rememberer(query)

        @mcp.tool(description="Deep reasoning to propose an approach.")
        def hypothesize(query):
            return hitl.run_hypothesizer(query)

        @mcp.tool(description="Direct chat with the hosted LLM.")
        def chat(query):
            return hitl.run_chatter(query)

        # Optional: a quick ping/health tool (some clients call this)
        @mcp.tool(description="Liveness check.")
        def ping(dummy: str = "ok") -> str:
            return "pong"

        # ---- ASGI app that serves the MCP Streamable HTTP endpoint ----
        # This is a complete ASGI app; uvicorn can serve it directly.
        asyncio.run(mcp.run_streamable_http_async())

    except KeyboardInterrupt:
        console.print("[grey50]Shutting down...[/grey50]")


def main():
    app()
