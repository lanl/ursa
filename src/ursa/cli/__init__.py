from pathlib import Path
from typing import Annotated, Optional

from rich.console import Console
from typer import Option, Typer

app = Typer()


# TODO: add help
@app.command()
def run(
    workspace: Annotated[
        Path, Option(help="Directory to store ursa ouput")
    ] = Path(".ursa"),
    llm_model_name: Annotated[
        str,
        Option(
            help="Name of LLM to use for agent tasks", envvar="URSA_LLM_NAME"
        ),
    ] = "gpt-5",
    llm_base_url: Annotated[
        str, Option(help="Base url for LLM.", envvar="URSA_LLM_BASE_URL")
    ] = "https://api.openai.com/v1",
    llm_api_key: Annotated[
        Optional[str], Option(help="API key for LLM", envvar="URSA_LLM_API_KEY")
    ] = None,
    max_completion_tokens: Annotated[
        int, Option(help="Maximum tokens for LLM to output")
    ] = 50000,
    emb_model_name: Annotated[
        str, Option(help="Embedding model name", envvar="URSA_EMB_NAME")
    ] = "text-embedding-3-small",
    emb_base_url: Annotated[
        str,
        Option(help="Base url for embedding model", envvar="URSA_EMB_BASE_URL"),
    ] = "https://api.openai.com/v1",
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


def main():
    app()
