from importlib.metadata import version as get_version
from pathlib import Path
from typing import Optional

from typer import Typer

from ursa.cli.hitl import HITL, UrsaRepl

# from ursa.cli.hitl import app as hitl_app


app = Typer()


@app.command()
def run(
    workspace: Path = Path(".ursa"),
    llm_model_name: str = "gpt-5",
    llm_base_url: str = "https://api.openai.com/v1",
    llm_api_key: Optional[str] = None,
    max_completion_tokens: int = 50000,
    emb_model_name: str = "text-embedding-3-small",
    emb_base_url: str = "https://api.openai.com/v1",
    emb_api_key: Optional[str] = None,
    share_key: bool = False,
    arxiv_summarize: bool = True,
    arxiv_process_images: bool = False,
    arxiv_max_results: int = 10,
    arxiv_database_path: Optional[Path] = None,
    arxiv_summaries_path: Optional[Path] = None,
    arxiv_vectorstore_path: Optional[Path] = None,
    arxiv_download_papers: bool = True,
) -> None:
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
    )
    UrsaRepl(hitl).run()


@app.command()
def version() -> None:
    print(get_version("ursa-ai"))


def main():
    app()
