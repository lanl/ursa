from pathlib import Path

from langchain.tools import ToolRuntime, tool
from rich import get_console
from rich.panel import Panel

from ursa.agents import ArxivAgent, OSTIAgent, WebSearchAgent

console = get_console()


@tool
def run_arxiv_search(
    prompt: str, query: str, runtime: ToolRuntime, max_results: int = 3
):
    """
    Search ArXiv for the first 'max_results' papers and summarize them in the context
    of the user prompt

    Arguments:
        prompt:
            string describing the information the agent is interested in from arxiv papers
        query:
            1 and 8 word search query for the Arxiv search API to find papers relevant to the prompt
        max_results:
            integer number of papers to return (defaults 3). Request fewer if searching for something
            very specific or a larger number if broadly searching for information. Do not exceeed 10.
    """
    try:
        agent = ArxivAgent(
            llm=runtime.state.get("model"),
            summarize=True,
            process_images=False,
            max_results=max_results,
            # rag_embedding=self.embedding,
            database_path=Path("./arxiv_downloaded"),
            summaries_path=Path("./arxiv_summaries"),
            download=True,
        )
        console.print(f"[bold cyan]Searching ArXiv for: [default]{query}")
        assert isinstance(query, str)

        arxiv_result = agent.invoke(
            arxiv_search_query=query,
            context=prompt,
        )
        console.print(
            Panel(
                f"{arxiv_result}",
                title=f"[bold cyan on black]ArXiv summary for {query}",
                border_style="cyan on black",
                style="cyan on black",
            )
        )
        return f"[ArXiv Agent Output]:\n {arxiv_result}"
    except Exception as e:
        return f"Unexpected error while running ArxivAgent: {e}"


@tool
def run_web_search(
    prompt: str,
    query: str,
    runtime: ToolRuntime,
    max_results: int = 3,
):
    """
    Search the internet for the first 'max_results' pages and summarize them in the context
    of the user prompt

    Arguments:
        prompt:
            string describing the information the agent is interested in from websites
        query:
            1 and 8 word search query for the web search engines to find papers relevant to the prompt
        max_results:
            integer number of pages to return (defaults 3). Request fewer if searching for something
            very specific or a larger number if broadly searching for information. Do not exceeed 10.
    """
    try:
        agent = WebSearchAgent(
            llm=runtime.state.get("model"),
            summarize=True,
            process_images=False,
            max_results=max_results,
            # rag_embedding=self.embedding,
            database_path=Path("./web_downloads"),
            summaries_path=Path("./web_summaries"),
            download=True,
        )
        console.print(f"[bold cyan]Searching Web for: [default]{query}")
        assert isinstance(query, str)

        web_result = agent.invoke(
            query=query,
            context=prompt,
        )
        console.print(
            Panel(
                f"{web_result}",
                title=f"[bold cyan on black]Web summary for {query}",
                border_style="cyan on black",
                style="cyan on black",
            )
        )
        return f"[Web Search Agent Output]:\n {web_result}"
    except Exception as e:
        return f"Unexpected error while running WebSearchAgent: {e}"


@tool
def run_osti_search(
    prompt: str,
    query: str,
    runtime: ToolRuntime,
    max_results: int = 3,
):
    """
    Search OSTI.gov for the first 'max_results' papers and summarize them in the context
    of the user prompt

    Arguments:
        prompt:
            string describing the information the agent is interested in from arxiv papers
        query:
            1 and 8 word search query for the OSTI.gov search API to find papers relevant to the prompt
        max_results:
            integer number of papers to return (defaults 3). Request fewer if searching for something
            very specific or a larger number if broadly searching for information. Do not exceeed 10.
    """
    max_results
    try:
        agent = OSTIAgent(
            llm=runtime.state.get("model"),
            summarize=True,
            process_images=False,
            max_results=max_results,
            # rag_embedding=self.embedding,
            database_path=Path("./osti_downloaded_papers"),
            summaries_path=Path("./osti_generated_summaries"),
            vectorstore_path=Path("./osti_vectorstores"),
            download=True,
        )
        console.print(f"[bold cyan]Searching OSTI.gov for: [default]{query}")
        assert isinstance(query, str)

        osti_result = agent.invoke(
            query=query,
            context=prompt,
        )
        console.print(
            Panel(
                f"[cyan on black]{osti_result}",
                title=f"[bold cyan on black]OSTI.gov summary for {query}",
                border_style="cyan on black",
                style="cyan on black",
            )
        )
        return f"[OSTI Agent Output]:\n {osti_result}"
    except Exception as e:
        return f"Unexpected error while running OSTIAgent: {e}"
