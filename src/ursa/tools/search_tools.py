from pathlib import Path

from langchain.tools import ToolRuntime, tool

from ursa.agents import ArxivAgent, OSTIAgent, WebSearchAgent
from ursa.util.events import ToolEvents


@tool
async def run_arxiv_search(
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
    events = ToolEvents.from_runtime("run_arxiv_search", runtime)
    try:
        agent = ArxivAgent(
            llm=runtime.context.llm,
            summarize=True,
            process_images=False,
            max_results=max_results,
            workspace=runtime.context.den,
            # rag_embedding=self.embedding,
            checkpointer=False,
            agent_name=None,
            database_path=Path("./arxiv_downloaded"),
            summaries_path=Path("./arxiv_summaries"),
            download=True,
        )
        events.emit(
            "Searching ArXiv",
            stage="search",
            query=query,
            max_results=max_results,
        )
        assert isinstance(query, str)

        arxiv_result = await agent.ainvoke(
            arxiv_search_query=query,
            context=prompt,
        )
        arxiv_result = arxiv_result["final_summary"]

        events.emit(
            "ArXiv search complete",
            stage="search_result",
            query=query,
            result_chars=len(arxiv_result),
        )
        return f"[ArXiv Agent Output]:\n {arxiv_result}"
    except Exception as e:  # noqa: BLE001
        events.emit(
            "ArXiv search failed",
            stage="search",
            phase="error",
            query=query,
            error_type=type(e).__name__,
            error=str(e),
        )
        return f"Unexpected error while running ArxivAgent: {e}"


@tool
async def run_web_search(
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
    events = ToolEvents.from_runtime("run_web_search", runtime)
    try:
        agent = WebSearchAgent(
            llm=runtime.context.llm,
            summarize=True,
            process_images=False,
            max_results=max_results,
            workspace=runtime.context.den,
            # rag_embedding=self.embedding,
            checkpointer=False,
            agent_name=None,
            database_path=Path("./web_downloads"),
            summaries_path=Path("./web_summaries"),
            download=True,
        )
        events.emit(
            "Searching Web",
            stage="search",
            query=query,
            max_results=max_results,
        )
        assert isinstance(query, str)

        web_result = await agent.ainvoke(
            query=query,
            context=prompt,
        )
        web_result = web_result["final_summary"]

        events.emit(
            "Web search complete",
            stage="search_result",
            query=query,
            result_chars=len(web_result),
        )
        return f"[Web Search Agent Output]:\n {web_result}"
    except Exception as e:  # noqa: BLE001
        events.emit(
            "Web search failed",
            stage="search",
            phase="error",
            query=query,
            error_type=type(e).__name__,
            error=str(e),
        )
        return f"Unexpected error while running WebSearchAgent: {e}"


@tool
async def run_osti_search(
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
    events = ToolEvents.from_runtime("run_osti_search", runtime)
    try:
        agent = OSTIAgent(
            llm=runtime.context.llm,
            summarize=True,
            process_images=False,
            max_results=max_results,
            workspace=runtime.context.den,
            # rag_embedding=self.embedding,
            checkpointer=False,
            agent_name=None,
            database_path=Path("./osti_downloaded_papers"),
            summaries_path=Path("./osti_generated_summaries"),
            vectorstore_path=Path("./osti_vectorstores"),
            download=True,
        )
        events.emit(
            "Searching OSTI.gov",
            stage="search",
            query=query,
            max_results=max_results,
        )
        assert isinstance(query, str)

        osti_result = await agent.ainvoke(
            query=query,
            context=prompt,
        )
        osti_result = osti_result["final_summary"]

        events.emit(
            "OSTI.gov search complete",
            stage="search_result",
            query=query,
            result_chars=len(osti_result),
        )
        return f"[OSTI Agent Output]:\n {osti_result}"
    except Exception as e:  # noqa: BLE001
        events.emit(
            "OSTI.gov search failed",
            stage="search",
            phase="error",
            query=query,
            error_type=type(e).__name__,
            error=str(e),
        )
        return f"Unexpected error while running OSTIAgent: {e}"
