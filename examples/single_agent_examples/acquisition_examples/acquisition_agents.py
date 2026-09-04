"""Compare URSA's web, OSTI, and arXiv acquisition agents."""

import asyncio
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel

from ursa.agents import ArxivAgent, OSTIAgent, WebSearchAgent
from ursa.cli.config import UrsaConfig
from ursa.util.events import configure_event_logging

QUERY = "graph neural networks for partial differential equations"
CONTEXT = (
    "Compare methods and benchmarks, emphasizing possible applications to "
    "shock hydrodynamics. Identify which claims come from each source."
)

configure_event_logging()


def print_summary(summary: str, title: str) -> None:
    """Render one agent's aggregate summary."""
    rprint(Panel(summary, title=title))


async def main() -> None:
    """Run the same research question through all three source types."""
    config = UrsaConfig.from_file(Path("config.yaml")).resolve()
    model = config.llm_model.init_chat_model()

    web_agent = WebSearchAgent(
        llm=model,
        max_results=5,
        database_path="web_db",
        summaries_path="web_summaries",
        enable_metrics=True,
    )
    result = await web_agent.ainvoke({"query": QUERY, "context": CONTEXT})
    print_summary(result["final_summary"], title="Web summary")

    osti_agent = OSTIAgent(
        llm=model,
        max_results=5,
        database_path="osti_db",
        summaries_path="osti_summaries",
        enable_metrics=True,
    )
    result = await osti_agent.ainvoke({"query": QUERY, "context": CONTEXT})
    print_summary(result["final_summary"], title="OSTI summary")

    arxiv_agent = ArxivAgent(
        llm=model,
        max_results=3,
        database_path="arxiv_papers",
        summaries_path="arxiv_generated_summaries",
        enable_metrics=True,
    )
    result = await arxiv_agent.ainvoke({"query": QUERY, "context": CONTEXT})
    print_summary(result["final_summary"], title="arXiv summary")


if __name__ == "__main__":
    asyncio.run(main())
