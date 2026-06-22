import asyncio

from langchain.chat_models import init_chat_model
from rich import print as rprint
from rich.panel import Panel

from ursa.agents import ArxivAgent, OSTIAgent, WebSearchAgent
from ursa.util.events import configure_event_logging

configure_event_logging()


def print_summary(summary: str, title: str):
    rprint(Panel(summary, title=title))


async def main():
    # Web search (ddgs) agent
    web_agent = WebSearchAgent(
        llm=init_chat_model("openai:gpt-5.4-mini"),
        max_results=20,
        database_path="web_db",
        summaries_path="web_summaries",
        enable_metrics=True,
    )
    result = await web_agent.ainvoke({
        "query": "graph neural networks for PDEs",
        "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
    })
    print_summary(result["final_summary"], title="Web Agent Summary")

    # OSTI agent
    osti_agent = OSTIAgent(
        llm=init_chat_model("openai:gpt-5.4-mini"),
        max_results=5,
        database_path="osti_db",
        summaries_path="osti_summaries",
        enable_metrics=True,
    )
    result = await osti_agent.ainvoke({
        "query": "quantum annealing materials",
        "context": "What are the key findings?",
    })
    print_summary(result["final_summary"], title="OSTI Agent Summary")

    # ArXiv agent
    arxiv_agent = ArxivAgent(
        llm=init_chat_model("openai:gpt-5.4-mini"),
        max_results=3,
        database_path="arxiv_papers",
        summaries_path="arxiv_generated_summaries",
        enable_metrics=True,
    )
    result = await arxiv_agent.ainvoke({
        "query": "graph neural networks for PDEs",
        "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
    })
    print_summary(result["final_summary"], title="Arxiv Agent Summary")


if __name__ == "__main__":
    asyncio.run(main())
