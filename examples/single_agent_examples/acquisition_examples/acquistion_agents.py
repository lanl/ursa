import asyncio

from langchain.chat_models import init_chat_model
from rich import print as rprint
from rich.panel import Panel

from ursa.agents import ArxivAgent, OSTIAgent, WebSearchAgent


def print_summary(summary: str, title: str):
    rprint(Panel(summary, title=title))


async def main():
    # Web search (ddgs) agent
    web_agent = WebSearchAgent(
        llm=init_chat_model("openai:gpt-5-mini"),
        max_results=20,
        database_path="web_db",
        summaries_path="web_summaries",
        enable_metrics=True,
    )
    summary = await web_agent.ainvoke({
        "query": "graph neural networks for PDEs",
        "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
    })
    print_summary(summary, title="Web Agent Summary")

    # OSTI agent
    osti_agent = OSTIAgent(
        llm=init_chat_model("openai:gpt-5-mini"),
        max_results=5,
        database_path="osti_db",
        summaries_path="osti_summaries",
        enable_metrics=True,
    )
    summary = await osti_agent.ainvoke({
        "query": "quantum annealing materials",
        "context": "What are the key findings?",
    })
    print_summary(summary, title="OSTI Agent Summary")

    # ArXiv agent
    arxiv_agent = ArxivAgent(
        llm=init_chat_model("openai:gpt-5-mini"),
        max_results=3,
        database_path="arxiv_papers",
        summaries_path="arxiv_generated_summaries",
        enable_metrics=True,
    )
    summary = await arxiv_agent.ainvoke({
        "query": "graph neural networks for PDEs",
        "context": "Summarize methods & benchmarks and potential for shock hydrodynamics",
    })
    print_summary(summary, title="Arxiv Agent Summary")


if __name__ == "__main__":
    asyncio.run(main())
