import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage

from ursa.agents import ChatAgent
from ursa.cli.config import UrsaConfig
from ursa.util.mcp import start_mcp_client


async def main() -> None:
    config = UrsaConfig.from_file(Path("config.yaml")).resolve()
    agent = ChatAgent(
        llm=config.llm_model.init_chat_model(),
        workspace=Path("ursa-script-workspace"),
    )

    mcp_client = start_mcp_client(config.mcp_servers)
    tool_sources = await agent.add_mcp_tools(mcp_client)
    print("Attached MCP tools:", tool_sources)

    result = await agent.ainvoke({
        "messages": [
            HumanMessage(
                content=(
                    "Use the laboratory tools to list the available measurements, "
                    "then summarize the strongest sample and any temperature "
                    "difference that limits a direct comparison."
                )
            )
        ],
        "thread_id": agent.thread_id,
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
