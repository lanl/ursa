"""Settings and tools for workflows that create their agents lazily."""

from typing import Any

from langchain_core.tools import BaseTool

from ursa.agents.base import AgentWithTools


class WorkflowToolsMixin:
    def __init__(
        self,
        *,
        use_web: bool = False,
        extra_tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        self._agent_kwargs = {
            key: kwargs[key]
            for key in (
                "enable_metrics",
                "metrics_dir",
                "autosave_metrics",
                "otel_metrics",
                "thread_id",
                "tokens_before_summarize",
                "messages_to_keep",
                "max_single_tool_message_tokens",
                "rag_tools",
                "rag_tool_group",
                "rag_tool_embedding",
                "rag_tool_return_k",
            )
            if key in kwargs
        }
        self._tool_agent_kwargs = {**self._agent_kwargs, "use_web": use_web}
        self._extra_tools: dict[str, BaseTool] = {}
        self.add_tool(extra_tools or [])
        super().__init__(**kwargs)

    def add_tool(self, tools: BaseTool | list[BaseTool]) -> None:
        """Retain tools for agents created during subsequent runs."""
        bundle = tools if isinstance(tools, list) else [tools]
        self._extra_tools.update({tool.name: tool for tool in bundle})

    async def add_mcp_tools(self, client, tool_name=None) -> dict[str, str]:
        # Reuse discovery/filtering and its event-loop boundary; it delivers
        # tools through add_tool, which buffers them until children exist.
        return await AgentWithTools.add_mcp_tools(self, client, tool_name)
