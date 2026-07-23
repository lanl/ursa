from typing import Annotated, Literal, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.graph.message import add_messages

from ursa.prompt_library.prompting_prompts import prompting_agent_prompt
from ursa.tools import edit_code, read_file, run_command, write_code
from ursa.tools.read_image_tool import read_image_tool
from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)

from .base import BaseAgent


class PromptingState(TypedDict, total=False):
    """State dictionary for the prompting agent.

    The agent keeps the full refinement conversation in ``messages`` and stores the
    latest proposed downstream-agent prompt in ``prompt``. ``approved`` indicates
    whether the human has accepted the current prompt.
    """

    messages: Annotated[list, add_messages]
    prompt: str
    approved: bool


_APPROVAL_TOKENS = {
    "approve",
    "approved",
    "looks good",
    "looks good to me",
    "good",
    "yes",
    "y",
    "ok",
    "okay",
    "works",
    "that works",
    "ship it",
    "use it",
    "done",
    "final",
    "confirm",
    "confirmed",
}


def _tool_line(tool: BaseTool) -> str:
    """Return a compact human-readable description of a tool."""
    description = " ".join((tool.description or "").strip().split())
    if len(description) > 220:
        description = description[:217].rstrip() + "..."
    return (
        f"- `{tool.name}`: {description}" if description else f"- `{tool.name}`"
    )


def _format_tool_group(title: str, tools: list[BaseTool]) -> str:
    """Format a list of tools as prompt context, not as callable tools."""
    if not tools:
        return f"### {title}\nNo tools are currently described."
    return f"### {title}\n" + "\n".join(_tool_line(tool) for tool in tools)


def _chat_agent_tool_references(use_web: bool) -> list[BaseTool]:
    tools = [
        run_command,
        write_code,
        edit_code,
        read_file,
        read_image_tool,
    ]
    if use_web:
        tools.extend([run_web_search, run_osti_search, run_arxiv_search])
    return tools


def _execution_agent_tool_references(
    use_web: bool,
    extra_tools: list[BaseTool] | None = None,
) -> list[BaseTool]:
    tools = [
        run_command,
        write_code,
        edit_code,
        read_file,
        read_image_tool,
    ]
    if use_web:
        tools.extend([run_web_search, run_osti_search, run_arxiv_search])
    if extra_tools:
        tools.extend(extra_tools)
    return tools


def _format_available_tool_context(
    *,
    use_web: bool,
    extra_execution_tools: list[BaseTool] | None = None,
) -> str:
    """Describe downstream ChatAgent/ExecutionAgent tools for prompt writing.

    The PromptingAgent does not bind or call these tools. This text is provided
    only so it can draft prompts that mention relevant downstream capabilities.
    """
    sections = [
        "The PromptingAgent cannot call tools directly. The following tools may "
        "be available to downstream URSA agents if the user chooses to run them:",
        _format_tool_group(
            "ChatAgent tools",
            _chat_agent_tool_references(use_web),
        ),
        _format_tool_group(
            "ExecutionAgent tools",
            _execution_agent_tool_references(use_web, extra_execution_tools),
        ),
    ]
    if not use_web:
        sections.append(
            "Web, arXiv, and OSTI search tools are not included in this tool "
            "context because web access has not been enabled."
        )
    return "\n\n".join(sections)


class PromptingAgent(BaseAgent[PromptingState]):
    """Prompting Agent

    Iterates with the user to turn an initial rough prompt into clear,
    self-contained instructions for a downstream agentic workflow. The agent
    proposes an improved prompt, accepts human feedback in subsequent turns, and
    marks the result approved when the user confirms it.
    """

    state_type = PromptingState

    def __init__(
        self,
        llm: BaseChatModel,
        use_web: bool = False,
        extra_execution_tools: list[BaseTool] | None = None,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        tool_context = _format_available_tool_context(
            use_web=use_web,
            extra_execution_tools=extra_execution_tools,
        )
        self.prompting_agent_prompt = (
            prompting_agent_prompt
            + "\n\nAvailable downstream tool context:\n\n"
            + tool_context
        )

    def format_query(
        self, prompt: str, state: PromptingState | None = None
    ) -> PromptingState:
        """Format a user message into the prompting-agent state.

        On the first turn, the user message is treated as the original prompt to
        refine. On later turns, it is treated as either approval or feedback on
        the previously proposed prompt.
        """
        if state is None:
            return PromptingState(
                messages=[HumanMessage(content=prompt)],
                approved=False,
            )

        state["messages"].append(HumanMessage(content=prompt))
        state["approved"] = False
        return state

    def format_result(self, state: PromptingState) -> str:
        """Return the current refined prompt as plain text."""
        prompt = state.get("prompt")
        if prompt:
            if state.get("approved"):
                return f"Approved prompt:\n\n{prompt}"
            return prompt
        if state.get("messages"):
            return state["messages"][-1].text
        return ""

    def route_node(self, state: PromptingState) -> PromptingState:
        """No-op entry node used to route approval before invoking the LLM."""
        return {}

    def proposal_node(
        self,
        state: PromptingState,
        config: RunnableConfig,
    ) -> PromptingState:
        """Generate or revise a downstream-agent prompt."""
        self.events(config).emit("Refining prompt", stage="refine_prompt")

        new_state, full_overwrite = self.prepare_messages_context(
            state,
            system_prompt=self.prompting_agent_prompt,
            # This agent only sees tool descriptions as prompt context. It does
            # not bind or call tools, so dangling-tool patching is unnecessary.
            patch_dangling=False,
        )

        messages = cast(list, new_state.get("messages", []))
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=self.prompting_agent_prompt)
        else:
            messages = [
                SystemMessage(content=self.prompting_agent_prompt)
            ] + messages

        response = self.llm.invoke(
            messages,
            self.build_config(tags=["prompting", "refine"]),
        )
        proposed_prompt = response.text

        return self.messages_update(
            new_state,
            [AIMessage(content=proposed_prompt)],
            full_overwrite=full_overwrite,
            extra={"prompt": proposed_prompt, "approved": False},
        )

    def approval_node(
        self,
        state: PromptingState,
        config: RunnableConfig,
    ) -> PromptingState:
        """Record human approval of the current prompt.

        Human review happens outside the graph: the user either invokes the agent
        again with feedback, causing another proposal cycle, or replies with a
        clear approval phrase, causing the existing prompt to be finalized.
        """
        prompt = state.get("prompt", "")
        self.events(config).emit("Prompt approved", stage="approve_prompt")
        return {
            "prompt": prompt,
            "approved": True,
            "messages": [AIMessage(content=f"Approved prompt:\n\n{prompt}")],
        }

    def _build_graph(self):
        self.add_node(self.route_node, "route")
        self.add_node(self.proposal_node, "propose")
        self.add_node(self.approval_node, "approve")
        self.graph.set_entry_point("route")
        self.graph.add_conditional_edges(
            "route",
            self._wrap_cond(
                _should_mark_approved,
                "should_mark_approved",
                "prompting_agent",
            ),
            {"approve": "approve", "propose": "propose"},
        )
        self.graph.add_edge("propose", END)
        self.graph.set_finish_point("approve")


def _is_approval_text(text: str) -> bool:
    normalized = " ".join(text.lower().strip().strip(".!?").split())
    if normalized in _APPROVAL_TOKENS:
        return True
    return normalized.startswith("approved") or normalized.startswith("approve")


def _should_mark_approved(
    state: PromptingState,
) -> Literal["approve", "propose"]:
    """Approve only when the latest human turn is a clear approval phrase."""
    messages = state.get("messages", [])
    latest_human = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human = message.text
            break

    if latest_human and state.get("prompt") and _is_approval_text(latest_human):
        return "approve"
    return "propose"
