"""Execution agent that builds a tool-enabled state graph to autonomously run tasks.

This module implements ExecutionAgent, a LangGraph-based agent that executes user
instructions by invoking LLM tool calls and coordinating a controlled workflow.

Key features:
- Workspace management with optional symlinking for external sources.
- Safety-checked shell execution via run_command with output size budgeting.
- Code authoring and edits through write_code and edit_code.
- Web search capability through DuckDuckGoSearchResults.
- Summarization of the session and optional memory logging.
- Configurable graph with nodes for agent, action, and summarize.

Implementation notes:
- LLM prompts are sourced from prompt_library.execution_prompts.
- Outputs from subprocess are trimmed under MAX_TOOL_MSG_CHARS to fit tool messages.
- The agent uses ToolNode and LangGraph StateGraph to loop until no tool calls remain.
- Safety gates block unsafe shell commands and surface the rationale to the user.

Environment:
- MAX_TOOL_MSG_CHARS caps combined stdout/stderr in tool responses.

Entry points:
- ExecutionAgent._invoke(...) runs the compiled graph.
- main() shows a minimal demo that writes and runs a script.
"""

# from langchain_core.runnables.graph import MermaidDrawMethod
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    TypedDict,
)

from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from ursa.agents.base import AgentContext, AgentWithTools, BaseAgent
from ursa.prompt_library.execution_prompts import (
    executor_prompt,
    get_review_prompt,
    recap_prompt,
)
from ursa.tools import (
    edit_code,
    edit_experience,
    list_experiences,
    read_experience,
    read_file,
    run_command,
    write_code,
    write_experience,
)
from ursa.tools.read_image_tool import read_image_tool
from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)
from ursa.util.memory_logger import AgentMemory


# Classes for typing
class ReviewAssessment(TypedDict):
    """Structured assessment of whether execution has addressed the request."""

    is_complete: bool
    reason: str


class ExecutionState(TypedDict):
    """TypedDict representing the execution agent's mutable run state used by nodes.

    Fields:
    - messages: list of messages (System/Human/AI/Tool).
    - symlinkdir: optional dict describing a symlink operation (source, dest,
      is_linked).
    - review: optional structured review of whether the work is complete.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    symlinkdir: dict
    review: NotRequired[ReviewAssessment]


def should_continue(state: ExecutionState) -> Literal["review", "continue"]:
    """Return 'review' if no tool calls in the last message, else 'continue'.

    Args:
        state: The current execution state containing messages.

    Returns:
        A literal "review" if the last message has no tool calls,
        otherwise "continue".
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then review the work before finishing.
    if not last_message.tool_calls:
        return "review"
    # Otherwise if there is, we continue.
    else:
        return "continue"


def review_complete(state: ExecutionState) -> Literal["recap", "continue"]:
    """Route to recap when review passes, otherwise continue execution."""
    review = state.get("review") or {}
    if review.get("is_complete"):
        return "recap"
    return "continue"


def command_safe(state: ExecutionState) -> Literal["safe", "unsafe"]:
    """Return 'safe' if the last command was safe, otherwise 'unsafe'.

    Args:
        state: The current execution state containing messages and tool calls.
    Returns:
        A literal "safe" if no '[UNSAFE]' tags are in the last command,
        otherwise "unsafe".
    """
    index = -1
    message = state["messages"][index]
    # Loop through all the consecutive tool messages in reverse order
    while isinstance(message, ToolMessage):
        if "[UNSAFE]" in message.text:
            return "unsafe"

        index -= 1
        message = state["messages"][index]

    return "safe"


# Main module class
class ExecutionAgent(AgentWithTools, BaseAgent[ExecutionState]):
    """Orchestrates model-driven code execution, tool calls, and state management.

    Orchestrates model-driven code execution, tool calls, and state management for
    iterative program synthesis and shell interaction.

    This agent wraps an LLM with a small execution graph that alternates
    between issuing model queries, invoking tools (read, run, write, edit, search),
    performing safety checks, and summarizing progress. It manages a
    workspace on disk, optional symlinks, and an optional memory backend to
    persist summaries.

    Args:
        llm (BaseChatModel): Model identifier or bound chat model
            instance. If a string is provided, the BaseAgent initializer will
            resolve it.
        agent_memory (Any | AgentMemory, optional): Memory backend used to
            store summarized agent interactions. If provided, summaries are
            saved here.
        log_state (bool): When True, the agent writes intermediate json state
            to disk for debugging and auditability.
        **kwargs: Passed through to the BaseAgent constructor (e.g., model
            configuration, checkpointer).

    Attributes:
        safe_codes (list[str]): List of trusted programming languages for the
            agent. Defaults to python and julia
        executor_prompt (str): Prompt used when invoking the executor LLM
            loop.
        recap_prompt (str): Prompt used to request concise summaries for
            memory or final output.
        tools (dict[str, Tool]): Tools available to the agent (run_command, write_code,
            edit_code, read_file, run_web_search, run_osti_search, run_arxiv_search),
            keyed by tool name for quick lookups.
        tool_node (ToolNode): Graph node that dispatches tool calls.
        llm (BaseChatModel): Base LLM instance used for summaries and recap.
        tool_llm (BaseChatModel): Tool-bound LLM used for the execution loop.

    Methods:
        query_executor(state): Send messages to the executor LLM, ensure
            workspace exists, and handle symlink setup before returning the
            model response.
        recap(state): Produce and optionally persist a summary of recent
            interactions to the memory backend.
        _build_graph(): Construct and compile the StateGraph for the agent
            loop.

    Raises:
        AttributeError: Accessing the .action attribute raises to encourage
            using .stream(...) or .invoke(...).
    """

    state_type = ExecutionState

    def __init__(
        self,
        llm: BaseChatModel,
        agent_memory: Any | AgentMemory | None = None,
        log_state: bool = False,
        extra_tools: list[BaseTool] | None = None,
        tokens_before_summarize: int = 50000,
        messages_to_keep: int = 20,
        use_web: bool = False,
        safe_codes: list[str] | None = None,
        **kwargs,
    ):
        default_tools = [
            run_command,
            write_code,
            edit_code,
            read_file,
            read_image_tool,
            read_experience,
            write_experience,
            edit_experience,
            list_experiences,
        ]
        if use_web:
            default_tools.extend([
                run_web_search,
                run_osti_search,
                run_arxiv_search,
            ])
        if extra_tools:
            default_tools.extend(extra_tools)

        super().__init__(llm=llm, tools=default_tools, **kwargs)
        self.agent_memory = agent_memory
        self.safe_codes = set(safe_codes or ["python", "julia"])
        self.executor_prompt = executor_prompt
        self.recap_prompt = recap_prompt
        self.extra_tools = extra_tools
        self.log_state = log_state
        self.tokens_before_summarize = tokens_before_summarize
        self.messages_to_keep = messages_to_keep

    # Define the function that calls the model
    def query_executor(
        self,
        state: ExecutionState,
        runtime: Runtime[AgentContext],
        config: RunnableConfig | None = None,
    ) -> ExecutionState:
        """Prepare workspace, handle optional symlinks, and invoke the executor LLM.

        This method copies the incoming state, ensures a workspace directory exists
        (creating one with a default name when absent), optionally creates a symlink
        described by state["symlinkdir"], sets or injects the executor system prompt
        as the first message, and invokes the bound LLM. When logging is enabled,
        it persists the pre-invocation state to disk.

        Args:
            state: The current execution state. Expected keys include:
                - "messages": Ordered list of System/Human/AI/Tool messages.
                - "workspace": Optional path to the working directory.
                - "symlinkdir": Optional dict with "source" and "dest" keys.

        Returns:
            ExecutionState: Partial state update containing:
                - "messages": A list with the model's response as the latest entry.
                - "workspace": The resolved workspace path.
        """
        # Add model to the state so it can be passed to tools like the URSA Arxiv or OSTI tools
        new_state = deepcopy(state)
        events = self.events(config)
        new_state.setdefault("symlinkdir", {})

        full_overwrite = False

        # 1.5) Check message history length, summarize, and patch dangling tool calls.
        new_state, full_overwrite = self.prepare_messages_context(
            new_state, system_prompt=self.executor_prompt
        )

        # 2) Optionally create a symlink if symlinkdir is provided and not yet linked.
        sd = new_state.get("symlinkdir")
        if sd and "is_linked" not in sd:
            # symlinkdir structure: {"source": "/path/to/src", "dest": "link/name"}
            symlinkdir = sd

            src = Path(symlinkdir["source"]).expanduser().resolve()
            dst = runtime.context.workspace.joinpath(symlinkdir["dest"])

            # If a file/link already exists at the destination, replace it.
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            # Ensure parent directories for the link exist.
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Create the symlink (tell pathlib if the target is a directory).
            dst.symlink_to(src, target_is_directory=src.is_dir())
            events.emit(
                "Workspace symlink created",
                stage="workspace_symlink",
                source=str(src),
                destination=str(dst),
            )
            new_state["symlinkdir"]["is_linked"] = True
            full_overwrite = True

        # 3) Ensure the executor prompt is the first SystemMessage.
        messages = deepcopy(new_state["messages"])
        if isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=self.executor_prompt)
        else:
            messages = [SystemMessage(content=self.executor_prompt)] + messages

        # 4) Invoke the LLM with the prepared message sequence.
        try:
            response = self.tool_llm.invoke(
                messages, self.build_config(tags=["agent"])
            )
            new_state["messages"].append(response)
        except Exception as e:  # noqa: BLE001
            response = AIMessage(content=f"Response error {e}")
            msg = new_state["messages"][-1].text
            events.emit(
                "Executor response failed",
                stage="executor_error",
                error_type=type(e).__name__,
                error=str(e),
                latest_message=msg[:2000],
            )
            new_state["messages"].append(response)

        # 5) Optionally persist the pre-invocation state for audit/debugging.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)
        if full_overwrite:
            return {
                "messages": Overwrite(new_state["messages"]),
                "symlinkdir": new_state["symlinkdir"],
            }
        else:
            return {"messages": response, "symlinkdir": new_state["symlinkdir"]}

    def _get_original_user_request(self, state: ExecutionState) -> str:
        """Return the first human message as the original user request."""
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                return str(getattr(msg, "text", None) or msg.content)
        messages = state.get("messages", [])
        if messages:
            msg = messages[0]
            return str(
                getattr(msg, "text", None) or getattr(msg, "content", "")
            )
        return ""

    def review_work(
        self, state: ExecutionState, config: RunnableConfig | None = None
    ) -> ExecutionState:
        """Review whether the completed work adequately addresses the request.

        This node runs after the executor emits an ordinary assistant response without
        requesting any tool calls. It asks the LLM for a structured assessment of the
        work so far. If the work is incomplete, the review rationale is appended as a
        HumanMessage so the executor can continue with explicit feedback. If the work
        is complete, only the structured review state is updated and the graph proceeds
        to recap.
        """
        new_state = deepcopy(state)
        events = self.events(config)
        full_overwrite = False

        new_state, full_overwrite = self.prepare_messages_context(
            new_state, system_prompt=self.executor_prompt
        )

        user_request = self._get_original_user_request(new_state)
        review_message = HumanMessage(content=get_review_prompt(user_request))
        review_messages = list(new_state["messages"]) + [review_message]

        try:
            review_result = self.llm.with_structured_output(
                ReviewAssessment
            ).invoke(
                review_messages,
                config=self.build_config(tags=["review"]),
            )
            if not isinstance(review_result, Mapping):
                raise ValueError(
                    "Review step returned no structured assessment."
                )
        except Exception as e:
            # Avoid trapping the graph in an infinite review loop if the review model
            # call fails or returns an unusable structured-output payload. The recap
            # will still surface the work completed so far.
            review_result = {
                "is_complete": True,
                "reason": f"Review step failed with error: {e}. Proceeding to recap.",
            }
            print("Review error: ", e)

        is_complete = bool(review_result.get("is_complete", False))
        reason = str(review_result.get("reason", ""))
        review: ReviewAssessment = {
            "is_complete": is_complete,
            "reason": reason,
        }

        events.emit(
            "Execution Review: Complete"
            if is_complete
            else "Execution Review: Continue",
            stage="review_work",
            approved=is_complete,
            reason=reason,
        )

        if is_complete:
            if full_overwrite:
                return {
                    "messages": Overwrite(new_state["messages"]),
                    "review": review,
                    "symlinkdir": new_state.get("symlinkdir", {}),
                }
            return {"review": review}

        feedback = HumanMessage(
            content=(
                "Review of the current work determined that the original user "
                "request has not yet been adequately addressed. Continue working "
                "and specifically address the following review feedback:\n\n"
                f"{reason}"
            )
        )
        return self.messages_update(
            new_state,
            [feedback],
            full_overwrite=full_overwrite,
            extra={
                "review": review,
                "symlinkdir": new_state.get("symlinkdir", {}),
            },
        )

    def recap(
        self,
        state: ExecutionState,
        config: RunnableConfig | None = None,
    ) -> ExecutionState:
        """Produce a concise summary of the conversation and optionally persist memory.

        This method builds a summarization prompt, invokes the LLM to obtain a compact
        summary of recent interactions, optionally logs salient details to the agent
        memory backend, and writes debug state when logging is enabled.

        Args:
            state (ExecutionState): The execution state containing message history.

        Returns:
            ExecutionState: A partial update with a single string message containing
                the recap.
        """
        new_state = deepcopy(state)
        events = self.events(config)
        full_overwrite = False

        # 0) Check message history length, summarize, and patch dangling tool calls.
        new_state, full_overwrite = self.prepare_messages_context(
            new_state, system_prompt=self.executor_prompt
        )

        # 1) Construct the summarization message list (system prompt + prior messages).
        recap_message = HumanMessage(content=self.recap_prompt)
        new_state["messages"] = new_state["messages"] + [recap_message]

        # 2) Invoke the LLM to generate a recap; capture content even on failure.
        try:
            response = self.llm.invoke(
                input=new_state["messages"],
                config=self.build_config(tags=["recap"]),
            )
            response_content = response.text
        except Exception:
            try:
                response = self.tool_llm.invoke(
                    input=new_state["messages"],
                    config=self.build_config(tags=["recap"]),
                )
                response_content = response.text
            except Exception as e:
                response_content = f"Response error {e}"
                response = AIMessage(content=response_content)
                events.emit(
                    "Recap failed",
                    stage="recap_error",
                    error_type=type(e).__name__,
                    error=str(e),
                    latest_message=new_state["messages"][-1].text[:2000],
                )

        # 3) Optionally persist salient details to the memory backend.
        if self.agent_memory:
            memories: list[str] = []
            # Collect human/system/tool message content; for AI tool calls, store args.
            for msg in new_state["messages"]:
                msg_content = msg.text
                if not isinstance(msg, AIMessage) or not msg.tool_calls:
                    memories.append(msg_content)
                else:
                    tool_strings = []
                    for tool in msg.tool_calls:
                        tool_strings.append("Tool Name: " + tool["name"])
                        for arg_name in tool["args"]:
                            tool_strings.append(
                                f"Arg: {arg_name!s}\nValue: "
                                f"{tool['args'][arg_name]!s}"
                            )
                    memories.append("\n".join(tool_strings))
            memories.append(response_content)
            self.agent_memory.add_memories(memories)

        if full_overwrite:
            # 4) Optionally write state to disk for debugging/auditing.
            new_state["messages"].append(response)
            if self.log_state:
                self.write_state("execution_agent.json", new_state)
            return Overwrite(new_state)
        else:
            if self.log_state:
                new_state["messages"].append(response)
                self.write_state("execution_agent.json", new_state)
            return {"messages": [recap_message, response]}

    def _build_graph(self):
        """Construct and compile the agent's LangGraph state machine."""

        # Keep self.llm unbound for summary/recap calls. The executor loop uses a
        # separate tool-bound model so provider-specific tool transcripts cannot
        # leak into summarization history.
        self.tool_llm = self.tool_llm.bind_tools(self.tools.values())

        # Register nodes:
        # - "agent": LLM planning/execution step
        # - "action": tool dispatch (run_command, write_code, etc.)
        # - "review": structured completeness review before final recap
        # - "recap": summary/finalization step
        self.add_node(self.query_executor, "agent")
        self.add_node(self.tool_node, "action")
        self.add_node(self.review_work, "review")
        self.add_node(self.recap, "recap")

        # Set entrypoint: execution starts with the "agent" node.
        self.graph.set_entry_point("agent")

        # From "agent", either continue with tools or review the completed work,
        # based on presence of tool calls in the last message.
        self.graph.add_conditional_edges(
            "agent",
            self._wrap_cond(should_continue, "should_continue", "execution"),
            {"continue": "action", "review": "review"},
        )

        # After tools run, return control to the agent for the next step.
        self.graph.add_edge("action", "agent")

        # After review, either ask the executor to continue with review feedback or
        # finish with the recap node.
        self.graph.add_conditional_edges(
            "review",
            self._wrap_cond(review_complete, "review_complete", "execution"),
            {"continue": "agent", "recap": "recap"},
        )

        # The graph completes at the "recap" node.
        self.graph.set_finish_point("recap")

    def format_result(self, state: ExecutionState) -> str:
        return state["messages"][-1].text

    def hook_storage_setup(self, store):
        # Record the edit operation
        if store is None:
            return
        for safe_code in self.safe_codes:
            store.put(
                ("workspace", "safe_codes"),
                safe_code,
                {},
            )
