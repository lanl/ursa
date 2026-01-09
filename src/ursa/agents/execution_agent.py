"""Execution agent that builds a tool-enabled state graph to autonomously run tasks.

This module implements ExecutionAgent, a LangGraph-based agent that executes user
instructions by invoking LLM tool calls and coordinating a controlled workflow.

Key features:
- Workspace management with optional symlinking for external sources.
- Safety-checked shell execution via run_command with output size budgeting.
- Code authoring and edits through write_code and edit_code with rich previews.
- Web search capability through DuckDuckGoSearchResults.
- Summarization of the session and optional memory logging.
- Configurable graph with nodes for agent, safety_check, action, and summarize.

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
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    TypedDict,
)

import randomname
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

# Rich
from rich import get_console
from rich.panel import Panel

from ursa.agents.base import AgentWithTools, BaseAgent
from ursa.prompt_library.execution_prompts import (
    executor_prompt,
    get_safety_prompt,
    recap_prompt,
)
from ursa.tools import edit_code, read_file, run_command, write_code
from ursa.tools.search_tools import (
    run_arxiv_search,
    run_osti_search,
    run_web_search,
)
from ursa.util.memory_logger import AgentMemory

console = get_console()  # always returns the same instance

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# Classes for typing
class ExecutionState(TypedDict):
    """TypedDict representing the execution agent's mutable run state used by nodes.

    Fields:
    - messages: list of messages (System/Human/AI/Tool) with add_messages metadata.
    - current_progress: short status string describing agent progress.
    - code_files: list of filenames created or edited in the workspace.
    - workspace: path to the working directory where files and commands run.
    - symlinkdir: optional dict describing a symlink operation (source, dest,
      is_linked).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    current_progress: str
    code_files: list[str]
    workspace: Path
    symlinkdir: dict
    model: BaseChatModel


def should_continue(state: ExecutionState) -> Literal["recap", "continue"]:
    """Return 'recap' if no tool calls in the last message, else 'continue'.

    Args:
        state: The current execution state containing messages.

    Returns:
        A literal "recap" if the last message has no tool calls,
        otherwise "continue".
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "recap"
    # Otherwise if there is, we continue
    else:
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
        if "[UNSAFE]" in message.content:
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
        llm (BaseChatModel): LLM instance bound to the available tools.

    Methods:
        query_executor(state): Send messages to the executor LLM, ensure
            workspace exists, and handle symlink setup before returning the
            model response.
        recap(state): Produce and optionally persist a summary of recent
            interactions to the memory backend.
        safety_check(state): Validate pending run_command calls via the safety
            prompt and append ToolMessages for unsafe commands.
        get_safety_prompt(query, safe_codes, created_files): Get the LLM prompt for safety_check
            that includes an editable list of available programming languages and gets the context
            of files that the agent has generated and can trust.
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
        agent_memory: Optional[Any | AgentMemory] = None,
        log_state: bool = False,
        extra_tools: Optional[list[BaseTool] | None] = None,
        tokens_before_summarize: int = 50000,
        messages_to_keep: int = 20,
        safe_codes: Optional[list[str]] = None,
        **kwargs,
    ):
        default_tools = [
            run_command,
            write_code,
            edit_code,
            read_file,
            run_web_search,
            run_osti_search,
            run_arxiv_search,
        ]
        if extra_tools:
            default_tools.extend(extra_tools)

        super().__init__(llm=llm, tools=default_tools, **kwargs)
        self.agent_memory = agent_memory
        self.safe_codes = safe_codes or ["python", "julia"]
        self.get_safety_prompt = get_safety_prompt
        self.executor_prompt = executor_prompt
        self.recap_prompt = recap_prompt
        self.extra_tools = extra_tools
        self.log_state = log_state
        self.context_summarizer = SummarizationMiddleware(
            model=self.llm,
            max_tokens_before_summary=tokens_before_summarize,
            messages_to_keep=messages_to_keep,
        )

    # Check message history length and summarize to shorten the token usage:
    def _summarize_context(self, state: ExecutionState) -> ExecutionState:
        summarized_messages = self.context_summarizer.before_model(state, None)
        if summarized_messages:
            tokens_before_summarize = self.context_summarizer.token_counter(
                state["messages"]
            )
            state["messages"] = summarized_messages["messages"]
            tokens_after_summarize = self.context_summarizer.token_counter(
                state["messages"][1:]
            )
            console.print(
                Panel(
                    (
                        f"Summarized Conversation History:\n"
                        f"Approximate tokens before: {tokens_before_summarize}\n"
                        f"Approximate tokens after: {tokens_after_summarize}\n"
                    ),
                    title="[bold yellow1 on black]Summarize Past Context",
                    border_style="yellow1",
                    style="bold yellow1 on black",
                )
            )
        else:
            tokens_after_summarize = self.context_summarizer.token_counter(
                state["messages"]
            )
        return state

    # Define the function that calls the model
    def query_executor(self, state: ExecutionState) -> ExecutionState:
        """Prepare workspace, handle optional symlinks, and invoke the executor LLM.

        This method copies the incoming state, ensures a workspace directory exists
        (creating one with a random name when absent), optionally creates a symlink
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
        state.setdefault("model", self.llm)
        new_state = state.copy()

        # 1) Ensure a workspace directory exists, creating a named one if absent.
        if "workspace" not in new_state.keys():
            new_state["workspace"] = self.workspace / randomname.get_name()
            print(
                f"{RED}Creating the folder "
                f"{BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} "
                f"for this project.{RESET}"
            )
        Path(new_state["workspace"]).mkdir(exist_ok=True)

        # 1.5) Check message history length and summarize to shorten the token usage:
        new_state = self._summarize_context(new_state)

        # 2) Optionally create a symlink if symlinkdir is provided and not yet linked.
        sd = new_state.get("symlinkdir")
        if isinstance(sd, dict) and "is_linked" not in sd:
            # symlinkdir structure: {"source": "/path/to/src", "dest": "link/name"}
            symlinkdir = sd

            src = Path(symlinkdir["source"]).expanduser().resolve()
            workspace_root = Path(new_state["workspace"]).expanduser().resolve()
            dst = (
                workspace_root / symlinkdir["dest"]
            )  # Link lives inside workspace.

            # If a file/link already exists at the destination, replace it.
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            # Ensure parent directories for the link exist.
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Create the symlink (tell pathlib if the target is a directory).
            dst.symlink_to(src, target_is_directory=src.is_dir())
            print(f"{RED}Symlinked {src} (source) --> {dst} (dest)")
            new_state["symlinkdir"]["is_linked"] = True

        # 3) Ensure the executor prompt is the first SystemMessage.
        if isinstance(new_state["messages"][0], SystemMessage):
            new_state["messages"][0] = SystemMessage(
                content=self.executor_prompt
            )
        else:
            new_state["messages"] = [
                SystemMessage(content=self.executor_prompt)
            ] + state["messages"]

        # 4) Invoke the LLM with the prepared message sequence.
        try:
            response = self.llm.invoke(
                new_state["messages"], self.build_config(tags=["agent"])
            )
            new_state["messages"].append(response)
        except Exception as e:
            print("Error: ", e, " ", new_state["messages"][-1].content)
            new_state["messages"].append(
                AIMessage(content=f"Response error {e}")
            )

        # 5) Optionally persist the pre-invocation state for audit/debugging.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)

        # Return the model's response and the workspace path as a partial state update.
        return new_state

    def recap(self, state: ExecutionState) -> ExecutionState:
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
        new_state = state.copy()

        # 0) Check message history length and summarize to shorten the token usage:
        new_state = self._summarize_context(new_state)

        # 1) Construct the summarization message list (system prompt + prior messages).
        messages = (
            new_state["messages"]
            if isinstance(new_state["messages"][0], SystemMessage)
            else [SystemMessage(content=recap_prompt)] + new_state["messages"]
        )

        # 2) Invoke the LLM to generate a recap; capture content even on failure.
        response_content = ""
        try:
            response = self.llm.invoke(
                messages, self.build_config(tags=["recap"])
            )
            response_content = response.content
            new_state["messages"].append(response)
        except Exception as e:
            print("Error: ", e, " ", messages[-1].content)
            new_state["messages"].append(
                AIMessage(content=f"Response error {e}")
            )

        # 3) Optionally persist salient details to the memory backend.
        if self.agent_memory:
            memories: list[str] = []
            # Collect human/system/tool message content; for AI tool calls, store args.
            for msg in new_state["messages"]:
                if not isinstance(msg, AIMessage):
                    memories.append(msg.content)
                elif not msg.tool_calls:
                    memories.append(msg.content)
                else:
                    tool_strings = []
                    for tool in msg.tool_calls:
                        tool_strings.append("Tool Name: " + tool["name"])
                        for arg_name in tool["args"]:
                            tool_strings.append(
                                f"Arg: {str(arg_name)}\nValue: "
                                f"{str(tool['args'][arg_name])}"
                            )
                    memories.append("\n".join(tool_strings))
            memories.append(response_content)
            self.agent_memory.add_memories(memories)

        # 4) Optionally write state to disk for debugging/auditing.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)

        # 5) Return a partial state update with only the summary content.
        return new_state

    def safety_check(self, state: ExecutionState) -> ExecutionState:
        """Assess pending shell commands for safety and inject ToolMessages with results.

        This method inspects the most recent AI tool calls, evaluates any run_command
        queries against the safety prompt, and constructs ToolMessages that either
        flag unsafe commands with reasons or confirm safe execution. If any command
        is unsafe, the generated ToolMessages are appended to the state so the agent
        can react without executing the command.

        Args:
            state (ExecutionState): Current execution state.

        Returns:
            ExecutionState: Either the unchanged state (all safe) or a copy with one
                or more ToolMessages appended when unsafe commands are detected.
        """
        # 1) Work on a shallow copy; inspect the most recent model message.
        new_state = state.copy()
        last_msg = new_state["messages"][-1]

        # 1.5) Check message history length and summarize to shorten the token usage:
        new_state = self._summarize_context(new_state)

        # 2) Evaluate any pending run_command tool calls for safety.
        tool_responses: list[ToolMessage] = []
        any_unsafe = False
        for tool_call in last_msg.tool_calls:
            if tool_call["name"] != "run_command":
                continue

            query = tool_call["args"]["query"]
            safety_result = self.llm.invoke(
                self.get_safety_prompt(
                    query, self.safe_codes, new_state.get("code_files", [])
                ),
                self.build_config(tags=["safety_check"]),
            )

            if "[NO]" in safety_result.content:
                any_unsafe = True
                tool_response = (
                    "[UNSAFE] That command `{q}` was deemed unsafe and cannot be run.\n"
                    "For reason: {r}"
                ).format(q=query, r=safety_result.content)
                console.print(
                    "[bold red][WARNING][/bold red] Command deemed unsafe:",
                    query,
                )
                # Also surface the model's rationale for transparency.
                console.print(
                    "[bold red][WARNING][/bold red] REASON:", tool_response
                )
            else:
                tool_response = f"Command `{query}` passed safety check."
                console.print(
                    f"[green]Command passed safety check:[/green] {query}"
                )

            tool_responses.append(
                ToolMessage(
                    content=tool_response,
                    tool_call_id=tool_call["id"],
                )
            )

        # 3) If any command is unsafe, append all tool responses; otherwise keep state.
        if any_unsafe:
            new_state["messages"].extend(tool_responses)

        return new_state

    def _build_graph(self):
        """Construct and compile the agent's LangGraph state machine."""

        # Bind tools to llm and context summarizer
        self.llm = self.llm.bind_tools(self.tools.values())
        self.context_summarizer.model = self.llm

        # Register nodes:
        # - "agent": LLM planning/execution step
        # - "action": tool dispatch (run_command, write_code, etc.)
        # - "recap": summary/finalization step
        # - "safety_check": gate for shell command safety
        self.add_node(self.query_executor, "agent")
        self.add_node(self.tool_node, "action")
        self.add_node(self.recap, "recap")
        self.add_node(self.safety_check, "safety_check")

        # Set entrypoint: execution starts with the "agent" node.
        self.graph.set_entry_point("agent")

        # From "agent", either continue (tools) or finish (recap),
        # based on presence of tool calls in the last message.
        self.graph.add_conditional_edges(
            "agent",
            self._wrap_cond(should_continue, "should_continue", "execution"),
            {"continue": "safety_check", "recap": "recap"},
        )

        # From "safety_check", route to tools if safe, otherwise back to agent
        # to revise the plan without executing unsafe commands.
        self.graph.add_conditional_edges(
            "safety_check",
            self._wrap_cond(command_safe, "command_safe", "execution"),
            {"safe": "action", "unsafe": "agent"},
        )

        # After tools run, return control to the agent for the next step.
        self.graph.add_edge("action", "agent")

        # The graph completes at the "recap" node.
        self.graph.set_finish_point("recap")

    def format_result(self, state: ExecutionState) -> str:
        return state["messages"][-1].content
