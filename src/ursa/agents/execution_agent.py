"""Execution agent that builds a tool-enabled state graph to autonomously run tasks.

This module implements ExecutionAgent, a LangGraph-based agent that executes user
instructions by invoking LLM tool calls and coordinating a controlled workflow.

Key features:
- Workspace management with optional symlinking for external sources.
- Safety-checked shell execution via run_cmd with output size budgeting.
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

import os

# from langchain_core.runnables.graph import MermaidDrawMethod
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional

import randomname
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_community.tools import (
    DuckDuckGoSearchResults,
)  # TavilySearchResults,
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

# Rich
from rich import get_console
from rich.panel import Panel
from rich.syntax import Syntax
from typing_extensions import TypedDict

from ..prompt_library.execution_prompts import (
    executor_prompt,
    get_safety_prompt,
    summarize_prompt,
)
from ..util.diff_renderer import DiffRenderer
from ..util.memory_logger import AgentMemory
from .base import BaseAgent

console = get_console()  # always returns the same instance

# --- ANSI color codes ---
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# Global variables for the module.

# Set a limit for message characters - the user could overload
# that in their env, or maybe we could pull this out of the LLM parameters
MAX_TOOL_MSG_CHARS = int(os.getenv("MAX_TOOL_MSG_CHARS", "50000"))

# Set a search tool.
search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(
#                   max_results=10,
#                   search_depth="advanced",
#                   include_answer=True)


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

    messages: Annotated[list, add_messages]
    current_progress: str
    code_files: list[str]
    workspace: str
    symlinkdir: dict


# Helper functions
def _strip_fences(snippet: str) -> str:
    """Remove markdown fences from a code snippet.

    This function strips leading triple backticks and any language
    identifiers from a markdown-formatted code snippet and returns
    only the contained code.

    Args:
        snippet: The markdown-formatted code snippet.

    Returns:
        The snippet content without leading markdown fences.
    """
    if "```" not in snippet:
        return snippet

    parts = snippet.split("```")
    if len(parts) < 3:
        return snippet

    body = parts[1]
    return "\n".join(body.split("\n")[1:]) if "\n" in body else body.strip()


def _snip_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text to a maximum length and indicate if truncation occurred.

    Args:
        text: The original text to potentially truncate.
        max_chars: The maximum characters allowed in the output.

    Returns:
        A tuple of (possibly truncated text, boolean flag indicating
        if truncation occurred).
    """
    if text is None:
        return "", False
    if max_chars <= 0:
        return "", len(text) > 0
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + f"\n... [snipped {len(text) - max_chars} chars] ...\n"
        + text[-tail:],
        True,
    )


def _fit_streams_to_budget(stdout: str, stderr: str, total_budget: int):
    """Allocate and truncate stdout and stderr to fit a total character budget.

    Args:
        stdout: The original stdout string.
        stderr: The original stderr string.
        total_budget: The combined character budget for stdout and stderr.

    Returns:
        A tuple of (possibly truncated stdout, possibly truncated stderr).
    """
    label_overhead = len("STDOUT:\n") + len("\nSTDERR:\n")
    budget = max(0, total_budget - label_overhead)

    if len(stdout) + len(stderr) <= budget:
        return stdout, stderr

    total_len = max(1, len(stdout) + len(stderr))
    stdout_budget = int(budget * (len(stdout) / total_len))
    stderr_budget = budget - stdout_budget

    stdout_snip, _ = _snip_text(stdout, stdout_budget)
    stderr_snip, _ = _snip_text(stderr, stderr_budget)

    return stdout_snip, stderr_snip


def should_continue(state: ExecutionState) -> Literal["summarize", "continue"]:
    """Return 'summarize' if no tool calls in the last message, else 'continue'.

    Args:
        state: The current execution state containing messages.

    Returns:
        A literal "summarize" if the last message has no tool calls,
        otherwise "continue".
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarize"
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


# Tools for ExecutionAgent
@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Execute a shell command in the workspace and return its combined output.

    Runs the specified command using subprocess.run in the given workspace
    directory, captures stdout and stderr, enforces a maximum character budget,
    and formats both streams into a single string. KeyboardInterrupt during
    execution is caught and reported.

    Args:
        query: The shell command to execute.
        state: A dict with injected state; must include the 'workspace' path.

    Returns:
        A formatted string with "STDOUT:" followed by the truncated stdout and
        "STDERR:" followed by the truncated stderr.
    """
    workspace_dir = state["workspace"]

    print("RUNNING: ", query)
    try:
        result = subprocess.run(
            query,
            text=True,
            shell=True,
            timeout=60000,
            capture_output=True,
            cwd=workspace_dir,
        )
        stdout, stderr = result.stdout, result.stderr
    except KeyboardInterrupt:
        print("Keyboard Interrupt of command: ", query)
        stdout, stderr = "", "KeyboardInterrupt:"

    # Fit BOTH streams under a single overall cap
    stdout_fit, stderr_fit = _fit_streams_to_budget(
        stdout or "", stderr or "", MAX_TOOL_MSG_CHARS
    )

    print("STDOUT: ", stdout_fit)
    print("STDERR: ", stderr_fit)

    return f"STDOUT:\n{stdout_fit}\nSTDERR:\n{stderr_fit}"


@tool
def write_code(
    code: str,
    filename: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Write source code to a file and update the agent’s workspace state.

    Args:
        code: The source code content to be written to disk.
        filename: Name of the target file (including its extension).
        tool_call_id: Identifier for this tool invocation.
        state: Agent state dict holding workspace path and file list.

    Returns:
        Command: Contains an updated state (including code_files) and
        a ToolMessage acknowledging success or failure.
    """
    # Determine the full path to the target file
    workspace_dir = state["workspace"]
    console.print("[cyan]Writing file:[/]", filename)

    # Clean up markdown fences on submitted code.
    code = _strip_fences(code)

    # Show syntax-highlighted preview before writing to file
    try:
        lexer_name = Syntax.guess_lexer(filename, code)
    except Exception:
        lexer_name = "text"

    console.print(
        Panel(
            Syntax(code, lexer_name, line_numbers=True),
            title="File Preview",
            border_style="cyan",
        )
    )

    # Write cleaned code to disk
    code_file = os.path.join(workspace_dir, filename)
    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as exc:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Failed to write file:[/]",
            exc,
        )
        return f"Failed to write {filename}."

    console.print(
        f"[bold bright_white on green] :heavy_check_mark: [/] "
        f"[green]File written:[/] {code_file}"
    )

    # Append the file to the list in agent's state for later reference
    file_list = state.get("code_files", [])
    if filename not in file_list:
        file_list.append(filename)

    # Create a tool message to send back to acknowledge success.
    msg = ToolMessage(
        content=f"File {filename} written successfully.",
        tool_call_id=tool_call_id,
    )

    # Return updated code files list & the message
    return Command(
        update={
            "code_files": file_list,
            "messages": [msg],
        }
    )


@tool
def edit_code(
    old_code: str,
    new_code: str,
    filename: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Replace the **first** occurrence of *old_code* with *new_code* in *filename*.

    Args:
        old_code: Code fragment to search for.
        new_code: Replacement fragment.
        filename: Target file inside the workspace.

    Returns:
        Success / failure message.
    """
    workspace_dir = state["workspace"]
    console.print("[cyan]Editing file:[/cyan]", filename)

    code_file = os.path.join(workspace_dir, filename)
    try:
        with open(code_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]File not found:[/]",
            filename,
        )
        return f"Failed: {filename} not found."

    # Clean up markdown fences
    old_code_clean = _strip_fences(old_code)
    new_code_clean = _strip_fences(new_code)

    if old_code_clean not in content:
        console.print(
            "[yellow] ⚠️ 'old_code' not found in file'; no changes made.[/]"
        )
        return f"No changes made to {filename}: 'old_code' not found in file."

    updated = content.replace(old_code_clean, new_code_clean, 1)

    console.print(
        Panel(
            DiffRenderer(content, updated, filename),
            title="Diff Preview",
            border_style="cyan",
        )
    )

    try:
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(updated)
    except Exception as exc:
        console.print(
            "[bold bright_white on red] :heavy_multiplication_x: [/] "
            "[red]Failed to write file:[/]",
            exc,
        )
        return f"Failed to edit {filename}."

    console.print(
        f"[bold bright_white on green] :heavy_check_mark: [/] "
        f"[green]File updated:[/] {code_file}"
    )
    file_list = state.get("code_files", [])
    if code_file not in file_list:
        file_list.append(filename)
    state["code_files"] = file_list

    return f"File {filename} updated successfully."


# Main module class
class ExecutionAgent(BaseAgent):
    """Orchestrates model-driven code execution, tool calls, and state management.

    Orchestrates model-driven code execution, tool calls, and state management for
    iterative program synthesis and shell interaction.

    This agent wraps an LLM with a small execution graph that alternates
    between issuing model queries, invoking tools (run, write, edit, search),
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
        summarize_prompt (str): Prompt used to request concise summaries for
            memory or final output.
        tools (list[Tool]): Tools available to the agent (run_cmd, write_code,
            edit_code, search_tool).
        tool_node (ToolNode): Graph node that dispatches tool calls.
        llm (BaseChatModel): LLM instance bound to the available tools.
        _action (StateGraph): Compiled execution graph that implements the
            main loop and branching logic.

    Methods:
        query_executor(state): Send messages to the executor LLM, ensure
            workspace exists, and handle symlink setup before returning the
            model response.
        summarize(state): Produce and optionally persist a summary of recent
            interactions to the memory backend.
        safety_check(state): Validate pending run_cmd calls via the safety
            prompt and append ToolMessages for unsafe commands.
        get_safety_prompt(query, safe_codes, created_files): Get the LLM prompt for safety_check
            that includes an editable list of available programming languages and gets the context
            of files that the agent has generated and can trust.
        _build_graph(): Construct and compile the StateGraph for the agent
            loop.
        _invoke(inputs, recursion_limit=...): Internal entry that invokes the
            compiled graph with a given recursion limit.
        action (property): Disabled; direct access is not supported. Use
            invoke or stream entry points instead.

    Raises:
        AttributeError: Accessing the .action attribute raises to encourage
            using .stream(...) or .invoke(...).
    """

    def __init__(
        self,
        llm: BaseChatModel = init_chat_model("openai:gpt-5-mini"),
        agent_memory: Optional[Any | AgentMemory] = None,
        log_state: bool = False,
        **kwargs,
    ):
        """ExecutionAgent class initialization."""
        super().__init__(llm, **kwargs)
        self.agent_memory = agent_memory
        self.safe_codes = kwargs.get("safe_codes", ["python", "julia"])
        self.get_safety_prompt = get_safety_prompt
        self.executor_prompt = executor_prompt
        self.summarize_prompt = summarize_prompt
        self.tools = [run_cmd, write_code, edit_code, search_tool]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)
        self.log_state = log_state
        self._action = self._build_graph()

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
        new_state = state.copy()

        # 1) Ensure a workspace directory exists, creating a named one if absent.
        if "workspace" not in new_state.keys():
            new_state["workspace"] = randomname.get_name()
            print(
                f"{RED}Creating the folder "
                f"{BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} "
                f"for this project.{RESET}"
            )
        os.makedirs(new_state["workspace"], exist_ok=True)

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
        except Exception as e:
            print("Error: ", e, " ", new_state["messages"][-1].content)

        # 5) Optionally persist the pre-invocation state for audit/debugging.
        if self.log_state:
            self.write_state("execution_agent.json", new_state)

        # Return the model's response and the workspace path as a partial state update.
        return {"messages": [response], "workspace": new_state["workspace"]}

    def summarize(self, state: ExecutionState) -> ExecutionState:
        """Produce a concise summary of the conversation and optionally persist memory.

        This method builds a summarization prompt, invokes the LLM to obtain a compact
        summary of recent interactions, optionally logs salient details to the agent
        memory backend, and writes debug state when logging is enabled.

        Args:
            state (ExecutionState): The execution state containing message history.

        Returns:
            ExecutionState: A partial update with a single string message containing
                the summary.
        """
        # 1) Construct the summarization message list (system prompt + prior messages).
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]

        # 2) Invoke the LLM to generate a summary; capture content even on failure.
        response_content = ""
        try:
            response = self.llm.invoke(
                messages, self.build_config(tags=["summarize"])
            )
            response_content = response.content
        except Exception as e:
            print("Error: ", e, " ", messages[-1].content)

        # 3) Optionally persist salient details to the memory backend.
        if self.agent_memory:
            memories: list[str] = []
            # Collect human/system/tool message content; for AI tool calls, store args.
            for msg in state["messages"]:
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
            save_state = state.copy()
            # Append the summary as an AI message for a complete trace.
            save_state["messages"] = save_state["messages"] + [
                AIMessage(content=response_content)
            ]
            self.write_state("execution_agent.json", save_state)

        # 5) Return a partial state update with only the summary content.
        return {"messages": [response_content]}

    def safety_check(self, state: ExecutionState) -> ExecutionState:
        """Assess pending shell commands for safety and inject ToolMessages with results.

        This method inspects the most recent AI tool calls, evaluates any run_cmd
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

        # 2) Evaluate any pending run_cmd tool calls for safety.
        tool_responses: list[ToolMessage] = []
        any_unsafe = False
        for tool_call in last_msg.tool_calls:
            if tool_call["name"] != "run_cmd":
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
        # Create a graph over the agent's execution state.
        graph = StateGraph(ExecutionState)

        # Register nodes:
        # - "agent": LLM planning/execution step
        # - "action": tool dispatch (run_cmd, write_code, etc.)
        # - "summarize": summary/finalization step
        # - "safety_check": gate for shell command safety
        self.add_node(graph, self.query_executor, "agent")
        self.add_node(graph, self.tool_node, "action")
        self.add_node(graph, self.summarize, "summarize")
        self.add_node(graph, self.safety_check, "safety_check")

        # Set entrypoint: execution starts with the "agent" node.
        graph.set_entry_point("agent")

        # From "agent", either continue (tools) or finish (summarize),
        # based on presence of tool calls in the last message.
        graph.add_conditional_edges(
            "agent",
            self._wrap_cond(should_continue, "should_continue", "execution"),
            {"continue": "safety_check", "summarize": "summarize"},
        )

        # From "safety_check", route to tools if safe, otherwise back to agent
        # to revise the plan without executing unsafe commands.
        graph.add_conditional_edges(
            "safety_check",
            self._wrap_cond(command_safe, "command_safe", "execution"),
            {"safe": "action", "unsafe": "agent"},
        )

        # After tools run, return control to the agent for the next step.
        graph.add_edge("action", "agent")

        # The graph completes at the "summarize" node.
        graph.set_finish_point("summarize")

        # Compile and return the executable graph (optionally with a checkpointer).
        return graph.compile(checkpointer=self.checkpointer)

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 999_999, **_
    ):
        """Invoke the compiled graph with inputs under a specified recursion limit.

        This method builds a LangGraph config with the provided recursion limit
        and a "graph" tag, then delegates to the compiled graph's invoke method.
        """
        # Build invocation config with a generous recursion limit for long runs.
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )

        # Delegate execution to the compiled graph.
        return self._action.invoke(inputs, config)

    # This property is trying to stop people bypassing invoke
    @property
    def action(self):
        """Property used to affirm `action` attribute is unsupported."""
        raise AttributeError(
            "Use .stream(...) or .invoke(...); direct .action access is unsupported."
        )
