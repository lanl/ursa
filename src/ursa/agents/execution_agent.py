import os

# from langchain_core.runnables.graph import MermaidDrawMethod
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import randomname
from langchain_community.tools import (
    DuckDuckGoSearchResults,
)  # TavilySearchResults,
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command
from litellm import ContentPolicyViolationError

# Rich
from rich import get_console
from rich.panel import Panel
from rich.syntax import Syntax
from typing_extensions import TypedDict

from ..prompt_library.execution_prompts import executor_prompt, summarize_prompt
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


class ExecutionState(TypedDict):
    messages: Annotated[list, add_messages]
    current_progress: str
    code_files: list[str]
    workspace: str
    symlinkdir: dict


class ExecutionAgent(BaseAgent):
    def __init__(
        self,
        llm: str | BaseChatModel = "openai/gpt-4o-mini",
        agent_memory: Optional[Any | AgentMemory] = None,
        log_state: bool = False,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.agent_memory = agent_memory
        self.executor_prompt = executor_prompt
        self.summarize_prompt = summarize_prompt
        self.tools = [run_cmd, write_code, edit_code, search_tool]
        self.tool_node = ToolNode(self.tools)
        self.llm = self.llm.bind_tools(self.tools)
        self.log_state = log_state

        self._initialize_agent()

    # Define the function that calls the model
    def query_executor(self, state: ExecutionState) -> ExecutionState:
        new_state = state.copy()
        if "workspace" not in new_state.keys():
            new_state["workspace"] = randomname.get_name()
            print(
                f"{RED}Creating the folder {BLUE}{BOLD}{new_state['workspace']}{RESET}{RED} for this project.{RESET}"
            )
        os.makedirs(new_state["workspace"], exist_ok=True)

        # code related to symlink
        sd = new_state.get("symlinkdir")
        if isinstance(sd, dict) and "is_linked" not in sd:
            # symlinkdir = {"source": "foo", "dest": "bar"}
            symlinkdir = new_state["symlinkdir"]
            # user provided a symlinkdir key - let's do the linking!

            src = Path(symlinkdir["source"]).expanduser().resolve()
            workspace_root = Path(new_state["workspace"]).expanduser().resolve()
            dst = workspace_root / symlinkdir["dest"]  # prepend workspace

            # if you want to replace an existing link/file, unlink it first
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            # create parent dirs for the link location if they don’t exist
            dst.parent.mkdir(parents=True, exist_ok=True)

            # actually make the link (tell pathlib it’s a directory target)
            dst.symlink_to(src, target_is_directory=src.is_dir())
            print(f"{RED}Symlinked {src} (source) --> {dst} (dest)")
            # note that we've done the symlink now, so don't need to do it later
            new_state["symlinkdir"]["is_linked"] = True

        if isinstance(new_state["messages"][0], SystemMessage):
            new_state["messages"][0] = SystemMessage(
                content=self.executor_prompt
            )
        else:
            new_state["messages"] = [
                SystemMessage(content=self.executor_prompt)
            ] + state["messages"]
        try:
            response = self.llm.invoke(
                new_state["messages"],
                {"configurable": {"thread_id": self.thread_id}},
            )
        except ContentPolicyViolationError as e:
            print("Error: ", e, " ", new_state["messages"][-1].content)
        if self.log_state:
            self.write_state("execution_agent.json", new_state)
        return {"messages": [response], "workspace": new_state["workspace"]}

    # Define the function that calls the model
    def summarize(self, state: ExecutionState) -> ExecutionState:
        messages = [SystemMessage(content=summarize_prompt)] + state["messages"]
        try:
            response = self.llm.invoke(
                messages, {"configurable": {"thread_id": self.thread_id}}
            )
        except ContentPolicyViolationError as e:
            print("Error: ", e, " ", messages[-1].content)
        if self.agent_memory:
            memories = []
            # Handle looping through the messages
            for x in state["messages"]:
                if not isinstance(x, AIMessage):
                    memories.append(x.content)
                elif not x.tool_calls:
                    memories.append(x.content)
                else:
                    tool_strings = []
                    for tool in x.tool_calls:
                        tool_name = "Tool Name: " + tool["name"]
                        tool_strings.append(tool_name)
                        for y in tool["args"]:
                            tool_strings.append(
                                f"Arg: {str(y)}\nValue: {str(tool['args'][y])}"
                            )
                    memories.append("\n".join(tool_strings))
            memories.append(response.content)
            self.agent_memory.add_memories(memories)
            save_state = state.copy()
            save_state["messages"].append(response)
        if self.log_state:
            self.write_state("execution_agent.json", save_state)
        return {"messages": [response.content]}

    # Define the function that calls the model
    def safety_check(self, state: ExecutionState) -> ExecutionState:
        """
        Validate the safety of a pending shell command.

        Args:
            state: Current execution state.

        Returns:
            Either the unchanged state (safe) or a state with tool message(s) (unsafe).
        """
        new_state = state.copy()
        last_msg = new_state["messages"][-1]

        tool_responses = []
        tool_failed = False
        for tool_call in last_msg.tool_calls:
            call_name = tool_call["name"]

            if call_name == "run_cmd":
                query = tool_call["args"]["query"]
                safety_check = self.llm.invoke(
                    (
                        "Assume commands to run/install python and Julia files are safe because "
                        "the files are from a trusted source. "
                        f"Explain why, followed by an answer [YES] or [NO]. Is this command safe to run: {query}"
                    )
                )

                if "[NO]" in safety_check.content:
                    tool_failed = True

                    tool_response = f"""
                    [UNSAFE] That command `{query}` was deemed unsafe and cannot be run.
                    For reason: {safety_check.content}
                    """
                    console.print(
                        "[bold red][WARNING][/bold red] Command deemed unsafe:",
                        query,
                    )
                    # and tell the user the reason
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

        if tool_failed:
            new_state["messages"].extend(tool_responses)

        return new_state

    def _initialize_agent(self):
        self.graph = StateGraph(ExecutionState)

        self.graph.add_node("agent", self.query_executor)
        self.graph.add_node("action", self.tool_node)
        self.graph.add_node("summarize", self.summarize)
        self.graph.add_node("safety_check", self.safety_check)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        self.graph.add_edge(START, "agent")

        self.graph.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "safety_check",
                "summarize": "summarize",
            },
        )

        self.graph.add_conditional_edges(
            "safety_check",
            command_safe,
            {
                "safe": "action",
                "unsafe": "agent",
            },
        )

        self.graph.add_edge("action", "agent")
        self.graph.add_edge("summarize", END)

        self.action = self.graph.compile(checkpointer=self.checkpointer)
        # self.action.get_graph().draw_mermaid_png(output_file_path="execution_agent_graph.png", draw_method=MermaidDrawMethod.PYPPETEER)

    def run(self, prompt, recursion_limit=1000):
        inputs = {"messages": [HumanMessage(content=prompt)]}
        return self.action.invoke(
            inputs,
            {
                "recursion_limit": recursion_limit,
                "configurable": {"thread_id": self.thread_id},
            },
        )


@tool
def run_cmd(query: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Run a commandline command from using the subprocess package in python

    Args:
        query: commandline command to be run as a string given to the subprocess.run command.
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

    print("STDOUT: ", stdout)
    print("STDERR: ", stderr)

    return f"STDOUT: {stdout} and STDERR: {stderr}"


def _strip_fences(snippet: str) -> str:
    """
    Remove leading markdown ``` fence
    """
    if "```" not in snippet:
        return snippet

    parts = snippet.split("```")
    if len(parts) < 3:
        return snippet

    body = parts[1]
    return "\n".join(body.split("\n")[1:]) if "\n" in body else body.strip()


@tool
def write_code(
    code: str,
    filename: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Write *code* to *filename*.

    Args:
        code: Source code as a string.
        filename: Target filename (including extension).

    Returns:
        Success / failure message.
    """
    workspace_dir = state["workspace"]
    console.print("[cyan]Writing file:[/]", filename)

    # Clean up markdown fences
    code = _strip_fences(code)

    # Syntax-highlighted preview
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

    # Append the file to the list in state
    file_list = state.get("code_files", [])
    file_list.append(filename)

    # Create a tool message to send back
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
    return f"File {filename} updated successfully."


search_tool = DuckDuckGoSearchResults(output_format="json", num_results=10)
# search_tool = TavilySearchResults(max_results=10, search_depth="advanced", include_answer=True)


# Define the function that determines whether to continue or not
def should_continue(state: ExecutionState) -> Literal["summarize", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "summarize"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that determines whether to continue or not
def command_safe(state: ExecutionState) -> Literal["safe", "unsafe"]:
    """
    Return graph edge "safe" if the last command was safe, otherwise return edge "unsafe"
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


def main():
    execution_agent = ExecutionAgent()
    problem_string = (
        "Write and execute a python script to print the first 10 integers."
    )
    inputs = {
        "messages": [HumanMessage(content=problem_string)]
    }  # , "workspace":"dummy_test"}
    result = execution_agent.action.invoke(
        inputs, {"configurable": {"thread_id": execution_agent.thread_id}}
    )
    print(result["messages"][-1].content)
    return result


if __name__ == "__main__":
    main()
