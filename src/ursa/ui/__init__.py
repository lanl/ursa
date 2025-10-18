import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import chainlit as cl
from chainlit.types import CommandDict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typer import Option, Typer

from ursa.agents import ExecutionAgent


def ruff_lint(path: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--silent",
            "--select",
            "I",
            str(path),
        ],
        check=False,
    )


def ruff_format(path: Path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--line-length",
            "80",
            "--silent",
            "--preview",
            str(path),
        ],
        check=False,
    )


def get_code_language(path: Path) -> str:
    extension = path.suffix.lower().replace(".", "")
    return {"py": "python"}.get(extension, extension)


def wrap_code(path: Path) -> str:
    ruff_lint(path)
    ruff_format(path)
    language = get_code_language(path)
    code = path.read_text()
    return f"""
*{path}*

```{language}
{code}
```
"""


# NOTE: View all icons here -- https://lucide.dev/icons/
commands = [
    CommandDict(
        id="Plan",
        icon="notebook-pen",
        description="Planning Agent",
        button=None,
        persistent=None,
    ),
    CommandDict(
        id="Execute",
        icon="file-terminal",
        description="Execution Agent",
        button=None,
        persistent=None,
    ),
    CommandDict(
        id="Implement",
        icon="bot",
        description="Plan, then Execute",
        button=None,
        persistent=None,
    ),
    CommandDict(
        id="Search",
        icon="globe",
        description="Websearch Agent",
        button=None,
        persistent=None,
    ),
    # {
    #     "id": "Image",
    #     "icon": "image",
    #     "description": "Use DALL-E",
    # },
]


def wrap_api_key(api_key: Optional[str]) -> Optional[SecretStr]:
    return None if api_key is None else SecretStr(api_key)


@cl.on_chat_start
async def on_chat_start():
    # TODO: User specify.
    llm_name = "gpt-4o-mini"
    model = ChatOpenAI(model=llm_name)

    cl.user_session.set("llm_name", llm_name)
    cl.user_session.set("model", model)
    cl.user_session.set("last_agent_result", "")
    cl.user_session.set("execution_agent", ExecutionAgent(llm=model))

    await cl.context.emitter.set_commands(commands)


def run_executor(prompt: str) -> str:
    # FIXME: This will be end up being shared among users. But we want each user
    # to have their own space.
    workspace = Path.home() / ".ursa" / "ui"
    workspace.mkdir(parents=True, exist_ok=True)
    state = cl.user_session.get(
        "executor_state",
        {"workspace": str(workspace)},
    )
    last_agent_result = cl.user_session.get("last_agent_result")

    if not isinstance(state, dict):
        raise RuntimeError("executor_state is not dict!")

    agent = cl.user_session.get("execution_agent")
    assert isinstance(agent, ExecutionAgent)

    if "messages" in state and isinstance(state["messages"], list):
        state["messages"].append(
            HumanMessage(
                f"The last agent output was: {last_agent_result}\n"
                f"The user stated: {prompt}"
            )
        )
        agent.invoke(state)

        if isinstance(content := state["messages"][-1].content, str):
            cl.user_session.set("last_agent_result", content)
        else:
            raise TypeError(
                f"content is supposed to be a str. Instead it is a {type(content)}!"
            )
    else:
        state = dict(
            messages=[
                HumanMessage(
                    f"The last agent output was: {last_agent_result}\n The user stated: {prompt}"
                )
            ],
            # FIXME: This will be end up being shared among users. But we want each user
            # to have their own space.
            workspace=str(workspace),
        )
        state = agent.invoke(state)
        cl.user_session.set("last_agent_result", state["messages"][-1].content)

    cl.user_session.set("executor_state", state)
    # return str(cl.user_session.get("last_agent_result"))
    response = "\n".join(
        str(msg.content)
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
    )
    code_files = "**The following files were written**:\n" + "\n\n".join(
        wrap_code(workspace / str(file)) for file in state["code_files"]
    )

    return response + "\n***\n" + code_files


def respond(query: str) -> str:
    model = cl.user_session.get("model")
    if model is None:
        raise RuntimeError("model is None!")
    return model.invoke([HumanMessage(query)]).content


@cl.on_message
async def on_message(message: cl.Message):
    match message.command:
        case None:
            # Send a response back to the user
            m = cl.Message(content="...")
            await m.send()
            m.content = respond(message.content)
            await m.update()

        case "Plan":
            m = cl.Message(
                content="Generating Plan...\n" + respond(message.content)
            )
            await m.send()
        case "Execute":
            m = cl.Message(content="Executing ...")
            await m.send()
            m.content = run_executor(message.content)
            await m.update()
        case "Web Search":
            await cl.Message(
                content="Searching Web...\n" + respond(message.content)
            ).send()
        case "Implement":
            await cl.Message(
                content="Will **plan**, then **execute**...\n"
                + respond(message.content)
            ).send()
        case _:
            raise ValueError(f"Command {message.command} is not supported")


app = Typer()


@app.command()
def serve(
    port: Annotated[int, Option(help="port to serve ursa web interface")],
    host: Annotated[
        Optional[str], Option(help="host to serve ursa web interface")
    ] = None,
    watch: Annotated[
        bool, Option(help="Whether or not to reload if source changes")
    ] = False,
):
    cmd = [
        sys.executable,
        "-m",
        "chainlit",
        "run",
        __file__,
        "--port",
        str(port),
        "--headless",
        # "--watch",
    ]

    if host is not None:
        cmd.extend(["--host", host])

    if watch:
        cmd.extend(["--watch"])

    result = subprocess.run(
        cmd,
        check=False,
    )
    sys.exit(result.returncode)


def main():
    app()
