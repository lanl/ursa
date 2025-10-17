import subprocess
import sys
from textwrap import dedent
from time import sleep
from typing import Annotated, Optional

import chainlit as cl
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typer import Option, Typer

# NOTE: View all icons here -- https://lucide.dev/icons/
commands = [
    {
        "id": "Plan",
        "icon": "notebook-pen",
        "description": "Planning Agent",
    },
    {
        "id": "Execute",
        "icon": "file-terminal",
        "description": "Execution Agent",
    },
    {
        "id": "Implement",
        "icon": "bot",
        "description": "Plan, then Execute",
    },
    {
        "id": "Search",
        "icon": "globe",
        "description": "Websearch Agent",
    },
    # {
    #     "id": "Image",
    #     "icon": "image",
    #     "description": "Use DALL-E",
    # },
]


def wrap_api_key(api_key: Optional[str]) -> Optional[SecretStr]:
    return None if api_key is None else SecretStr(api_key)


# FIXME: This doesn't work.
@cl.on_chat_start
def on_chat_start():
    # TODO: User specify.
    llm_name = "gpt-4o-mini"
    model = ChatOpenAI(model=llm_name)

    print("Setting LLM")
    cl.user_session.set("llm_name", llm_name)
    cl.user_session.set("model", model)
    print("Finished setting LLM")


@cl.on_chat_start
async def start():
    await cl.context.emitter.set_commands(commands)


@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...

    def respond(query: str) -> str:
        print("HERE: ", cl.user_session.get("user"))
        model = cl.user_session.get("model")
        return model.invoke(messages=[dict(role="user", content=query)])

    match message.command:
        case None:
            # Send a response back to the user
            await cl.Message(content=respond(message.content)).send()
        case "Plan":
            await cl.Message(
                content="Generating Plan...\n" + respond(message.content)
            ).send()
        case "Execute":
            await cl.Message(
                content="Executing...\n" + respond(message.content)
            ).send()
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
        "--watch",
    ]

    if host is not None:
        cmd.extend(["--host", host])

    result = subprocess.run(
        cmd,
        check=False,
    )
    sys.exit(result.returncode)


def main():
    app()
