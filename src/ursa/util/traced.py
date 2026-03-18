import json
from functools import cached_property
from pathlib import Path

from langchain.messages import AnyMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def append_message(msg, messages: list[AnyMessage]):
    match msg:
        case BaseMessage():
            messages.append(msg)
        case str():
            messages.append(HumanMessage(msg))
        case list():
            for m in msg:
                append_message(m, messages)
        case _:
            print(f"Skipping as msg is of type {type(msg)}")


def _traced_invoke(
    messages: list[AnyMessage], llm, input, config=None, **kwargs
):
    append_message(input, messages)
    output = llm.invoke(input, config=config, **kwargs)
    append_message(output, messages)
    return output


class Traced:
    @cached_property
    def messages(self) -> list[AnyMessage]:
        return []

    def save_messages(self, path: Path, indent: int = 2):
        json.dump(
            [msg.model_dump() for msg in self.messages],
            path.open("w"),
            indent=indent,
        )


class UrsaOllama(ChatOllama, Traced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def invoke(self, input, config=None, **kwargs):
        return _traced_invoke(
            self.messages, super(), input, config=config, **kwargs
        )


class UrsaOpenAI(ChatOpenAI, Traced):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def invoke(self, input, config=None, **kwargs):
        # return self.wrapped_invoke(super(), input, **kwargs)
        return _traced_invoke(
            self.messages, super(), input, config=config, **kwargs
        )
