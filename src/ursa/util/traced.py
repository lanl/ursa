import json
from pathlib import Path
from typing import cast

from langchain.chat_models import BaseChatModel
from langchain.messages import AnyMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class _Traced:
    _messages: list[AnyMessage]

    @property
    def messages(self):
        """All messages to/from the llm"""
        return self._messages

    def _append_message(self, msg):
        match msg:
            case BaseMessage():
                self._messages.append(msg)
            case str():
                self._messages.append(HumanMessage(msg))
            case list():
                for m in msg:
                    self._append_message(m)
            case _:
                print(f"Skipping as msg is of type {type(msg)}")

    def save_messages(self, path: Path, indent: int | None = None):
        """Save all messages to json

        Arguments
        =========
        path (Path): json file to save message history
        indent (int|None): indentation for json file. No indentation if None
        supplied.
        """
        json.dump(
            [msg.model_dump() for msg in self.messages],
            path.open("w"),
            indent=indent,
        )

    def invoke(self, input, config=None, **kwargs):
        """wrapped invoke method to trace messages

        See BaseChatModel.invoke for help

        """
        self._append_message(input)

        # NOTE: Assuming the parent class is BaseChatModel. Do not use the
        # _Traced mixin with other classes. This line simply informs type
        # checkers, and does not actually type cast.
        parent = cast(BaseChatModel, super())

        output = parent.invoke(input, config=config, **kwargs)
        self._append_message(output)
        return output

    # TODO: ainvoke?


class TracedChatOllama(_Traced, ChatOllama):
    def __init__(self, messages: list[AnyMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._messages = messages or []


class TracedChatOpenAI(_Traced, ChatOpenAI):
    def __init__(self, messages: list[AnyMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._messages = messages or []
