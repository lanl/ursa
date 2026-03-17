import json
from abc import ABC, abstractmethod
from pathlib import Path

from langchain.messages import AnyMessage
from rich.console import Console


class UrsaLogger(ABC):
    def __init__(self):
        self.messages = []

    def append_messages(self, messages: list[AnyMessage]):
        self.messages.extend(messages)

    @abstractmethod
    def show(self, text: str): ...

    def save_messages(self, path: Path, indent: int = 2):
        json.dump(
            [msg.model_dump() for msg in self.messages],
            path.open("w"),
            indent=indent,
        )


class QuietLogger(UrsaLogger):
    def show(self, text: str):
        pass


class UrsaTerminalLogger(UrsaLogger):
    def __init__(self, console: Console):
        self.console = console

    def show(self, text: str):
        self.console.print(text)
