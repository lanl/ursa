from __future__ import annotations

import re
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from xml.sax.saxutils import escape, quoteattr

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.runnables.config import ensure_config

from .catalog import SkillCatalog, SkillScope, parse_skill

_CURRENT: ContextVar[tuple[object, "SkillRun"] | None] = ContextVar(
    "ursa_skill_run", default=None
)
_CODE = re.compile(
    r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[^\n]*(?:\n|$)|(`+)[^`]*?\2"
)
_REFERENCE = re.compile(r"(?<![\\\w$])\$([a-z0-9]+(?:-[a-z0-9]+)*)(?![\w-])")


def explicit_skills(text: str) -> tuple[str, ...]:
    text = _CODE.sub(" ", text)
    return tuple(dict.fromkeys(_REFERENCE.findall(text)))


def request_text(inputs: Any) -> str:
    if isinstance(inputs, str):
        return inputs
    if not isinstance(inputs, Mapping):
        return ""
    for key in ("question", "query"):
        if isinstance(inputs.get(key), str):
            return inputs[key]
    for message in reversed(convert_to_messages(inputs.get("messages") or [])):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


@dataclass
class SkillRun:
    catalog: SkillCatalog
    pending_events: list[str] = field(default_factory=list)
    active: dict[str, str] = field(default_factory=dict)
    resources: dict[tuple[str, str], str] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)

    def activate(self, name: str) -> str:
        with self.lock:
            skill = self.catalog.get(name)
            if name not in self.active:
                metadata, body = parse_skill(skill.path)
                if metadata["name"] != name:
                    raise ValueError(
                        f"Skill '{name}' changed during this request; retry the request"
                    )
                self.active[name] = body
                self.pending_events.append(name)
            return f"Skill '{name}' is active for this request. Its instructions are included in the next model request. Directory: {skill.directory}"

    def read_resource(self, name: str, path: str) -> str:
        with self.lock:
            skill = self.catalog.get(name)
            if name not in self.active:
                raise ValueError(
                    f"Load skill '{name}' before reading its resources"
                )
            relative = Path(path)
            target = (skill.directory / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(
                skill.directory.resolve()
            ):
                raise ValueError(
                    "Resource path must stay within the skill directory"
                )
            self.resources[(name, path)] = target.read_text(encoding="utf-8")
            return f"Loaded resource '{path}' for '{name}'; its contents are included in the next model request."

    def prompt(self) -> str:
        with self.lock:
            parts = [self.catalog.prompt()]
            for name, body in self.active.items():
                skill = self.catalog.get(name)
                parts.append(
                    f"<active_skill name={quoteattr(name)} directory={quoteattr(str(skill.directory))}>\n"
                    f"{escape(body)}\n</active_skill>"
                )
            for (name, path), content in self.resources.items():
                parts.append(
                    f"<skill_resource skill={quoteattr(name)} path={quoteattr(path)}>\n{escape(content)}\n</skill_resource>"
                )
            return "\n\n".join(part for part in parts if part)


def current_run(owner: object) -> SkillRun | None:
    entry = _CURRENT.get()
    return entry[1] if entry is not None and entry[0] is owner else None


@contextmanager
def skill_run(
    owner: object,
    scope: SkillScope | None,
    inputs: Any,
    config: dict | None,
) -> Iterator[None]:
    if scope is None:
        yield
        return
    run = SkillRun(scope.discover())
    token = _CURRENT.set((owner, run))
    try:
        text = (config or {}).get("configurable", {}).get("ursa_skill_request")
        for name in explicit_skills(
            request_text(inputs) if text is None else text
        ):
            run.activate(name)
        yield
    finally:
        _CURRENT.reset(token)


def augment_messages(owner: object, messages: list) -> list:
    run = current_run(owner)
    if run is not None:
        config = ensure_config()
        if config.get("callbacks") is not None:
            with run.lock:
                for name in run.pending_events:
                    owner.events(config).emit(
                        f"Loaded skill: {name}",
                        stage="skill_activation",
                        skill=name,
                        path=str(run.catalog.get(name).path),
                    )
                run.pending_events.clear()
    prompt = run.prompt() if run is not None else ""
    if not prompt:
        return messages
    result = list(messages)
    if result and isinstance(result[0], SystemMessage):
        original = result[0]
        if isinstance(original.content, str):
            content = original.content + "\n\n" + prompt
        else:
            content = [*original.content, {"type": "text", "text": prompt}]
        result[0] = original.model_copy(update={"content": content})
    else:
        result.insert(0, SystemMessage(content=prompt))
    return result


def skill_stream(owner, scope, inputs, config, stream):
    def generate():
        with skill_run(owner, scope, inputs, config):
            yield from stream()

    context = copy_context()
    iterator = generate()
    try:
        while True:
            try:
                item = context.run(next, iterator)
            except StopIteration:
                return
            yield item
    finally:
        context.run(iterator.close)
