from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel

from ursa.security import group_environments_dir, validate_group_name
from ursa.workflows.base_workflow import BaseWorkflow, InputLike

from .config import EnvironmentMemberConfig, load_object, make_llm


class BaseEnvironment(BaseWorkflow):
    """Base class for multi-agent URSA environments.

    Environments compose agents and/or other environments while exposing the same
    simple ``invoke`` surface used by workflows. They deliberately keep persistent
    configuration separate from agent graph checkpoints: environment definitions live
    under ``~/.cache/ursa/<group>/environments/``, while member agents use the
    shared ``~/.cache/ursa/<group>/agents/<agent_name>`` persistence mechanism.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        name: str,
        group: str = "default",
        workspace: str | Path | None = None,
        persist_members: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.name = name
        self.group = validate_group_name(group)
        self.workspace = Path(
            workspace
            or group_environments_dir(self.group) / "workspaces" / name
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.persist_members = persist_members

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        if isinstance(inputs, str):
            return {"task": inputs}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    def _member_workspace(self, member_name: str) -> Path:
        return self.workspace / member_name

    def _member_agent_name(self, member_name: str) -> str | None:
        if not self.persist_members:
            return None
        return f"{self.name}_{member_name}"

    def build_member(self, member: EnvironmentMemberConfig) -> Any:
        """Instantiate a configured member agent or nested environment.

        BaseAgent subclasses use ``agent_name`` for checkpoint persistence, while
        nested environments use ``name`` and manage their own member persistence.
        The distinction keeps teams usable as symposium members without requiring
        environment constructors to accept BaseAgent-specific keywords.
        """
        cls = load_object(member.agent)
        llm = make_llm(self.llm, member.model)
        kwargs = dict(member.config or {})
        kwargs.setdefault("workspace", self._member_workspace(member.name))
        kwargs.setdefault("group", self.group)
        if isinstance(cls, type) and issubclass(cls, BaseEnvironment):
            kwargs.setdefault("name", member.name)
            kwargs.setdefault("persist_members", self.persist_members)
        elif self.persist_members:
            kwargs.setdefault(
                "agent_name", self._member_agent_name(member.name)
            )
        return cls(llm=llm, **kwargs)

    def _run_ainvoke_from_sync(
        self, inputs: Mapping[str, Any], **config: Any
    ) -> Any:
        """Run this environment's async implementation for sync callers.

        Environments are natively async internally so nested agents/tools can use
        async-only implementations. The public ``invoke`` surface remains useful
        for scripts by creating an event loop at the outer boundary. If a loop is
        already running, callers must use ``await environment.ainvoke(...)``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ainvoke(inputs, **config))

        raise RuntimeError(
            "This environment uses async execution internally, but `.invoke()` "
            "was called from an async context. Use "
            "`await environment.ainvoke(...)` instead."
        )

    async def _invoke_member_async(
        self, member: Any, prompt: str, **kwargs: Any
    ) -> Any:
        """Invoke a member through its async API when available.

        BaseAgent and BaseEnvironment instances expose ``ainvoke``. Lightweight
        test doubles or custom sync-only members may expose only ``invoke``; run
        those in a worker thread so async environment phases remain non-blocking.
        """
        ainvoke = getattr(member, "ainvoke", None)
        if callable(ainvoke):
            result = ainvoke(prompt, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        invoke = getattr(member, "invoke")
        return await asyncio.to_thread(invoke, prompt, **kwargs)


def invocation_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop empty control kwargs before forwarding nested invocations."""
    return {key: value for key, value in config.items() if value is not None}


def result_to_text(result: Any) -> str:
    """Best-effort extraction of human-readable text from an agent result."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        if "prompt" in result and isinstance(result["prompt"], str):
            return result["prompt"]
        if "final" in result and isinstance(result["final"], str):
            return result["final"]
        if "result" in result and isinstance(result["result"], str):
            return result["result"]
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            text = getattr(last, "text", None)
            if text:
                return str(text)
            content = getattr(last, "content", None)
            if content:
                return str(content)
    text = getattr(result, "text", None)
    if text:
        return str(text)
    content = getattr(result, "content", None)
    if content:
        return str(content)
    return str(result)
