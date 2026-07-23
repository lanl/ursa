from __future__ import annotations

import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ursa.util.events import EnvironmentEvents

from .base import (
    BaseEnvironment,
    bind_current_environment_config,
    current_environment_config,
    invocation_kwargs,
    reset_current_environment_config,
    result_to_text,
    runnable_config_from_kwargs,
)
from .config import (
    AgentTeamConfig,
    EnvironmentMemberConfig,
    load_object,
    load_team_config,
    make_llm,
)

logger = logging.getLogger(__name__)


class DelegateInput(BaseModel):
    """Input schema for PI-to-member delegation tools."""

    task: str = Field(
        ...,
        description="Specific task or question to delegate to this team member.",
    )
    context: str = Field(
        "",
        description="Relevant global context, constraints, prior results, or success criteria.",
    )


def _slug_tool_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or "member"


class AgentTeamEnvironment(BaseEnvironment):
    """Hierarchical multi-agent team coordinated by a PI agent.

    The PI is user-facing. Team members are exposed to the PI as tools, one tool
    per member, so the PI can plan, delegate, compare returned work, ask follow-up
    questions, and synthesize a final answer. The environment itself exposes a
    normal ``invoke`` method and can therefore be nested inside other environments
    such as an Agent Symposium.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        config: AgentTeamConfig | Mapping[str, Any] | str | Path | None = None,
        name: str | None = None,
        group: str | None = None,
        pi: EnvironmentMemberConfig | Mapping[str, Any] | None = None,
        members: list[EnvironmentMemberConfig | Mapping[str, Any]]
        | None = None,
        workspace: str | Path | None = None,
        persist_members: bool = True,
        trace_delegation: bool = True,
        trace_character_limit: int = 4000,
        **kwargs: Any,
    ):
        team_config = self._coerce_config(
            config=config,
            name=name,
            group=group,
            pi=pi,
            members=members,
            workspace=workspace,
        )
        super().__init__(
            llm,
            name=team_config.name,
            group=team_config.group,
            workspace=team_config.workspace or workspace,
            persist_members=persist_members,
            **kwargs,
        )
        self.config = team_config
        self.trace_delegation = trace_delegation
        self.trace_character_limit = trace_character_limit
        self.members = {
            member.name: self._build_team_member(member)
            for member in self.config.members
        }
        self.member_configs = {
            member.name: member for member in self.config.members
        }
        self.pi = self._build_pi()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> "AgentTeamEnvironment":
        return cls(llm=llm, config=load_team_config(path), **kwargs)

    def _coerce_config(
        self,
        *,
        config: AgentTeamConfig | Mapping[str, Any] | str | Path | None,
        name: str | None,
        group: str | None,
        pi: EnvironmentMemberConfig | Mapping[str, Any] | None,
        members: list[EnvironmentMemberConfig | Mapping[str, Any]] | None,
        workspace: str | Path | None,
    ) -> AgentTeamConfig:
        if isinstance(config, (str, Path)):
            base = load_team_config(config)
        elif isinstance(config, Mapping):
            base = AgentTeamConfig.from_mapping(config)
        elif isinstance(config, AgentTeamConfig):
            base = config
        else:
            pi_cfg = self._coerce_member(
                pi
                or {
                    "name": "pi",
                    "role": "Principal investigator / team lead",
                    "agent": "ExecutionAgent",
                }
            )
            member_cfgs = [self._coerce_member(m) for m in (members or [])]
            base = AgentTeamConfig(
                name=name or "agent_team",
                group=group or "default",
                pi=pi_cfg,
                members=member_cfgs,
                workspace=str(workspace) if workspace else None,
            )
        return AgentTeamConfig(
            name=name or base.name,
            group=group or base.group,
            description=base.description,
            pi=base.pi,
            members=base.members,
            workspace=str(workspace) if workspace else base.workspace,
            defaults=base.defaults,
        )

    @staticmethod
    def _coerce_member(
        member: EnvironmentMemberConfig | Mapping[str, Any],
    ) -> EnvironmentMemberConfig:
        if isinstance(member, EnvironmentMemberConfig):
            return member
        return EnvironmentMemberConfig.from_mapping(member)

    def _build_team_member(self, member: EnvironmentMemberConfig) -> Any:
        """Instantiate a team member in the shared team workspace.

        Team member names are stable URSA agent names. This lets users reuse
        existing named agents directly from YAML, e.g. ``team_nif_expert`` rather
        than creating a new ``<team>_<member>`` checkpoint. All team members see
        the same team workspace so they can collaborate through shared files.
        """
        return self._build_named_team_agent(member, agent_name=member.name)

    def _build_team_pi(self, member: EnvironmentMemberConfig) -> Any:
        """Instantiate the PI in the shared team workspace.

        The PI also uses its configured name directly. If that named PI does not
        yet exist, BaseAgent will create its checkpoint under the usual URSA agent
        cache for the configured group.
        """
        return self._build_named_team_agent(member, agent_name=member.name)

    def _build_named_team_agent(
        self,
        member: EnvironmentMemberConfig,
        *,
        agent_name: str,
    ) -> Any:
        cls = load_object(member.agent)
        llm = make_llm(self.llm, member.model)
        kwargs = dict(member.config or {})
        kwargs.setdefault("workspace", self.workspace)
        kwargs.setdefault("group", self.group)
        if isinstance(cls, type) and issubclass(cls, BaseEnvironment):
            kwargs.setdefault("name", member.name)
            kwargs.setdefault("persist_members", self.persist_members)
        elif self.persist_members:
            kwargs.setdefault("agent_name", agent_name)
        return cls(llm=llm, **kwargs)

    def _build_pi(self) -> Any:
        pi_config = self.config.pi
        cls_kwargs = dict(pi_config.config or {})
        delegation_tools = [
            self._make_delegation_tool(member_config)
            for member_config in self.config.members
        ]

        # ExecutionAgent explicitly supports `extra_tools`, making it the preferred
        # PI implementation. Other AgentWithTools-style agents can receive tools
        # after construction via `add_tool` if they expose that method.
        pass_extra_tools = (
            pi_config.agent.endswith("ExecutionAgent")
            or pi_config.agent == "ExecutionAgent"
        )
        if pass_extra_tools:
            cls_kwargs.setdefault("extra_tools", delegation_tools)

        pi_member = EnvironmentMemberConfig(
            name=pi_config.name,
            role=pi_config.role,
            agent=pi_config.agent,
            model=pi_config.model,
            config=cls_kwargs,
            prompt=pi_config.prompt,
        )
        pi_agent = self._build_team_pi(pi_member)
        if (
            not pass_extra_tools
            and delegation_tools
            and hasattr(pi_agent, "add_tool")
        ):
            pi_agent.add_tool(delegation_tools)
        elif not pass_extra_tools and delegation_tools:
            raise TypeError(
                f"Configured PI agent {pi_config.agent!r} cannot accept delegation tools. "
                "Use ExecutionAgent or another AgentWithTools-compatible agent."
            )
        return pi_agent

    def _events(self, config: Mapping[str, Any] | None) -> EnvironmentEvents:
        return EnvironmentEvents(
            environment=self.name,
            config=config,
            environment_type="agent_team",
            environment_id=self.name,
            path=[self.name],
        )

    def _source(self, name: str, *, kind: str = "agent") -> dict[str, Any]:
        return {
            "id": f"{self.name}.{name}",
            "name": name,
            "kind": kind,
            "path": [self.name, name],
        }

    def _member_runtime_config(
        self,
        base_config: Mapping[str, Any] | None,
        member: EnvironmentMemberConfig,
    ) -> dict[str, Any] | None:
        """Attach stable team-member identity to nested agent/tool events."""
        if base_config is None:
            return None
        member_id = f"{self.name}.{member.name}"
        merged = dict(base_config)
        base_metadata = merged.get("metadata")
        metadata = (
            dict(base_metadata) if isinstance(base_metadata, Mapping) else {}
        )
        metadata.update({
            "environment_id": self.name,
            "environment_member": member.name,
            "environment_member_id": member_id,
            "environment_member_role": member.role,
            "environment_member_path": [self.name, member.name],
            "agent": member.name,
            "agent_id": member_id,
        })
        merged["metadata"] = metadata

        base_tags = merged.get("tags")
        if isinstance(base_tags, str):
            tags = [base_tags]
        else:
            tags = list(base_tags) if base_tags else []
        for tag in (member.name, member_id, "environment_member"):
            if tag not in tags:
                tags.append(tag)
        merged["tags"] = tags
        return merged

    def _member_node(self, member: EnvironmentMemberConfig) -> dict[str, Any]:
        kind = (
            "environment"
            if isinstance(self.members.get(member.name), BaseEnvironment)
            else "agent"
        )
        return {
            "id": f"{self.name}.{member.name}",
            "name": member.name,
            "kind": kind,
            "role": member.role,
            "agent_class": member.agent,
            "path": [self.name, member.name],
        }

    def _topology_payload(self) -> dict[str, Any]:
        pi = self.config.pi
        nodes = [
            {
                "id": f"{self.name}.{pi.name}",
                "name": pi.name,
                "kind": "agent",
                "role": pi.role,
                "agent_class": pi.agent,
                "path": [self.name, pi.name],
            },
            *[self._member_node(member) for member in self.config.members],
        ]
        return {
            "kind": "agent_team",
            "name": self.name,
            "description": self.config.description,
            "nodes": nodes,
            "edges": [
                {
                    "source": f"{self.name}.{pi.name}",
                    "target": f"{self.name}.{member.name}",
                    "kind": "delegates_to",
                }
                for member in self.config.members
            ],
        }

    def _make_delegation_tool(
        self, member: EnvironmentMemberConfig
    ) -> StructuredTool:
        member_name = member.name
        role = member.role
        tool_name = f"delegate_to_{_slug_tool_name(member_name)}"

        def delegate(task: str, context: str = "") -> str:
            prompt = self._delegation_prompt(member, task=task, context=context)
            runtime_config = current_environment_config()
            events = self._events(runtime_config)
            source = self._source(self.config.pi.name)
            target = self._source(member_name)
            start = perf_counter()
            events.emit(
                f"Delegating to {member_name}",
                stage="delegation",
                phase="start",
                event_type="delegation_started",
                source=source,
                target=target,
                task=task,
                context=context,
                prompt=prompt,
            )
            self._trace_delegation(
                f"PI -> {member_name}",
                f"Task:\n{task}\n\nContext:\n{context or 'No additional context provided.'}",
            )
            try:
                member_config = self._member_runtime_config(
                    runtime_config, member
                )
                kwargs = {"config": member_config} if member_config else {}
                result = self.members[member_name].invoke(prompt, **kwargs)
            except BaseException as exc:
                events.emit(
                    f"Delegation to {member_name} failed",
                    stage="delegation",
                    phase="error",
                    event_type="delegation_failed",
                    level="error",
                    source=source,
                    target=target,
                    task=task,
                    context=context,
                    error=str(exc),
                    elapsed_seconds=perf_counter() - start,
                )
                raise
            text = result_to_text(result)
            events.emit(
                f"Delegation to {member_name} completed",
                stage="delegation",
                phase="end",
                event_type="delegation_completed",
                source=target,
                target=source,
                task=task,
                context=context,
                result=text,
                elapsed_seconds=perf_counter() - start,
            )
            self._trace_delegation(f"{member_name} -> PI", text)
            return text

        async def adelegate(task: str, context: str = "") -> str:
            prompt = self._delegation_prompt(member, task=task, context=context)
            runtime_config = current_environment_config()
            events = self._events(runtime_config)
            source = self._source(self.config.pi.name)
            target = self._source(member_name)
            start = perf_counter()
            await events.aemit(
                f"Delegating to {member_name}",
                stage="delegation",
                phase="start",
                event_type="delegation_started",
                source=source,
                target=target,
                task=task,
                context=context,
                prompt=prompt,
            )
            self._trace_delegation(
                f"PI -> {member_name}",
                f"Task:\n{task}\n\nContext:\n{context or 'No additional context provided.'}",
            )
            try:
                member_config = self._member_runtime_config(
                    runtime_config, member
                )
                kwargs = {"config": member_config} if member_config else {}
                result = await self._invoke_member_async(
                    self.members[member_name], prompt, **kwargs
                )
            except BaseException as exc:
                await events.aemit(
                    f"Delegation to {member_name} failed",
                    stage="delegation",
                    phase="error",
                    event_type="delegation_failed",
                    level="error",
                    source=source,
                    target=target,
                    task=task,
                    context=context,
                    error=str(exc),
                    elapsed_seconds=perf_counter() - start,
                )
                raise
            text = result_to_text(result)
            await events.aemit(
                f"Delegation to {member_name} completed",
                stage="delegation",
                phase="end",
                event_type="delegation_completed",
                source=target,
                target=source,
                task=task,
                context=context,
                result=text,
                elapsed_seconds=perf_counter() - start,
            )
            self._trace_delegation(f"{member_name} -> PI", text)
            return text

        return StructuredTool.from_function(
            func=delegate,
            coroutine=adelegate,
            name=tool_name,
            description=(
                f"Delegate work to team member '{member_name}' ({role}). "
                "Use this when that member's specialty is relevant. Provide a "
                "self-contained task and any context needed for independent work."
            ),
            args_schema=DelegateInput,
        )

    def _trace_delegation(self, label: str, message: str) -> None:
        """Log a small, explicit delegation trace."""
        if not self.trace_delegation:
            return
        text = message
        if (
            self.trace_character_limit > 0
            and len(text) > self.trace_character_limit
        ):
            text = text[: self.trace_character_limit] + "\n... [truncated]"
        logger.info(f"\n[AgentTeam:{self.name}] {label}\n{text}\n")

    def _delegation_prompt(
        self,
        member: EnvironmentMemberConfig,
        *,
        task: str,
        context: str,
    ) -> str:
        guidance = (
            f"\n\nMember-specific guidance:\n{member.prompt}"
            if member.prompt
            else ""
        )
        return (
            f"You are acting as team member '{member.name}' with role: {member.role}.\n"
            "You have been delegated a task by the team PI. Complete the delegated "
            "task thoroughly, using your available tools when appropriate, and return "
            "a clear writeup of methods, evidence, outputs, limitations, and any files "
            "created or commands needed to reproduce the work.\n"
            f"{guidance}\n\n"
            f"Overall context:\n{context or 'No additional context provided.'}\n\n"
            f"Delegated task:\n{task}"
        )

    def _team_roster(self) -> str:
        if not self.config.members:
            return "No team members are configured. Solve directly as PI."
        return "\n".join(
            f"- {member.name}: {member.role} ({member.agent})"
            for member in self.config.members
        )

    def _pi_prompt(self, task: str) -> str:
        description = (
            f"\nTeam description: {self.config.description}\n"
            if self.config.description
            else ""
        )
        pi_extra = (
            f"\nPI-specific guidance:\n{self.config.pi.prompt}\n"
            if self.config.pi.prompt
            else ""
        )
        return (
            "You are the PI/team leader of a hierarchical Agent Team. "
            "You are the user-facing coordinator responsible for satisfying the "
            "user's overall goal. Formulate an approach, decide which team members "
            "to assign work to, call member-delegation tools as needed, review their "
            "returns critically, request follow-up work if necessary, and synthesize "
            "a final answer for the user.\n"
            "Do not claim a member completed work unless you have delegated it and "
            "reviewed the result. You are personally responsible for final "
            "organization and presentation: integrate the delegated work into one "
            "clean, coherent, easily shareable answer with a clear structure, "
            "actionable conclusions, supporting evidence, limitations, and "
            "reproducibility details where relevant.\n"
            f"{description}{pi_extra}\n"
            "Available team members:\n"
            f"{self._team_roster()}\n\n"
            f"User task:\n{task}"
        )

    def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        return self._run_ainvoke_from_sync(inputs, **config)

    async def _ainvoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        task = str(inputs.get("task") or inputs.get("prompt") or inputs)
        runtime_config = runnable_config_from_kwargs(config)
        events = self._events(runtime_config)
        token = bind_current_environment_config(runtime_config)
        start = perf_counter()
        await events.aemit(
            f"Agent team {self.name} started",
            stage="team",
            phase="start",
            event_type="team_started",
            task=task,
            topology=self._topology_payload(),
        )
        await events.aemit(
            f"Agent team {self.name} topology declared",
            stage="team",
            phase="topology",
            event_type="topology_declared",
            topology=self._topology_payload(),
        )
        try:
            pi_kwargs = invocation_kwargs(config)
            pi_runtime_config = self._member_runtime_config(
                runtime_config, self.config.pi
            )
            if pi_runtime_config is not None:
                pi_kwargs["config"] = pi_runtime_config
            result = await self._invoke_member_async(
                self.pi, self._pi_prompt(task), **pi_kwargs
            )
        except BaseException as exc:
            await events.aemit(
                f"Agent team {self.name} failed",
                stage="team",
                phase="error",
                event_type="team_failed",
                level="error",
                task=task,
                error=str(exc),
                elapsed_seconds=perf_counter() - start,
            )
            raise
        finally:
            reset_current_environment_config(token)
        await events.aemit(
            f"Agent team {self.name} completed",
            stage="team",
            phase="end",
            event_type="team_completed",
            task=task,
            result=result_to_text(result),
            elapsed_seconds=perf_counter() - start,
        )
        return result
