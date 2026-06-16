from __future__ import annotations

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel

from ursa.util.events import EnvironmentEvents

from .base import (
    BaseEnvironment,
    invocation_kwargs,
    result_to_text,
    runnable_config_from_kwargs,
)
from .config import (
    AgentSymposiumConfig,
    EnvironmentMemberConfig,
    load_symposium_config,
)


class AgentSymposiumEnvironment(BaseEnvironment):
    """Competitive/collaborative multi-agent symposium environment.

    A symposium organizer dispatches the same complex problem to several members
    or nested teams. Members first work independently. The environment then sends
    all writeups to each member for critical review, explicitly instructing them
    not to modify reviewed work and to only view/run code as needed for assessment.
    Finally, each member revises its own work using the feedback it received and
    insights gained from reviewing others. The organizer synthesizes the final
    symposium report.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        config: AgentSymposiumConfig
        | Mapping[str, Any]
        | str
        | Path
        | None = None,
        name: str | None = None,
        group: str | None = None,
        organizer: EnvironmentMemberConfig | Mapping[str, Any] | None = None,
        members: list[EnvironmentMemberConfig | Mapping[str, Any]]
        | None = None,
        workspace: str | Path | None = None,
        revision_rounds: int | None = None,
        persist_members: bool = True,
        **kwargs: Any,
    ):
        symposium_config = self._coerce_config(
            config=config,
            name=name,
            group=group,
            organizer=organizer,
            members=members,
            workspace=workspace,
            revision_rounds=revision_rounds,
        )
        super().__init__(
            llm,
            name=symposium_config.name,
            group=symposium_config.group,
            workspace=symposium_config.workspace or workspace,
            persist_members=persist_members,
            **kwargs,
        )
        self.config = symposium_config
        self.members = {
            member.name: self.build_member(member)
            for member in self.config.members
        }
        self.member_configs = {
            member.name: member for member in self.config.members
        }
        self.organizer = self.build_member(self.config.organizer)

    def _events(self, config: Mapping[str, Any] | None) -> EnvironmentEvents:
        return EnvironmentEvents(
            environment=self.name,
            config=config,
            environment_type="agent_symposium",
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

    def _environment_source(self) -> dict[str, Any]:
        return {
            "id": self.name,
            "name": self.name,
            "kind": "environment",
            "path": [self.name],
        }

    def _member_node(self, member: EnvironmentMemberConfig) -> dict[str, Any]:
        member_obj = self.members.get(member.name)
        kind = (
            "environment"
            if isinstance(member_obj, BaseEnvironment)
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
        organizer = self.config.organizer
        organizer_kind = (
            "environment"
            if isinstance(self.organizer, BaseEnvironment)
            else "agent"
        )
        nodes = [
            {
                "id": f"{self.name}.{organizer.name}",
                "name": organizer.name,
                "kind": organizer_kind,
                "role": organizer.role,
                "agent_class": organizer.agent,
                "path": [self.name, organizer.name],
            },
            *[self._member_node(member) for member in self.config.members],
        ]
        edges = []
        for member in self.config.members:
            edges.append({
                "source": self.name,
                "target": f"{self.name}.{member.name}",
                "kind": "dispatches_to",
            })
            edges.append({
                "source": f"{self.name}.{member.name}",
                "target": f"{self.name}.{organizer.name}",
                "kind": "synthesized_by",
            })
        return {
            "kind": "agent_symposium",
            "name": self.name,
            "description": self.config.description,
            "revision_rounds": self.config.revision_rounds,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> "AgentSymposiumEnvironment":
        return cls(llm=llm, config=load_symposium_config(path), **kwargs)

    def _coerce_config(
        self,
        *,
        config: AgentSymposiumConfig | Mapping[str, Any] | str | Path | None,
        name: str | None,
        group: str | None,
        organizer: EnvironmentMemberConfig | Mapping[str, Any] | None,
        members: list[EnvironmentMemberConfig | Mapping[str, Any]] | None,
        workspace: str | Path | None,
        revision_rounds: int | None,
    ) -> AgentSymposiumConfig:
        if isinstance(config, (str, Path)):
            base = load_symposium_config(config)
        elif isinstance(config, Mapping):
            base = AgentSymposiumConfig.from_mapping(config)
        elif isinstance(config, AgentSymposiumConfig):
            base = config
        else:
            organizer_cfg = self._coerce_member(
                organizer
                or {
                    "name": "organizer",
                    "role": "Symposium organizer / final synthesizer",
                    "agent": "ChatAgent",
                }
            )
            member_cfgs = [self._coerce_member(m) for m in (members or [])]
            base = AgentSymposiumConfig(
                name=name or "agent_symposium",
                group=group or "default",
                organizer=organizer_cfg,
                members=member_cfgs,
                workspace=str(workspace) if workspace else None,
                revision_rounds=revision_rounds or 1,
            )
        return AgentSymposiumConfig(
            name=name or base.name,
            group=group or base.group,
            description=base.description,
            organizer=base.organizer,
            members=base.members,
            workspace=str(workspace) if workspace else base.workspace,
            defaults=base.defaults,
            revision_rounds=revision_rounds
            if revision_rounds is not None
            else base.revision_rounds,
        )

    @staticmethod
    def _coerce_member(
        member: EnvironmentMemberConfig | Mapping[str, Any],
    ) -> EnvironmentMemberConfig:
        if isinstance(member, EnvironmentMemberConfig):
            return member
        return EnvironmentMemberConfig.from_mapping(member)

    def _member_roster(self) -> str:
        return (
            "\n".join(
                f"- {member.name}: {member.role} ({member.agent})"
                for member in self.config.members
            )
            or "No symposium members configured."
        )

    def _initial_prompt(
        self, member: EnvironmentMemberConfig, task: str
    ) -> str:
        extra = (
            f"\n\nMember-specific guidance:\n{member.prompt}"
            if member.prompt
            else ""
        )
        return (
            f"You are symposium participant '{member.name}' with role: {member.role}.\n"
            "Work independently on the complex problem below. Produce a detailed, "
            "self-contained writeup of your approach, methods, assumptions, files or "
            "commands used, findings, uncertainty, and reproducibility instructions.\n"
            "Your work will be reviewed by other symposium participants. Clean, "
            "compact organization, documentation, and reproducibility are important "
            "parts of how your contribution will be assessed.\n"
            f"{extra}\n\nProblem:\n{task}"
        )

    def _review_prompt(
        self,
        reviewer: EnvironmentMemberConfig,
        task: str,
        writeups: Mapping[str, str],
    ) -> str:
        writeup_block = "\n\n".join(
            f"## Writeup from {name}\n{writeup}"
            for name, writeup in writeups.items()
        )
        return (
            f"You are symposium reviewer '{reviewer.name}' with role: {reviewer.role}.\n"
            "Read all submitted writeups, including your own if present. Assess the "
            "quality of the findings, compare methods and conclusions, and write a "
            "fair but critical review for each writeup. Discuss strengths, weaknesses, "
            "reproducibility, evidence quality, missing checks, and concrete ways to "
            "improve the solution.\n\n"
            "Important review constraints:\n"
            "- Do not change, edit, overwrite, or reorganize any work you are reviewing.\n"
            "- If code or files are referenced, you may inspect or run them only to assess "
            "correctness/reproducibility.\n"
            "- Make clear which findings are well supported and which are speculative.\n"
            "- Your own work will also be judged on clarity, compact organization, "
            "documentation, and reproducibility.\n\n"
            f"Original problem:\n{task}\n\n"
            f"Submitted writeups:\n{writeup_block}"
        )

    def _revision_prompt(
        self,
        member: EnvironmentMemberConfig,
        task: str,
        own_writeup: str,
        all_writeups: Mapping[str, str],
        reviews: Mapping[str, str],
        round_index: int,
    ) -> str:
        other_writeups = "\n\n".join(
            f"## {name}\n{writeup}" for name, writeup in all_writeups.items()
        )
        review_block = "\n\n".join(
            f"## Review from {name}\n{review}"
            for name, review in reviews.items()
        )
        return (
            f"You are symposium participant '{member.name}' revising your own work "
            f"after review round {round_index}.\n"
            "Use the feedback you received and anything you learned from reviewing "
            "other submissions to improve your own solution. You may change only your "
            "own work/artifacts. Do not modify the work of other symposium members.\n"
            "Return a revised detailed writeup with clear improvements, evidence, "
            "limitations, and reproducibility instructions.\n\n"
            f"Original problem:\n{task}\n\n"
            f"Your previous writeup:\n{own_writeup}\n\n"
            f"All writeups you saw:\n{other_writeups}\n\n"
            f"Reviews from symposium members:\n{review_block}"
        )

    def _synthesis_prompt(
        self,
        task: str,
        writeups: Mapping[str, str],
        reviews: Mapping[str, str],
    ) -> str:
        description = (
            f"\nSymposium description: {self.config.description}\n"
            if self.config.description
            else ""
        )
        writeup_block = "\n\n".join(
            f"## Final writeup from {name}\n{writeup}"
            for name, writeup in writeups.items()
        )
        review_block = "\n\n".join(
            f"## Review by {name}\n{review}" for name, review in reviews.items()
        )
        return (
            "You are the organizer of an URSA Agent Symposium. Synthesize the final "
            "results for the user after independent work, peer review, and revision. "
            "Compare participant outputs, identify consensus and disagreement, judge "
            "evidence quality, and provide a final recommendation or solution.\n"
            f"{description}\n"
            "Symposium members:\n"
            f"{self._member_roster()}\n\n"
            f"Original problem:\n{task}\n\n"
            f"Final participant writeups:\n{writeup_block}\n\n"
            f"Peer reviews:\n{review_block}"
        )

    def _invoke(
        self, inputs: Mapping[str, Any], **config: Any
    ) -> dict[str, Any]:
        return self._run_ainvoke_from_sync(inputs, **config)

    async def _member_writeup(
        self,
        member_config: EnvironmentMemberConfig,
        prompt: str,
        invoke_kwargs: Mapping[str, Any],
        events: EnvironmentEvents,
        *,
        event_prefix: str,
        phase_name: str,
        round_index: int | None = None,
    ) -> tuple[str, str]:
        member_name = member_config.name
        source = self._source(member_name)
        start = perf_counter()
        await events.aemit(
            f"{member_name} started {phase_name}",
            stage="symposium",
            phase=phase_name,
            event_type=f"{event_prefix}_started",
            source=source,
            target=self._environment_source(),
            member=member_name,
            round_index=round_index,
            prompt=prompt,
        )
        try:
            result = await self._invoke_member_async(
                self.members[member_name], prompt, **invoke_kwargs
            )
        except BaseException as exc:
            await events.aemit(
                f"{member_name} failed {phase_name}",
                stage="symposium",
                phase=phase_name,
                event_type=f"{event_prefix}_failed",
                level="error",
                source=source,
                target=self._environment_source(),
                member=member_name,
                round_index=round_index,
                error=str(exc),
                elapsed_seconds=perf_counter() - start,
            )
            raise
        text = result_to_text(result)
        await events.aemit(
            f"{member_name} completed {phase_name}",
            stage="symposium",
            phase=phase_name,
            event_type=f"{event_prefix}_completed",
            source=source,
            target=self._environment_source(),
            member=member_name,
            round_index=round_index,
            result=text,
            elapsed_seconds=perf_counter() - start,
        )
        return member_name, text

    async def _ainvoke(
        self, inputs: Mapping[str, Any], **config: Any
    ) -> dict[str, Any]:
        task = str(inputs.get("task") or inputs.get("prompt") or inputs)
        if not self.config.members:
            raise ValueError(
                "AgentSymposiumEnvironment requires at least one member."
            )

        runtime_config = runnable_config_from_kwargs(config)
        events = self._events(runtime_config)
        invoke_kwargs = invocation_kwargs(config)
        start = perf_counter()
        await events.aemit(
            f"Agent symposium {self.name} started",
            stage="symposium",
            phase="start",
            event_type="symposium_started",
            task=task,
            topology=self._topology_payload(),
        )
        await events.aemit(
            f"Agent symposium {self.name} topology declared",
            stage="symposium",
            phase="topology",
            event_type="topology_declared",
            topology=self._topology_payload(),
        )
        try:
            await events.aemit(
                "Initial symposium work started",
                stage="symposium",
                phase="initial_work",
                event_type="symposium_phase_started",
                task=task,
            )
            initial_pairs = await asyncio.gather(*[
                self._member_writeup(
                    member_config,
                    self._initial_prompt(member_config, task),
                    invoke_kwargs,
                    events,
                    event_prefix="initial_work",
                    phase_name="initial_work",
                )
                for member_config in self.config.members
            ])
            initial_writeups = dict(initial_pairs)
            await events.aemit(
                "Initial symposium work completed",
                stage="symposium",
                phase="initial_work",
                event_type="symposium_phase_completed",
                result=initial_writeups,
            )

            current_writeups = dict(initial_writeups)
            review_rounds: list[dict[str, str]] = []
            latest_reviews: dict[str, str] = {}

            for round_index in range(
                1, max(1, self.config.revision_rounds) + 1
            ):
                await events.aemit(
                    f"Review round {round_index} started",
                    stage="symposium",
                    phase="review",
                    event_type="review_round_started",
                    round_index=round_index,
                    writeups=current_writeups,
                )
                review_pairs = await asyncio.gather(*[
                    self._member_writeup(
                        reviewer_config,
                        self._review_prompt(
                            reviewer_config, task, current_writeups
                        ),
                        invoke_kwargs,
                        events,
                        event_prefix="review",
                        phase_name="review",
                        round_index=round_index,
                    )
                    for reviewer_config in self.config.members
                ])
                round_reviews = dict(review_pairs)
                latest_reviews = round_reviews
                review_rounds.append(round_reviews)
                await events.aemit(
                    f"Review round {round_index} completed",
                    stage="symposium",
                    phase="review",
                    event_type="review_round_completed",
                    round_index=round_index,
                    reviews=round_reviews,
                )

                await events.aemit(
                    f"Revision round {round_index} started",
                    stage="symposium",
                    phase="revision",
                    event_type="revision_round_started",
                    round_index=round_index,
                    reviews=round_reviews,
                )
                revision_pairs = await asyncio.gather(*[
                    self._member_writeup(
                        member_config,
                        self._revision_prompt(
                            member_config,
                            task,
                            current_writeups[member_config.name],
                            current_writeups,
                            round_reviews,
                            round_index,
                        ),
                        invoke_kwargs,
                        events,
                        event_prefix="revision",
                        phase_name="revision",
                        round_index=round_index,
                    )
                    for member_config in self.config.members
                ])
                current_writeups = dict(revision_pairs)
                await events.aemit(
                    f"Revision round {round_index} completed",
                    stage="symposium",
                    phase="revision",
                    event_type="revision_round_completed",
                    round_index=round_index,
                    writeups=current_writeups,
                )

            synthesis_prompt = self._synthesis_prompt(
                task, current_writeups, latest_reviews
            )
            synth_start = perf_counter()
            await events.aemit(
                "Symposium synthesis started",
                stage="symposium",
                phase="synthesis",
                event_type="synthesis_started",
                source=self._source(self.config.organizer.name),
                task=task,
                prompt=synthesis_prompt,
            )
            organizer_result = await self._invoke_member_async(
                self.organizer,
                synthesis_prompt,
                **invoke_kwargs,
            )
            final = result_to_text(organizer_result)
            await events.aemit(
                "Symposium synthesis completed",
                stage="symposium",
                phase="synthesis",
                event_type="synthesis_completed",
                source=self._source(self.config.organizer.name),
                result=final,
                elapsed_seconds=perf_counter() - synth_start,
            )
            result = {
                "task": task,
                "initial_writeups": initial_writeups,
                "review_rounds": review_rounds,
                "reviews": latest_reviews,
                "final_writeups": current_writeups,
                "organizer_result": organizer_result,
                "final": final,
            }
        except BaseException as exc:
            await events.aemit(
                f"Agent symposium {self.name} failed",
                stage="symposium",
                phase="error",
                event_type="symposium_failed",
                level="error",
                task=task,
                error=str(exc),
                elapsed_seconds=perf_counter() - start,
            )
            raise
        await events.aemit(
            f"Agent symposium {self.name} completed",
            stage="symposium",
            phase="end",
            event_type="symposium_completed",
            task=task,
            result=result["final"],
            elapsed_seconds=perf_counter() - start,
        )
        return result
