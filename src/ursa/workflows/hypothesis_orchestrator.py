"""Sub-agent orchestrator variant of the hypothesis workflow (Plan D).

This workflow is a *dynamic-topology* counterpart to
:class:`~ursa.workflows.hypothesis_symposium.HypothesisSymposiumWorkflow`. It is
built for empirical comparison: instead of a fixed fan-out + symposium graph, an
orchestrating agent is given tools to spawn persistent, per-hypothesis
investigation agents on demand, then task each of them with reviewing the
others.

Design:

* ``build_hypotheses`` runs the :class:`HypothesizerAgent` to produce the
  hypothesis space (same as the symposium workflow).
* A run-scoped :class:`_InvestigatorRegistry` mints a *timestamped* persistent
  ``agent_name`` per hypothesis (mirroring the ``ursa save-agent`` timestamp
  convention ``%Y%m%d_%H%M%S``). Timestamping guarantees uniqueness across
  concurrent dashboard runs; caching by hypothesis label guarantees the *same*
  agent both investigates a hypothesis and later reviews the others, so it
  accumulates durable memory across phases.
* The orchestrator is an :class:`ExecutionAgent` given two extra tools:
  ``spawn_investigator`` (create-or-reuse a persistent investigator and invoke
  it) and ``list_investigators`` (recall minted identities). The orchestrator is
  prompted to record the minted identities in its experiences and to avoid
  calling the same investigator twice in one parallel batch of tool calls.
* Each investigator works in its own private sub-workspace, but before every
  delegation its peers' workspaces are symlinked under ``peers/<label>/`` inside
  it. That lets the review round read peers' *real* artifacts instead of only the
  summary text, while the write-tool sandbox keeps those paths read-only.

Observability/testability is deliberately weaker than the static symposium graph
(dynamic topology), which is the expected tradeoff for adaptive orchestration.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ursa.agents.execution_agent import ExecutionAgent
from ursa.agents.hypothesizer_agent import (
    DEFAULT_HYPOTHESIS_EXPERIENCE,
    HypothesizerAgent,
)
from ursa.util.events import EnvironmentEvents
from ursa.workflows.base_workflow import BaseWorkflow, InputLike
from ursa.workflows.hypothesis_symposium import Hypothesis

logger = logging.getLogger(__name__)

__all__ = [
    "HypothesisOrchestratorWorkflow",
    "SpawnInvestigatorInput",
]


def _timestamp() -> str:
    """Match the ``ursa save-agent`` timestamp convention."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str, *, max_words: int = 4) -> str:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    return "_".join(words[:max_words]) or "hypothesis"


class SpawnInvestigatorInput(BaseModel):
    """Input schema for the ``spawn_investigator`` orchestrator tool."""

    hypothesis_label: str = Field(
        ...,
        description=(
            "Short stable label for the hypothesis this investigator owns, e.g. "
            "'H1' or 'H2_expected_improvement'. Reuse the SAME label to reach the "
            "same persistent agent again (e.g. to have it review others). Do not "
            "call the same label twice within one parallel batch of tool calls."
        ),
    )
    task: str = Field(
        ...,
        description=(
            "Self-contained instructions for the investigator: what to "
            "investigate or review, the evidence to gather, and the deliverable."
        ),
    )


class _InvestigatorRegistry:
    """Run-scoped registry of persistent per-hypothesis investigators.

    One timestamped persistent ``agent_name`` is minted per hypothesis label and
    reused for the whole run so the same agent investigates then reviews.

    Each investigator writes in its own private sub-workspace. So that the review
    phase can inspect peers' *actual artifacts* (plots, CSVs, scripts) instead of
    only the summary text pasted into the prompt, every investigator workspace
    gets a ``peers/`` directory holding one symlink per other investigator. Reads
    through those symlinks succeed; writes are refused by the write-tool sandbox
    (it ``resolve()``s the path and rejects anything outside the agent's own
    workspace), so peer work is effectively read-only.
    """

    #: Sub-directory inside each investigator workspace holding peer symlinks.
    PEER_DIRNAME = "peers"

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        workspace: Path | None,
        group: str | None,
        run_stamp: str,
    ) -> None:
        self._llm = llm
        self._workspace = workspace
        self._group = group
        self._run_stamp = run_stamp
        self._agents: dict[str, ExecutionAgent] = {}
        self._names: dict[str, str] = {}

    @property
    def identities(self) -> dict[str, str]:
        """Mapping of hypothesis label -> minted persistent agent_name."""
        return dict(self._names)

    def _mint_name(self, label: str) -> str:
        return f"hyp_{_slug(label)}_{self._run_stamp}"[:60]

    def get_or_create(self, label: str) -> ExecutionAgent:
        key = label.strip().lower()
        if key in self._agents:
            return self._agents[key]
        agent_name = self._mint_name(label)
        kwargs: dict[str, Any] = {"agent_name": agent_name}
        if self._workspace is not None:
            # Give each investigator its own sub-workspace so their artifacts do
            # not collide, while remaining under the run workspace.
            kwargs["workspace"] = str(self._workspace / agent_name)
        if self._group is not None:
            kwargs["group"] = self._group
        agent = ExecutionAgent(self._llm, **kwargs)
        self._agents[key] = agent
        self._names[key] = agent_name
        return agent

    def _peer_label(self, key: str) -> str:
        """Directory-safe label used for a peer symlink name."""
        return _slug(key, max_words=4)

    def workspace_for(self, label: str) -> Path | None:
        """Private workspace of one investigator, if workspaces are in use."""
        if self._workspace is None:
            return None
        agent_name = self._names.get(label.strip().lower())
        if agent_name is None:
            return None
        return self._workspace / agent_name

    def link_peers(self, label: str) -> dict[str, str]:
        """Expose every *other* investigator's workspace under ``peers/``.

        Called immediately before an investigator runs, so the links reflect
        whichever peers exist at that moment: the initial investigation sees an
        empty/partial ``peers/`` while the review round sees everyone.

        Returns a mapping of peer label -> relative path (``peers/<label>``) for
        the peers that were successfully linked, so the caller can tell the agent
        where to look. Failures are non-fatal: symlink creation needs privileges
        on some platforms, and losing peer visibility must not fail the run.
        """
        own = self.workspace_for(label)
        if own is None:
            return {}
        key = label.strip().lower()
        exposed: dict[str, str] = {}
        peer_root = own / self.PEER_DIRNAME
        for peer_key in self._names:
            if peer_key == key:
                continue
            target = self.workspace_for(peer_key)
            if target is None or not target.exists():
                continue
            peer_name = self._peer_label(peer_key)
            link = peer_root / peer_name
            try:
                peer_root.mkdir(parents=True, exist_ok=True)
                # Refresh the link so it always points at the current target.
                if link.is_symlink() or link.exists():
                    if link.is_symlink() or link.is_file():
                        link.unlink()
                    else:
                        # A real directory here is not ours to delete; skip it
                        # rather than risk destroying an agent's own files.
                        continue
                link.symlink_to(target.resolve(), target_is_directory=True)
            except OSError as exc:
                logger.warning(
                    "Could not expose peer workspace %s to %s: %s",
                    peer_key,
                    label,
                    exc,
                )
                continue
            exposed[peer_key] = f"{self.PEER_DIRNAME}/{peer_name}"
        return exposed


_ORCHESTRATOR_PROMPT = """\
You are the ORCHESTRATOR for a hypothesis investigation. You do not do the \
scientific investigation yourself; you coordinate a set of persistent \
per-hypothesis investigator sub-agents via tools.

Competing hypotheses for the question:
{hypothesis_block}

Follow this protocol:
1. For EACH hypothesis above, call `spawn_investigator` once with a stable \
`hypothesis_label` (use the H-number, e.g. 'H1', 'H2', ...) and a self-contained \
task instructing that investigator to gather concrete evidence FOR and AGAINST \
its own hypothesis (write and run code, produce data/plots, quantify results). \
You may issue several `spawn_investigator` calls in parallel, but NEVER call the \
same `hypothesis_label` more than once in a single parallel batch: one \
in-flight call per label at a time (each label is one persistent agent).
2. After every investigator has produced its evidence, run a review round: call \
each investigator AGAIN (reuse the SAME `hypothesis_label`) and give it the \
other investigators' findings, asking it to critically review them and to say \
whether the combined evidence supports or weakens its own hypothesis. \
Each investigator can also READ its peers' actual workspace artifacts (data \
files, plots, scripts) at the relative paths `peers/<label>/` inside its own \
workspace, so instruct it to inspect the underlying evidence directly and not \
rely only on the summaries you paste in. Tell it explicitly that peer paths are \
READ-ONLY: it must not write to, edit, or delete anything under `peers/` (such \
writes are refused) and should copy anything it needs into its own workspace \
first. Also warn it that peer directories do not show up in directory listings \
and that it should avoid unbounded recursive scans (`grep -r`, `find`) from its \
workspace root, using the explicit peer paths instead.
3. Record the minted investigator identities you learn from `list_investigators` \
in your experiences (write_experience) so you never lose track of which agent \
owns which hypothesis across phases.
4. Synthesize a final verdict: which hypothesis is best supported by the \
combined, cross-reviewed evidence, with justification and remaining uncertainty.

Question:
{query}
"""


class HypothesisOrchestratorWorkflow(BaseWorkflow):
    """Hypothesize, then let an orchestrator spawn/review per-hypothesis agents.

    A dynamic-topology alternative to
    :class:`~ursa.workflows.hypothesis_symposium.HypothesisSymposiumWorkflow`,
    intended for empirical comparison. Every scientific phase is exposed as an
    overridable hook.

    Because the topology is dynamic there is no environment object to hand to
    ``run_with_visualization``, so this workflow records its own environment run:
    it opens an :func:`environment_run_recorder` for the whole orchestration and
    threads the resulting run config into both the orchestrator agent and every
    spawned investigator. That captures (a) explicit orchestrator->investigator
    delegation events emitted here and (b) all structured events emitted *by* the
    sub-agents themselves, streamed live to ``events.jsonl``. Delegation traffic
    is additionally rendered to the stdout logs so progress is visible without a
    dashboard.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        hypothesizer: HypothesizerAgent | None = None,
        experience_filename: str = DEFAULT_HYPOTHESIS_EXPERIENCE,
        max_hypotheses: int = 5,
        orchestrator: ExecutionAgent | None = None,
        workspace: str | Path | None = None,
        group: str | None = None,
        name: str = "hypothesis_orchestrator",
        visualize: bool = True,
        trace: bool = True,
        trace_character_limit: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.workspace = Path(workspace) if workspace else None
        self.group = group
        # `name`/`group` are what the visualization recorder reads off an
        # "environment"; this workflow acts as its own environment.
        self.name = name
        self.visualize = bool(visualize)
        self.trace = bool(trace)
        self.trace_character_limit = int(trace_character_limit)
        self.orchestrator_run_id: str | None = None
        # Per-run observability context, published by `_invoke` so overridable
        # hooks keep their original signatures.
        self._run_config: Mapping[str, Any] | None = None
        self._run_events: EnvironmentEvents | None = None
        self.experience_filename = (
            HypothesizerAgent._validate_experience_filename(experience_filename)
        )
        self.hypothesizer = hypothesizer or HypothesizerAgent(
            llm, experience_filename=self.experience_filename
        )
        self.max_hypotheses = max(1, int(max_hypotheses))
        self._orchestrator_override = orchestrator

    # ------------------------------------------------------- overridable hooks

    def format_result(self, result: Mapping[str, Any]) -> str:
        """Return only the final write-up, not the whole state.

        The full state carries bulky intermediate data (raw orchestrator result,
        parsed hypotheses, the hypothesis-space markdown, investigator
        identities). Returning all of it makes the dashboard output unreadable,
        so surface just the detailed final answer, falling back to ``summary``
        if a subclass produced one instead.
        """
        detailed = result.get("final")
        summary = result.get("summary")
        text = detailed or summary
        if not text:
            raise ValueError("Orchestration completed without a response.")
        return str(text).strip()

    def build_hypotheses(
        self, query: str, context: str, config: Mapping[str, Any] | None
    ) -> list[Hypothesis]:
        """Produce competing hypotheses via the HypothesizerAgent."""
        result = self.hypothesizer.invoke(
            {
                "query": query,
                "new_information": query,
                "context": context,
                "experience_filename": self.experience_filename,
                "revision_history": [],
            },
            config=config,
        )
        artifact = self._artifact_text(result)
        self._last_hypothesis_space = artifact
        return self._parse_hypotheses(artifact, query)

    @staticmethod
    def _artifact_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            text = result.get("hypothesis_space_markdown")
            if text:
                return str(text)
            summary = result.get("summary")
            if summary:
                return str(summary)
        return HypothesizerAgent._response_text(result)

    # ------------------------------------------------------------- run surface

    def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        cfg = config.get("config")
        query = str(inputs.get("query") or inputs.get("task") or "")
        context = str(inputs.get("context", ""))
        self._last_hypothesis_space = ""

        hypotheses = self.build_hypotheses(query, context, cfg)
        registry = self._build_registry()

        prompt = _ORCHESTRATOR_PROMPT.format(
            hypothesis_block=self._hypothesis_block(hypotheses),
            query=query,
        )
        # Open a recorder for the whole orchestration. `run_config` carries the
        # recorder callback, so it is threaded into the orchestrator AND every
        # investigator: their own structured events land in the same stream.
        with self._recording(task=query, config=cfg) as run_config:
            events = self._events(run_config)
            self._emit_topology(events, hypotheses)
            # Published as instance state (not hook arguments) so that user
            # subclasses overriding `_build_orchestrator`/`_make_spawn_tools`
            # with the original single-argument signature keep working.
            self._run_config = run_config
            self._run_events = events
            orchestrator = self._build_orchestrator(registry)
            start = perf_counter()
            try:
                result = orchestrator.invoke(prompt, config=run_config)
            except BaseException as exc:
                events.emit(
                    "Orchestration failed",
                    stage="orchestration",
                    phase="error",
                    event_type="orchestration_failed",
                    level="error",
                    error=str(exc),
                    elapsed_seconds=perf_counter() - start,
                )
                raise
            events.emit(
                "Orchestration completed",
                stage="orchestration",
                phase="end",
                event_type="orchestration_completed",
                investigators=registry.identities,
                elapsed_seconds=perf_counter() - start,
            )

        return {
            "query": query,
            "hypotheses": [h.__dict__ for h in hypotheses],
            "hypothesis_space_markdown": self._last_hypothesis_space,
            "investigator_identities": registry.identities,
            "orchestrator_run_id": self.orchestrator_run_id,
            "orchestrator_result": result,
            "final": self._result_text(result),
        }

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        if isinstance(inputs, str):
            return {"query": inputs}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    # --------------------------------------------------------------- internals

    @contextmanager
    def _recording(
        self, *, task: str, config: Mapping[str, Any] | None
    ) -> Iterator[Mapping[str, Any] | None]:
        """Yield a run config carrying the visualization recorder callback.

        Degrades gracefully: if visualization is disabled or the recorder cannot
        be created, the original config is yielded unchanged so the run still
        proceeds (and stdout tracing still provides progress feedback).
        """
        if not self.visualize:
            yield config
            return
        try:
            from ursa.environments import environment_run_recorder
        except Exception:  # pragma: no cover - defensive import guard
            logger.warning(
                "Orchestrator visualization unavailable; running unrecorded.",
                exc_info=True,
            )
            yield config
            return

        run_id = self._new_run_id()
        # Validate viability *before* entering the recorder so a setup failure
        # cannot be confused with a failure from the orchestration body.
        try:
            from ursa.security import validate_group_name

            validate_group_name(self.group or "default")
        except Exception:
            logger.warning(
                "Orchestrator group %r is not recordable; running unrecorded.",
                self.group,
                exc_info=True,
            )
            self.orchestrator_run_id = None
            yield config
            return

        with environment_run_recorder(
            self, task=task, config=config, run_id=run_id
        ) as (recorder, run_config):
            self.orchestrator_run_id = recorder.run_id
            logger.info(
                "Orchestrator visualization run started: group=%s "
                "run_id=%s (events stream live to the dashboard)",
                recorder.group,
                recorder.run_id,
            )
            yield run_config

    @staticmethod
    def _new_run_id() -> str:
        """Readable, sortable, unique run id for one orchestration."""
        return (
            f"hyporch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid4().hex[:8]}"
        )

    def _events(self, config: Mapping[str, Any] | None) -> EnvironmentEvents:
        """Event emitter identifying this workflow as the environment."""
        return EnvironmentEvents(
            environment=self.name,
            config=config,
            environment_type="hypothesis_orchestrator",
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

    def _emit_topology(
        self, events: EnvironmentEvents, hypotheses: Sequence[Hypothesis]
    ) -> None:
        """Declare the (initially known) orchestrator/investigator topology."""
        nodes = [
            {
                **self._source("orchestrator"),
                "role": "orchestrator",
            },
            *[
                {
                    **self._source(f"H{h.index}"),
                    "role": f"investigator for H{h.index}: {h.statement}",
                }
                for h in hypotheses
            ],
        ]
        topology = {
            "kind": "hypothesis_orchestrator",
            "name": self.name,
            "description": "Orchestrator spawning per-hypothesis investigators",
            "nodes": nodes,
            "edges": [
                {
                    "source": f"{self.name}.orchestrator",
                    "target": f"{self.name}.H{h.index}",
                    "kind": "spawns",
                }
                for h in hypotheses
            ],
        }
        events.emit(
            f"Hypothesis orchestrator {self.name} topology declared",
            stage="orchestration",
            phase="topology",
            event_type="topology_declared",
            topology=topology,
        )
        events.emit(
            f"Hypothesis orchestrator {self.name} started",
            stage="orchestration",
            phase="start",
            event_type="orchestration_started",
            hypothesis_count=len(hypotheses),
        )

    def _trace(self, label: str, message: str) -> None:
        """Render one orchestrator<->investigator message to the stdout logs."""
        if not self.trace:
            return
        text = message
        if (
            self.trace_character_limit > 0
            and len(text) > self.trace_character_limit
        ):
            text = text[: self.trace_character_limit] + "\n... [truncated]"
        logger.info(f"\n[HypothesisOrchestrator:{self.name}] {label}\n{text}\n")

    @staticmethod
    def _result_text(result: Any) -> str:
        if isinstance(result, Mapping) and result.get("messages"):
            last = result["messages"][-1]
            text = getattr(last, "text", None)
            if isinstance(text, str) and text:
                return text
            content = getattr(last, "content", last)
            return content if isinstance(content, str) else str(content)
        return HypothesizerAgent._response_text(result)

    def _build_registry(self) -> _InvestigatorRegistry:
        return _InvestigatorRegistry(
            self.llm,
            workspace=self.workspace,
            group=self.group,
            run_stamp=_timestamp(),
        )

    def _build_orchestrator(
        self, registry: _InvestigatorRegistry
    ) -> ExecutionAgent:
        tools = self._make_spawn_tools(registry)
        if self._orchestrator_override is not None:
            orchestrator = self._orchestrator_override
            orchestrator.add_tool(tools)
            return orchestrator
        kwargs: dict[str, Any] = {"extra_tools": tools}
        if self.workspace is not None:
            kwargs["workspace"] = str(self.workspace / "orchestrator")
        if self.group is not None:
            kwargs["group"] = self.group
        return ExecutionAgent(self.llm, **kwargs)

    def _investigator_config(
        self, run_config: Mapping[str, Any] | None, label: str, agent_name: str
    ) -> dict[str, Any] | None:
        """Attach stable investigator identity to nested sub-agent events.

        Threading this config into the investigator is what makes the sub-agent's
        *own* structured events show up in the run stream, attributed to it.
        """
        if run_config is None:
            return None
        member_id = f"{self.name}.{label}"
        merged = dict(run_config)
        base_metadata = merged.get("metadata")
        metadata = (
            dict(base_metadata) if isinstance(base_metadata, Mapping) else {}
        )
        metadata.update({
            "environment_id": self.name,
            "environment_member": label,
            "environment_member_id": member_id,
            "environment_member_role": f"investigator for {label}",
            "environment_member_path": [self.name, label],
            "agent": label,
            "agent_id": member_id,
            "investigator_agent_name": agent_name,
        })
        merged["metadata"] = metadata

        base_tags = merged.get("tags")
        if isinstance(base_tags, str):
            tags = [base_tags]
        else:
            tags = list(base_tags) if base_tags else []
        for tag in (label, member_id, "environment_member"):
            if tag not in tags:
                tags.append(tag)
        merged["tags"] = tags
        return merged

    def _with_peer_access(self, registry: Any, label: str, task: str) -> str:
        """Link peer workspaces and append the read-only access note to the task.

        Guarded so that alternative/stub registries without peer support (and any
        filesystem failure) degrade to the original task rather than failing the
        delegation.
        """
        linker = getattr(registry, "link_peers", None)
        if not callable(linker):
            return task
        try:
            exposed = linker(label)
        except Exception as exc:  # noqa: BLE001 - never fail a delegation
            logger.warning(
                "Peer workspace linking failed for %s: %s", label, exc
            )
            return task
        if not isinstance(exposed, Mapping) or not exposed:
            return task
        return f"{task}{self._peer_access_note(exposed)}"

    @staticmethod
    def _peer_access_note(exposed: Mapping[str, str]) -> str:
        """Tell an investigator where peer artifacts live and that they are read-only.

        This has to be explicit: neither ``list_workspace_files`` nor the
        dashboard follows symlinks, so the peer directories are invisible to
        directory listings even though ``read_file``/``run_command`` can read
        through them by path.
        """
        if not exposed:
            return ""
        lines = "\n".join(
            f"- {label.upper()}: {path}/"
            for label, path in sorted(exposed.items())
        )
        return (
            "\n\nPEER WORKSPACE ACCESS (READ-ONLY)\n"
            "The other investigators' workspaces are mounted inside your own "
            "workspace at these relative paths:\n"
            f"{lines}\n"
            "You can read their real artifacts there (data files, plots, "
            "scripts, notes) with read_file, or inspect them with run_command "
            "(e.g. `ls peers/h1/`), which is far better evidence than a summary "
            "alone. Two caveats:\n"
            "1. These paths are READ-ONLY. Any attempt to write to, edit, or "
            "delete a file under a peer path will be refused. Never modify a "
            "peer's files; copy what you need into your own workspace instead.\n"
            "2. Directory listing tools do not traverse these mounts, so they "
            "will not appear in list_workspace_files output. Use the explicit "
            "paths above.\n"
            "Do not use `grep -r`, `find`, or other unbounded recursive scans "
            "from your workspace root, because peer mounts can nest; target the "
            "specific peer path you care about instead."
        )

    def _make_spawn_tools(
        self, registry: _InvestigatorRegistry
    ) -> list[StructuredTool]:
        run_config = self._run_config
        emitter = (
            self._run_events
            if self._run_events is not None
            else self._events(run_config)
        )

        def _start(label: str, task: str, agent_name: str) -> float:
            emitter.emit(
                f"Orchestrator -> {label}",
                stage="delegation",
                phase="start",
                event_type="delegation_started",
                source=self._source("orchestrator"),
                target=self._source(label),
                hypothesis_label=label,
                investigator_agent_name=agent_name,
                task=task,
            )
            self._trace(f"orchestrator -> {label} ({agent_name})", task)
            return perf_counter()

        def _end(label: str, task: str, text: str, started: float) -> None:
            emitter.emit(
                f"{label} -> Orchestrator",
                stage="delegation",
                phase="end",
                event_type="delegation_completed",
                source=self._source(label),
                target=self._source("orchestrator"),
                hypothesis_label=label,
                task=task,
                result=text,
                elapsed_seconds=perf_counter() - started,
            )
            self._trace(f"{label} -> orchestrator", text)

        def _error(
            label: str, task: str, exc: BaseException, started: float
        ) -> None:
            emitter.emit(
                f"Investigator {label} failed",
                stage="delegation",
                phase="error",
                event_type="delegation_failed",
                level="error",
                source=self._source(label),
                target=self._source("orchestrator"),
                hypothesis_label=label,
                task=task,
                error=str(exc),
                elapsed_seconds=perf_counter() - started,
            )
            self._trace(f"{label} -> orchestrator [FAILED]", str(exc))

        def spawn_investigator(hypothesis_label: str, task: str) -> str:
            agent = registry.get_or_create(hypothesis_label)
            agent_name = registry.identities.get(
                hypothesis_label.strip().lower(), hypothesis_label
            )
            task = self._with_peer_access(registry, hypothesis_label, task)
            started = _start(hypothesis_label, task, agent_name)
            member_config = self._investigator_config(
                run_config, hypothesis_label, agent_name
            )
            kwargs = {"config": member_config} if member_config else {}
            try:
                result = agent.invoke(task, **kwargs)
            except BaseException as exc:
                _error(hypothesis_label, task, exc, started)
                raise
            text = self._result_text(result)
            _end(hypothesis_label, task, text, started)
            return text

        async def aspawn_investigator(hypothesis_label: str, task: str) -> str:
            agent = registry.get_or_create(hypothesis_label)
            agent_name = registry.identities.get(
                hypothesis_label.strip().lower(), hypothesis_label
            )
            task = self._with_peer_access(registry, hypothesis_label, task)
            started = _start(hypothesis_label, task, agent_name)
            member_config = self._investigator_config(
                run_config, hypothesis_label, agent_name
            )
            kwargs = {"config": member_config} if member_config else {}
            try:
                result = await agent.ainvoke(task, **kwargs)
            except BaseException as exc:
                _error(hypothesis_label, task, exc, started)
                raise
            text = self._result_text(result)
            _end(hypothesis_label, task, text, started)
            return text

        spawn = StructuredTool.from_function(
            func=spawn_investigator,
            coroutine=aspawn_investigator,
            name="spawn_investigator",
            description=(
                "Create (or reuse) a persistent per-hypothesis investigator "
                "agent identified by a stable hypothesis_label, run the given "
                "task on it, and return its result. Reusing a label reaches the "
                "same agent so it remembers its earlier investigation."
            ),
            args_schema=SpawnInvestigatorInput,
        )

        def list_investigators() -> str:
            identities = registry.identities
            if not identities:
                text = "No investigators spawned yet."
            else:
                text = "\n".join(
                    f"- label '{label}' -> agent_name '{name}'"
                    for label, name in identities.items()
                )
            emitter.emit(
                "Orchestrator listed investigators",
                stage="registry",
                phase="update",
                event_type="investigators_listed",
                source=self._source("orchestrator"),
                investigators=identities,
            )
            self._trace("orchestrator: list_investigators", text)
            return text

        listing = StructuredTool.from_function(
            func=list_investigators,
            name="list_investigators",
            description=(
                "List the hypothesis labels spawned so far and their minted "
                "persistent agent_names. Record these in your experiences so you "
                "never lose track of which agent owns which hypothesis."
            ),
        )
        return [spawn, listing]

    def _hypothesis_block(self, hypotheses: Sequence[Hypothesis]) -> str:
        if not hypotheses:
            return "(none parsed; treat the question itself as one hypothesis)"
        return "\n".join(
            f"H{h.index}: {h.statement}"
            + (f"\n    {h.detail}" if h.detail else "")
            for h in hypotheses
        )

    def _parse_hypotheses(self, artifact: str, query: str) -> list[Hypothesis]:
        hypotheses: list[Hypothesis] = []
        current: Hypothesis | None = None
        detail: list[str] = []
        for raw in artifact.splitlines():
            line = raw.rstrip()
            if line.startswith("### H"):
                if current is not None:
                    current.detail = "\n".join(detail).strip()
                    hypotheses.append(current)
                    detail = []
                heading = line[3:].strip()
                statement = heading
                if ":" in heading:
                    statement = heading.split(":", 1)[1].strip()
                current = Hypothesis(
                    index=len(hypotheses) + 1, statement=statement or heading
                )
            elif current is not None:
                detail.append(line)
        if current is not None:
            current.detail = "\n".join(detail).strip()
            hypotheses.append(current)
        if not hypotheses:
            hypotheses = [
                Hypothesis(
                    index=1,
                    statement=query or "primary hypothesis",
                    detail=artifact,
                )
            ]
        return hypotheses[: self.max_hypotheses]
