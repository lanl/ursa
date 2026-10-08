"""Hypothesis-driven symposium workflow (Plan A).

This workflow builds a durable hypothesis space, fans out one investigation
branch per competing hypothesis, converges the per-hypothesis findings, runs a
symposium-style cross-review / multi-round revision over the collected
evidence, synthesizes a verdict, and finally folds every piece of new
information back into the hypothesis space.

Design goals baked into the structure:

* The multi-round symposium loop is deliberately isolated behind overridable
  hooks (:meth:`build_hypotheses`, :meth:`investigate_hypothesis`,
  :meth:`converge`, :meth:`run_symposium`, :meth:`synthesize`,
  :meth:`update_hypothesis_space`). A future Plan D layer can subclass this
  workflow and replace :meth:`run_symposium` with orchestrated all-member
  dispatch plus adaptive follow-up without touching the fan-out plumbing.
* Per-hypothesis investigation is *quarantined*: an exception in one branch is
  captured as recorded failure data (via a reducer channel) instead of
  aborting the run, so healthy sibling branches and downstream review survive.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence, TypedDict
from uuid import uuid4

from langchain.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Overwrite, RetryPolicy, Send

from ursa.agents.hypothesizer_agent import (
    DEFAULT_HYPOTHESIS_EXPERIENCE,
    HypothesizerAgent,
)
from ursa.workflows.base_workflow import BaseWorkflow, InputLike

logger = logging.getLogger(__name__)

__all__ = [
    "Hypothesis",
    "HypothesisInvestigation",
    "HypothesisSymposiumState",
    "HypothesisSymposiumWorkflow",
]


@dataclass
class Hypothesis:
    """One competing hypothesis to investigate in its own fan-out branch."""

    index: int
    statement: str
    detail: str = ""

    def as_task(self, query: str) -> str:
        lines = [
            f"Investigate the following hypothesis for the question: {query}",
            "",
            f"Hypothesis H{self.index}: {self.statement}",
        ]
        if self.detail:
            lines += ["", self.detail]
        lines += [
            "",
            "Gather and weigh evidence for and against this specific "
            "hypothesis. Report findings, confidence, and any evidence that "
            "would change your assessment.",
        ]
        return "\n".join(lines)


@dataclass
class HypothesisInvestigation:
    """Recorded outcome of a single (quarantined) investigation branch."""

    index: int
    statement: str
    findings: str = ""
    ok: bool = True
    error: str | None = None

    def as_markdown(self) -> str:
        status = "ok" if self.ok else "FAILED"
        body = self.findings if self.ok else (self.error or "unknown error")
        return f"### H{self.index} ({status}): {self.statement}\n\n{body}"


class HypothesisSymposiumState(TypedDict, total=False):
    """Parent graph state.

    Fan-in requires reducer channels, so investigation results and failures use
    ``operator.add`` accumulators. ``Overwrite([])`` resets them between phases.
    """

    query: str
    context: str
    experience_filename: str

    hypotheses: list[dict[str, Any]]
    investigations: Annotated[list[dict[str, Any]], operator.add]
    failures: Annotated[list[dict[str, Any]], operator.add]

    evidence_digest: str
    symposium_result: dict[str, Any]
    symposium_run_id: str
    synthesis: str
    hypothesis_space_markdown: str
    summary: str


@dataclass
class _SymposiumSpec:
    """Lazily-built defaults for the internal symposium reviewer/reviser loop."""

    revision_rounds: int = 1
    member_names: Sequence[str] = field(
        default_factory=lambda: ("analyst", "skeptic")
    )


class HypothesisSymposiumWorkflow(BaseWorkflow):
    """Hypothesize -> fan out -> investigate -> converge -> review -> update.

    Every scientific phase is exposed as an overridable hook so a later Plan D
    layer can replace the fixed multi-round symposium with orchestrated
    all-member dispatch plus adaptive follow-up. The default
    :meth:`run_symposium` uses the hardened :class:`AgentSymposiumEnvironment`
    so member failures are contained rather than fatal.
    """

    state_type = HypothesisSymposiumState

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        hypothesizer: HypothesizerAgent | None = None,
        experience_filename: str = DEFAULT_HYPOTHESIS_EXPERIENCE,
        max_hypotheses: int = 5,
        revision_rounds: int = 1,
        symposium: Any | None = None,
        symposium_member_names: Sequence[str] | None = None,
        workspace: str | Path | None = None,
        group: str | None = None,
        name: str = "hypothesis_symposium",
        checkpointer: Any | None = None,
        recursion_limit: int = 200,
        visualize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        # Preserve the caller/dashboard-provided workspace so the internal
        # symposium (and its per-hypothesis members) run under it instead of
        # silently falling back to the shared cache directory. Previously this
        # was swallowed into **kwargs and discarded by BaseWorkflow.__init__.
        self.workspace = Path(workspace) if workspace else None
        self.group = group
        # Display name for the internal symposium environment. It surfaces on
        # the dashboard Runs page, so keep it short and human-readable rather
        # than deriving it from the (long) class name.
        self.name = name
        self.experience_filename = (
            HypothesizerAgent._validate_experience_filename(experience_filename)
        )
        self.hypothesizer = hypothesizer or HypothesizerAgent(
            llm, experience_filename=self.experience_filename
        )
        self.max_hypotheses = max(1, int(max_hypotheses))
        self._symposium_spec = _SymposiumSpec(
            revision_rounds=max(1, int(revision_rounds)),
            member_names=tuple(
                symposium_member_names or _SymposiumSpec().member_names
            ),
        )
        self._symposium = symposium
        self._default_symposium: Any | None = None
        # When true, the symposium stage is wrapped in the environment
        # visualization recorder so the dashboard can follow it *while running*
        # (the recorder flushes each event to events.jsonl immediately).
        self.visualize = bool(visualize)
        self.symposium_run_id: str | None = None
        self.recursion_limit = int(recursion_limit)
        self.checkpointer = checkpointer or InMemorySaver()
        # Branch retries: RetryPolicy's default predicate refuses common builtin
        # exceptions, so retry broadly on any Exception (cancellations excluded
        # by langgraph itself).
        self._branch_retry = RetryPolicy(
            max_attempts=2,
            initial_interval=0.1,
            retry_on=lambda exc: isinstance(exc, Exception),
        )
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        builder = StateGraph(self.state_type)
        builder.add_node("hypothesize", self._node_hypothesize)
        builder.add_node(
            "investigate",
            RunnableLambda(
                self._node_investigate_sync,
                afunc=self._node_investigate_async,
            ),
            retry_policy=self._branch_retry,
        )
        builder.add_node("converge", self._node_converge)
        builder.add_node("symposium", self._node_symposium)
        builder.add_node("synthesize", self._node_synthesize)
        builder.add_node("update_space", self._node_update_space)

        builder.add_edge(START, "hypothesize")
        # A node cannot both return a dict and Send list, so the fan-out lives
        # in a conditional edge keyed off the hypotheses produced upstream.
        builder.add_conditional_edges(
            "hypothesize", self._fan_out, ["investigate", "converge"]
        )
        builder.add_edge("investigate", "converge")
        builder.add_edge("converge", "symposium")
        builder.add_edge("symposium", "synthesize")
        builder.add_edge("synthesize", "update_space")
        builder.add_edge("update_space", END)
        return builder.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------ graph nodes

    def _node_hypothesize(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        hypotheses = self.build_hypotheses(state, config)
        return {
            "hypotheses": [
                {
                    "index": h.index,
                    "statement": h.statement,
                    "detail": h.detail,
                }
                for h in hypotheses
            ],
            # Reset accumulators so re-entry / re-runs start clean.
            "investigations": Overwrite([]),
            "failures": Overwrite([]),
        }

    def _fan_out(self, state: HypothesisSymposiumState) -> list[Send] | str:
        hypotheses = state.get("hypotheses") or []
        if not hypotheses:
            return "converge"
        return [
            Send("investigate", {"query": state.get("query", ""), **h})
            for h in hypotheses
        ]

    def _investigate(
        self, payload: Mapping[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        hypothesis = Hypothesis(
            index=int(payload.get("index", 0)),
            statement=str(payload.get("statement", "")),
            detail=str(payload.get("detail", "")),
        )
        query = str(payload.get("query", ""))
        try:
            outcome = self.investigate_hypothesis(hypothesis, query, config)
        except Exception as exc:  # noqa: BLE001
            # Quarantine: record the failing branch as data. langgraph's
            # error_handler/timeout cannot suppress a sync failure, so we do it
            # here to preserve sibling branches (RetryPolicy already retried).
            failed = HypothesisInvestigation(
                index=hypothesis.index,
                statement=hypothesis.statement,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"failures": [failed.__dict__]}
        return {"investigations": [outcome.__dict__]}

    def _node_investigate_sync(
        self, payload: Mapping[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        return self._investigate(payload, config)

    async def _node_investigate_async(
        self, payload: Mapping[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        hypothesis = Hypothesis(
            index=int(payload.get("index", 0)),
            statement=str(payload.get("statement", "")),
            detail=str(payload.get("detail", "")),
        )
        query = str(payload.get("query", ""))
        try:
            outcome = await self.ainvestigate_hypothesis(
                hypothesis, query, config
            )
        except Exception as exc:  # noqa: BLE001
            failed = HypothesisInvestigation(
                index=hypothesis.index,
                statement=hypothesis.statement,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return {"failures": [failed.__dict__]}
        return {"investigations": [outcome.__dict__]}

    def _node_converge(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        return {"evidence_digest": self.converge(state, config)}

    async def _node_symposium(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        result = await self.run_symposium(state, config)
        update: dict[str, Any] = {"symposium_result": result}
        # Surface the visualization run id in state so the dashboard/caller can
        # locate the live event stream for this symposium stage.
        if self.symposium_run_id:
            update["symposium_run_id"] = self.symposium_run_id
        return update

    async def _node_synthesize(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        return {"synthesis": await self.synthesize(state, config)}

    async def _node_update_space(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        return await self.update_hypothesis_space(state, config)

    # ------------------------------------------------------- overridable hooks

    def format_result(self, result: HypothesisSymposiumState) -> str:
        """Return only the final write-up, not the whole state.

        Prefers the symposium's detailed final answer and falls back to the
        shorter synthesized ``summary`` when the symposium produced no final
        text (for example if the stage was skipped).
        """
        symposium_result = result.get("symposium_result")
        detailed = (
            symposium_result.get("final")
            if isinstance(symposium_result, Mapping)
            else None
        )
        summary = result.get("summary")
        text = detailed or summary
        if not text:
            raise ValueError("Symposium completed without a response.")
        return str(text).strip()

    def build_hypotheses(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> list[Hypothesis]:
        """Produce competing hypotheses via the HypothesizerAgent.

        The hypothesizer maintains a durable markdown artifact whose competing
        hypotheses are ``### H...`` lines; those become the fan-out branches.
        """
        query = str(state.get("query", ""))
        result = self.hypothesizer.invoke(
            {
                "query": query,
                "new_information": query,
                "context": str(state.get("context", "")),
                "experience_filename": self.experience_filename,
                "revision_history": [],
            },
            config=config,
        )
        artifact = self._artifact_text(result)
        state["hypothesis_space_markdown"] = artifact
        return self._parse_hypotheses(artifact, query)

    def investigate_hypothesis(
        self,
        hypothesis: Hypothesis,
        query: str,
        config: RunnableConfig,
    ) -> HypothesisInvestigation:
        """Default synchronous investigation (assignment record).

        In the default design (A) the *substantive* per-hypothesis
        investigation is performed by the symposium member that owns this
        hypothesis (during the symposium's initial-work phase), so this fan-out
        branch only records the assignment that will be handed to that member.
        This keeps the fan-out / quarantine / fan-in plumbing intact as the
        Plan D seam: a subclass can override this (or
        :meth:`ainvestigate_hypothesis`) to dispatch a dedicated investigation
        agent per branch instead of (or in addition to) the symposium.
        """
        return HypothesisInvestigation(
            index=hypothesis.index,
            statement=hypothesis.statement,
            findings=(
                f"Assigned to a dedicated symposium member for investigation: "
                f"{hypothesis.as_task(query)}"
            ),
            ok=True,
        )

    async def ainvestigate_hypothesis(
        self,
        hypothesis: Hypothesis,
        query: str,
        config: RunnableConfig,
    ) -> HypothesisInvestigation:
        """Async investigation hook; defaults to the sync implementation."""
        return self.investigate_hypothesis(hypothesis, query, config)

    def converge(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> str:
        """Fan-in: assemble the per-hypothesis findings into one digest."""
        investigations = [
            HypothesisInvestigation(**d)
            for d in state.get("investigations", [])
        ]
        failures = [
            HypothesisInvestigation(**d) for d in state.get("failures", [])
        ]
        ordered = sorted(investigations + failures, key=lambda i: i.index)
        parts = [i.as_markdown() for i in ordered]
        if failures:
            parts.append(
                f"\n_{len(failures)} of {len(ordered)} investigation "
                "branch(es) failed and were recorded above._"
            )
        return "\n\n".join(parts)

    async def run_symposium(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Cross-review + multi-round revision over the collected evidence.

        The default symposium assigns one member per hypothesis (design A): each
        member investigates its own hypothesis during initial work, then reads
        every member's investigation during review to adjudicate which
        hypothesis the combined evidence best supports. Isolated behind this
        hook so a Plan D layer can swap the fixed loop for orchestrated
        all-member dispatch + adaptive follow-up. The underlying environment is
        the hardened symposium (member failures are contained).
        """
        hypotheses = [
            Hypothesis(
                index=int(h.get("index", 0)),
                statement=str(h.get("statement", "")),
                detail=str(h.get("detail", "")),
            )
            for h in (state.get("hypotheses") or [])
        ]
        symposium = self._get_symposium(hypotheses)
        digest = str(state.get("evidence_digest", ""))
        digest_block = (
            "\n\nPrior per-hypothesis notes gathered before this symposium "
            f"(shared context):\n{digest}"
            if digest.strip()
            else ""
        )
        task = (
            f"Question: {state.get('query', '')}\n\n"
            "Each member has been assigned one competing hypothesis as its "
            "expertise. Investigate your own hypothesis, then cross-review the "
            "others, challenge weak evidence, and converge on which hypothesis "
            f"is best supported.{digest_block}"
        )
        return await self._ainvoke_symposium(symposium, task, config)

    async def _ainvoke_symposium(
        self, symposium: Any, task: str, config: RunnableConfig
    ) -> dict[str, Any]:
        """Invoke the symposium, recording a live visualization run when enabled.

        ``arun_with_visualization`` attaches an ``EnvironmentEventRecorder``
        callback to the run config and flushes every structured progress event
        to ``events.jsonl`` as it happens, so the dashboard can render the
        symposium *while it is still running* instead of only at the end. The
        run id is stashed on the workflow (and logged) so callers can locate the
        stream. Any failure to set up visualization must never break the
        science, so we degrade to a plain ``ainvoke``.
        """
        if not self.visualize:
            return await symposium.ainvoke(task, config=config)

        try:
            from ursa.environments import arun_with_visualization
        except Exception:  # pragma: no cover - defensive import guard
            logger.warning(
                "Symposium visualization unavailable; running unrecorded.",
                exc_info=True,
            )
            return await symposium.ainvoke(task, config=config)

        run_id = self._new_symposium_run_id()
        group = getattr(symposium, "group", None) or "default"
        # Decide whether visualization is viable *before* invoking: a failure
        # after the run has started would either lose the result or re-run the
        # whole (expensive) symposium.
        try:
            from ursa.security import validate_group_name

            validate_group_name(group)
        except Exception:
            logger.warning(
                "Symposium group %r is not recordable; running unrecorded.",
                group,
                exc_info=True,
            )
            return await symposium.ainvoke(task, config=config)

        self.symposium_run_id = run_id
        logger.info(
            "Symposium visualization run started: group=%s run_id=%s "
            "(events stream live to the dashboard)",
            group,
            run_id,
        )
        return await arun_with_visualization(
            symposium, task, config=config, run_id=run_id
        )

    @staticmethod
    def _new_symposium_run_id() -> str:
        """Readable, sortable, unique run id for one symposium stage."""
        return f"hypsym_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

    async def synthesize(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> str:
        """Turn the symposium result into a final verdict string."""
        result = state.get("symposium_result") or {}
        final = result.get("final") if isinstance(result, Mapping) else None
        if final:
            return str(final)
        return str(state.get("evidence_digest", ""))

    async def update_hypothesis_space(
        self, state: HypothesisSymposiumState, config: RunnableConfig
    ) -> dict[str, Any]:
        """Fold ALL new information back into the durable hypothesis space."""
        query = str(state.get("query", ""))
        new_information = self._final_new_information(state)
        result = await self.hypothesizer.ainvoke(
            {
                "query": query,
                "new_information": new_information,
                "context": str(state.get("context", "")),
                "experience_filename": self.experience_filename,
                "revision_history": [],
            },
            config=config,
        )
        artifact = self._artifact_text(result)
        summary = ""
        if isinstance(result, Mapping):
            summary = str(result.get("summary", "") or "")
        return {
            "hypothesis_space_markdown": artifact,
            "summary": summary,
        }

    # ------------------------------------------------------------- helpers

    def _final_new_information(self, state: HypothesisSymposiumState) -> str:
        sections = [
            "New information gathered during hypothesis investigation and "
            "symposium review:",
            "",
            "## Per-hypothesis investigation digest",
            str(state.get("evidence_digest", "")),
            "",
            "## Symposium synthesis / verdict",
            str(state.get("synthesis", "")),
        ]
        return "\n".join(sections)

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
                heading = line[3:].strip()  # drop leading "###"
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
            # Fall back to a single "the artifact is the hypothesis" branch so
            # the fan-out always has at least one thing to investigate.
            hypotheses = [
                Hypothesis(
                    index=1,
                    statement=query or "primary hypothesis",
                    detail=artifact,
                )
            ]
        return hypotheses[: self.max_hypotheses]

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

    def _get_symposium(
        self, hypotheses: Sequence[Hypothesis] | None = None
    ) -> Any:
        # An explicitly injected symposium is used as-is. Otherwise the default
        # symposium is built *per run* from the hypotheses so that each member
        # can be assigned one hypothesis as its expertise (design A). We cache
        # the built default so repeated calls within one run reuse the same
        # members (and therefore the same in-process investigation memory).
        if self._symposium is not None:
            return self._symposium
        if self._default_symposium is None:
            self._default_symposium = self._build_default_symposium(
                hypotheses or []
            )
        return self._default_symposium

    def _hypothesis_member_name(self, hypothesis: Hypothesis) -> str:
        """Stable, filesystem/tool-safe member name for one hypothesis."""
        words = "".join(
            ch.lower() if ch.isalnum() else " " for ch in hypothesis.statement
        ).split()
        slug = "_".join(words[:4]) or "hypothesis"
        return f"h{hypothesis.index}_{slug}"[:48]

    def _build_default_symposium(self, hypotheses: Sequence[Hypothesis]) -> Any:
        """Build a symposium with one member per hypothesis (design A).

        Each member is assigned exactly one hypothesis as its expertise. During
        the symposium's initial-work phase the member investigates *its* own
        hypothesis (gathering evidence for and against it); during the review
        phase every member reads every other member's investigation, so all
        members can learn from the shared evidence while adjudicating which
        hypothesis is best supported. This is the "specialist-per-hypothesis"
        design: guaranteed coverage of every hypothesis, real per-hypothesis
        parallel investigation, and a naturally adversarial cross-review.
        """
        from ursa.environments.agent_symposium import (
            AgentSymposiumEnvironment,
        )
        from ursa.environments.config import EnvironmentMemberConfig

        members = []
        for hypothesis in hypotheses:
            detail = (
                f"\n\nSupporting detail from the hypothesis space:\n{hypothesis.detail}"
                if hypothesis.detail
                else ""
            )
            members.append(
                EnvironmentMemberConfig(
                    name=self._hypothesis_member_name(hypothesis),
                    role=(
                        f"Advocate/investigator for hypothesis H{hypothesis.index}: "
                        f"{hypothesis.statement}"
                    ),
                    agent="ChatAgent",
                    reviewer=True,
                    prompt=(
                        "Your assigned hypothesis (your expertise for this "
                        f"symposium) is H{hypothesis.index}: "
                        f"{hypothesis.statement}.{detail}\n\n"
                        "During your initial work, rigorously investigate THIS "
                        "hypothesis only: gather concrete evidence for and "
                        "against it (write and run code, produce data/plots, and "
                        "quantify results where possible). Do not investigate the "
                        "other hypotheses now — other members own those. During "
                        "the review phase you will read every member's "
                        "investigation and help judge which hypothesis the "
                        "combined evidence best supports."
                    ),
                )
            )
        if not members:
            # No hypotheses parsed: fall back to a single generalist member so
            # the symposium still produces a usable adjudication.
            members.append(
                EnvironmentMemberConfig(
                    name="generalist",
                    role="Hypothesis investigator and reviewer",
                    agent="ChatAgent",
                    reviewer=True,
                )
            )
        return AgentSymposiumEnvironment(
            self.llm,
            name=self.name,
            members=members,
            revision_rounds=self._symposium_spec.revision_rounds,
            persist_members=False,
            # Thread the workflow workspace/group through so members run under
            # the caller/dashboard-provided workspace instead of the shared
            # cache directory. Passing None preserves BaseEnvironment defaults.
            workspace=self.workspace,
            group=self.group,
        )

    # ------------------------------------------------------------- run surface

    def _base_config(self, config: Mapping[str, Any] | None) -> RunnableConfig:
        cfg: dict[str, Any] = dict(config or {})
        cfg.setdefault("recursion_limit", self.recursion_limit)
        configurable = dict(cfg.get("configurable", {}))
        configurable.setdefault("thread_id", "hypothesis-symposium")
        cfg["configurable"] = configurable
        return cfg  # type: ignore[return-value]

    def _prepare_state(
        self, inputs: Mapping[str, Any]
    ) -> HypothesisSymposiumState:
        query = str(inputs.get("query") or inputs.get("task") or "")
        return HypothesisSymposiumState(
            query=query,
            context=str(inputs.get("context", "")),
            experience_filename=self.experience_filename,
            investigations=[],
            failures=[],
        )

    def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        # The graph contains async-only nodes (symposium/synthesis/update run
        # async agent calls), so sync callers drive the async graph through a
        # boundary event loop, mirroring BaseEnvironment.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._ainvoke(inputs, **config))
        raise RuntimeError(
            "HypothesisSymposiumWorkflow uses async execution internally, but "
            "`.invoke()` was called from an async context. Use "
            "`await workflow.ainvoke(...)` instead."
        )

    async def _ainvoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        cfg = self._base_config(config.get("config"))
        return await self._graph.ainvoke(self._prepare_state(inputs), cfg)

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        if isinstance(inputs, str):
            return {"query": inputs}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")
