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

Observability/testability is deliberately weaker than the static symposium graph
(dynamic topology), which is the expected tradeoff for adaptive orchestration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ursa.agents.execution_agent import ExecutionAgent
from ursa.agents.hypothesizer_agent import (
    DEFAULT_HYPOTHESIS_EXPERIENCE,
    HypothesizerAgent,
)
from ursa.workflows.base_workflow import BaseWorkflow, InputLike
from ursa.workflows.hypothesis_symposium import Hypothesis

__all__ = [
    "HypothesisOrchestratorWorkflow",
    "SpawnInvestigatorInput",
]


def _timestamp() -> str:
    """Match the ``ursa save-agent`` timestamp convention."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str, *, max_words: int = 4) -> str:
    words = "".join(
        ch.lower() if ch.isalnum() else " " for ch in text
    ).split()
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
    """

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
whether the combined evidence supports or weakens its own hypothesis.
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm = llm
        self.workspace = Path(workspace) if workspace else None
        self.group = group
        self.experience_filename = (
            HypothesizerAgent._validate_experience_filename(experience_filename)
        )
        self.hypothesizer = hypothesizer or HypothesizerAgent(
            llm, experience_filename=self.experience_filename
        )
        self.max_hypotheses = max(1, int(max_hypotheses))
        self._orchestrator_override = orchestrator

    # ------------------------------------------------------- overridable hooks

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
        orchestrator = self._build_orchestrator(registry)

        prompt = _ORCHESTRATOR_PROMPT.format(
            hypothesis_block=self._hypothesis_block(hypotheses),
            query=query,
        )
        result = orchestrator.invoke(prompt, config=cfg)
        return {
            "query": query,
            "hypotheses": [h.__dict__ for h in hypotheses],
            "hypothesis_space_markdown": self._last_hypothesis_space,
            "investigator_identities": registry.identities,
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
        if self._orchestrator_override is not None:
            orchestrator = self._orchestrator_override
            orchestrator.add_tool(self._make_spawn_tools(registry))
            return orchestrator
        kwargs: dict[str, Any] = {
            "extra_tools": self._make_spawn_tools(registry)
        }
        if self.workspace is not None:
            kwargs["workspace"] = str(self.workspace / "orchestrator")
        if self.group is not None:
            kwargs["group"] = self.group
        return ExecutionAgent(self.llm, **kwargs)

    def _make_spawn_tools(
        self, registry: _InvestigatorRegistry
    ) -> list[StructuredTool]:
        def spawn_investigator(hypothesis_label: str, task: str) -> str:
            agent = registry.get_or_create(hypothesis_label)
            return self._result_text(agent.invoke(task))

        async def aspawn_investigator(hypothesis_label: str, task: str) -> str:
            agent = registry.get_or_create(hypothesis_label)
            return self._result_text(await agent.ainvoke(task))

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
                return "No investigators spawned yet."
            return "\n".join(
                f"- label '{label}' -> agent_name '{name}'"
                for label, name in identities.items()
            )

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

    def _parse_hypotheses(
        self, artifact: str, query: str
    ) -> list[Hypothesis]:
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
