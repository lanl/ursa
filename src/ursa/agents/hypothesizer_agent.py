import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .base import BaseAgent


class HypothesizerState(TypedDict, total=False):
    """State for persistent hypothesis-space maintenance.

    The durable hypothesis space is intentionally offloaded to an experience file
    rather than kept entirely in graph state. This makes it easy for other URSA
    agent behaviors, such as chat and execution, to read the current hypothesis
    space back into context via the existing experience tools.
    """

    query: str
    """Original or current user question/topic."""

    new_information: str
    """New evidence, clarification, or instruction to incorporate."""

    context: str
    """Optional additional context from another agent behavior or user-provided notes."""

    experience_filename: str
    """Markdown experience file used as the durable hypothesis artifact."""

    hypothesis_space_markdown: str
    """Latest hypothesis-space artifact markdown."""

    summary: str
    """Compact state summary of the current hypothesis space."""

    revision_history: list[str]
    """Short descriptions of updates made during this run/thread."""

    last_updated: str
    """ISO timestamp for the latest update."""


DEFAULT_HYPOTHESIS_EXPERIENCE = "hypothesis_space.md"


class HypothesizerAgent(BaseAgent[HypothesizerState]):
    """Maintain a persistent, shareable hypothesis space.

    Unlike the legacy/current-review workflow, this agent tracks alternative
    hypotheses, relative likelihoods, and evidence for/against each hypothesis.
    The full artifact is stored in ``den/experiences/<experience_filename>`` so
    other agents can bring it back into context with ``read_experience`` even if
    conversational context has been summarized away.
    """

    state_type = HypothesizerState

    def __init__(
        self,
        llm: BaseChatModel,
        experience_filename: str = DEFAULT_HYPOTHESIS_EXPERIENCE,
        **kwargs,
    ):
        super().__init__(llm, **kwargs)
        self.experience_filename = self._validate_experience_filename(
            experience_filename
        )

    def _normalize_inputs(self, inputs) -> HypothesizerState:
        if isinstance(inputs, str):
            return HypothesizerState(
                query=inputs,
                new_information=inputs,
                experience_filename=self.experience_filename,
                revision_history=[],
            )

        state = dict(cast(dict[str, Any], inputs))
        if "query" not in state and "question" in state:
            state["query"] = state["question"]
        if "new_information" not in state:
            state["new_information"] = state.get("query", "")
        state.setdefault("experience_filename", self.experience_filename)
        state.setdefault("revision_history", [])
        return cast(HypothesizerState, state)

    def format_query(
        self,
        prompt: str,
        state: HypothesizerState | None = None,
    ) -> HypothesizerState:
        """Treat follow-up prompts as new information for the existing space."""
        if state is None:
            return self._normalize_inputs(prompt)
        updated = dict(state)
        updated["new_information"] = prompt
        updated.setdefault("query", state.get("query", prompt))
        updated.setdefault("experience_filename", self.experience_filename)
        updated.setdefault("revision_history", [])
        return cast(HypothesizerState, updated)

    def format_result(self, result: HypothesizerState) -> str:
        artifact = self._response_text(result.get("hypothesis_space_markdown"))
        if artifact:
            return artifact
        summary = self._response_text(result.get("summary"))
        return (
            summary
            or "Hypothesizer failed to produce a hypothesis-space artifact."
        )

    @staticmethod
    def _validate_experience_filename(filename: str) -> str:
        name = filename.strip()
        path = Path(name)
        if not name:
            raise ValueError("Experience filename must not be empty.")
        if path.is_absolute() or path.name != name or name in {".", ".."}:
            raise ValueError(
                "Experience filename must be a simple relative file name."
            )
        if path.suffix.lower() != ".md":
            raise ValueError("Experience filename must use the .md extension.")
        return name

    @property
    def _experiences_dir(self) -> Path:
        path = self.den / "experiences"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _experience_path(self, filename: str) -> Path:
        safe_filename = self._validate_experience_filename(filename)
        return self._experiences_dir / safe_filename

    def _read_existing_hypothesis_space(self, filename: str) -> str:
        path = self._experience_path(filename)
        if not path.exists():
            return ""
        try:
            return self._response_text(path.read_text(encoding="utf-8"))
        except OSError:
            return ""

    def _write_hypothesis_space(self, filename: str, content: str) -> Path:
        path = self._experience_path(filename)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _clean_markdown_text(text: str) -> str:
        """Normalize markdown text returned by model/client layers."""
        text = text.strip()
        if "\\n" in text and text.count("\\n") >= text.count("\n"):
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
        return text.strip()

    @classmethod
    def _response_text(cls, value: Any) -> str:
        """Extract user-visible text from LLM response content.

        Some chat models return structured content blocks, e.g. reasoning blocks
        plus text blocks. Avoid ``str(list_of_blocks)`` because that surfaces
        Python/JSON-ish wrapper data to the CLI and hypothesis artifact.
        """
        content = getattr(value, "content", value)
        if content is None:
            return ""
        if isinstance(content, str):
            raw_text = content.strip()
            if (
                raw_text.startswith(("[", "{"))
                and "'type'" in raw_text
                and "'text'" in raw_text
            ):
                try:
                    parsed = ast.literal_eval(raw_text)
                except (SyntaxError, ValueError):
                    return cls._clean_markdown_text(raw_text)
                parsed_text = cls._response_text(parsed)
                return parsed_text or cls._clean_markdown_text(raw_text)
            if (
                raw_text.startswith(("[", "{"))
                and '"type"' in raw_text
                and '"text"' in raw_text
            ):
                try:
                    parsed = ast.literal_eval(raw_text)
                except (SyntaxError, ValueError):
                    return cls._clean_markdown_text(raw_text)
                parsed_text = cls._response_text(parsed)
                return parsed_text or cls._clean_markdown_text(raw_text)
            return cls._clean_markdown_text(raw_text)
        if isinstance(content, list):
            parts = [cls._response_text(item) for item in content]
            return "\n\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            block_type = content.get("type")
            if block_type in {"text", "output_text"} and "text" in content:
                return cls._response_text(content["text"])
            if block_type == "reasoning":
                return ""
            if "text" in content:
                return cls._response_text(content["text"])
            if "content" in content:
                return cls._response_text(content["content"])
            return ""
        return cls._clean_markdown_text(str(content))

    def _fallback_hypothesis_space(
        self,
        *,
        query: str,
        new_information: str,
        context: str,
        previous: str,
        now: str,
    ) -> str:
        basis = new_information or query or "the current user question"
        context_note = (
            context or "No additional cross-agent context was provided."
        )
        previous_note = (
            "A prior hypothesis-space artifact existed and should remain part "
            "of the ongoing context."
            if previous.strip()
            else "No previous hypothesis-space artifact was found."
        )
        return f"""# Hypothesis Space

## Question / Topic

{query or basis}

## Current Update

{basis}

## Additional Context

{context_note}

## Hypotheses

### H1: Primary working hypothesis

- **Relative likelihood:** 0.34
- **Rationale:** This is the most direct explanation currently available from the supplied prompt/context.
- **Evidence for:**
  - The current user-provided information is consistent with this hypothesis.
- **Evidence against:**
  - No explicit contradictory evidence has been recorded yet.
- **Assumptions / uncertainties:**
  - This likelihood is provisional and should be updated as more evidence is gathered.

### H2: Alternative mechanism or explanation

- **Relative likelihood:** 0.33
- **Rationale:** A plausible alternative may explain the same observations through a different causal path.
- **Evidence for:**
  - The current evidence does not rule it out.
- **Evidence against:**
  - No specific supporting evidence has been isolated yet.
- **Assumptions / uncertainties:**
  - Additional targeted evidence is needed to compare it against H1.

### H3: Null, mixed, or confounded explanation

- **Relative likelihood:** 0.33
- **Rationale:** The available observations may be incomplete, confounded, or explained by multiple factors.
- **Evidence for:**
  - {previous_note}
- **Evidence against:**
  - No decisive evidence yet.
- **Assumptions / uncertainties:**
  - More precise observations, source documents, experiments, or logs would reduce uncertainty.

## Evidence Log

- **E1:** {basis}
  - Supports: H1, H2, H3 to varying degrees.
  - Contradicts: none recorded yet.
  - Strength: provisional.

## Change Summary

Initialized or updated the hypothesis space with the latest user-provided information. Treat all likelihoods as provisional until additional evidence is gathered.

## Recommended Next Evidence

- Identify observations that would distinguish H1 from H2.
- Look for evidence that would falsify the primary working hypothesis.
- Record source documents, commands, results, or experiments as future evidence updates.
"""

    def _build_update_prompt(
        self,
        *,
        query: str,
        new_information: str,
        context: str,
        previous: str,
        now: str,
        filename: str,
    ) -> list:
        system = SystemMessage(
            content=(
                "You are a persistent hypothesis-space maintainer for URSA. "
                "Your job is to maintain a structured set of competing hypotheses, "
                "relative likelihoods, and evidence for/against each hypothesis. "
                "Return only Markdown. Do not use markdown code fences around the whole response. "
                "The artifact will be written as an experience file so other agents can read it later."
            )
        )
        human = HumanMessage(
            content=f"""Update the hypothesis-space artifact.

The artifact will be saved to the configured hypothesis-space experience file.
Do not include implementation metadata such as file paths, timestamps, JSON, or HTML comments in the output.

Question / topic:
{query}

New user information / evidence / instruction:
{new_information}

Additional context from another agent behavior or user-provided notes:
{context or "(none provided)"}

Previous hypothesis-space artifact:
{previous or "(none found; initialize a new hypothesis space)"}

Requirements:
- Maintain clear hypothesis IDs such as H1, H2, H3.
- Keep or update relative likelihoods.
- State whether likelihoods are mutually exclusive probabilities or independent plausibility scores.
- Track evidence for and against each hypothesis.
- Preserve useful prior evidence unless contradicted.
- Explain exactly what changed in this update.
- Include recommended next evidence or work that chat/execution agents could gather.
- Keep the artifact concise enough to be read back into context later, but complete enough to be useful.
"""
        )
        return [system, human]

    def _summarize_artifact(self, artifact: str, filename: str) -> str:
        lines = [line.strip() for line in artifact.splitlines() if line.strip()]
        hypotheses = [line for line in lines if line.startswith("### H")]
        if hypotheses:
            hyp_text = "; ".join(hypotheses[:5])
            return (
                f"Updated hypothesis space in experiences/{filename}. "
                f"Current hypotheses: {hyp_text}"
            )
        return f"Updated hypothesis space in experiences/{filename}."

    def update_hypothesis_space(
        self,
        state: HypothesizerState,
        config: RunnableConfig | None = None,
    ) -> HypothesizerState:
        events = self.events(config)
        filename = self._validate_experience_filename(
            state.get("experience_filename", self.experience_filename)
        )
        query = state.get("query") or state.get("new_information", "")
        new_information = state.get("new_information") or query
        context = state.get("context", "")
        now = datetime.now(UTC).isoformat()
        previous = self._read_existing_hypothesis_space(filename)

        events.emit(
            "Updating hypothesis space",
            stage="hypothesis_update",
            experience_filename=filename,
            has_existing_artifact=bool(previous.strip()),
        )

        messages = self._build_update_prompt(
            query=query,
            new_information=new_information,
            context=context,
            previous=previous,
            now=now,
            filename=filename,
        )
        response = self.llm.invoke(messages, config=config)
        artifact = self._response_text(response)

        if not artifact or artifact.lower() in {"ok", "stub"}:
            artifact = self._fallback_hypothesis_space(
                query=query,
                new_information=new_information,
                context=context,
                previous=previous,
                now=now,
            )

        if not artifact.lstrip().startswith("#"):
            artifact = f"# Hypothesis Space\n\n{artifact}"

        artifact = artifact.rstrip()

        path = self._write_hypothesis_space(filename, artifact)
        summary = self._summarize_artifact(artifact, filename)
        revision_entry = (
            f"{now}: updated {filename} with new information: "
            f"{new_information[:160]}"
        )
        revision_history = list(state.get("revision_history", []))
        revision_history.append(revision_entry)

        events.emit(
            "Hypothesis space updated",
            stage="hypothesis_update",
            experience_filename=filename,
            output_path=str(path),
            summary=summary,
        )

        return HypothesizerState(
            query=query,
            new_information=new_information,
            context=context,
            experience_filename=filename,
            hypothesis_space_markdown=artifact,
            summary=summary,
            revision_history=revision_history,
            last_updated=now,
        )

    def _build_graph(self):
        self.add_node(self.update_hypothesis_space, "update_hypothesis_space")
        self.graph.set_entry_point("update_hypothesis_space")
        self.graph.set_finish_point("update_hypothesis_space")


class LegacyHypothesizerAgentWarning(DeprecationWarning):
    pass


def should_continue(state: HypothesizerState) -> Literal["finish"]:
    """Compatibility helper for callers that imported the old symbol."""
    return "finish"
