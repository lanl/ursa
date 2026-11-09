"""Shared hypothesis records and artifact parsing for hypothesis workflows."""

from dataclasses import dataclass
from typing import Any, Mapping

from ursa.agents.hypothesizer_agent import HypothesizerAgent


@dataclass
class Hypothesis:
    """One competing hypothesis to investigate."""

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


def parse_hypotheses(
    artifact: str, query: str, *, max_hypotheses: int
) -> list[Hypothesis]:
    """Parse hypothesis headings, falling back to the question and artifact."""
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
    return hypotheses[:max_hypotheses]


def artifact_text(result: Any) -> str:
    """Extract hypothesis markdown, a summary, or the model's response text."""
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
