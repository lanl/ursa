from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import BaseChatModel

from ursa.agents.chat_agent import ChatAgent

from .base import result_to_text


@dataclass(frozen=True)
class JudgeDecision:
    """Structured result returned by the Elo judge."""

    winner: str
    reasoning: str


class AgentEloJudge:
    """Ephemeral ChatAgent-based judge for Elo matches."""

    OUTPUT_INSTRUCTIONS = (
        "\n\n"
        "After evaluating the candidates, return exactly one JSON object "
        "with this schema:\n"
        "{\n"
        '  "winner": "A" | "B" | "DRAW",\n'
        '  "reasoning": "brief explanation"\n'
        "}\n"
        "Do not include markdown fences or text outside the JSON object."
    )

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        workspace: Path,
        group: str,
        judge_prompt: str,
    ):
        self.llm = llm
        self.workspace = Path(workspace)
        self.group = group
        self.judge_prompt = judge_prompt

    def _judge_prompt(
        self,
        *,
        task: str,
        player_a: str,
        agent_type_a: str,
        output_a: str,
        player_b: str,
        agent_type_b: str,
        output_b: str,
    ) -> str:
        return (
            f"{self.judge_prompt}\n\n"
            "You are judging one Elo match.\n\n"
            "First evaluate the candidates using their declared final "
            "outputs below. If those outputs provide enough evidence for "
            "a clear decision, decide immediately without unnecessary "
            "tool use.\n\n"
            "If quick verification or clarification is useful, you may "
            "inspect files in the candidates' workspaces. Your current "
            "workspace is the parent Elo workspace.\n\n"
            "The only candidate workspace directories relevant to this "
            "match are:\n"
            f"- Candidate A: {player_a}/\n"
            f"- Candidate B: {player_b}/\n\n"
            "Be efficient. Do not reproduce the candidates' entire work, "
            "perform extensive new research, or launch long-running "
            "calculations. Use tools only when they materially help resolve "
            "the comparison.\n\n"
            "Do not modify candidate files.\n\n"
            f"Original task:\n{task}\n\n"
            "Candidate A\n"
            f"Name: {player_a}\n"
            f"Agent type: {agent_type_a}\n"
            f"Workspace: {player_a}/\n"
            f"Declared final output:\n{output_a}\n\n"
            "Candidate B\n"
            f"Name: {player_b}\n"
            f"Agent type: {agent_type_b}\n"
            f"Workspace: {player_b}/\n"
            f"Declared final output:\n{output_b}"
            f"{self.OUTPUT_INSTRUCTIONS}"
        )

    @staticmethod
    def _parse_decision(
        text: str,
    ) -> JudgeDecision:
        try:
            judgment = json.loads(text)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Elo ChatAgent judge returned invalid JSON:\n"
                f"{text}"
            ) from exc

        winner = str(
            judgment.get("winner", "")
        ).strip().upper()

        reasoning = str(
            judgment.get("reasoning", "")
        ).strip()

        if winner not in {
            "A",
            "B",
            "DRAW",
        }:
            raise ValueError(
                "Elo judge must return winner as "
                "'A', 'B', or 'DRAW'. "
                f"Received: {winner!r}"
            )

        return JudgeDecision(
            winner=winner,
            reasoning=reasoning,
        )

    async def judge_match(
        self,
        *,
        task: str,
        player_a: str,
        agent_type_a: str,
        output_a: str,
        player_b: str,
        agent_type_b: str,
        output_b: str,
    ) -> JudgeDecision:
        """Judge one match using a fresh ChatAgent."""

        judge = ChatAgent(
            llm=self.llm,
            workspace=self.workspace,
            group=self.group,
            use_web=False,
        )

        prompt = self._judge_prompt(
            task=task,
            player_a=player_a,
            agent_type_a=agent_type_a,
            output_a=output_a,
            player_b=player_b,
            agent_type_b=agent_type_b,
            output_b=output_b,
        )

        try:
            result = await judge.ainvoke(
                prompt
            )

            formatter = getattr(
                judge,
                "format_result",
                None,
            )

            if callable(formatter):
                text = str(
                    formatter(result)
                )
            else:
                text = result_to_text(
                    result
                )

            return self._parse_decision(
                text
            )

        finally:
            close = getattr(
                judge,
                "close",
                None,
            )

            if callable(close):
                close()