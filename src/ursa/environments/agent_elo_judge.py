from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
)

from ursa.agents.chat_agent import ChatAgent

from .base import result_to_text


@dataclass(frozen=True)
class JudgeDecision:
    """Structured result returned by the Elo judge."""

    winner: str
    reasoning: str
    method: str = "chat_agent"


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
        guidance = (
            f"\nAdditional judging guidance:\n{self.judge_prompt}\n"
            if self.judge_prompt
            else ""
        )
        return (
            "Compare the candidates using the task's evaluation criteria. "
            "If none are specified, judge how well each fulfills the task. "
            "A timeout alone is not a loss.\n\n"
            "Use the supplied final responses or progress reports first. "
            "If neither is available, briefly inspect the candidate's workspace. "
            "Otherwise, inspect files only when useful for verification. "
            "Prioritize relevant summaries and results; avoid extensive searches "
            "or lengthy calculations. If evidence remains unavailable, state that "
            "limitation. Do not modify candidate files.\n\n"
            "Your workspace is the parent of the candidate directories below.\n"
            f"{guidance}\n"
            f"Task:\n{task}\n\n"
            f"Candidate A: {player_a}\n"
            f"Agent type: {agent_type_a}\n"
            f"Workspace: {player_a}/\n"
            f"Submission:\n{output_a}\n\n"
            f"Candidate B: {player_b}\n"
            f"Agent type: {agent_type_b}\n"
            f"Workspace: {player_b}/\n"
            f"Submission:\n{output_b}\n"
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
                f"Elo ChatAgent judge returned invalid JSON:\n{text}"
            ) from exc

        winner = str(judgment.get("winner", "")).strip().upper()

        reasoning = str(judgment.get("reasoning", "")).strip()

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
        """Judge one match with fault-tolerant fallback."""

        prompt = self._judge_prompt(
            task=task,
            player_a=player_a,
            agent_type_a=agent_type_a,
            output_a=output_a,
            player_b=player_b,
            agent_type_b=agent_type_b,
            output_b=output_b,
        )

        judge = None

        try:
            try:
                judge = ChatAgent(
                    llm=self.llm,
                    workspace=self.workspace,
                    group=self.group,
                    use_web=False,
                )
                result = await judge.ainvoke(prompt)

                formatter = getattr(
                    judge,
                    "format_result",
                    None,
                )

                if callable(formatter):
                    text = str(formatter(result))
                else:
                    text = result_to_text(result)

                decision = self._parse_decision(text)

                return JudgeDecision(
                    winner=decision.winner,
                    reasoning=(decision.reasoning),
                    method="chat_agent",
                )

            except Exception as agent_exc:
                try:
                    fallback = await self._judge_with_llm(prompt)

                    return JudgeDecision(
                        winner=(fallback.winner),
                        reasoning=(
                            "Agentic judge failed "
                            f"({type(agent_exc).__name__}: "
                            f"{agent_exc}). "
                            "Decision obtained from "
                            "raw-LLM fallback. "
                            f"{fallback.reasoning}"
                        ),
                        method="llm_fallback",
                    )

                except Exception as fallback_exc:
                    return JudgeDecision(
                        winner="DRAW",
                        reasoning=(
                            "Judging failed in both "
                            "the ChatAgent judge and "
                            "raw-LLM fallback. "
                            "Match recorded as a draw. "
                            "ChatAgent error: "
                            f"{type(agent_exc).__name__}: "
                            f"{agent_exc}. "
                            "Fallback error: "
                            f"{type(fallback_exc).__name__}: "
                            f"{fallback_exc}."
                        ),
                        method="failed_draw",
                    )

        finally:
            close = getattr(
                judge,
                "close",
                None,
            )

            if callable(close):
                try:
                    close()
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Failed to close Elo judge",
                        exc_info=True,
                    )

    async def _judge_with_llm(
        self,
        prompt: str,
    ) -> JudgeDecision:
        """Fallback to a single raw LLM judgment."""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])

        text = result_to_text(response)

        decision = self._parse_decision(text)

        return JudgeDecision(
            winner=decision.winner,
            reasoning=decision.reasoning,
            method="llm_fallback",
        )
