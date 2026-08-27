from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import (
    BaseEnvironment,
    invocation_kwargs,
    result_to_text,
)
from .config import EnvironmentMemberConfig


@dataclass
class EloPlayer:
    name: str
    rating: float = 1500.0


@dataclass
class MatchResult:
    player_a: str
    player_b: str
    score_a: float
    reasoning: str

    @property
    def winner(self) -> str | None:
        if self.score_a == 1.0:
            return self.player_a
        if self.score_a == 0.0:
            return self.player_b
        return None


class AgentEloEnvironment(BaseEnvironment):
    """Pairwise competitive environment with Elo ratings.

    Stage 1:
    - All members independently answer the same task.
    - Members are paired.
    - An LLM judge returns A, B, or DRAW for each pair.
    - Elo ratings are updated deterministically.
    - No elimination or reproduction yet.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        name: str = "agent_elo",
        group: str = "default",
        members: list[EnvironmentMemberConfig | Mapping[str, Any]] | None = None,
        workspace: str | Path | None = None,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        judge_prompt: str | None = None,
        persist_members: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            llm,
            name=name,
            group=group,
            workspace=workspace,
            persist_members=persist_members,
            **kwargs,
        )

        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)

        self.member_configs = [
            self._coerce_member(member)
            for member in (members or [])
        ]

        self.members = {
            member.name: self.build_member(member)
            for member in self.member_configs
        }

        self.players = {
            member.name: EloPlayer(
                name=member.name,
                rating=self.initial_rating,
            )
            for member in self.member_configs
        }

        default_judge_prompt = (
            "You are an impartial judge comparing two candidate solutions "
            "to the same task.\n\n"
            "Judge primarily on correctness, quality of reasoning, completeness, "
            "use of evidence, and adherence to the task. Do not prefer a candidate "
            "merely because it is longer or better written."
        )
        
        judge_instructions = judge_prompt or default_judge_prompt
        
        output_instructions = (
            "\n\n"
            "After evaluating the candidates, you must return exactly one JSON "
            "object with this schema:\n"
            "{\n"
            '  "winner": "A" | "B" | "DRAW",\n'
            '  "reasoning": "brief explanation"\n'
            "}\n"
            "Do not include markdown fences or any text outside the JSON object."
        )
        
        self.judge_prompt = judge_instructions + output_instructions

    @staticmethod
    def _coerce_member(
        member: EnvironmentMemberConfig | Mapping[str, Any],
    ) -> EnvironmentMemberConfig:
        if isinstance(member, EnvironmentMemberConfig):
            return member

        return EnvironmentMemberConfig.from_mapping(member)

    @staticmethod
    def expected_score(
        rating_a: float,
        rating_b: float,
    ) -> float:
        """Return A's expected Elo score against B."""
        return 1.0 / (
            1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
        )

    def update_elo(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,
    ) -> tuple[float, float]:
        """Return updated Elo ratings for a match.

        score_a:
            1.0 -> A wins
            0.5 -> draw
            0.0 -> B wins
        """
        if score_a not in {0.0, 0.5, 1.0}:
            raise ValueError(
                "score_a must be one of 0.0, 0.5, or 1.0"
            )

        expected_a = self.expected_score(
            rating_a,
            rating_b,
        )

        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = (
            rating_a
            + self.k_factor * (score_a - expected_a)
        )

        new_b = (
            rating_b
            + self.k_factor * (score_b - expected_b)
        )

        return new_a, new_b

    def _make_pairs(
        self,
        names: list[str],
    ) -> list[tuple[str, str]]:
        """Pair members in their current order.

        Stage 1 keeps matchmaking intentionally simple.

        If the population is odd, the final member receives
        a bye and does not play this round.
        """
        return [
            (names[index], names[index + 1])
            for index in range(0, len(names) - 1, 2)
        ]

    def _member_prompt(
        self,
        member: EnvironmentMemberConfig,
        task: str,
    ) -> str:
        extra = (
            f"\n\nMember-specific guidance:\n{member.prompt}"
            if member.prompt
            else ""
        )

        return (
            f"You are competitor '{member.name}'.\n"
            f"Your role is: {member.role}.\n\n"
            "Work independently on the task below. "
            "Produce your best complete solution. "
            "Your response will be compared against another competitor."
            f"{extra}\n\n"
            f"Task:\n{task}"
        )

    async def _run_member(
        self,
        member: EnvironmentMemberConfig,
        task: str,
        invoke_kwargs: Mapping[str, Any],
    ) -> tuple[str, str]:
        prompt = self._member_prompt(
            member,
            task,
        )

        result = await self._invoke_member_async(
            self.members[member.name],
            prompt,
            **invoke_kwargs,
        )

        return member.name, result_to_text(result)

    def _judge_messages(
        self,
        task: str,
        player_a: str,
        output_a: str,
        player_b: str,
        output_b: str,
    ) -> list[Any]:
        comparison = (
            f"Original task:\n{task}\n\n"
            f"Candidate A ({player_a}):\n"
            f"{output_a}\n\n"
            f"Candidate B ({player_b}):\n"
            f"{output_b}"
        )

        return [
            SystemMessage(content=self.judge_prompt),
            HumanMessage(content=comparison),
        ]

    async def _judge_match(
        self,
        task: str,
        player_a: str,
        output_a: str,
        player_b: str,
        output_b: str,
    ) -> MatchResult:
        response = await self.llm.ainvoke(
            self._judge_messages(
                task,
                player_a,
                output_a,
                player_b,
                output_b,
            )
        )

        text = result_to_text(response)

        try:
            judgment = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Elo judge returned invalid JSON:\n"
                f"{text}"
            ) from exc

        winner = str(
            judgment.get("winner", "")
        ).strip().upper()

        reasoning = str(
            judgment.get("reasoning", "")
        ).strip()

        if winner == "A":
            score_a = 1.0
        elif winner == "B":
            score_a = 0.0
        elif winner == "DRAW":
            score_a = 0.5
        else:
            raise ValueError(
                "Elo judge must return winner as "
                "'A', 'B', or 'DRAW'. "
                f"Received: {winner!r}"
            )

        return MatchResult(
            player_a=player_a,
            player_b=player_b,
            score_a=score_a,
            reasoning=reasoning,
        )

    def _apply_match_result(
        self,
        result: MatchResult,
    ) -> None:
        player_a = self.players[result.player_a]
        player_b = self.players[result.player_b]

        new_a, new_b = self.update_elo(
            player_a.rating,
            player_b.rating,
            result.score_a,
        )

        player_a.rating = new_a
        player_b.rating = new_b

    def standings(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self.players.values(),
            key=lambda player: player.rating,
            reverse=True,
        )

        return [
            {
                "rank": rank,
                "name": player.name,
                "rating": player.rating,
            }
            for rank, player in enumerate(
                ordered,
                start=1,
            )
        ]

    def _invoke(
        self,
        inputs: Mapping[str, Any],
        **config: Any,
    ) -> dict[str, Any]:
        return self._run_ainvoke_from_sync(
            inputs,
            **config,
        )

    async def _ainvoke(
        self,
        inputs: Mapping[str, Any],
        **config: Any,
    ) -> dict[str, Any]:
        task = str(
            inputs.get("task")
            or inputs.get("prompt")
            or inputs
        )

        if len(self.member_configs) < 2:
            raise ValueError(
                "AgentEloEnvironment requires at least two members."
            )

        invoke_kwargs = invocation_kwargs(config)

        # Phase 1: all competitors work independently.
        member_results = await asyncio.gather(
            *[
                self._run_member(
                    member,
                    task,
                    invoke_kwargs,
                )
                for member in self.member_configs
            ]
        )

        outputs = dict(member_results)

        # Stage 1 matchmaking: pair in configured order.
        pairs = self._make_pairs(
            [
                member.name
                for member in self.member_configs
            ]
        )

        match_results: list[MatchResult] = []

        # Judge sequentially for now. Easy to parallelize later.
        for player_a, player_b in pairs:
            result = await self._judge_match(
                task=task,
                player_a=player_a,
                output_a=outputs[player_a],
                player_b=player_b,
                output_b=outputs[player_b],
            )

            self._apply_match_result(result)
            match_results.append(result)

        return {
            "task": task,
            "outputs": outputs,
            "matches": [
                {
                    "player_a": result.player_a,
                    "player_b": result.player_b,
                    "winner": result.winner,
                    "score_a": result.score_a,
                    "reasoning": result.reasoning,
                }
                for result in match_results
            ],
            "standings": self.standings(),
        }