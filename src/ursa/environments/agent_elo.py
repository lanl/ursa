from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import (
    AgentEloConfig,
    EnvironmentMemberConfig,
    load_elo_config,
)

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
    generation: int = 0
    parent: str | None = None
    latest_output: str | None = None


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

    @property
    def loser(self) -> str | None:
        if self.score_a == 1.0:
            return self.player_b
        if self.score_a == 0.0:
            return self.player_a
        return None


class AgentEloEnvironment(BaseEnvironment):
    """Pairwise competitive environment with Elo-based reproduction.

    Stage 2:
    - All members independently answer the same task.
    - Members are paired.
    - An LLM judge returns A, B, or DRAW.
    - Elo ratings are updated deterministically.
    - A fixed number of match losers are eliminated.
    - The highest-rated survivors reproduce.
    - Children inherit the parent's latest output and workspace.
    - True agent checkpoint/state cloning is deferred to Stage 4.
    """

    JUDGE_OUTPUT_INSTRUCTIONS = (
        "\n\n"
        "After evaluating the candidates, you must return exactly one JSON "
        "object with this schema:\n"
        "{\n"
        '  "winner": "A" | "B" | "DRAW",\n'
        '  "reasoning": "brief explanation"\n'
        "}\n"
        "Do not include markdown fences or any text outside the JSON object."
    )

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        config: AgentEloConfig
        | Mapping[str, Any]
        | str
        | Path
        | None = None,
        name: str | None = None,
        group: str | None = None,
        members: list[
            EnvironmentMemberConfig
            | Mapping[str, Any]
        ]
        | None = None,
        workspace: str | Path | None = None,
        initial_rating: float | None = None,
        k_factor: float | None = None,
        deaths_per_round: int | None = None,
        judge_prompt: str | None = None,
        persist_members: bool = True,
        **kwargs: Any,
    ):


        elo_config = self._coerce_config(
            config=config,
            name=name,
            group=group,
            members=members,
            workspace=workspace,
            initial_rating=initial_rating,
            k_factor=k_factor,
            deaths_per_round=deaths_per_round,
            judge_prompt=judge_prompt,
        )
                    
        super().__init__(
            llm,
            name=elo_config.name,
            group=elo_config.group,
            workspace=elo_config.workspace or workspace,
            persist_members=persist_members,
            **kwargs,
        )


        self.config = elo_config
        
        self.initial_rating = float(
            elo_config.initial_rating
        )
        
        self.k_factor = float(
            elo_config.k_factor
        )
        
        self.deaths_per_round = int(
            elo_config.deaths_per_round
        )

        
        if self.deaths_per_round < 0:
            raise ValueError(
                "deaths_per_round must be non-negative."
            )
        

        self.member_configs = list(
            self.config.members
        )

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

        # Used only to guarantee unique descendant names.
        self._offspring_counts: dict[str, int] = {}

        default_judge_prompt = (
            "You are an impartial judge comparing two candidate solutions "
            "to the same task.\n\n"
            "Judge primarily on correctness, quality of reasoning, completeness, "
            "use of evidence, and adherence to the task. Do not prefer a candidate "
            "merely because it is longer or better written."
        )

        judge_instructions = (
            self.config.judge_prompt
            or default_judge_prompt
        )
        
        self.judge_prompt = (
            judge_instructions
            + self.JUDGE_OUTPUT_INSTRUCTIONS
        )


    def _coerce_config(
        self,
        *,
        config: AgentEloConfig
        | Mapping[str, Any]
        | str
        | Path
        | None,
        name: str | None,
        group: str | None,
        members: list[
            EnvironmentMemberConfig
            | Mapping[str, Any]
        ]
        | None,
        workspace: str | Path | None,
        initial_rating: float | None,
        k_factor: float | None,
        deaths_per_round: int | None,
        judge_prompt: str | None,
    ) -> AgentEloConfig:
        if isinstance(config, (str, Path)):
            base = load_elo_config(config)
    
        elif isinstance(config, Mapping):
            base = AgentEloConfig.from_mapping(
                config
            )
    
        elif isinstance(config, AgentEloConfig):
            base = config
    
        else:
            member_cfgs = [
                self._coerce_member(member)
                for member in (members or [])
            ]
    
            base = AgentEloConfig(
                name=name or "agent_elo",
                group=group or "default",
                members=member_cfgs,
                workspace=(
                    str(workspace)
                    if workspace
                    else None
                ),
                initial_rating=(
                    initial_rating
                    if initial_rating is not None
                    else 1500.0
                ),
                k_factor=(
                    k_factor
                    if k_factor is not None
                    else 32.0
                ),
                deaths_per_round=(
                    deaths_per_round
                    if deaths_per_round is not None
                    else 1
                ),
                judge_prompt=judge_prompt,
            )
    
        return AgentEloConfig(
            name=name or base.name,
            group=group or base.group,
            description=base.description,
            members=base.members,
            workspace=(
                str(workspace)
                if workspace
                else base.workspace
            ),
            defaults=base.defaults,
            initial_rating=(
                initial_rating
                if initial_rating is not None
                else base.initial_rating
            ),
            k_factor=(
                k_factor
                if k_factor is not None
                else base.k_factor
            ),
            deaths_per_round=(
                deaths_per_round
                if deaths_per_round is not None
                else base.deaths_per_round
            ),
            judge_prompt=(
                judge_prompt
                if judge_prompt is not None
                else base.judge_prompt
            ),
        )    


    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        llm: BaseChatModel,
        **kwargs: Any,
    ) -> "AgentEloEnvironment":
        return cls(
            llm=llm,
            config=load_elo_config(path),
            **kwargs,
        )    
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

        Stage 2 still keeps matchmaking intentionally simple.

        If the population is odd, the final member receives a bye.
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

        player = self.players[member.name]

        inherited_state = ""

        if (
            player.parent is not None
            and player.latest_output
        ):
            inherited_state = (
                "\n\n"
                "You are a descendant of another research agent. "
                "You inherit the following research state from your parent. "
                "Use it as prior work, but independently continue the problem "
                "and improve, revise, or extend it where appropriate.\n\n"
                "Inherited research state:\n"
                "-------------------------\n"
                f"{player.latest_output}\n"
                "-------------------------"
            )

        return (
            f"You are competitor '{member.name}'.\n"
            f"Your role is: {member.role}.\n\n"
            "Work independently on the task below. "
            "Produce your best complete solution. "
            "Your response will be compared against another competitor."
            f"{extra}"
            f"{inherited_state}\n\n"
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

    def _select_losers(
        self,
        match_results: list[MatchResult],
    ) -> list[str]:
        """Choose which match losers are eliminated.

        Draws produce no loser.

        If there are more decisive losers than deaths_per_round,
        eliminate the lowest-rated losers first.
        """
        losers = [
            result.loser
            for result in match_results
            if result.loser is not None
        ]

        # Defensive deduplication. Under the current matchmaking
        # each player appears in at most one match.
        losers = list(dict.fromkeys(losers))

        losers.sort(
            key=lambda name: self.players[name].rating
        )

        return losers[: self.deaths_per_round]

    def _eliminate(
        self,
        losers: list[str],
    ) -> None:
        """Remove players from the active population.

        Their workspaces are intentionally left on disk so the history
        of extinct lineages remains inspectable.
        """
        loser_set = set(losers)

        for name in losers:
            self.players.pop(name, None)
            self.members.pop(name, None)

        self.member_configs = [
            member
            for member in self.member_configs
            if member.name not in loser_set
        ]

    def _top_survivors(
        self,
        count: int,
    ) -> list[EloPlayer]:
        """Return the highest-rated surviving players."""
        return sorted(
            self.players.values(),
            key=lambda player: player.rating,
            reverse=True,
        )[:count]

    def _next_child_name(
        self,
        parent: EloPlayer,
    ) -> str:
        """Generate a unique descendant name."""
        count = self._offspring_counts.get(
            parent.name,
            0,
        ) + 1

        self._offspring_counts[parent.name] = count

        return (
            f"{parent.name}_g"
            f"{parent.generation + 1}_"
            f"{count}"
        )

    def _copy_parent_workspace(
        self,
        parent_name: str,
        child_name: str,
    ) -> None:
        """Copy the parent's workspace into the child's workspace."""
        parent_workspace = self._member_workspace(
            parent_name
        )

        child_workspace = self._member_workspace(
            child_name
        )

        if parent_workspace.exists():
            shutil.copytree(
                parent_workspace,
                child_workspace,
                dirs_exist_ok=True,
            )
        else:
            child_workspace.mkdir(
                parents=True,
                exist_ok=True,
            )

    def _reproduce(
        self,
        parents: list[EloPlayer],
    ) -> list[str]:
        """Create one child for every selected parent.

        Stage-2 inheritance consists of:
        - inherited Elo rating,
        - parent/generation metadata,
        - parent's latest output,
        - copied workspace.

        True checkpoint inheritance is deferred to Stage 4.
        """
        children: list[str] = []

        config_by_name = {
            member.name: member
            for member in self.member_configs
        }

        for parent in parents:
            parent_config = config_by_name[parent.name]

            child_name = self._next_child_name(parent)

            child_config = replace(
                parent_config,
                name=child_name,
            )

            self._copy_parent_workspace(
                parent.name,
                child_name,
            )

            child = self.build_member(
                child_config
            )

            self.members[child_name] = child
            self.member_configs.append(
                child_config
            )

            self.players[child_name] = EloPlayer(
                name=child_name,
                rating=parent.rating,
                generation=parent.generation + 1,
                parent=parent.name,
                latest_output=parent.latest_output,
            )

            children.append(child_name)

        return children

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
                "generation": player.generation,
                "parent": player.parent,
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

        initial_population_size = len(
            self.players
        )

        invoke_kwargs = invocation_kwargs(
            config
        )

        # ---------------------------------------------------------
        # Phase 1: competitors work independently.
        # ---------------------------------------------------------

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

        for name, output in outputs.items():
            self.players[name].latest_output = (
                output
            )

        # ---------------------------------------------------------
        # Phase 2: pairwise matches.
        # ---------------------------------------------------------

        pairs = self._make_pairs(
            [
                member.name
                for member in self.member_configs
            ]
        )

        match_results: list[MatchResult] = []

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

        standings_after_matches = (
            self.standings()
        )

        # ---------------------------------------------------------
        # Phase 3: selection.
        # ---------------------------------------------------------

        eliminated = self._select_losers(
            match_results
        )

        self._eliminate(eliminated)

        # ---------------------------------------------------------
        # Phase 4: reproduction.
        #
        # Reproduce exactly as many agents as were actually killed,
        # thereby conserving population size even when draws occur.
        # ---------------------------------------------------------

        parents = self._top_survivors(
            len(eliminated)
        )

        parent_names = [
            parent.name
            for parent in parents
        ]

        children = self._reproduce(
            parents
        )

        final_population_size = len(
            self.players
        )

        if final_population_size != initial_population_size:
            raise RuntimeError(
                "Population size changed unexpectedly: "
                f"{initial_population_size} -> "
                f"{final_population_size}"
            )

        return {
            "task": task,
            "outputs": outputs,
            "matches": [
                {
                    "player_a": result.player_a,
                    "player_b": result.player_b,
                    "winner": result.winner,
                    "loser": result.loser,
                    "score_a": result.score_a,
                    "reasoning": result.reasoning,
                }
                for result in match_results
            ],
            "standings_after_matches": standings_after_matches,
            "eliminated": eliminated,
            "reproducing_parents": parent_names,
            "children": children,
            "standings": self.standings(),
            "population_size": final_population_size,
        }