from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ursa.security import group_agents_dir

from .base import (
    BaseEnvironment,
    invocation_kwargs,
    result_to_text,
)
from .config import (
    AgentEloConfig,
    EnvironmentMemberConfig,
    load_elo_config,
)


@dataclass
class EloPlayer:
    """Evolutionary metadata associated with one active agent."""

    name: str
    rating: float = 1500.0
    generation: int = 0
    parent: str | None = None


@dataclass
class MatchResult:
    """Result of one pairwise competition."""

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
    """Evolutionary pairwise agent environment using Elo ratings.

    Each round:

    1. Every active member independently works on the task.
    2. Members compete pairwise.
    3. An LLM judge returns A, B, or DRAW.
    4. Elo ratings are updated deterministically.
    5. Up to ``deaths_per_round`` decisive losers are eliminated.
    6. The highest-rated survivors reproduce to restore population size.

    For persistent URSA agents, descendants inherit:

    - a copy of the parent's workspace,
    - a fork of the parent's LangGraph checkpoint database,
    - a fork of the parent's LangGraph store,
    - the parent's Elo rating,
    - lineage metadata.

    Parent and child subsequently evolve independently.
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
        config: (
            AgentEloConfig
            | Mapping[str, Any]
            | str
            | Path
            | None
        ) = None,
        name: str | None = None,
        group: str | None = None,
        members: (
            list[
                EnvironmentMemberConfig
                | Mapping[str, Any]
            ]
            | None
        ) = None,
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

        if self.k_factor <= 0:
            raise ValueError(
                "k_factor must be positive."
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

        # Used to guarantee unique descendant names during
        # the lifetime of this environment.
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

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _coerce_config(
        self,
        *,
        config: (
            AgentEloConfig
            | Mapping[str, Any]
            | str
            | Path
            | None
        ),
        name: str | None,
        group: str | None,
        members: (
            list[
                EnvironmentMemberConfig
                | Mapping[str, Any]
            ]
            | None
        ),
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
                    if workspace is not None
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

        if members is not None:
            resolved_members = [
                self._coerce_member(member)
                for member in members
            ]
        else:
            resolved_members = base.members

        return AgentEloConfig(
            name=(
                name
                if name is not None
                else base.name
            ),
            group=(
                group
                if group is not None
                else base.group
            ),
            description=base.description,
            members=resolved_members,
            workspace=(
                str(workspace)
                if workspace is not None
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

        return EnvironmentMemberConfig.from_mapping(
            member
        )

    # ------------------------------------------------------------------
    # Persistent URSA state
    # ------------------------------------------------------------------

    def _member_den(
        self,
        member_name: str,
    ) -> Path:
        """Return the persistent URSA den for a member."""
        agent_name = self._member_agent_name(
            member_name
        )

        if agent_name is None:
            raise RuntimeError(
                "Agent persistence is disabled. "
                "Persistent inheritance requires "
                "persist_members=True."
            )

        return (
            group_agents_dir(self.group)
            / agent_name
        )

    @staticmethod
    def _backup_sqlite_database(
        source: Path,
        destination: Path,
    ) -> None:
        """Create an independent SQLite snapshot."""
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(
                f"SQLite source does not exist: "
                f"{source}"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise FileExistsError(
                f"SQLite destination already exists: "
                f"{destination}"
            )

        with sqlite3.connect(
            source
        ) as source_conn:
            with sqlite3.connect(
                destination
            ) as destination_conn:
                source_conn.backup(
                    destination_conn
                )

    def _fork_parent_persistence(
        self,
        parent_name: str,
        child_name: str,
    ) -> None:
        """Fork parent URSA persistence into a new child den."""
        parent_den = self._member_den(
            parent_name
        )

        child_den = self._member_den(
            child_name
        )

        if not parent_den.exists():
            raise FileNotFoundError(
                f"Parent den does not exist: "
                f"{parent_den}"
            )

        if child_den.exists():
            raise FileExistsError(
                f"Child den already exists: "
                f"{child_den}"
            )

        child_den.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:
            self._backup_sqlite_database(
                parent_den
                / "db"
                / "checkpointer.db",
                child_den
                / "db"
                / "checkpointer.db",
            )

            graph_store_source = (
                parent_den
                / "graph_store.sqlite"
            )

            if graph_store_source.exists():
                self._backup_sqlite_database(
                    graph_store_source,
                    child_den
                    / "graph_store.sqlite",
                )

        except Exception:
            # Do not leave a half-forked lineage behind.
            shutil.rmtree(
                child_den,
                ignore_errors=True,
            )
            raise

    # ------------------------------------------------------------------
    # Elo
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(
        rating_a: float,
        rating_b: float,
    ) -> float:
        """Return A's expected Elo score against B."""
        return 1.0 / (
            1.0
            + 10.0 ** (
                (rating_b - rating_a) / 400.0
            )
        )

    def update_elo(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float,
    ) -> tuple[float, float]:
        """Return updated Elo ratings for a match.

        ``score_a``:
            1.0 -> A wins
            0.5 -> draw
            0.0 -> B wins
        """
        if score_a not in {
            0.0,
            0.5,
            1.0,
        }:
            raise ValueError(
                "score_a must be one of "
                "0.0, 0.5, or 1.0"
            )

        expected_a = self.expected_score(
            rating_a,
            rating_b,
        )

        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = (
            rating_a
            + self.k_factor
            * (score_a - expected_a)
        )

        new_b = (
            rating_b
            + self.k_factor
            * (score_b - expected_b)
        )

        return new_a, new_b

    def _apply_match_result(
        self,
        result: MatchResult,
    ) -> None:
        player_a = self.players[
            result.player_a
        ]

        player_b = self.players[
            result.player_b
        ]

        new_a, new_b = self.update_elo(
            player_a.rating,
            player_b.rating,
            result.score_a,
        )

        player_a.rating = new_a
        player_b.rating = new_b

    # ------------------------------------------------------------------
    # Member execution
    # ------------------------------------------------------------------

    def _make_pairs(
        self,
        names: list[str],
    ) -> list[tuple[str, str]]:
        """Pair active members in their current order.

        For an odd population, the final member receives a bye.
        """
        return [
            (
                names[index],
                names[index + 1],
            )
            for index in range(
                0,
                len(names) - 1,
                2,
            )
        ]

    def _member_prompt(
        self,
        member: EnvironmentMemberConfig,
        task: str,
    ) -> str:
        extra = (
            f"\n\nMember-specific guidance:\n"
            f"{member.prompt}"
            if member.prompt
            else ""
        )

        return (
            f"You are competitor '{member.name}'.\n"
            f"Your role is: {member.role}.\n\n"
            "Work independently on the task below. "
            "Produce your best complete solution. "
            "Your response will be compared against "
            "another competitor."
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

        return (
            member.name,
            result_to_text(result),
        )

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    def _judge_messages(
        self,
        task: str,
        player_a: str,
        output_a: str,
        player_b: str,
        output_b: str,
    ) -> list[Any]:
        comparison = (
            f"Original task:\n"
            f"{task}\n\n"
            f"Candidate A ({player_a}):\n"
            f"{output_a}\n\n"
            f"Candidate B ({player_b}):\n"
            f"{output_b}"
        )

        return [
            SystemMessage(
                content=self.judge_prompt
            ),
            HumanMessage(
                content=comparison
            ),
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

        text = result_to_text(
            response
        )

        try:
            judgment = json.loads(
                text
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Elo judge returned invalid JSON:\n"
                f"{text}"
            ) from exc

        winner = str(
            judgment.get(
                "winner",
                "",
            )
        ).strip().upper()

        reasoning = str(
            judgment.get(
                "reasoning",
                "",
            )
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

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_losers(
        self,
        match_results: list[MatchResult],
    ) -> list[str]:
        """Choose decisive losers for elimination.

        Draws produce no loser.

        If more decisive losers exist than the configured number
        of deaths, the lowest-rated losing agents are eliminated.
        """
        losers = [
            result.loser
            for result in match_results
            if result.loser is not None
        ]

        # Defensive deduplication.
        losers = list(
            dict.fromkeys(losers)
        )

        losers.sort(
            key=lambda name: (
                self.players[name].rating
            )
        )

        return losers[
            : self.deaths_per_round
        ]

    def _eliminate(
        self,
        losers: list[str],
    ) -> None:
        """Remove agents from the active population.

        Persistent dens and workspaces are intentionally left on disk
        so extinct lineages remain inspectable.
        """
        loser_set = set(
            losers
        )

        for name in losers:
            member = self.members.pop(
                name,
                None,
            )

            if member is not None:
                close = getattr(
                    member,
                    "close",
                    None,
                )

                if callable(close):
                    close()

            self.players.pop(
                name,
                None,
            )

        self.member_configs = [
            member
            for member in self.member_configs
            if member.name not in loser_set
        ]

    def _top_survivors(
        self,
        count: int,
    ) -> list[EloPlayer]:
        """Return the highest-rated surviving agents."""
        return sorted(
            self.players.values(),
            key=lambda player: player.rating,
            reverse=True,
        )[:count]

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------

    def _next_child_name(
        self,
        parent: EloPlayer,
    ) -> str:
        """Generate a unique descendant name."""
        count = (
            self._offspring_counts.get(
                parent.name,
                0,
            )
            + 1
        )

        self._offspring_counts[
            parent.name
        ] = count

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
        """Fork the parent's working filesystem."""
        parent_workspace = (
            self._member_workspace(
                parent_name
            )
        )

        child_workspace = (
            self._member_workspace(
                child_name
            )
        )

        if child_workspace.exists():
            raise FileExistsError(
                "Child workspace already exists: "
                f"{child_workspace}"
            )

        if parent_workspace.exists():
            shutil.copytree(
                parent_workspace,
                child_workspace,
            )

        else:
            child_workspace.mkdir(
                parents=True,
                exist_ok=False,
            )

    def _cleanup_failed_child(
        self,
        child_name: str,
    ) -> None:
        """Best-effort rollback of a failed reproduction."""
        child_workspace = (
            self._member_workspace(
                child_name
            )
        )

        shutil.rmtree(
            child_workspace,
            ignore_errors=True,
        )

        if self.persist_members:
            try:
                child_den = self._member_den(
                    child_name
                )
            except Exception:
                return

            shutil.rmtree(
                child_den,
                ignore_errors=True,
            )

    def _reproduce(
        self,
        parents: list[EloPlayer],
    ) -> list[str]:
        """Create one independent child for every selected parent.

        Children inherit:

        - Elo rating,
        - lineage metadata,
        - workspace,
        - URSA checkpoint state,
        - LangGraph persistent store.
        """
        children: list[str] = []

        config_by_name = {
            member.name: member
            for member in self.member_configs
        }

        for parent in parents:
            parent_config = config_by_name[
                parent.name
            ]

            child_name = (
                self._next_child_name(
                    parent
                )
            )

            child_config = replace(
                parent_config,
                name=child_name,
            )

            try:
                # The child filesystem and persistent state must
                # exist before build_member() constructs the agent.
                self._copy_parent_workspace(
                    parent.name,
                    child_name,
                )

                if self.persist_members:
                    self._fork_parent_persistence(
                        parent.name,
                        child_name,
                    )

                child = self.build_member(
                    child_config
                )

            except Exception:
                self._cleanup_failed_child(
                    child_name
                )
                raise

            self.members[
                child_name
            ] = child

            self.member_configs.append(
                child_config
            )

            self.players[
                child_name
            ] = EloPlayer(
                name=child_name,
                rating=parent.rating,
                generation=(
                    parent.generation + 1
                ),
                parent=parent.name,
            )

            children.append(
                child_name
            )

        return children

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def standings(
        self,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            self.players.values(),
            key=lambda player: (
                player.rating
            ),
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

    # ------------------------------------------------------------------
    # Environment invocation
    # ------------------------------------------------------------------

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
                "AgentEloEnvironment requires "
                "at least two active members."
            )

        initial_population_size = len(
            self.players
        )

        invoke_kwargs = (
            invocation_kwargs(config)
        )

        # --------------------------------------------------------------
        # Phase 1: independent agent work
        # --------------------------------------------------------------

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

        outputs = dict(
            member_results
        )

        # --------------------------------------------------------------
        # Phase 2: pairwise competition
        # --------------------------------------------------------------

        pairs = self._make_pairs(
            [
                member.name
                for member in self.member_configs
            ]
        )

        match_results: list[
            MatchResult
        ] = []

        for player_a, player_b in pairs:
            result = await self._judge_match(
                task=task,
                player_a=player_a,
                output_a=outputs[
                    player_a
                ],
                player_b=player_b,
                output_b=outputs[
                    player_b
                ],
            )

            self._apply_match_result(
                result
            )

            match_results.append(
                result
            )

        standings_after_matches = (
            self.standings()
        )

        # --------------------------------------------------------------
        # Phase 3: elimination
        # --------------------------------------------------------------

        eliminated = self._select_losers(
            match_results
        )

        self._eliminate(
            eliminated
        )

        # --------------------------------------------------------------
        # Phase 4: reproduction
        #
        # We reproduce exactly as many agents as were actually killed.
        # Therefore draws can reduce turnover without changing the
        # population size.
        # --------------------------------------------------------------

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

        if (
            final_population_size
            != initial_population_size
        ):
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
                    "player_a": (
                        result.player_a
                    ),
                    "player_b": (
                        result.player_b
                    ),
                    "winner": (
                        result.winner
                    ),
                    "loser": (
                        result.loser
                    ),
                    "score_a": (
                        result.score_a
                    ),
                    "reasoning": (
                        result.reasoning
                    ),
                }
                for result in match_results
            ],
            "standings_after_matches": (
                standings_after_matches
            ),
            "eliminated": eliminated,
            "reproducing_parents": (
                parent_names
            ),
            "children": children,
            "standings": self.standings(),
            "population_size": (
                final_population_size
            ),
        }