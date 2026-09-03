from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping
import random
from datetime import datetime, timedelta, timezone

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


@dataclass
class MemberRunResult:
    """Result of one member's work during a generation."""

    name: str
    status: str
    output: str | None
    deadline: str | None = None
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def timed_out(self) -> bool:
        return self.status == "timed_out"

    @property
    def failed(self) -> bool:
        return self.status == "failed"


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
        seed: int | None = None,
        generations: int | None = None,
        member_timeout_seconds: float | None = None,
        restart_from_json: str | Path | None = None,
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
            seed=seed,
            generations=generations,
            member_timeout_seconds=member_timeout_seconds,
            restart_from_json=restart_from_json,
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


        if self.k_factor <= 0:
            raise ValueError(
                "k_factor must be positive."
            )


        self.deaths_per_round = int(
            elo_config.deaths_per_round
        )

        if self.deaths_per_round < 0:
            raise ValueError(
                "deaths_per_round must be non-negative."
            )


        self.generations = int(
            elo_config.generations
        )
        
        if self.generations < 1:
            raise ValueError(
                "generations must be at least 1."
            )

        self.member_timeout_seconds = (
            None
            if elo_config.member_timeout_seconds is None
            else float(elo_config.member_timeout_seconds)
        )
        
        if (
            self.member_timeout_seconds is not None
            and self.member_timeout_seconds <= 0
        ):
            raise ValueError(
                "member_timeout_seconds must be positive or None."
            )

        self.seed = elo_config.seed

        self._rng = random.Random(self.seed)

        self.generation_index = 0

        self._offspring_counts: dict[str,int,] = {}
        
        if (
            self.config.restart_from_json
            is not None
        ):
            if not self.persist_members:
                raise ValueError(
                    "restart_from_json requires "
                    "persist_members=True."
                )
        
            state = self._load_environment_state(
                self.config.restart_from_json
            )
        
            self._restore_environment_state(
                state
            )
        
        else:
            self.member_configs = list(
                self.config.members
            )

            self._validate_population_size(
                len(self.member_configs)
            )
                   
            self.members = {
                member.name: self.build_member(
                    member
                )
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


    @staticmethod
    def _validate_population_size(
        population_size: int,
    ) -> None:
        """Validate the active Elo population size."""
    
        if population_size < 2:
            raise ValueError(
                "AgentEloEnvironment requires at least two active members."
            )
    
        if population_size % 2 != 0:
            raise ValueError(
                "AgentEloEnvironment requires an even number of active members. "
                f"Received {population_size}."
            )

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
        seed: int | None,
        generations: int | None,
        member_timeout_seconds: float | None,
        restart_from_json: str | Path | None,
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
                seed=seed,
                generations=(
                    generations
                    if generations is not None
                    else 1
                ),
                member_timeout_seconds=member_timeout_seconds,
                restart_from_json=(
                    str(restart_from_json)
                    if restart_from_json is not None
                    else None
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
            seed=(
                seed
                if seed is not None
                else base.seed
            ),
            generations=(
                generations
                if generations is not None
                else base.generations
            ),
            member_timeout_seconds=(
                member_timeout_seconds
                if member_timeout_seconds is not None
                else base.member_timeout_seconds
            ),
            restart_from_json=(
                str(restart_from_json)
                if restart_from_json is not None
                else base.restart_from_json
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
    # For lightweight environment level persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _member_config_to_mapping(
        member: EnvironmentMemberConfig,
    ) -> dict[str, Any]:
        """Convert a member config into JSON-safe data."""
    
        model = None
    
        if member.model is not None:
            model = member.model.model_dump(
                mode="json",
                exclude_none=True,
            )
    
        return {
            "name": member.name,
            "role": member.role,
            "agent": member.agent,
            "model": model,
            "config": member.config,
            "prompt": member.prompt,
            "reviewer": member.reviewer,
        }

    @staticmethod
    def _rng_state_to_json(
        value: Any,
    ) -> Any:
        if isinstance(value, tuple):
            return [
                AgentEloEnvironment._rng_state_to_json(
                    item
                )
                for item in value
            ]
    
        return value
    
    
    @staticmethod
    def _rng_state_from_json(
        value: Any,
    ) -> Any:
        if isinstance(value, list):
            return tuple(
                AgentEloEnvironment._rng_state_from_json(
                    item
                )
                for item in value
            )
    
        return value


    def _environment_state_path(
        self,
    ) -> Path:
        return (
            Path(self.workspace)
            / "environment_state.json"
        )



    def _environment_state_payload(
        self,
    ) -> dict[str, Any]:
        """Return lightweight resumable environment state."""
    
        config_by_name = {
            member.name: member
            for member in self.member_configs
        }
    
        active_players = []
    
        # Use member_configs ordering so active population order
        # is preserved exactly across restart.
        for member in self.member_configs:
            player = self.players[
                member.name
            ]
    
            active_players.append(
                {
                    "name": player.name,
                    "rating": player.rating,
                    "generation": player.generation,
                    "parent": player.parent,
                    "member_config": (
                        self._member_config_to_mapping(
                            config_by_name[
                                player.name
                            ]
                        )
                    ),
                }
            )
    
        return {
            "schema_version": 1,
            "environment_name": self.name,
            "group": self.group,
            "generation": self.generation_index,
            "seed": self.seed,
            "rng_state": self._rng_state_to_json(
                self._rng.getstate()
            ),
            "active_players": active_players,
        }


    def _save_environment_state(
        self,
    ) -> Path:
        """Atomically save lightweight environment restart state."""
    
        target = self._environment_state_path()
    
        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    
        temporary = target.with_suffix(
            ".json.tmp"
        )
    
        payload = self._environment_state_payload()
    
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    
        temporary.replace(
            target
        )
    
        return target


    @staticmethod
    def _load_environment_state(
        path: str | Path,
    ) -> dict[str, Any]:
        state_path = Path(
            path
        ).expanduser()
    
        if not state_path.exists():
            raise FileNotFoundError(
                "Elo environment restart file "
                f"does not exist: {state_path}"
            )
    
        state = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )
    
        if not isinstance(state, dict):
            raise ValueError(
                "Elo environment restart file "
                "must contain a JSON object."
            )
    
        if state.get(
            "schema_version"
        ) != 1:
            raise ValueError(
                "Unsupported Elo environment "
                "state schema version: "
                f"{state.get('schema_version')!r}"
            )
    
        return state



    def _restore_environment_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """Restore the active evolutionary population.
    
        This method restores only environment metadata.
    
        Individual URSA agents reopen their own persistence
        independently when build_member() is called.
        """
    
        saved_name = state.get(
            "environment_name"
        )
    
        saved_group = state.get(
            "group"
        )
    
        if saved_name != self.name:
            raise ValueError(
                "Restart environment name mismatch: "
                f"snapshot={saved_name!r}, "
                f"current={self.name!r}"
            )
    
        if saved_group != self.group:
            raise ValueError(
                "Restart group mismatch: "
                f"snapshot={saved_group!r}, "
                f"current={self.group!r}"
            )
    
        raw_players = state.get(
            "active_players"
        )
    
        if not isinstance(
            raw_players,
            list,
        ):
            raise ValueError(
                "Restart state is missing "
                "'active_players'."
            )

        self._validate_population_size(
            len(raw_players)
        )
            
        member_configs: list[
            EnvironmentMemberConfig
        ] = []
    
        players: dict[
            str,
            EloPlayer,
        ] = {}
    
        for raw_player in raw_players:
            if not isinstance(
                raw_player,
                Mapping,
            ):
                raise ValueError(
                    "Each active player entry must "
                    "be a mapping."
                )
    
            raw_member_config = raw_player.get(
                "member_config"
            )
    
            if not isinstance(
                raw_member_config,
                Mapping,
            ):
                raise ValueError(
                    "Restart player is missing "
                    "'member_config'."
                )
    
            member_config = (
                EnvironmentMemberConfig.from_mapping(
                    raw_member_config,
                    group=self.group,
                )
            )
    
            name = str(
                raw_player["name"]
            )
    
            if member_config.name != name:
                raise ValueError(
                    "Restart player/member config "
                    f"name mismatch: {name!r} vs "
                    f"{member_config.name!r}"
                )
    
            # A restart should resume an existing persistent
            # URSA agent, never silently create a fresh one.
            den = self._member_den(
                name
            )
    
            if not den.exists():
                raise FileNotFoundError(
                    "Persistent URSA agent den "
                    "required for restart does not exist: "
                    f"{den}"
                )
    
            member_workspace = (
                self._member_workspace(
                    name
                )
            )
    
            if not member_workspace.exists():
                raise FileNotFoundError(
                    "Agent workspace required for "
                    "restart does not exist: "
                    f"{member_workspace}"
                )
    
            member_configs.append(
                member_config
            )
    
            players[name] = EloPlayer(
                name=name,
                rating=float(
                    raw_player["rating"]
                ),
                generation=int(
                    raw_player["generation"]
                ),
                parent=raw_player.get(
                    "parent"
                ),
            )
    
        self.member_configs = member_configs
    
        self.players = players
    
        # Constructing by the same names causes URSA to reopen
        # the agents' existing persistent state.
        self.members = {
            member.name: self.build_member(
                member
            )
            for member in self.member_configs
        }
    
        self.generation_index = int(
            state.get(
                "generation",
                0,
            )
        )
    
        rng_state = state.get(
            "rng_state"
        )
    
        if rng_state is not None:
            self._rng.setstate(
                self._rng_state_from_json(
                    rng_state
                )
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
        """Randomly pair all active members.
    
        AgentEloEnvironment requires an even population, so every
        active member participates in exactly one match per generation.
        """
    
        self._validate_population_size(
            len(names)
        )
    
        shuffled = list(names)
    
        self._rng.shuffle(
            shuffled
        )
    
        return [
            (
                shuffled[index],
                shuffled[index + 1],
            )
            for index in range(
                0,
                len(shuffled),
                2,
            )
        ]

    
    def _member_prompt(
        self,
        member: EnvironmentMemberConfig,
        task: str,
        *,
        deadline: datetime | None = None,
    ) -> str:

        player = self.players[
            member.name
        ]
    
        extra = (
            f"\n\nMember-specific guidance:\n"
            f"{member.prompt}"
            if member.prompt
            else ""
        )
    
        lineage = (
            "Evolutionary status:\n"
            f"- Name: {player.name}\n"
            f"- Lineage generation: {player.generation}\n"
            f"- Parent: {player.parent or 'None'}\n"
            f"- Environment generations already completed: "
            f"{self.generation_index}\n"
        )


        if deadline is None:
            execution_budget = ""
        
        else:
            deadline_text = deadline.isoformat()
        
            execution_budget = (
                "\n\nExecution budget:\n"
                f"- Hard deadline: {deadline_text}\n"
                "- The deadline is expressed in UTC.\n"
                "- You may run `date -u` to check the current wall-clock time.\n"
                "- Plan your work so that useful, judgeable results exist "
                "before the deadline.\n"
                "- If you do not finish working before the deadline, "
                "you may automatically lose this round.\n"
            )
            
        if self.generation_index == 0:
            evolutionary_instruction = (
                "This is the founding round of the evolutionary run.\n"
                "Develop the strongest independent solution you can. "
                "Explore the problem directly and execute the work required "
                "by the task."
            )
    
        elif player.parent is not None:
            evolutionary_instruction = (
                "You are a descendant of a previously successful agent. "
                "You inherited your parent's research state and workspace.\n\n"
                "Do not merely repeat, rerun, or summarize the inherited work. "
                "Treat it as a successful starting point.\n\n"
                "Identify at least one substantive scientific, numerical, or "
                "methodological limitation in the inherited work."
                "Choose a modification that addresses it and execute that work."
            )
    
        else:
            evolutionary_instruction = (
                "You are a surviving founding agent from an earlier round. "
                "Continue developing your existing research rather than starting "
                "over or merely summarizing it.\n\n"
                "Identify a substantive limitation, uncertainty, weakness, or "
                "untested assumption in your current work. Make and execute a "
                "scientifically motivated improvement."
            )
    
        return (
            f"You are competitor '{member.name}'.\n"
            f"Your role is: {member.role}.\n\n"
            f"{lineage}\n"
            f"Evolutionary instructions:\n"
            f"{evolutionary_instruction}"
            f"{execution_budget}"
            f"{extra}\n\n"
            f"Scientific task:\n{task}"
        )

    async def _run_member(
        self,
        member: EnvironmentMemberConfig,
        task: str,
        invoke_kwargs: Mapping[str, Any],
        *,
        deadline: datetime | None = None,
    ) -> MemberRunResult:
        prompt = self._member_prompt(
            member,
            task,
            deadline=deadline,
        )
    
        async def invoke() -> Any:
            return await self._invoke_member_async(
                self.members[member.name],
                prompt,
                **invoke_kwargs,
            )
    
        deadline_text = (
            deadline.isoformat()
            if deadline is not None
            else None
        )
    
        try:
            if deadline is None:
                result = await invoke()
    
            else:
                remaining_seconds = (
                    deadline
                    - datetime.now(timezone.utc)
                ).total_seconds()
    
                if remaining_seconds <= 0:
                    return MemberRunResult(
                        name=member.name,
                        status="timed_out",
                        output=None,
                        deadline=deadline_text,
                    )
    
                # Cancellation stops the environment from awaiting this
                # member. A blocking subprocess already running in an
                # executor may continue until that subprocess exits or
                # reaches its own timeout.
                result = await asyncio.wait_for(
                    invoke(),
                    timeout=remaining_seconds,
                )
    
            return MemberRunResult(
                name=member.name,
                status="completed",
                output=result_to_text(result),
                deadline=deadline_text,
            )
    
        except TimeoutError:
            return MemberRunResult(
                name=member.name,
                status="timed_out",
                output=None,
                deadline=deadline_text,
            )
    
        except Exception as exc:
            return MemberRunResult(
                name=member.name,
                status="failed",
                output=None,
                deadline=deadline_text,
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
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

    async def _resolve_match(
        self,
        *,
        task: str,
        player_a: str,
        run_a: MemberRunResult,
        player_b: str,
        run_b: MemberRunResult,
    ) -> MatchResult:
        """Resolve a match using execution status before LLM judging."""
    
        # Both produced valid completed outputs: use the normal judge.
        if run_a.completed and run_b.completed:
            assert run_a.output is not None
            assert run_b.output is not None
    
            return await self._judge_match(
                task=task,
                player_a=player_a,
                output_a=run_a.output,
                player_b=player_b,
                output_b=run_b.output,
            )
    
        # A completed and B did not.
        if run_a.completed:
            return MatchResult(
                player_a=player_a,
                player_b=player_b,
                score_a=1.0,
                reasoning=(
                    f"{player_a} completed the generation successfully; "
                    f"{player_b} did not complete successfully "
                    f"(status={run_b.status}). "
                    f"{player_a} therefore wins automatically."
                ),
            )
    
        # B completed and A did not.
        if run_b.completed:
            return MatchResult(
                player_a=player_a,
                player_b=player_b,
                score_a=0.0,
                reasoning=(
                    f"{player_b} completed the generation successfully; "
                    f"{player_a} did not complete successfully "
                    f"(status={run_a.status}). "
                    f"{player_b} therefore wins automatically."
                ),
            )
    
        # Neither completed.
        return MatchResult(
            player_a=player_a,
            player_b=player_b,
            score_a=0.5,
            reasoning=(
                "Neither competitor completed the generation successfully. "
                f"{player_a} status={run_a.status}; "
                f"{player_b} status={run_b.status}. "
                "Match recorded as a draw."
            ),
        )

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

        self._rng.shuffle(
            losers
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
        """Return the highest-rated surviving agents.
    
        Equal-rated survivors are ordered randomly using
        the environment RNG.
        """
        candidates = list(
            self.players.values()
        )
    
        self._rng.shuffle(
            candidates
        )
    
        candidates.sort(
            key=lambda player: player.rating,
            reverse=True,
        )
    
        return candidates[:count]

    # ------------------------------------------------------------------
    # Reproduction
    # ------------------------------------------------------------------


    def _child_name_available(
        self,
        child_name: str,
    ) -> bool:
        """Return whether a name is safe to use for a new birth.
    
        A generated child name must not collide with:
    
        - an active player,
        - an active member,
        - an active member config,
        - an existing environment workspace,
        - an existing persistent URSA den.
    
        Existing persistent dens are checked only when member
        persistence is enabled.
        """
    
        if child_name in self.players:
            return False
    
        if child_name in self.members:
            return False
    
        if any(
            member.name == child_name
            for member in self.member_configs
        ):
            return False
    
        child_workspace = self._member_workspace(
            child_name
        )
    
        if child_workspace.exists():
            return False
    
        if self.persist_members:
            child_den = self._member_den(
                child_name
            )
    
            if child_den.exists():
                return False
    
        return True



    def _next_child_name(
        self,
        parent: EloPlayer,
    ) -> str:
        """Generate the next unused descendant identity.
    
        The search starts after the highest child count generated
        by this environment instance, but also checks existing
        workspaces and persistent agent dens so resumed runs cannot
        accidentally reuse an older descendant.
        """
    
        count = (
            self._offspring_counts.get(
                parent.name,
                0,
            )
            + 1
        )
    
        child_generation = (
            parent.generation + 1
        )
    
        while True:
            child_name = (
                f"{parent.name}_g"
                f"{child_generation}_"
                f"{count}"
            )
    
            if self._child_name_available(
                child_name
            ):
                self._offspring_counts[
                    parent.name
                ] = count
    
                return child_name
    
            count += 1
            

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

    async def _run_generation(
        self,
        task: str,
        invoke_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one complete evolutionary generation."""
    
        self._validate_population_size(
            len(self.member_configs)
        )

        if not (
            len(self.member_configs)
            == len(self.members)
            == len(self.players)
        ):
            raise RuntimeError(
                "AgentEloEnvironment active population state is inconsistent: "
                f"member_configs={len(self.member_configs)}, "
                f"members={len(self.members)}, "
                f"players={len(self.players)}."
            )

        generation_number = (
            self.generation_index + 1
        )
    
        initial_population_size = len(
            self.players
        )
    
        # Capture ratings before competition for reporting.
        ratings_before = {
            name: player.rating
            for name, player in self.players.items()
        }

        generation_deadline: datetime | None = None
        
        if self.member_timeout_seconds is not None:
            generation_deadline = (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=self.member_timeout_seconds
                )
            )
        
                
        # ----------------------------------------------------------
        # Phase 1: independent research
        # ----------------------------------------------------------
        
        member_results = await asyncio.gather(
            *[
                self._run_member(
                    member,
                    task,
                    invoke_kwargs,
                    deadline=generation_deadline,
                )
                for member in self.member_configs
            ]
        )
                
        runs = {
            result.name: result
            for result in member_results
        }


        outputs = {
            name: (
                result.output
                if result.completed
                else None
            )
            for name, result in runs.items()
        }


        timed_out = [
            name
            for name, result in runs.items()
            if result.timed_out
        ]

        failed = [
            name
            for name, result in runs.items()
            if result.failed
        ]
        
        # ----------------------------------------------------------
        # Phase 2: randomized pairwise competition
        # ----------------------------------------------------------
    
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
            result = await self._resolve_match(
                task=task,
                player_a=player_a,
                run_a=runs[player_a],
                player_b=player_b,
                run_b=runs[player_b],
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
    
        # ----------------------------------------------------------
        # Phase 3: elimination
        # ----------------------------------------------------------
    
        eliminated = self._select_losers(
            match_results
        )
    
        self._eliminate(
            eliminated
        )
    
        # ----------------------------------------------------------
        # Phase 4: reproduction
        # ----------------------------------------------------------
    
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
    
        # ----------------------------------------------------------
        # Generation successfully completed.
        #
        # Increment BEFORE snapshotting so restart begins at the
        # following generation.
        # ----------------------------------------------------------
    
        self.generation_index += 1
    
        state_path = (
            self._save_environment_state()
        )
    
        return {
            "generation": generation_number,
            "task": task,
            "outputs": outputs,
            "pairs": [
                list(pair)
                for pair in pairs
            ],
            "ratings_before": ratings_before,
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
            "environment_state": str(
                state_path
            ),
            "member_runs": {
                name: {
                    "status": run.status,
                    "deadline": run.deadline,
                    "error": run.error,
                }
                for name, run in runs.items()
            },
            "timed_out": timed_out,
            "failed": failed,
            "generation_deadline": (
                generation_deadline.isoformat()
                if generation_deadline is not None
                else None
            ),
        }


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
    
        invoke_kwargs = (
            invocation_kwargs(config)
        )
    
        generation_results = []
    
        starting_generation = (
            self.generation_index
        )
    
        for _ in range(
            self.generations
        ):
            result = await self._run_generation(
                task,
                invoke_kwargs,
            )
    
            generation_results.append(
                result
            )
    
        return {
            "task": task,
            "starting_generation": (
                starting_generation
            ),
            "completed_generations": (
                len(generation_results)
            ),
            "ending_generation": (
                self.generation_index
            ),
            "generations": (
                generation_results
            ),
            "standings": self.standings(),
            "population_size": len(
                self.players
            ),
            "environment_state": str(
                self._environment_state_path()
            ),
        }