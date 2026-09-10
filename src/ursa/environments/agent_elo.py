from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from langchain.chat_models import BaseChatModel

from ursa.security import group_agents_dir
from ursa.util.events import EnvironmentEvents

from .agent_elo_judge import AgentEloJudge
from .base import (
    BaseEnvironment,
    invocation_kwargs,
    result_to_text,
    runnable_config_from_kwargs,
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
    progress_report: str | None = None

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

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        config: (AgentEloConfig | Mapping[str, Any] | str | Path | None) = None,
        name: str | None = None,
        group: str | None = None,
        members: (
            list[EnvironmentMemberConfig | Mapping[str, Any]] | None
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

        self.initial_rating = float(elo_config.initial_rating)

        self.k_factor = float(elo_config.k_factor)

        if self.k_factor <= 0:
            raise ValueError("k_factor must be positive.")

        self.deaths_per_round = int(elo_config.deaths_per_round)

        if self.deaths_per_round < 0:
            raise ValueError("deaths_per_round must be non-negative.")

        self.generations = int(elo_config.generations)

        if self.generations < 1:
            raise ValueError("generations must be at least 1.")

        self.member_timeout_seconds = (
            None
            if elo_config.member_timeout_seconds is None
            else float(elo_config.member_timeout_seconds)
        )

        if (
            self.member_timeout_seconds is not None
            and self.member_timeout_seconds <= 0
        ):
            raise ValueError("member_timeout_seconds must be positive or None.")

        self.seed = elo_config.seed

        self._rng = random.Random(self.seed)

        self.generation_index = 0

        self._offspring_counts: dict[
            str,
            int,
        ] = {}

        if self.config.restart_from_json is not None:
            if not self.persist_members:
                raise ValueError(
                    "restart_from_json requires persist_members=True."
                )

            state = self._load_environment_state(self.config.restart_from_json)

            self._restore_environment_state(state)

        else:
            self.member_configs = list(self.config.members)

            self._validate_population_size(len(self.member_configs))

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

        self.judge_prompt = self.config.judge_prompt or ""

        self.judge = AgentEloJudge(
            llm=self.llm,
            workspace=self.workspace,
            group=self.group,
            judge_prompt=self.judge_prompt,
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
        config: (AgentEloConfig | Mapping[str, Any] | str | Path | None),
        name: str | None,
        group: str | None,
        members: (list[EnvironmentMemberConfig | Mapping[str, Any]] | None),
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
            base = AgentEloConfig.from_mapping(config)

        elif isinstance(config, AgentEloConfig):
            base = config

        else:
            member_cfgs = [
                self._coerce_member(member) for member in (members or [])
            ]

            base = AgentEloConfig(
                name=name or "agent_elo",
                group=group or "default",
                members=member_cfgs,
                workspace=(str(workspace) if workspace is not None else None),
                initial_rating=(
                    initial_rating if initial_rating is not None else 1500.0
                ),
                k_factor=(k_factor if k_factor is not None else 32.0),
                deaths_per_round=(
                    deaths_per_round if deaths_per_round is not None else 1
                ),
                seed=seed,
                generations=(generations if generations is not None else 1),
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
                self._coerce_member(member) for member in members
            ]
        else:
            resolved_members = base.members

        return AgentEloConfig(
            name=(name if name is not None else base.name),
            group=(group if group is not None else base.group),
            description=base.description,
            members=resolved_members,
            workspace=(
                str(workspace) if workspace is not None else base.workspace
            ),
            defaults=base.defaults,
            initial_rating=(
                initial_rating
                if initial_rating is not None
                else base.initial_rating
            ),
            k_factor=(k_factor if k_factor is not None else base.k_factor),
            deaths_per_round=(
                deaths_per_round
                if deaths_per_round is not None
                else base.deaths_per_round
            ),
            seed=(seed if seed is not None else base.seed),
            generations=(
                generations if generations is not None else base.generations
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
                judge_prompt if judge_prompt is not None else base.judge_prompt
            ),
            inference_providers=(base.inference_providers),
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

    # ------------------------------------------------------------------
    # For lightweight environment level persistence
    # ------------------------------------------------------------------

    def _events(
        self,
        config: Mapping[str, Any] | None,
    ) -> EnvironmentEvents:
        return EnvironmentEvents(
            environment=self.name,
            config=config,
            environment_type="agent_elo",
            environment_id=self.name,
            path=[self.name],
        )

    def _source(
        self,
        name: str,
        *,
        kind: str = "agent",
    ) -> dict[str, Any]:
        return {
            "id": f"{self.name}.{name}",
            "name": name,
            "kind": kind,
            "path": [
                self.name,
                name,
            ],
        }

    def _member_runtime_config(
        self,
        base_config: Mapping[str, Any] | None,
        member: EnvironmentMemberConfig,
    ) -> dict[str, Any] | None:
        """Attach stable Elo-member identity to nested agent/tool events."""

        if base_config is None:
            return None

        member_id = f"{self.name}.{member.name}"

        merged = dict(base_config)

        base_metadata = merged.get("metadata")

        metadata = (
            dict(base_metadata)
            if isinstance(
                base_metadata,
                Mapping,
            )
            else {}
        )

        metadata.update({
            "environment_id": self.name,
            "environment_member": (member.name),
            "environment_member_id": (member_id),
            "environment_member_role": (member.role),
            "environment_member_path": [
                self.name,
                member.name,
            ],
            "agent": member.name,
            "agent_id": member_id,
        })

        merged["metadata"] = metadata

        base_tags = merged.get("tags")

        if isinstance(base_tags, str):
            tags = [base_tags]
        else:
            tags = list(base_tags) if base_tags else []

        for tag in (
            member.name,
            member_id,
            "environment_member",
            "elo_member",
        ):
            if tag not in tags:
                tags.append(tag)

        merged["tags"] = tags

        return merged

    def _topology_payload(
        self,
        *,
        generation: int | None = None,
    ) -> dict[str, Any]:
        active_names = {member.name for member in self.member_configs}

        nodes: list[dict[str, Any]] = []

        # Active population.
        for member in self.member_configs:
            player = self.players[member.name]

            nodes.append({
                "id": (f"{self.name}.{member.name}"),
                "name": member.name,
                "kind": "agent",
                "role": member.role,
                "agent_class": (member.agent),
                "path": [
                    self.name,
                    member.name,
                ],
                "rating": player.rating,
                "lineage_generation": (player.generation),
                "parent": player.parent,
                "active": True,
            })

        # A restart can contain a child whose parent is no
        # longer active. Include a lightweight historical
        # parent node so its lineage edge still has a source.
        historical_parents = {
            player.parent
            for player in self.players.values()
            if (player.parent and player.parent not in active_names)
        }

        for parent in sorted(historical_parents):
            nodes.append({
                "id": (f"{self.name}.{parent}"),
                "name": parent,
                "kind": "agent",
                "role": ("Historical Elo parent"),
                "path": [
                    self.name,
                    parent,
                ],
                "active": False,
            })

        edges = []

        for player in self.players.values():
            if player.parent:
                edges.append({
                    "source": (f"{self.name}.{player.parent}"),
                    "target": (f"{self.name}.{player.name}"),
                    "kind": "parent_of",
                })

        return {
            "kind": "agent_elo",
            "name": self.name,
            "description": (self.config.description),
            "generation": (
                self.generation_index if generation is None else generation
            ),
            "nodes": nodes,
            "edges": edges,
        }

    def _member_config(
        self,
        name: str,
    ) -> EnvironmentMemberConfig:
        """Return the active config for one Elo member."""

        for member in self.member_configs:
            if member.name == name:
                return member

        raise KeyError(f"Unknown Elo member: {name}")

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
                # Persist resolved settings without requiring the original
                # provider registry (or conflicting with its resolved URL).
                exclude={"inference_provider"},
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
                AgentEloEnvironment._rng_state_to_json(item) for item in value
            ]

        return value

    @staticmethod
    def _rng_state_from_json(
        value: Any,
    ) -> Any:
        if isinstance(value, list):
            return tuple(
                AgentEloEnvironment._rng_state_from_json(item) for item in value
            )

        return value

    def _environment_state_path(
        self,
    ) -> Path:
        return Path(self.workspace) / "environment_state.json"

    def _environment_state_payload(
        self,
    ) -> dict[str, Any]:
        """Return lightweight resumable environment state."""

        config_by_name = {member.name: member for member in self.member_configs}

        active_players = []

        # Use member_configs ordering so active population order
        # is preserved exactly across restart.
        for member in self.member_configs:
            player = self.players[member.name]

            active_players.append({
                "name": player.name,
                "rating": player.rating,
                "generation": player.generation,
                "parent": player.parent,
                "member_config": (
                    self._member_config_to_mapping(config_by_name[player.name])
                ),
            })

        return {
            "schema_version": 1,
            "environment_name": self.name,
            "group": self.group,
            "generation": self.generation_index,
            "seed": self.seed,
            "rng_state": self._rng_state_to_json(self._rng.getstate()),
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

        temporary = target.with_suffix(".json.tmp")

        payload = self._environment_state_payload()

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(target)

        return target

    @staticmethod
    def _load_environment_state(
        path: str | Path,
    ) -> dict[str, Any]:
        state_path = Path(path).expanduser()

        if not state_path.exists():
            raise FileNotFoundError(
                f"Elo environment restart file does not exist: {state_path}"
            )

        state = json.loads(state_path.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            raise ValueError(
                "Elo environment restart file must contain a JSON object."
            )

        if state.get("schema_version") != 1:
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

        saved_name = state.get("environment_name")

        saved_group = state.get("group")

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

        raw_players = state.get("active_players")

        if not isinstance(
            raw_players,
            list,
        ):
            raise ValueError("Restart state is missing 'active_players'.")

        self._validate_population_size(len(raw_players))

        member_configs: list[EnvironmentMemberConfig] = []

        players: dict[
            str,
            EloPlayer,
        ] = {}

        for raw_player in raw_players:
            if not isinstance(
                raw_player,
                Mapping,
            ):
                raise ValueError("Each active player entry must be a mapping.")

            raw_member_config = raw_player.get("member_config")

            if not isinstance(
                raw_member_config,
                Mapping,
            ):
                raise ValueError("Restart player is missing 'member_config'.")

            restored_member_config = dict(raw_member_config)
            saved_model = restored_member_config.get("model")
            if isinstance(saved_model, Mapping):
                # Older snapshots retained the provider name alongside the
                # resolved settings. They too must load without a registry.
                saved_model = dict(saved_model)
                saved_model.pop("inference_provider", None)
                restored_member_config["model"] = saved_model

            member_config = EnvironmentMemberConfig.from_mapping(
                restored_member_config,
                group=self.group,
            )
            AgentEloConfig.validate_member_config(member_config)

            name = str(raw_player["name"])

            if name in players:
                raise ValueError(
                    f"Elo member name {name!r} is duplicated in restart state. "
                    "Each member must have a unique name."
                )

            if member_config.name != name:
                raise ValueError(
                    "Restart player/member config "
                    f"name mismatch: {name!r} vs "
                    f"{member_config.name!r}"
                )

            # A restart should resume an existing persistent
            # URSA agent, never silently create a fresh one.
            den = self._member_den(name)

            if not den.exists():
                raise FileNotFoundError(
                    "Persistent URSA agent den "
                    "required for restart does not exist: "
                    f"{den}"
                )

            member_workspace = self._member_workspace(name)

            if not member_workspace.exists():
                raise FileNotFoundError(
                    "Agent workspace required for "
                    "restart does not exist: "
                    f"{member_workspace}"
                )

            member_configs.append(member_config)

            players[name] = EloPlayer(
                name=name,
                rating=float(raw_player["rating"]),
                generation=int(raw_player["generation"]),
                parent=raw_player.get("parent"),
            )

        self.member_configs = member_configs

        self.players = players

        # Constructing by the same names causes URSA to reopen
        # the agents' existing persistent state.
        self.members = {
            member.name: self.build_member(member)
            for member in self.member_configs
        }

        self.generation_index = int(
            state.get(
                "generation",
                0,
            )
        )

        # The restart snapshot is authoritative for the
        # evolutionary RNG metadata.
        self.seed = state.get("seed")

        rng_state = state.get("rng_state")

        if rng_state is not None:
            self._rng.setstate(self._rng_state_from_json(rng_state))
        else:
            # Defensive support for snapshots that contain a seed
            # but no serialized RNG state.
            self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Persistent URSA state
    # ------------------------------------------------------------------

    def _member_den(
        self,
        member_name: str,
    ) -> Path:
        """Return the persistent URSA den for a member."""
        agent_name = self._member_agent_name(member_name)

        if agent_name is None:
            raise RuntimeError(
                "Agent persistence is disabled. "
                "Persistent inheritance requires "
                "persist_members=True."
            )

        return group_agents_dir(self.group) / agent_name

    @staticmethod
    def _backup_sqlite_database(
        source: Path,
        destination: Path,
    ) -> None:
        """Create an independent SQLite snapshot."""
        source = Path(source)
        destination = Path(destination)

        if not source.exists():
            raise FileNotFoundError(f"SQLite source does not exist: {source}")

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise FileExistsError(
                f"SQLite destination already exists: {destination}"
            )

        # SQLite's connection context manager only manages transactions.
        # Close both handles so rollback can remove child databases on Windows.
        with closing(sqlite3.connect(source)) as source_conn:
            with closing(sqlite3.connect(destination)) as destination_conn:
                source_conn.backup(destination_conn)

    def _fork_parent_persistence(
        self,
        parent_name: str,
        child_name: str,
    ) -> None:
        """Fork parent URSA persistence into a new child den."""
        parent_den = self._member_den(parent_name)

        child_den = self._member_den(child_name)

        if not parent_den.exists():
            raise FileNotFoundError(f"Parent den does not exist: {parent_den}")

        if child_den.exists():
            raise FileExistsError(f"Child den already exists: {child_den}")

        child_den.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:
            self._backup_sqlite_database(
                parent_den / "db" / "checkpointer.db",
                child_den / "db" / "checkpointer.db",
            )

            graph_store_source = parent_den / "graph_store.sqlite"

            if graph_store_source.exists():
                self._backup_sqlite_database(
                    graph_store_source,
                    child_den / "graph_store.sqlite",
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
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

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
            raise ValueError("score_a must be one of 0.0, 0.5, or 1.0")

        expected_a = self.expected_score(
            rating_a,
            rating_b,
        )

        expected_b = 1.0 - expected_a
        score_b = 1.0 - score_a

        new_a = rating_a + self.k_factor * (score_a - expected_a)

        new_b = rating_b + self.k_factor * (score_b - expected_b)

        return new_a, new_b

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

        self._validate_population_size(len(names))

        shuffled = list(names)

        self._rng.shuffle(shuffled)

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

    def _progress_report_relative_path(self) -> Path:
        return (
            Path("_elo_progress") / f"generation_{self.generation_index + 1}.md"
        )

    def _read_progress_report(self, member_name: str) -> str | None:
        path = (
            self._member_workspace(member_name)
            / self._progress_report_relative_path()
        )
        try:
            report = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        return report or None

    def _member_prompt(
        self,
        member: EnvironmentMemberConfig,
        task: str,
        *,
        deadline: datetime | None = None,
    ) -> str:
        player = self.players[member.name]
        if self.generation_index == 0:
            approach = "Develop and validate a strong independent approach to the task."
        elif player.parent is not None:
            approach = (
                "Build on your parent's work while exploring a substantive "
                "improvement or alternative approach. Preserve useful results "
                "and validate your changes; novelty alone is not success."
            )
        else:
            approach = (
                "Build on your existing work. Favor refinement and stronger "
                "validation, making larger changes when justified."
            )
        execution_budget = (
            f"\nDeadline (UTC): {deadline.isoformat()}\n"
            "Your run will be terminated at this deadline. Save useful results "
            "as you work and aim to provide a final response before time runs out.\n"
            if deadline is not None
            else ""
        )
        extra = (
            f"\nAdditional guidance:\n{member.prompt}\n"
            if member.prompt
            else ""
        )
        report_path = self._progress_report_relative_path().as_posix()
        return (
            f"You are competitor '{member.name}'.\n"
            f"Role: {member.role}\n"
            f"Environment generation: {self.generation_index + 1}\n"
            f"Lineage generation: {player.generation}\n"
            f"Parent: {player.parent or 'None'}\n\n"
            f"Task:\n{task}\n\n"
            f"Approach:\n{approach}\n\n"
            "Follow the task's evaluation criteria. Distinguish verified results "
            "from plans or untested claims.\n"
            f"{execution_budget}\n"
            "If you can write files, maintain a concise progress report at:\n"
            f"{report_path}\n\n"
            "Update it after meaningful milestones with completed work, verified "
            "results, remaining limitations, and supporting file paths. "
            "Distinguish new progress from inherited work. Write updates to a "
            "temporary file, then rename it into place.\n\n"
            "Your final response should summarize results, evidence, limitations, "
            "and files produced. If you time out, the judge will use your progress "
            "report or briefly inspect your workspace. A timeout alone is not a loss.\n"
            f"{extra}"
        )

    def _format_member_result(
        self,
        member_name: str,
        result: Any,
    ) -> str:
        """Return the member agent's canonical user-facing result."""

        member = self.members[member_name]

        formatter = getattr(
            member,
            "format_result",
            None,
        )

        if callable(formatter):
            formatted = formatter(result)

            if formatted is not None:
                return str(formatted)

        return result_to_text(result)

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

        deadline_text = deadline.isoformat() if deadline is not None else None

        try:
            if deadline is None:
                result = await invoke()

            else:
                remaining_seconds = (
                    deadline - datetime.now(timezone.utc)
                ).total_seconds()

                if remaining_seconds <= 0:
                    return MemberRunResult(
                        name=member.name,
                        status="timed_out",
                        output=None,
                        deadline=deadline_text,
                        progress_report=self._read_progress_report(member.name),
                    )

                # Cancellation stops the environment from awaiting this
                # member. A blocking subprocess already running in an
                # executor may continue until that subprocess exits or
                # reaches its own timeout.
                result = await asyncio.wait_for(
                    invoke(),
                    timeout=remaining_seconds,
                )

            output = self._format_member_result(member.name, result)
            return MemberRunResult(
                name=member.name,
                status="completed",
                output=output,
                deadline=deadline_text,
                progress_report=(
                    self._read_progress_report(member.name)
                    if not output.strip()
                    else None
                ),
            )

        except TimeoutError:
            return MemberRunResult(
                name=member.name,
                status="timed_out",
                output=None,
                deadline=deadline_text,
                progress_report=self._read_progress_report(member.name),
            )

        except Exception as exc:
            return MemberRunResult(
                name=member.name,
                status="failed",
                output=None,
                deadline=deadline_text,
                error=(f"{type(exc).__name__}: {exc}"),
            )

    # ------------------------------------------------------------------
    # Judging
    # ------------------------------------------------------------------

    @staticmethod
    def _judge_submission(run: MemberRunResult) -> str:
        if run.output and run.output.strip():
            evidence = f"Final response:\n{run.output}"
        elif run.progress_report:
            evidence = f"Saved progress report:\n{run.progress_report}"
        else:
            evidence = (
                "No final response or progress report is available. "
                "Briefly inspect this candidate's workspace for evidence."
            )
        return f"Execution status: {run.status}\n\n{evidence}"

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

        # Timeouts remain eligible for judging using partial work.
        if (run_a.completed or run_a.timed_out) and (
            run_b.completed or run_b.timed_out
        ):
            return await self._judge_match(
                task=task,
                player_a=player_a,
                output_a=self._judge_submission(run_a),
                player_b=player_b,
                output_b=self._judge_submission(run_b),
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
        config_a = self._member_config(player_a)

        config_b = self._member_config(player_b)

        decision = await self.judge.judge_match(
            task=task,
            player_a=player_a,
            agent_type_a=config_a.agent,
            output_a=output_a,
            player_b=player_b,
            agent_type_b=config_b.agent,
            output_b=output_b,
        )

        if decision.winner == "A":
            score_a = 1.0

        elif decision.winner == "B":
            score_a = 0.0

        else:
            score_a = 0.5

        return MatchResult(
            player_a=player_a,
            player_b=player_b,
            score_a=score_a,
            reasoning=decision.reasoning,
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
            result.loser for result in match_results if result.loser is not None
        ]

        # Defensive deduplication.
        losers = list(dict.fromkeys(losers))

        self._rng.shuffle(losers)

        losers.sort(key=lambda name: (self.players[name].rating))

        return losers[: self.deaths_per_round]

    def _eliminate(
        self,
        losers: list[str],
    ) -> None:
        """Remove agents from the active population.

        Persistent dens and workspaces are intentionally left on disk
        so extinct lineages remain inspectable.
        """
        loser_set = set(losers)

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
                    try:
                        close()
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Failed to close eliminated Elo member %s",
                            name,
                            exc_info=True,
                        )

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
        *,
        excluded: list[str] | None = None,
    ) -> list[EloPlayer]:
        """Return the highest-rated surviving agents.

        Equal-rated survivors are ordered randomly using
        the environment RNG.
        """
        excluded_names = set(excluded or [])
        candidates = [
            player
            for player in self.players.values()
            if player.name not in excluded_names
        ]

        self._rng.shuffle(candidates)

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

        if any(member.name == child_name for member in self.member_configs):
            return False

        child_workspace = self._member_workspace(child_name)

        if child_workspace.exists():
            return False

        if self.persist_members:
            child_den = self._member_den(child_name)

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

        child_generation = parent.generation + 1

        while True:
            child_name = f"{parent.name}_g{child_generation}_{count}"

            if self._child_name_available(child_name):
                self._offspring_counts[parent.name] = count

                return child_name

            count += 1

    def _copy_parent_workspace(
        self,
        parent_name: str,
        child_name: str,
    ) -> None:
        """Fork the parent's working filesystem."""
        parent_workspace = self._member_workspace(parent_name)

        child_workspace = self._member_workspace(child_name)

        if child_workspace.exists():
            raise FileExistsError(
                f"Child workspace already exists: {child_workspace}"
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
        child_workspace = self._member_workspace(child_name)

        shutil.rmtree(
            child_workspace,
            ignore_errors=True,
        )

        if self.persist_members:
            try:
                child_den = self._member_den(child_name)
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
        prepared: list[tuple[EnvironmentMemberConfig, Any, EloPlayer]] = []
        offspring_counts_before = dict(self._offspring_counts)
        config_by_name = {member.name: member for member in self.member_configs}

        try:
            for parent in parents:
                parent_config = config_by_name[parent.name]
                child_name = self._next_child_name(parent)
                children.append(child_name)
                child_config = replace(parent_config, name=child_name)

                # Prepare every child before changing the active population.
                self._copy_parent_workspace(parent.name, child_name)
                if self.persist_members:
                    self._fork_parent_persistence(parent.name, child_name)
                child = self.build_member(child_config)
                prepared.append((
                    child_config,
                    child,
                    EloPlayer(
                        name=child_name,
                        rating=parent.rating,
                        generation=parent.generation + 1,
                        parent=parent.name,
                    ),
                ))
        except BaseException:
            # Close constructed children before deleting their persistence.
            # A cleanup error must not prevent rollback of the other children.
            for _, child, _ in prepared:
                try:
                    close = getattr(child, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
            for child_name in children:
                self._cleanup_failed_child(child_name)
            self._offspring_counts = offspring_counts_before
            raise

        for child_config, child, player in prepared:
            self.members[player.name] = child
            self.member_configs.append(child_config)
            self.players[player.name] = player

        return children

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def standings(
        self,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            self.players.values(),
            key=lambda player: (player.rating),
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
        *,
        runtime_config: Mapping[
            str,
            Any,
        ]
        | None = None,
    ) -> dict[str, Any]:
        """Run one complete evolutionary generation."""

        self._validate_population_size(len(self.member_configs))

        if not (
            len(self.member_configs) == len(self.members) == len(self.players)
        ):
            raise RuntimeError(
                "AgentEloEnvironment active population state is inconsistent: "
                f"member_configs={len(self.member_configs)}, "
                f"members={len(self.members)}, "
                f"players={len(self.players)}."
            )

        events = self._events(runtime_config)

        generation_start = perf_counter()

        generation_number = self.generation_index + 1

        initial_population_size = len(self.players)
        rng_state_before = self._rng.getstate()

        # Capture ratings before competition for reporting.
        ratings_before = {
            name: player.rating for name, player in self.players.items()
        }

        generation_deadline: datetime | None = None

        if self.member_timeout_seconds is not None:
            generation_deadline = datetime.now(timezone.utc) + timedelta(
                seconds=self.member_timeout_seconds
            )

        await events.aemit(
            (f"Elo generation {generation_number} started"),
            stage="generation",
            phase="start",
            event_type="generation_started",
            generation=generation_number,
            population_size=len(self.players),
            ratings={
                name: player.rating for name, player in self.players.items()
            },
            deadline=(
                generation_deadline.isoformat()
                if generation_deadline is not None
                else None
            ),
        )

        async def run_member(
            member: EnvironmentMemberConfig,
        ) -> MemberRunResult:
            source = self._source(member.name)

            start = perf_counter()

            await events.aemit(
                (f"Elo member {member.name} started"),
                stage="member",
                phase="start",
                event_type="member_started",
                generation=generation_number,
                source=source,
            )

            member_kwargs = dict(invoke_kwargs)

            member_runtime_config = self._member_runtime_config(
                runtime_config,
                member,
            )

            if member_runtime_config is not None:
                member_kwargs["config"] = member_runtime_config

            result = await self._run_member(
                member,
                task,
                member_kwargs,
                deadline=generation_deadline,
            )

            if result.completed:
                event_type = "member_completed"
                phase = "end"
                level = "info"

            elif result.timed_out:
                event_type = "member_timed_out"
                phase = "error"
                level = "warning"

            else:
                event_type = "member_failed"
                phase = "error"
                level = "error"

            await events.aemit(
                (f"Elo member {member.name} {result.status}"),
                stage="member",
                phase=phase,
                event_type=event_type,
                level=level,
                generation=generation_number,
                source=source,
                status=result.status,
                result=result.output,
                error=result.error,
                deadline=result.deadline,
                elapsed_seconds=(perf_counter() - start),
            )

            return result

        # ----------------------------------------------------------
        # Phase 1: independent research
        # ----------------------------------------------------------

        member_results = await asyncio.gather(*[
            run_member(member) for member in self.member_configs
        ])

        runs = {result.name: result for result in member_results}

        outputs = {
            name: (result.output if result.completed else None)
            for name, result in runs.items()
        }

        timed_out = [name for name, result in runs.items() if result.timed_out]

        failed = [name for name, result in runs.items() if result.failed]

        # ----------------------------------------------------------
        # Phase 2: randomized pairwise competition
        # ----------------------------------------------------------

        pairs = self._make_pairs([
            member.name for member in self.member_configs
        ])

        await events.aemit(
            (f"Elo generation {generation_number} pairings created"),
            stage="pairing",
            phase="declared",
            event_type="pairings_declared",
            generation=generation_number,
            pairs=[list(pair) for pair in pairs],
        )

        match_results: list[MatchResult] = []

        for player_a, player_b in pairs:
            rating_a_before = self.players[player_a].rating

            rating_b_before = self.players[player_b].rating

            await events.aemit(
                (f"Elo match {player_a} vs {player_b} started"),
                stage="match",
                phase="start",
                event_type="match_started",
                generation=generation_number,
                source=self._source(player_a),
                target=self._source(player_b),
            )

            result = await self._resolve_match(
                task=task,
                player_a=player_a,
                run_a=runs[player_a],
                player_b=player_b,
                run_b=runs[player_b],
            )

            self._apply_match_result(result)

            await events.aemit(
                (f"Elo match {player_a} vs {player_b} completed"),
                stage="match",
                phase="end",
                event_type="match_completed",
                generation=generation_number,
                source=self._source(player_a),
                target=self._source(player_b),
                winner=result.winner,
                loser=result.loser,
                score_a=result.score_a,
                reasoning=result.reasoning,
                rating_a_before=(rating_a_before),
                rating_a_after=(self.players[player_a].rating),
                rating_b_before=(rating_b_before),
                rating_b_after=(self.players[player_b].rating),
            )

            match_results.append(result)

        standings_after_matches = self.standings()

        # ----------------------------------------------------------
        # Phase 3: prepare replacement children, then eliminate losers
        # ----------------------------------------------------------

        eliminated = self._select_losers(match_results)
        parents = self._top_survivors(len(eliminated), excluded=eliminated)
        parent_names = [parent.name for parent in parents]

        try:
            children = self._reproduce(parents)
        except BaseException:
            # Keep failed generations retryable without applying Elo twice.
            for name, rating in ratings_before.items():
                self.players[name].rating = rating
            self._rng.setstate(rng_state_before)
            raise

        eliminated_ratings = {
            name: self.players[name].rating for name in eliminated
        }
        self._eliminate(eliminated)

        for name in eliminated:
            await events.aemit(
                (f"Elo member {name} eliminated"),
                stage="selection",
                phase="end",
                event_type="member_eliminated",
                generation=generation_number,
                source=self._source(name),
                rating=eliminated_ratings[name],
            )

        for child_name in children:
            child = self.players[child_name]

            await events.aemit(
                (f"Elo child {child_name} created"),
                stage="reproduction",
                phase="end",
                event_type="child_created",
                generation=generation_number,
                source=(self._source(child.parent) if child.parent else None),
                target=self._source(child_name),
                child=child_name,
                parent=child.parent,
                rating=child.rating,
                lineage_generation=(child.generation),
            )

        await events.aemit(
            (f"Elo generation {generation_number} topology updated"),
            stage="elo",
            phase="topology",
            event_type="topology_declared",
            generation=generation_number,
            topology=self._topology_payload(
                generation=generation_number,
            ),
        )

        final_population_size = len(self.players)

        if final_population_size != initial_population_size:
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

        state_path = self._save_environment_state()

        await events.aemit(
            (f"Elo generation {generation_number} completed"),
            stage="generation",
            phase="end",
            event_type="generation_completed",
            generation=generation_number,
            eliminated=eliminated,
            children=children,
            standings=self.standings(),
            elapsed_seconds=(perf_counter() - generation_start),
        )

        return {
            "generation": generation_number,
            "task": task,
            "outputs": outputs,
            "pairs": [list(pair) for pair in pairs],
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
            "standings_after_matches": (standings_after_matches),
            "eliminated": eliminated,
            "reproducing_parents": (parent_names),
            "children": children,
            "standings": self.standings(),
            "population_size": (final_population_size),
            "environment_state": str(state_path),
            "member_runs": {
                name: {
                    "status": run.status,
                    "deadline": run.deadline,
                    "error": run.error,
                    "progress_report": run.progress_report,
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
        task = str(inputs.get("task") or inputs.get("prompt") or inputs)

        runtime_config = runnable_config_from_kwargs(config)

        events = self._events(runtime_config)

        run_start = perf_counter()

        invoke_kwargs = invocation_kwargs(config)

        generation_results = []

        starting_generation = self.generation_index

        await events.aemit(
            f"Agent Elo {self.name} started",
            stage="elo",
            phase="start",
            event_type="elo_started",
            task=task,
            starting_generation=(self.generation_index),
            requested_generations=(self.generations),
            topology=(self._topology_payload()),
        )

        await events.aemit(
            (f"Agent Elo {self.name} topology declared"),
            stage="elo",
            phase="topology",
            event_type="topology_declared",
            topology=(self._topology_payload()),
        )

        try:
            for _ in range(self.generations):
                result = await self._run_generation(
                    task,
                    invoke_kwargs,
                    runtime_config=(runtime_config),
                )

                generation_results.append(result)

        except BaseException as exc:
            await events.aemit(
                f"Agent Elo {self.name} failed",
                stage="elo",
                phase="error",
                event_type="elo_failed",
                level="error",
                task=task,
                error=str(exc),
                elapsed_seconds=(perf_counter() - run_start),
            )

            raise

        final_result = {
            "task": task,
            "starting_generation": (starting_generation),
            "completed_generations": (len(generation_results)),
            "ending_generation": (self.generation_index),
            "generations": (generation_results),
            "standings": self.standings(),
            "population_size": len(self.players),
            "environment_state": str(self._environment_state_path()),
        }

        await events.aemit(
            f"Agent Elo {self.name} completed",
            stage="elo",
            phase="end",
            event_type="elo_completed",
            task=task,
            completed_generations=(len(generation_results)),
            ending_generation=(self.generation_index),
            standings=self.standings(),
            result={
                "ending_generation": (self.generation_index),
                "standings": (self.standings()),
            },
            elapsed_seconds=(perf_counter() - run_start),
        )

        return final_result
