"""Discover Agent Skills metadata; load instruction bodies only on activation."""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property, lru_cache
from importlib.resources import files
from pathlib import Path
from threading import Lock
from types import MappingProxyType

import yaml

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
_CLIENT_DIRS = (".claude", ".codex", ".agents", ".ursa")
_FRONTMATTER = re.compile(
    r"\A(?:\r?\n)?---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
_VALID_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class SkillScope(StrEnum):
    BUNDLED = "bundled"
    USER = "user"
    PROJECT = "project"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    scope: SkillScope

    @property
    def directory(self) -> Path:
        return self.path.parent

    @cached_property
    def instructions(self) -> str:
        _, body = split_frontmatter(
            self.path.read_text(encoding="utf-8-sig", errors="replace")
        )
        return body.strip()


def is_valid_skill_name(name: str) -> bool:
    return 1 <= len(name) <= 64 and bool(_VALID_NAME.fullmatch(name))


def user_skills_root() -> Path:
    return Path.home() / ".ursa" / "skills"


def project_skills_root(project_root: Path | None = None) -> Path:
    return Path(project_root or Path.cwd()) / ".ursa" / "skills"


def skill_roots(
    project_root: Path | None = None,
) -> tuple[tuple[Path, SkillScope], ...]:
    home = Path.home()
    project = Path(project_root or Path.cwd()).resolve()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    if not xdg.is_absolute():
        xdg = home / ".config"
    return (
        (Path(str(files("ursa.skills"))), SkillScope.BUNDLED),
        *(
            (home / client / "skills", SkillScope.USER)
            for client in _CLIENT_DIRS
        ),
        (xdg / "ursa" / "skills", SkillScope.USER),
        *(
            (project / client / "skills", SkillScope.PROJECT)
            for client in _CLIENT_DIRS
        ),
    )


def split_frontmatter(text: str) -> tuple[dict, str]:
    text = text.removeprefix("\ufeff")
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}, text
    loaded = yaml.safe_load(match.group(1))
    return loaded if isinstance(loaded, dict) else {}, text[match.end() :]


def validate_metadata(metadata: dict, directory_name: str) -> None:
    name = metadata.get("name")
    if not isinstance(name, str) or not is_valid_skill_name(name):
        raise ValueError(
            "name must be 1–64 lowercase letters/digits separated by single hyphens"
        )
    if name != directory_name:
        raise ValueError("name must match the skill directory")
    description = metadata.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise ValueError(
            "description must be a non-empty string of at most 1024 characters"
        )
    compatibility = metadata.get("compatibility")
    if "compatibility" in metadata and (
        not isinstance(compatibility, str) or not 1 <= len(compatibility) <= 500
    ):
        raise ValueError("compatibility must be a string of 1–500 characters")
    for field in ("license", "allowed-tools"):
        if field in metadata and not isinstance(metadata[field], str):
            raise ValueError(f"{field} must be a string")
    if "metadata" in metadata:
        extra = metadata["metadata"]
        if not isinstance(extra, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in extra.items()
        ):
            raise ValueError("metadata must map strings to strings")


def load_skill(directory: Path, scope: SkillScope) -> Skill | None:
    path = directory / SKILL_FILENAME
    try:
        # Stop at the closing delimiter, without reading instruction bodies.
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            opening = next(stream, "")
            if opening in {"\n", "\r\n"}:
                opening = next(stream, "")
            if opening.rstrip(" \t\r\n") != "---":
                raise ValueError("missing YAML frontmatter")
            lines = [opening]
            for line in stream:
                lines.append(line)
                if line.rstrip(" \t\r\n") == "---":
                    break
            else:
                raise ValueError("unclosed YAML frontmatter")
        metadata, _ = split_frontmatter("".join(lines))
        validate_metadata(metadata, directory.name)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.warning("Ignoring invalid skill %s: %s", path, exc)
        return None
    return Skill(
        metadata["name"], metadata["description"], path.absolute(), scope
    )


def discover_skills_in(root: Path, scope: SkillScope) -> dict[str, Skill]:
    found = {}
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if (entry / SKILL_FILENAME).is_file():
            if (skill := load_skill(entry, scope)) is not None:
                found[skill.name] = skill
    return found


class SkillCatalog:
    def __init__(self, roots: tuple[tuple[Path, SkillScope], ...]):
        self.roots = roots
        self._skills = None
        self._lock = Lock()

    def load(self) -> MappingProxyType[str, Skill]:
        with self._lock:
            if self._skills is None:
                skills = {}
                for root, scope in self.roots:
                    for name, skill in discover_skills_in(root, scope).items():
                        if name in skills and skill.path != skills[name].path:
                            logger.warning(
                                "Skill %s at %s shadows %s",
                                name,
                                skill.path,
                                skills[name].path,
                            )
                        skills[name] = skill
                self._skills = MappingProxyType(dict(sorted(skills.items())))
            return self._skills

    async def aload(self) -> MappingProxyType[str, Skill]:
        return await asyncio.to_thread(self.load)


@lru_cache(maxsize=32)
def _catalog_for_roots(
    roots: tuple[tuple[Path, SkillScope], ...],
) -> SkillCatalog:
    return SkillCatalog(roots)


def get_skill_catalog(project_root: Path | None = None) -> SkillCatalog:
    return _catalog_for_roots(skill_roots(project_root))


def discover_skills(project_root: Path | None = None) -> dict[str, Skill]:
    return dict(SkillCatalog(skill_roots(project_root)).load())


async def adiscover_skills(
    project_root: Path | None = None,
) -> dict[str, Skill]:
    return await asyncio.to_thread(discover_skills, project_root)
