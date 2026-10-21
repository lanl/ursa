from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

logger = logging.getLogger(__name__)
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.path.parent

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[Skill, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def get(self, name: str) -> Skill:
        for skill in self.skills:
            if skill.name == name:
                return skill
        raise ValueError(
            f"Unknown skill '${name}'. Use /skills or agent.list_skills() to list available skills."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "skills": [skill.as_dict() for skill in self.skills],
            "diagnostics": list(self.diagnostics),
        }

    def markdown(self) -> str:
        lines = ["Available skills", ""]
        for skill in self.skills:
            lines.extend([
                f"- **${skill.name}** — {skill.description}",
                f"  Source: {skill.source}; `{skill.path}`",
            ])
        if not self.skills:
            lines.append("No skills found in the selected skill directories.")
        if self.diagnostics:
            lines.extend(["", "Discovery diagnostics", ""])
            lines.extend(f"- {message}" for message in self.diagnostics)
        return "\n".join(lines)

    def prompt(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "Available skills for this request. When a skill's description matches "
            "the task or conversation context, call load_skill before doing that "
            "work. Follow loaded skill instructions for this request. Skills do "
            "not grant tools or bypass existing permissions. Earlier skill-load "
            "receipts are history, not current activations; load relevant skills "
            "again for this request. Resolve supporting files relative to the "
            "loaded skill directory, using read_skill_resource as needed.",
            "<available_skills>",
        ]
        for skill in self.skills:
            lines.append(
                f"<skill><name>{escape(skill.name)}</name>"
                f"<description>{escape(skill.description)}</description></skill>"
            )
        return "\n".join([*lines, "</available_skills>"])


def parse_skill(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    end = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
    )
    if end is None:
        raise ValueError("Unterminated YAML frontmatter")
    metadata = yaml.safe_load("".join(lines[1:end]))
    if not isinstance(metadata, dict):
        raise ValueError("Frontmatter must be a mapping")
    name = metadata.get("name")
    if not isinstance(name, str) or len(name) > 64 or not NAME.fullmatch(name):
        raise ValueError(
            "name must be 1–64 lowercase letters, digits, or single hyphens"
        )
    if name != path.parent.name:
        raise ValueError("name must match the skill directory name")
    description = metadata.get("description")
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > 1024
    ):
        raise ValueError(
            "description must be a nonempty string of at most 1024 characters"
        )
    for key in ("license", "compatibility", "allowed-tools"):
        if key in metadata and not isinstance(metadata[key], str):
            raise ValueError(f"{key} must be a string")
    if (
        "compatibility" in metadata
        and not 1 <= len(metadata["compatibility"]) <= 500
    ):
        raise ValueError("compatibility must contain 1–500 characters")
    extra = metadata.get("metadata", {})
    if not isinstance(extra, dict) or any(
        not isinstance(k, str) or not isinstance(v, str)
        for k, v in extra.items()
    ):
        raise ValueError("metadata must map strings to strings")
    return metadata, "".join(lines[end + 1 :]).strip()


@dataclass(frozen=True)
class SkillScope:
    workspace: Path | None = None
    cwd: Path = field(default_factory=lambda: Path.cwd())
    home: Path = field(default_factory=lambda: Path.home())

    def __post_init__(self) -> None:
        for key in ("workspace", "cwd", "home"):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(
                    self, key, Path(value).expanduser().resolve()
                )

    def discover(self) -> SkillCatalog:
        roots = (
            [("workspace", self.workspace / ".ursa" / "skills")]
            if self.workspace is not None
            else [
                ("home", self.home / ".ursa" / "skills"),
                ("project", self.cwd / ".ursa" / "skills"),
            ]
        )
        found: dict[str, Skill] = {}
        diagnostics: list[str] = []
        seen: set[Path] = set()
        for source, root in roots:
            try:
                children = sorted(root.iterdir())
            except FileNotFoundError:
                continue
            except OSError as exc:
                diagnostics.append(f"Cannot read {root}: {exc}")
                continue
            for directory in children:
                path = directory / "SKILL.md"
                try:
                    if not directory.is_dir() or not path.is_file():
                        continue
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    metadata, _ = parse_skill(path)
                    seen.add(resolved)
                    name = metadata["name"]
                    skill = Skill(
                        name,
                        metadata["description"],
                        path,
                        source,
                        metadata,
                    )
                    if name in found:
                        diagnostics.append(
                            f"Skill '{name}' at {found[name].path} is shadowed by {path}"
                        )
                    found[name] = skill
                except (
                    OSError,
                    UnicodeError,
                    ValueError,
                    yaml.YAMLError,
                ) as exc:
                    diagnostics.append(f"Skipping {path}: {exc}")
        for diagnostic in diagnostics:
            logger.warning("%s", diagnostic)
        return SkillCatalog(
            tuple(found[name] for name in sorted(found)), tuple(diagnostics)
        )


def discover_skills(
    *,
    workspace: str | Path | None = None,
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> SkillCatalog:
    return SkillScope(
        workspace=Path(workspace) if workspace is not None else None,
        cwd=Path(cwd) if cwd is not None else Path.cwd(),
        home=Path(home) if home is not None else Path.home(),
    ).discover()
