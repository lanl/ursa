# References: https://agentskills.io/client-implementation/adding-skills-support

import re
from pathlib import Path
from typing import TypedDict

import yaml

_INVALID_SKILLS: dict[str, str] = {}


class SkillError(Exception):
    """Raise when skill parsing is unsuccessful."""


class SkillDict(TypedDict):
    name: str
    description: str
    location: str


class Skill:
    name: str
    description: str
    _name_re: re.Pattern = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

    def __init__(
        self,
        path: Path,
        name_max_length: int = 64,
        description_max_length: int = 1024,
    ):
        self.path = path.expanduser().resolve()
        self.name_max_length = name_max_length
        self.description_max_length = description_max_length
        if self.path.name != "SKILL.md":
            raise SkillError(
                f"Skill file must be named SKILL.md. Found: {self.path}"
            )
        self.parse_skill()

    def is_valid_skill_name(self, name: str):
        return (
            isinstance(name, str)
            and (1 <= len(name) <= self.name_max_length)
            and self._name_re.fullmatch(name)
        )

    def is_valid_description(self, description: str):
        return (
            isinstance(description, str)
            and len(description) <= self.description_max_length
            and "\n" not in description
            and "\r" not in description
            and description.isprintable()
        )

    def parse_skill(self):
        content = self.path.read_text()
        lines = content.strip().splitlines()

        # Check for frontmatter start
        if not lines or lines[0] != "---":
            raise SkillError(
                f"SKILL.md lacks YAML frontmatter '---': {self.path}"
            )

        # Check for frontmatter end
        try:
            frontmatter_last_line_nbr = lines.index("---", 1)
        except ValueError:
            raise SkillError(
                f"Closing frontmatter '---' not found: {self.path}"
            )

        # Validate frontmatter content
        try:
            meta = yaml.safe_load("\n".join(lines[:frontmatter_last_line_nbr]))
            self.name = meta["name"].strip()
            self.description = meta["description"].strip()
        except Exception:  # noqa: BLE001
            raise SkillError(
                f"Missing name or description in frontmatter: {self.path}"
            )

        if not self.is_valid_skill_name(self.name):
            raise SkillError(f"Invalid skill name in {self.path}: {self.name}")

        if not self.is_valid_description(self.description):
            raise SkillError(
                f"Invalid skill description in {self.path}: " + self.description
            )


def create_skill_catalog(directories: list[Path]) -> dict[str, SkillDict]:
    """Create skill catalog from a list of directories.

    When multiple directories contain a skill with the same name, register the
    one from the earliest directory in the provided list.
    """
    skills = {}
    for directory in reversed(directories):
        d = directory.expanduser().resolve()
        skill_md_paths = [p for p in d.rglob("SKILL.md") if p.is_file()]
        for p in skill_md_paths:
            try:
                skill = Skill(p)
                skills[skill.name] = SkillDict(
                    name=skill.name,
                    description=skill.description,
                    location=str(skill.path),
                )
            except SkillError as e:
                # NOTE: Where to log this information?
                _INVALID_SKILLS[str(p)] = str(e)
    return skills
