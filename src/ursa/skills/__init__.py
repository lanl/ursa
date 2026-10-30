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
    body: str
    name: str
    description: str
    _name_re: re.Pattern = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

    def __init__(
        self,
        skill_md_path: Path,
        max_description_length: int = 550,
        max_name_length: int = 64,
    ):
        if skill_md_path.name != "SKILL.md":
            raise ValueError(
                "Skill file must be named SKILL.md. Found: "
                + skill_md_path.name
            )

        if not skill_md_path.is_file():
            raise FileNotFoundError("SKILL.md not found.")
        self.skill_md_path = skill_md_path.expanduser().resolve()
        self.skill_dir = self.skill_md_path.parent
        self.max_description_length = max_description_length
        self.max_name_length = max_name_length
        self.parse_skill()

    def is_valid_skill_name(self, name: str):
        return (
            isinstance(name, str)
            and (1 <= len(name) <= self.max_name_length)
            and self._name_re.fullmatch(name)
        )

    def is_valid_description(self, description: str):
        return (
            isinstance(description, str)
            and len(description) <= self.max_description_length
            and "\n" not in description
            and "\r" not in description
            and description.isprintable()
        )

    def parse_skill(self):
        content = self.skill_md_path.read_text()
        lines = content.strip().splitlines()

        # SKILL.md should have at least these lines:
        #
        # 1 ---
        # 2 name: skill-name
        # 3 description: Some description here
        # 4 ---
        # 5 Do this.
        #
        # So, at least 5 lines.
        min_lines_in_skill_md = 5

        if len(lines) <= min_lines_in_skill_md:
            raise SkillError(f"{self.skill_md_path} is too short.")

        if lines[0] != "---":
            raise SkillError(
                f"Missing frontmatter '---' in: {self.skill_md_path}"
            )

        try:
            frontmatter_last_line_nbr = lines.index("---", 1)
        except ValueError:
            raise SkillError(
                f"Closing frontmatter '---' not found: {self.skill_md_path}"
            )

        try:
            meta = yaml.safe_load("\n".join(lines[:frontmatter_last_line_nbr]))
            self.name = meta["name"].strip()
            self.description = meta["description"].strip()
        except Exception:  # noqa: BLE001
            raise SkillError(
                "Missing name or description in frontmatter: "
                + str(self.skill_md_path.resolve())
            )

        if not self.is_valid_skill_name(self.name):
            raise SkillError(
                f"Invalid skill name in {self.skill_md_path}: {self.name}"
            )

        if not self.is_valid_description(self.description):
            raise SkillError(
                f"Invalid skill description in {self.skill_md_path}: "
                + self.description
            )

        self.body = "\n".join(lines[frontmatter_last_line_nbr:]).strip()
        if len(self.body) == 0:
            raise SkillError(f"Error parsing skill: {self.skill_md_path}")


def create_skill_catalog(directories: list[Path]) -> dict[str, SkillDict]:
    """Create skill catalog from a list of directories.

    When multiple directories contain a skill with the same name, register the
    one from the earliest directory in the provided list.
    """
    skills = {}
    for directory in reversed(directories):
        d = directory.expanduser().resolve()
        skills_dirs = [p.parent for p in d.rglob("SKILL.md")]
        for d in skills_dirs:
            try:
                skill = Skill(d / "SKILL.md")
                skills[skill.name] = SkillDict(
                    name=skill.name,
                    description=skill.description,
                    location=str(skill.skill_md_path),
                )
            except SkillError as e:
                _INVALID_SKILLS[str(d)] = str(e)
    return skills


# if __name__ == "__main__":
#     skills = list(
#         discover_skills(
#             [
#                 Path("~/.codex/skills"),
#                 Path("~/.claude/skills"),
#             ],
#         )
#     )
#
#     for i, s in enumerate(skills, start=1):
#         print(
#             f"{i:3d} | {s.name:15s} | {str(s.skill_dir)[:15]:15s} | {s.description[:20]}"
#         )
