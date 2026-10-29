import re
from pathlib import Path

import yaml


class SkillError(Exception):
    """Raise when skill parsing is unsuccessful."""


_INVALID_SKILLS: dict[str, str] = {}


class Skill:
    body: str
    name: str
    description: str
    _name_re: re.Pattern = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

    def __init__(
        self,
        skill_dir: Path,
        max_description_length: int = 550,
        max_name_length: int = 64,
    ):
        if not self.is_skills_dir(skill_dir):
            raise FileNotFoundError("SKILL.md not found.")

        self.skill_dir = skill_dir.expanduser().resolve()
        self.skill_md_path = self.skill_dir / "SKILL.md"
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
        )

    @staticmethod
    def is_skills_dir(directory: Path) -> bool:
        d = directory.expanduser()
        return d.is_dir() and (d / "SKILL.md").is_file()

    def parse_skill(self):
        content = self.skill_md_path.read_text()
        lines = content.strip().splitlines()

        if len(lines) <= 5:
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
            self.name = meta["name"]
            self.description = meta["description"]
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


def discover_skills(directories: list[Path]):
    for directory in directories:
        d = directory.expanduser().resolve()
        skills_dirs = [p.parent for p in d.rglob("SKILL.md")]
        for d in skills_dirs:
            try:
                yield Skill(d)
            except SkillError as e:
                _INVALID_SKILLS[str(d)] = str(e)


if __name__ == "__main__":
    skills = list(
        discover_skills(
            [
                Path("~/.codex/skills"),
                Path("~/.claude/skills"),
            ],
        )
    )

    for i, s in enumerate(skills, start=1):
        print(
            f"{i:3d} | {s.name:15s} | {str(s.skill_dir)[:15]:15s} | {s.description[:20]}"
        )
