from pathlib import Path

import pytest


@pytest.fixture
def make_skill():
    def create(
        root: Path,
        name="lab-report",
        body="SKILL_BODY_SENTINEL",
        description="Write laboratory reports",
    ):
        directory = root / ".ursa" / "skills" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
            encoding="utf-8",
        )
        return directory

    return create
