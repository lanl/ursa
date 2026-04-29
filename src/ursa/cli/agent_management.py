import re
import shutil
from datetime import datetime
from pathlib import Path

from jsonargparse import ArgumentParser

AGENT_GROUPS_DIR = Path("~/.cache/ursa_agents").expanduser()
_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def add_agent_management_subcommands(subparsers) -> None:
    list_agents_parser = ArgumentParser()
    list_agents_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Group to list agents from",
    )
    subparsers.add_subcommand(
        "list-agents",
        list_agents_parser,
        help="List agents in a group",
        dest="subcommand",
    )

    show_agent_parser = ArgumentParser()
    show_agent_parser.add_argument("--name", required=True, type=str, help="Agent name")
    show_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Group containing the agent",
    )
    subparsers.add_subcommand(
        "show-agent",
        show_agent_parser,
        help="Show details about an agent",
        dest="subcommand",
    )

    save_agent_parser = ArgumentParser()
    save_agent_parser.add_argument("--name", required=True, type=str, help="Agent name")
    save_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Group containing the agent",
    )
    subparsers.add_subcommand(
        "save-agent",
        save_agent_parser,
        help="Create a timestamped checkpoint copy of an agent",
        dest="subcommand",
    )

    copy_agent_parser = ArgumentParser()
    copy_agent_parser.add_argument("--name", required=True, type=str, help="New agent name")
    copy_agent_parser.add_argument(
        "--from",
        dest="source_agent",
        required=True,
        type=str,
        help="Source agent name to copy from",
    )
    copy_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Destination group for the copied agent",
    )
    copy_agent_parser.add_argument(
        "--from-group",
        default="default",
        type=str,
        help="Source group containing the source agent",
    )
    subparsers.add_subcommand(
        "copy-agent",
        copy_agent_parser,
        help="Copy an agent to a new name, optionally across groups",
        dest="subcommand",
    )


def ensure_group_dir(group_name: str) -> Path:
    if not group_name.strip():
        raise ValueError("Group name must not be empty")
    if Path(group_name).name != group_name or group_name in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")

    AGENT_GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    group_dir = AGENT_GROUPS_DIR / group_name
    if not group_dir.exists():
        raise FileNotFoundError(f"Group does not exist: {group_name}")
    if not group_dir.is_dir():
        raise ValueError(f"Group path is not a directory: {group_dir}")
    return group_dir


def validate_agent_name(name: str) -> str:
    if not name or not name.strip():
        raise ValueError("Agent name must not be empty")
    name = name.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Agent name must be a simple directory name")
    if not _AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            "Agent name may only contain letters, numbers, dot, underscore, and hyphen"
        )
    return name


def agent_dir(group_name: str, agent_name: str) -> Path:
    return ensure_group_dir(group_name) / validate_agent_name(agent_name)


def _copy_directory(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Source agent does not exist: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(src, dst)


def list_agents(group_name: str = "default") -> None:
    group_dir = ensure_group_dir(group_name)
    for path in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        print(path.name)


def show_agent(agent_name: str, group_name: str = "default") -> None:
    path = agent_dir(group_name, agent_name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Agent does not exist: {agent_name} in group {group_name}")

    print(f"name: {path.name}")
    print(f"group: {group_name}")
    print(f"path: {path}")

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    if entries:
        print("contents:")
        for entry in entries:
            kind = "dir" if entry.is_dir() else "file"
            print(f"  - [{kind}] {entry.name}")


def save_agent(agent_name: str, group_name: str = "default") -> None:
    src = agent_dir(group_name, agent_name)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"Agent does not exist: {agent_name} in group {group_name}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{validate_agent_name(agent_name)}_{timestamp}"
    dst = src.parent / checkpoint_name
    _copy_directory(src, dst)
    print(f"Saved agent checkpoint: {checkpoint_name}")
    print(f"Source: {src}")
    print(f"Checkpoint: {dst}")


def copy_agent(
    new_agent_name: str,
    source_agent_name: str,
    group_name: str = "default",
    from_group_name: str = "default",
) -> None:
    src = agent_dir(from_group_name, source_agent_name)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"Source agent does not exist: {source_agent_name} in group {from_group_name}"
        )

    dst_group = ensure_group_dir(group_name)
    dst = dst_group / validate_agent_name(new_agent_name)
    _copy_directory(src, dst)
    print(f"Copied agent '{source_agent_name}' to '{new_agent_name}'")
    print(f"Source: {src}")
    print(f"Destination: {dst}")
