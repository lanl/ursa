import shutil
from pathlib import Path

import yaml

AGENT_GROUPS_DIR = Path("~/.cache/ursa_agents").expanduser()


def add_group_subcommands(subparsers) -> None:
    from jsonargparse import ArgumentParser

    list_groups_parser = ArgumentParser()
    subparsers.add_subcommand(
        "list-groups",
        list_groups_parser,
        help="List available URSA agent groups",
        dest="subcommand",
    )

    create_group_parser = ArgumentParser()
    create_group_parser.add_argument("group_name", type=str, help="Name of the group")
    create_group_parser.add_argument(
        "config_file",
        type=Path,
        help="Path to a YAML config file containing allowed_base_urls",
    )
    subparsers.add_subcommand(
        "create-group",
        create_group_parser,
        help="Create a new URSA agent group from a YAML config file",
        dest="subcommand",
    )

    delete_group_parser = ArgumentParser()
    delete_group_parser.add_argument("group_name", type=str, help="Name of the group")
    subparsers.add_subcommand(
        "delete-group",
        delete_group_parser,
        help="Delete an existing URSA agent group",
        dest="subcommand",
    )

    show_group_parser = ArgumentParser()
    show_group_parser.add_argument("group_name", type=str, help="Name of the group")
    subparsers.add_subcommand(
        "show-group",
        show_group_parser,
        help="Show details about a URSA agent group",
        dest="subcommand",
    )

    update_group_parser = ArgumentParser()
    update_group_parser.add_argument("group_name", type=str, help="Name of the group")
    update_group_parser.add_argument(
        "config_file",
        type=Path,
        help="Path to a YAML config file containing allowed_base_urls",
    )
    subparsers.add_subcommand(
        "update-group",
        update_group_parser,
        help="Update an existing URSA agent group config",
        dest="subcommand",
    )


def validate_group_config(config_file: Path) -> dict:
    if not config_file.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_file}")
    if not config_file.is_file():
        raise ValueError(f"Config path is not a file: {config_file}")
    if config_file.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Group config file must be a YAML file")

    with open(config_file, "r", encoding="utf-8") as fid:
        data = yaml.safe_load(fid)

    if not isinstance(data, dict):
        raise ValueError("Group config must be a YAML mapping")

    allowed_base_urls = data.get("allowed_base_urls")
    if not isinstance(allowed_base_urls, list) or not allowed_base_urls:
        raise ValueError(
            "Group config must define a non-empty 'allowed_base_urls' list"
        )

    if not all(isinstance(url, str) and url.strip() for url in allowed_base_urls):
        raise ValueError("Each entry in 'allowed_base_urls' must be a non-empty string")

    return data


def list_groups() -> None:
    AGENT_GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    default_dir = AGENT_GROUPS_DIR / "default"
    default_dir.mkdir(exist_ok=True)

    for path in sorted(p for p in AGENT_GROUPS_DIR.iterdir() if p.is_dir()):
        print(path.name)


def create_group(group_name: str, config_file: Path) -> None:
    if not group_name.strip():
        raise ValueError("Group name must not be empty")
    if Path(group_name).name != group_name or group_name in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")

    validate_group_config(config_file)

    AGENT_GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    default_dir = AGENT_GROUPS_DIR / "default"
    default_dir.mkdir(exist_ok=True)

    group_dir = AGENT_GROUPS_DIR / group_name
    if group_dir.exists():
        raise FileExistsError(f"Group already exists: {group_name}")

    group_dir.mkdir()
    destination = group_dir / config_file.name
    shutil.copy2(config_file, destination)
    print(f"Created group '{group_name}' at {group_dir}")
    print(f"Copied config to {destination}")


def delete_group(group_name: str) -> None:
    if not group_name.strip():
        raise ValueError("Group name must not be empty")
    if Path(group_name).name != group_name or group_name in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")
    if group_name == "default":
        raise ValueError("The default group cannot be deleted")

    group_dir = AGENT_GROUPS_DIR / group_name
    if not group_dir.exists() or not group_dir.is_dir():
        raise FileNotFoundError(f"Group does not exist: {group_name}")

    shutil.rmtree(group_dir)
    print(f"Deleted group '{group_name}'")


def show_group(group_name: str) -> None:
    if not group_name.strip():
        raise ValueError("Group name must not be empty")
    if Path(group_name).name != group_name or group_name in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")

    group_dir = AGENT_GROUPS_DIR / group_name
    if not group_dir.exists() or not group_dir.is_dir():
        raise FileNotFoundError(f"Group does not exist: {group_name}")

    print(f"name: {group_name}")
    print(f"path: {group_dir}")
    entries = sorted(group_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    if entries:
        print("contents:")
        for entry in entries:
            kind = "dir" if entry.is_dir() else "file"
            print(f"  - [{kind}] {entry.name}")


def update_group(group_name: str, config_file: Path) -> None:
    if not group_name.strip():
        raise ValueError("Group name must not be empty")
    if Path(group_name).name != group_name or group_name in {".", ".."}:
        raise ValueError("Group name must be a simple directory name")

    validate_group_config(config_file)

    group_dir = AGENT_GROUPS_DIR / group_name
    if not group_dir.exists() or not group_dir.is_dir():
        raise FileNotFoundError(f"Group does not exist: {group_name}")

    destination = group_dir / config_file.name
    shutil.copy2(config_file, destination)
    print(f"Updated group '{group_name}' with config {destination}")
