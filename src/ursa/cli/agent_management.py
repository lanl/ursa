import re
import shutil
import sqlite3
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

from jsonargparse import ArgumentParser

from ursa.security import (
    DEFAULT_GROUP_NAME,
    URSA_CACHE_DIR,
    validate_group_name,
)

_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _group_root_dir(group_name: str) -> Path:
    return URSA_CACHE_DIR / validate_group_name(group_name)


def _group_agents_dir(group_name: str) -> Path:
    return _group_root_dir(group_name) / "agents"


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
    show_agent_parser.add_argument(
        "--name", required=True, type=str, help="Agent name"
    )
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

    delete_agent_parser = ArgumentParser()
    delete_agent_parser.add_argument(
        "--name", required=True, type=str, help="Agent name"
    )
    delete_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Group containing the agent",
    )
    subparsers.add_subcommand(
        "delete-agent",
        delete_agent_parser,
        help="Delete an agent from a group",
        dest="subcommand",
    )

    save_agent_parser = ArgumentParser()
    save_agent_parser.add_argument(
        "--name", required=True, type=str, help="Agent name"
    )
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
    copy_agent_parser.add_argument(
        "--name", required=True, type=str, help="New agent name"
    )
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

    share_agent_parser = ArgumentParser()
    share_agent_parser.add_argument(
        "--name", required=True, type=str, help="Agent name"
    )
    share_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Group containing the agent",
    )
    share_agent_parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Share only the experiences subdirectory instead of the full agent directory",
    )
    subparsers.add_subcommand(
        "share-agent",
        share_agent_parser,
        help="Create a shareable tar.gz archive of an agent in the current working directory",
        dest="subcommand",
    )

    import_agent_parser = ArgumentParser()
    import_agent_parser.add_argument(
        "archive_file",
        type=Path,
        help="Path to a shared agent tar.gz archive or SQLite checkpoint .db file",
    )
    import_agent_parser.add_argument(
        "--name",
        default=None,
        type=str,
        help="Name to assign the imported agent; defaults to the shared name",
    )
    import_agent_parser.add_argument(
        "--group",
        default="default",
        type=str,
        help="Destination group for the imported agent",
    )
    subparsers.add_subcommand(
        "import-agent",
        import_agent_parser,
        help="Import a shared agent archive or SQLite checkpoint database into a group",
        dest="subcommand",
    )


def ensure_group_dir(group_name: str) -> Path:
    group_name = validate_group_name(group_name)

    URSA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root_dir = _group_root_dir(group_name)
    if not root_dir.exists():
        if group_name == DEFAULT_GROUP_NAME:
            root_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Group does not exist: {group_name}")
    if not root_dir.is_dir():
        raise ValueError(f"Group path is not a directory: {root_dir}")

    agents_dir = _group_agents_dir(group_name)
    agents_dir.mkdir(parents=True, exist_ok=True)
    return agents_dir


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


def _share_archive_name(
    group_name: str, agent_name: str, no_checkpoint: bool
) -> str:
    suffix = "experiences" if no_checkpoint else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ursa_agent_{group_name}_{agent_name}_{suffix}_{timestamp}.tar.gz"


def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
    destination = path.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not member_path.is_relative_to(destination):
            raise ValueError("Archive contains unsafe paths")
    tar.extractall(destination)


def _normalize_checkpoint_threads(db_path: Path) -> None:
    """Rewrite imported checkpoint threads to URSA's default thread id."""
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = {"checkpoints", "writes"} - tables
        if missing:
            raise ValueError(
                "Checkpoint database is missing required table(s): "
                + ", ".join(sorted(missing))
            )

        for table in ("checkpoints", "writes"):
            columns = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if "thread_id" not in columns:
                raise ValueError(
                    f"Checkpoint database table '{table}' is missing thread_id column"
                )

        conn.execute("UPDATE checkpoints SET thread_id = ?", ("ursa",))
        conn.execute("UPDATE writes SET thread_id = ?", ("ursa",))
        conn.commit()
    except sqlite3.DatabaseError as e:
        raise ValueError(f"Invalid checkpoint database: {db_path}") from e
    finally:
        conn.close()


def _import_checkpoint_db(
    db_file: Path, dst_group: Path, group_name: str, agent_name: str | None
) -> None:
    if agent_name is None:
        raise ValueError(
            "Importing a checkpoint database requires --name to set the new agent name"
        )

    final_name = validate_agent_name(agent_name)
    dst = dst_group / final_name
    if dst.exists():
        raise FileExistsError(
            f"Destination agent already exists: {final_name} in group {group_name}"
        )

    checkpoint_dir = dst / "db"
    checkpoint_path = checkpoint_dir / "checkpointer.db"
    dst.mkdir(parents=False, exist_ok=False)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(db_file, checkpoint_path)
        _normalize_checkpoint_threads(checkpoint_path)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        raise

    print(
        f"Imported checkpoint database as agent '{final_name}' into group '{group_name}'"
    )
    print(f"Source database: {db_file}")
    print(f"Destination: {dst}")
    print(f"Checkpoint: {checkpoint_path}")


def list_agents(group_name: str = "default") -> None:
    group_dir = ensure_group_dir(group_name)
    for path in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        print(path.name)


def show_agent(agent_name: str, group_name: str = "default") -> None:
    path = agent_dir(group_name, agent_name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Agent does not exist: {agent_name} in group {group_name}"
        )

    print(f"name: {path.name}")
    print(f"group: {group_name}")
    print(f"path: {path}")

    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    if entries:
        print("contents:")
        for entry in entries:
            kind = "dir" if entry.is_dir() else "file"
            print(f"  - [{kind}] {entry.name}")


def delete_agent(agent_name: str, group_name: str = "default") -> None:
    path = agent_dir(group_name, agent_name)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Agent does not exist: {agent_name} in group {group_name}"
        )

    shutil.rmtree(path)
    print(f"Deleted agent '{agent_name}' from group '{group_name}'")
    print(f"Path: {path}")


def save_agent(agent_name: str, group_name: str = "default") -> None:
    src = agent_dir(group_name, agent_name)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"Agent does not exist: {agent_name} in group {group_name}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{validate_agent_name(agent_name)}.{timestamp}"
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


def share_agent(
    agent_name: str,
    group_name: str = "default",
    no_checkpoint: bool = False,
) -> None:
    src = agent_dir(group_name, agent_name)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(
            f"Agent does not exist: {agent_name} in group {group_name}"
        )

    archive_name = _share_archive_name(
        group_name, validate_agent_name(agent_name), no_checkpoint
    )
    archive_path = Path.cwd() / archive_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        export_root = tmp_root / validate_agent_name(agent_name)
        export_root.mkdir(parents=True, exist_ok=True)

        if no_checkpoint:
            experiences_src = src / "experiences"
            if experiences_src.exists() and experiences_src.is_dir():
                shutil.copytree(experiences_src, export_root / "experiences")
            manifest = export_root / "SHARED_AGENT_INFO.txt"
            manifest.write_text(
                "This archive contains only the agent experiences directory.\n"
                f"Original agent: {agent_name}\n"
                f"Original group: {group_name}\n",
                encoding="utf-8",
            )
        else:
            for item in src.iterdir():
                dst = export_root / item.name
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)

        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(export_root, arcname=export_root.name)

    print(f"Created shared agent archive: {archive_path}")


def import_agent(
    archive_file: Path,
    group_name: str = "default",
    agent_name: str | None = None,
) -> None:
    archive_file = archive_file.expanduser().resolve()
    if not archive_file.exists() or not archive_file.is_file():
        raise FileNotFoundError(f"Archive file not found: {archive_file}")

    dst_group = ensure_group_dir(group_name)

    if archive_file.suffix.lower() == ".db":
        _import_checkpoint_db(archive_file, dst_group, group_name, agent_name)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        with tarfile.open(archive_file, "r:gz") as tar:
            _safe_extract_tar(tar, tmp_root)

        extracted_dirs = [p for p in tmp_root.iterdir() if p.is_dir()]
        if len(extracted_dirs) != 1:
            raise ValueError(
                "Shared agent archive must contain exactly one top-level directory"
            )

        src_root = extracted_dirs[0]
        inferred_name = validate_agent_name(src_root.name)
        final_name = (
            validate_agent_name(agent_name) if agent_name else inferred_name
        )
        dst = dst_group / final_name
        if dst.exists():
            raise FileExistsError(
                f"Destination agent already exists: {final_name} in group {group_name}"
            )

        shutil.copytree(src_root, dst)

    print(f"Imported shared agent '{final_name}' into group '{group_name}'")
    print(f"Source archive: {archive_file}")
    print(f"Destination: {dst}")
