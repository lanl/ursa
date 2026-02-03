from __future__ import annotations

from pathlib import Path

import randomname
from rich.panel import Panel


def setup_workspace(
    *,
    user_specified_workspace: str | None,
    project: str = "run",
    model_name: str = "openai:gpt-5-mini",
    console,
) -> str:
    """
    Create (or reuse) a workspace directory for a run.

    Behavior:
      - If user_specified_workspace is None, make a new folder named:
            "<project>_<randomname>"
      - Otherwise use the provided workspace path.
      - Always mkdir(parents=True, exist_ok=True)
      - Prints a Rich panel with the chosen workspace + model.

    Returns:
      workspace (str)
    """
    if user_specified_workspace is None:
        print("No workspace specified, creating one for this project!")
        print(
            "Make sure to pass this string to restart using --workspace <this workspace string>"
        )
        workspace = f"{project}_{randomname.get_name(adj=('colors', 'emotions', 'character', 'speed', 'size', 'weather', 'appearance', 'sound', 'age', 'taste'), noun=('cats', 'dogs', 'apex_predators', 'birds', 'fish', 'fruit'))}"
    else:
        workspace = user_specified_workspace
        print(f"User specified workspace: {workspace}")

    Path(workspace).mkdir(parents=True, exist_ok=True)

    # Choose a fun emoji based on the model family
    if model_name.startswith("openai"):
        model_emoji = "🤖"
    elif "llama" in model_name.lower():
        model_emoji = "🦙"
    else:
        model_emoji = "🧠"

    console.print(
        Panel.fit(
            f":rocket:  [bold bright_blue]{workspace}[/bold bright_blue]  :rocket:\n"
            f"{model_emoji}  [bold cyan]{model_name}[/bold cyan]",
            title="[bold green]ACTIVE WORKSPACE[/bold green]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )

    return workspace


def ensure_symlink(
    *, workspace: str | Path, symlink_cfg: dict | None
) -> dict | None:
    """
    Ensure a symlink described by symlink_cfg exists inside workspace.

    symlink_cfg shape:
      {"source": "...", "dest": "..."}  # dest is relative to workspace
    Adds/returns:
      {"source": ..., "dest": ..., "is_linked": True}

    If symlink_cfg is None/empty, returns None.
    """
    if not symlink_cfg or not isinstance(symlink_cfg, dict):
        return None

    # If caller already marked it linked, do nothing.
    if symlink_cfg.get("is_linked"):
        return symlink_cfg

    src = Path(symlink_cfg["source"]).expanduser().resolve()
    ws_root = Path(workspace).expanduser().resolve()
    dst = ws_root / symlink_cfg["dest"]

    # Ensure parent dir exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If something exists there, replace it
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    dst.symlink_to(src, target_is_directory=src.is_dir())
    print(f"Symlinked {src} (source) --> {dst} (dest)")

    out = dict(symlink_cfg)
    out["is_linked"] = True
    return out
