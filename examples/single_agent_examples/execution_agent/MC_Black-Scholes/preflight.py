"""Preflight checks for the ExecutionAgent Monte Carlo example.

Design goals:
- No stdout/stderr output (silent by default).
- Validate minimal prerequisites for running the example.
- Persist detected mode/config to outputs/preflight.json so later steps
  (runner + README) can reference what happened.

This module is safe to import. The main entrypoint can be run directly:

  python examples/execution_agent_monte_carlo/preflight.py

It will exit non-zero on failure, but will not print.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    details: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_dir_writable(path: Path) -> tuple[bool, str | None]:
    """Check directory is writable by creating and deleting a tiny file."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok\n")
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def _check_matplotlib_headless() -> dict[str, Any]:
    """Return matplotlib availability and enforce Agg backend if importable.

    Requirement: if matplotlib is importable, enforce headless backend via
    matplotlib.use('Agg') BEFORE importing pyplot.
    """
    out: dict[str, Any] = {
        "importable": False,
        "pyplot_importable": False,
        "backend": None,
        "error": None,
    }

    try:
        import importlib.util

        spec = importlib.util.find_spec("matplotlib")
        if spec is None:
            return out

        out["importable"] = True

        import matplotlib  # type: ignore

        # Must be called before importing pyplot.
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt  # type: ignore

        out["pyplot_importable"] = True
        out["backend"] = str(matplotlib.get_backend())
        # Close any implicit figures.
        try:
            plt.close("all")
        except Exception:
            pass

        return out
    except Exception as e:  # pragma: no cover
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def run_preflight(outputs_dir: Path, strict: bool = True) -> PreflightResult:
    """Run checks and write outputs/preflight.json.

    Args:
      outputs_dir: directory where JSON should be written.
      strict: if True, raise RuntimeError when required checks fail.

    Returns:
      PreflightResult with ok flag and details.
    """

    required_env_vars = ["OPENAI_API_KEY"]
    env_status = {
        name: {"present": bool(os.environ.get(name))}
        for name in required_env_vars
    }

    outputs_writable, outputs_writable_error = _is_dir_writable(outputs_dir)

    mpl = _check_matplotlib_headless()
    plotting_enabled = bool(mpl.get("pyplot_importable"))

    details: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "requirements": {
            "env": env_status,
            "outputs_dir": str(outputs_dir),
            "outputs_writable": outputs_writable,
            "outputs_writable_error": outputs_writable_error,
        },
        "optional": {
            "matplotlib": mpl,
            "plotting_enabled": plotting_enabled,
        },
        "constraints": {
            "network_allowed": False,
            "notes": "Example is designed to run without network calls; do not download data.",
        },
    }

    # Required checks
    missing = [k for k, v in env_status.items() if not v["present"]]
    ok = (len(missing) == 0) and outputs_writable
    details["ok"] = ok
    details["missing_required_env_vars"] = missing

    # Persist JSON so the runner/README can reference what happened.
    outputs_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = outputs_dir / "preflight.json"
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, sort_keys=True)
        f.write("\n")

    if strict and not ok:
        raise RuntimeError(
            "Preflight failed: "
            + (f"missing env vars: {missing}; " if missing else "")
            + ("outputs dir not writable" if not outputs_writable else "")
        )

    return PreflightResult(ok=ok, details=details)


def main() -> int:
    # Default outputs dir: alongside this file.
    here = Path(__file__).resolve().parent
    outputs_dir = here / "outputs"

    try:
        run_preflight(outputs_dir=outputs_dir, strict=True)
        return 0
    except Exception as e:
        # Do not print; persist error to JSON.
        try:
            outputs_dir.mkdir(parents=True, exist_ok=True)
            err_path = outputs_dir / "preflight_error.json"
            payload = {
                "timestamp_utc": _utc_now_iso(),
                "error": f"{type(e).__name__}: {e}",
            }
            with open(err_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
