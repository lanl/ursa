# src/ursa/util/config_loader.py

# helper functions related to YAML config loading.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> NS:
    """
    Load a YAML config file and return a SimpleNamespace where top-level keys
    are accessible as attributes. Nested objects remain dict/list.
    Mirrors the behavior you had inline in the runner script.
    """
    path = Path(config_path)
    try:
        raw_text = path.read_text(encoding="utf-8")
        raw_cfg = yaml.safe_load(raw_text) or {}
        if not isinstance(raw_cfg, dict):
            raise ValueError("Top-level YAML must be a mapping/object.")
        return NS(**raw_cfg)
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error loading YAML: {e}", file=sys.stderr)
        raise


def get_models_cfg(cfg: Any) -> dict:
    return getattr(cfg, "models", {}) or {}


def get_default_models(models_cfg: dict) -> tuple[str, ...]:
    """
    Equivalent to:
      DEFAULT_MODELS = tuple(models_cfg.get("choices") or (...fallback...))
    """
    fallback = (
        "openai:gpt-5",
        "openai:gpt-5-mini",
        "openai:o3",
        "openai:o3-mini",
    )
    choices = models_cfg.get("choices")
    if choices:
        return tuple(choices)
    return fallback


def get_default_model(models_cfg: dict) -> str | None:
    return models_cfg.get("default")


def get_config_planning_mode(cfg: Any) -> str | None:
    """
    Preserve your logic:
      planning_cfg = getattr(cfg, "planning", None)
      if dict: config_mode = planning_cfg.get("mode")
      else: config_mode = getattr(cfg, "planning_mode", None)
    """
    config_mode = None
    planning_cfg = getattr(cfg, "planning", None)
    if isinstance(planning_cfg, dict):
        config_mode = planning_cfg.get("mode")
    if not config_mode:
        config_mode = getattr(cfg, "planning_mode", None)
    return config_mode
