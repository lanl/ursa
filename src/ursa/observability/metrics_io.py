# metrics_io.py
from __future__ import annotations

import json
from typing import Any, Dict


def load_metrics(path: str) -> Dict[str, Any]:
    """Load a telemetry metrics JSON into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
