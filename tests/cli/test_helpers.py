from dataclasses import dataclass

import pytest

from ursa.cli.helpers import (
    _fuzzy_match,
    _plan_step_text,
    _token_usage,
    _token_usage_breakdown,
    _truncate_middle,
)


def test_fuzzy_match_and_token_usage_support_common_shapes():
    assert _fuzzy_match("sre", "src/example.py")
    assert not _fuzzy_match("xyz", "src/example.py")
    assert _token_usage({"usage": {"total_tokens": 42}}) == 42

    usage = _token_usage_breakdown({
        "usage_metadata": {
            "input_tokens": 31,
            "output_tokens": 11,
            "total_tokens": 42,
            "input_token_details": {"cache_read": 17},
        }
    })
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_tokens,
        usage.total_tokens,
    ) == (31, 11, 17, 42)


@dataclass
class PlanStep:
    name: str
    description: str

    def model_dump(self):
        return {"name": self.name, "description": self.description}


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (
            {"name": "Inspect", "description": "  read\nfiles  "},
            "2. Inspect: read files",
        ),
        ({"description": "Validate"}, "2. Step 2: Validate"),
        (PlanStep("Ship", "run tests"), "2. Ship: run tests"),
        ("Fallback", "2. Fallback"),
    ],
)
def test_plan_step_text_normalizes_supported_step_shapes(step, expected):
    assert _plan_step_text(2, step) == expected


def test_truncate_middle_preserves_short_text_and_both_ends():
    assert _truncate_middle("short", 20) == "short"

    result = _truncate_middle("alpha beta gamma delta epsilon", 24)

    assert result.startswith("alpha")
    assert result.endswith("ilon")
    assert "truncated" in result


@pytest.mark.parametrize(
    ("width", "expected"),
    [(0, ""), (1, "…"), (5, "alph…")],
)
def test_truncate_middle_falls_back_to_end_truncation(width, expected):
    assert _truncate_middle("alpha beta gamma", width) == expected


def test_truncate_middle_uses_marker_when_it_just_fits():
    assert "truncated" in _truncate_middle("alpha beta gamma", 15)
