import pytest

from ursa.tools.feasibility_checker import heuristic_feasibility_check
from ursa.tools.feasibility_tools import feasibility_check_auto


def test_heuristic_feasibility_reports_unknown_variable_type():
    result = heuristic_feasibility_check.func(
        constraints=["x <= 1"],
        variable_name=["x"],
        variable_type=["unsupported"],
        variable_bounds=[[0, 1]],
        samples=1,
    )

    assert result.startswith("Error: unknown variable type 'unsupported'")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"variable_type": []}, "must have the same length"),
        ({"variable_bounds": [[2, 1]]}, "lower bound 2 exceeds upper bound 1"),
        ({"samples": 0}, "samples must be at least 1"),
        ({"variable_name": [""]}, "Error parsing constraints"),
    ],
)
def test_heuristic_feasibility_reports_invalid_correlated_input(
    overrides, expected
):
    inputs = {
        "constraints": ["x <= 1"],
        "variable_name": ["x"],
        "variable_type": ["real"],
        "variable_bounds": [[0, 1]],
    }
    inputs.update(overrides)
    result = heuristic_feasibility_check.func(**inputs)

    assert expected in result


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"variable_bounds": []}, "must have the same length"),
        ({"variable_bounds": [[2, 1]]}, "lower bound 2 exceeds upper bound 1"),
        ({"constraints": ["y <= 1"]}, "undeclared variables: y"),
    ],
)
def test_feasibility_auto_reports_invalid_input_before_calling_a_solver(
    overrides, expected
):
    inputs = {
        "constraints": ["x <= 1"],
        "variable_name": ["x"],
        "variable_type": ["real"],
        "variable_bounds": [[0, 1]],
        "heuristic_enabled": False,
    }
    inputs.update(overrides)
    result = feasibility_check_auto.func(**inputs)

    assert expected in result


@pytest.mark.parametrize(
    ("constraints", "variable_bounds"),
    [(["x >= x"], [[None, None]]), (["True"], [[0, 1]])],
)
def test_feasibility_auto_accepts_unbounded_bounds_and_literal_booleans(
    constraints, variable_bounds
):
    result = feasibility_check_auto.func(
        constraints=constraints,
        variable_name=["x"],
        variable_type=["real"],
        variable_bounds=variable_bounds,
        heuristic_samples=1,
    )

    assert not result.startswith(("Input error:", "Parse error:"))
