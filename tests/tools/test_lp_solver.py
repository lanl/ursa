"""Tests for the LP solver backend and its integration with the
CMM supply-chain optimization tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ursa.tools.cmm_supply_chain_optimization_tool import (
    OptimizationOutput,
    solve_cmm_supply_chain_optimization,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_ND_SCENARIOS_PATH = _CONFIGS_DIR / "nd_china_2025_scenarios.json"


def _base_input() -> dict:
    return {
        "commodity": "CO",
        "demand": {"NA": 100, "EU": 80},
        "suppliers": [
            {
                "name": "US_mine",
                "capacity": 120,
                "unit_cost": 8.0,
                "risk_score": 0.2,
            },
            {
                "name": "Allied_import",
                "capacity": 90,
                "unit_cost": 9.2,
                "risk_score": 0.1,
            },
        ],
        "shipping_cost": {
            "US_mine": {"NA": 1.0, "EU": 2.0},
            "Allied_import": {"NA": 1.4, "EU": 1.2},
        },
        "risk_weight": 2.0,
        "max_supplier_share": 0.8,
    }


def test_lp_optimal_status():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "optimal_lp"
    assert result["feasible"] is True


def test_lp_objective_leq_greedy():
    payload = _base_input()

    payload_greedy = {**payload, "solver_backend": "greedy"}
    result_greedy = solve_cmm_supply_chain_optimization(payload_greedy)

    payload_lp = {**payload, "solver_backend": "lp"}
    result_lp = solve_cmm_supply_chain_optimization(payload_lp)

    assert (
        result_lp["objective_value"] <= result_greedy["objective_value"] + 1e-6
    )


def test_lp_shadow_prices_present():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result = solve_cmm_supply_chain_optimization(payload)

    sp = result.get("shadow_prices")
    assert sp is not None
    assert "demand_balance" in sp
    assert "supplier_capacity" in sp
    assert "supplier_share_cap" in sp
    assert "composition" in sp


def test_lp_determinism():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result_a = solve_cmm_supply_chain_optimization(payload)
    result_b = solve_cmm_supply_chain_optimization(payload)

    assert result_a == result_b


def test_lp_demand_balance():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result = solve_cmm_supply_chain_optimization(payload)

    demand = payload["demand"]
    for market, required in demand.items():
        allocated = sum(
            a["amount"] for a in result["allocations"] if a["market"] == market
        )
        unmet = result["unmet_demand"].get(market, 0.0)
        assert abs(allocated + unmet - required) < 1e-3


def test_lp_capacity_respected():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result = solve_cmm_supply_chain_optimization(payload)

    for sup in payload["suppliers"]:
        allocated = sum(
            a["amount"]
            for a in result["allocations"]
            if a["supplier"] == sup["name"]
        )
        assert allocated <= sup["capacity"] + 1e-3


def test_lp_with_composition_constraints():
    payload = {
        "commodity": "ALLOY",
        "demand": {"US": 100},
        "suppliers": [
            {
                "name": "la_rich",
                "capacity": 100,
                "unit_cost": 1.0,
                "risk_score": 0.1,
                "composition_profile": {
                    "LA": 0.10,
                    "Y": 0.00,
                },
            },
            {
                "name": "y_rich",
                "capacity": 100,
                "unit_cost": 5.0,
                "risk_score": 0.1,
                "composition_profile": {
                    "LA": 0.00,
                    "Y": 0.10,
                },
            },
        ],
        "shipping_cost": {
            "la_rich": {"US": 0.0},
            "y_rich": {"US": 0.0},
        },
        "risk_weight": 0.0,
        "max_supplier_share": 1.0,
        "composition_targets": {"LA": 0.05, "Y": 0.05},
        "composition_tolerance": 0.001,
        "solver_backend": "lp",
    }

    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "optimal_lp"
    assert result["feasible"] is True
    comp = result["composition"]
    assert comp is not None
    assert abs(comp["actual"]["LA"] - 0.05) <= 0.002
    assert abs(comp["actual"]["Y"] - 0.05) <= 0.002


def test_greedy_fallback_when_scipy_unavailable():
    payload = _base_input()
    # Default backend (auto) should fall back to greedy

    with (
        patch("ursa.tools._lp_solver._HAS_SCIPY", False),
        patch(
            "ursa.tools.cmm_supply_chain_optimization_tool.scipy_available",
            return_value=False,
        ),
    ):
        result = solve_cmm_supply_chain_optimization(
            payload,
        )

    assert result["status"] == "optimal_greedy"
    assert result["feasible"] is True
    assert result.get("shadow_prices") is None


def test_lp_infeasible_capacity():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    payload["suppliers"] = [
        {
            "name": "tiny",
            "capacity": 10,
            "unit_cost": 8.0,
            "risk_score": 0.0,
        },
    ]
    payload["shipping_cost"] = {"tiny": {"NA": 0.0, "EU": 0.0}}

    result = solve_cmm_supply_chain_optimization(payload)

    # LP should handle this via unmet demand slack
    assert result["status"] == "infeasible_unmet_demand"
    assert result["feasible"] is False
    assert sum(result["unmet_demand"].values()) > 0


@pytest.mark.skipif(
    not _ND_SCENARIOS_PATH.exists(),
    reason="Nd scenario config not found",
)
def test_nd_scenarios_solve():
    with open(_ND_SCENARIOS_PATH) as fh:
        scenarios = json.load(fh)

    # Pre-shock: should be feasible
    pre = solve_cmm_supply_chain_optimization(
        scenarios["nd_preshock_baseline"]["optimization_input"],
    )
    assert pre["feasible"] is True

    # Post-December: demand exceeds constrained capacity
    post_dec = solve_cmm_supply_chain_optimization(
        scenarios["nd_post_december_2025"]["optimization_input"],
    )
    # Should have some unmet demand due to share caps
    assert post_dec["sensitivity_summary"]["unmet_demand_total"] > 0


def test_output_validates_through_pydantic():
    payload = _base_input()
    payload["solver_backend"] = "lp"
    result = solve_cmm_supply_chain_optimization(payload)

    validated = OptimizationOutput.model_validate(result)
    assert validated.commodity == "CO"
    assert validated.shadow_prices is not None
    assert isinstance(
        validated.shadow_prices.demand_balance,
        dict,
    )
