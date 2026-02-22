from ursa.tools.cmm_supply_chain_optimization_tool import (
    solve_cmm_supply_chain_optimization,
)


def _base_input():
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


def test_cmm_optimization_output_schema_and_determinism():
    payload = _base_input()
    result_a = solve_cmm_supply_chain_optimization(payload)
    result_b = solve_cmm_supply_chain_optimization(payload)

    assert result_a == result_b
    assert "objective_value" in result_a
    assert "allocations" in result_a
    assert "constraint_residuals" in result_a
    assert "sensitivity_summary" in result_a
    assert isinstance(result_a["feasible"], bool)


def test_cmm_optimization_reports_infeasible_with_unmet_demand():
    payload = _base_input()
    payload["suppliers"] = [
        {
            "name": "low_capacity_1",
            "capacity": 50,
            "unit_cost": 8.0,
            "risk_score": 0.2,
        },
        {
            "name": "low_capacity_2",
            "capacity": 30,
            "unit_cost": 9.2,
            "risk_score": 0.1,
        },
    ]

    result = solve_cmm_supply_chain_optimization(payload)

    assert result["feasible"] is False
    assert result["status"] == "infeasible_unmet_demand"
    assert sum(result["unmet_demand"].values()) > 0


def test_cmm_optimization_enforces_composition_targets_when_possible():
    payload = {
        "commodity": "ND2FE14B_LA5_Y5",
        "demand": {"US": 100},
        "suppliers": [
            {
                "name": "la_rich",
                "capacity": 100,
                "unit_cost": 1.0,
                "risk_score": 0.1,
                "composition_profile": {"LA": 0.10, "Y": 0.00},
            },
            {
                "name": "y_rich",
                "capacity": 100,
                "unit_cost": 5.0,
                "risk_score": 0.1,
                "composition_profile": {"LA": 0.00, "Y": 0.10},
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
    }

    result = solve_cmm_supply_chain_optimization(payload)

    assert result["feasible"] is True
    assert result["status"] in ("optimal", "optimal_greedy", "optimal_mip_feasible")
    composition = result["composition"]
    assert composition is not None
    assert composition["feasible"] is True
    assert abs(composition["actual"]["LA"] - 0.05) <= 0.001
    assert abs(composition["actual"]["Y"] - 0.05) <= 0.0011


def test_cmm_optimization_reports_infeasible_composition_constraints():
    payload = {
        "commodity": "ND2FE14B_LA5_Y5",
        "demand": {"US": 100},
        "suppliers": [
            {
                "name": "supplier_a",
                "capacity": 100,
                "unit_cost": 1.0,
                "risk_score": 0.1,
                "composition_profile": {"LA": 0.02, "Y": 0.01},
            },
            {
                "name": "supplier_b",
                "capacity": 100,
                "unit_cost": 1.1,
                "risk_score": 0.1,
                "composition_profile": {"LA": 0.03, "Y": 0.02},
            },
        ],
        "shipping_cost": {
            "supplier_a": {"US": 0.0},
            "supplier_b": {"US": 0.0},
        },
        "risk_weight": 0.0,
        "max_supplier_share": 1.0,
        "composition_targets": {"LA": 0.05, "Y": 0.05},
        "composition_tolerance": 0.001,
    }

    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] in (
        "infeasible_composition_constraints",
        "infeasible_unmet_and_composition",
    )
    assert result["feasible"] is False
    composition = result["composition"]
    assert composition is not None
    assert composition["feasible"] is False
