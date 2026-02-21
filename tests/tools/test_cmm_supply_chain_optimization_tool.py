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
