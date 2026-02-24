from ursa.tools.cmm_supply_chain_optimization_tool import (
    OptimizationOutput,
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
    assert result["status"] == "optimal_greedy"
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

    assert result["status"] == "infeasible_composition_constraints"
    assert result["feasible"] is False
    composition = result["composition"]
    assert composition is not None
    assert composition["feasible"] is False
    assert abs(composition["residuals"]["LA"]) > 0.001


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


def test_validation_error_negative_capacity():
    payload = _base_input()
    payload["suppliers"][0]["capacity"] = -50
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False
    assert len(result["errors"]) >= 1


def test_validation_error_fraction_above_one():
    payload = _base_input()
    payload["max_supplier_share"] = 1.5
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_non_numeric_cost():
    payload = _base_input()
    payload["suppliers"][0]["unit_cost"] = "not_a_number"
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_empty_demand():
    payload = _base_input()
    payload["demand"] = {}
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_empty_suppliers():
    payload = _base_input()
    payload["suppliers"] = []
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_unknown_field():
    payload = _base_input()
    payload["bogus_field"] = 42
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_composition_fraction_out_of_range():
    payload = _base_input()
    payload["composition_targets"] = {"LA": 1.5}
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_unmet_demand_penalty_below_min():
    payload = _base_input()
    payload["unmet_demand_penalty"] = 0.5
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "validation_error"
    assert result["feasible"] is False


def test_validation_error_preserves_commodity_in_response():
    payload = _base_input()
    payload["suppliers"][0]["capacity"] = -10
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["commodity"] == "CO"


# ---------------------------------------------------------------------------
# Backward-compatibility tests
# ---------------------------------------------------------------------------


def test_int_demand_values_coerced_to_float():
    payload = _base_input()
    # _base_input already uses int demand values (100, 80)
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "optimal_greedy"
    assert "objective_value" in result


def test_supplier_name_auto_generated():
    payload = {
        "commodity": "CMM",
        "demand": {"NA": 50},
        "suppliers": [
            {"capacity": 100, "unit_cost": 5.0},
            {"capacity": 80, "unit_cost": 6.0},
        ],
    }
    result = solve_cmm_supply_chain_optimization(payload)

    assert result["status"] == "optimal_greedy"
    supplier_names = {a["supplier"] for a in result["allocations"]}
    assert "supplier_1" in supplier_names or "supplier_2" in supplier_names


def test_component_names_normalized_to_uppercase():
    payload = {
        "commodity": "CMM",
        "demand": {"US": 100},
        "suppliers": [
            {
                "name": "s1",
                "capacity": 100,
                "unit_cost": 1.0,
                "composition_profile": {"la": 0.05, "y": 0.05},
            },
        ],
        "composition_targets": {"la": 0.05, "y": 0.05},
        "composition_tolerance": 0.01,
    }
    result = solve_cmm_supply_chain_optimization(payload)

    composition = result["composition"]
    assert composition is not None
    assert "LA" in composition["actual"]
    assert "Y" in composition["actual"]


# ---------------------------------------------------------------------------
# Output model validation test
# ---------------------------------------------------------------------------


def test_output_has_correct_types():
    payload = _base_input()
    result = solve_cmm_supply_chain_optimization(payload)

    # Re-validate through the output model to confirm structural correctness
    validated = OptimizationOutput.model_validate(result)
    assert validated.commodity == "CO"
    assert isinstance(validated.feasible, bool)
    assert isinstance(validated.objective_value, float)
    assert len(validated.allocations) > 0
