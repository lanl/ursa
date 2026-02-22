from __future__ import annotations

from unittest.mock import patch

import pytest

from ursa.tools.cmm_ortools_solver import ORToolsResult, ortools_available, solve_with_ortools

_skip_no_ortools = pytest.mark.skipif(
    not ortools_available(), reason="ortools not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _base_kwargs(
    *,
    suppliers: list[dict[str, object]] | None = None,
    demand: dict[str, float] | None = None,
    composition_targets: dict[str, float] | None = None,
    composition_tolerance: float = 0.0,
    composition_profiles: dict[str, dict[str, float]] | None = None,
    max_supplier_share: float = 1.0,
) -> dict[str, object]:
    """Build a minimal keyword-argument dict for ``solve_with_ortools``."""
    if suppliers is None:
        suppliers = [
            {"name": "S1", "capacity": 100.0, "unit_cost": 5.0, "risk_score": 0.1},
            {"name": "S2", "capacity": 80.0, "unit_cost": 7.0, "risk_score": 0.2},
        ]
    if demand is None:
        demand = {"NA": 60.0, "EU": 40.0}
    if composition_targets is None:
        composition_targets = {}
    if composition_profiles is None:
        composition_profiles = {s["name"]: {} for s in suppliers}

    names = [str(s["name"]) for s in suppliers]
    return dict(
        markets=sorted(demand.keys()),
        supplier_names=names,
        supplier_capacities={str(s["name"]): float(s["capacity"]) for s in suppliers},
        supplier_unit_costs={str(s["name"]): float(s["unit_cost"]) for s in suppliers},
        supplier_risk_scores={str(s["name"]): float(s["risk_score"]) for s in suppliers},
        demand=demand,
        shipping_cost={n: {m: 1.0 for m in demand} for n in names},
        risk_weight=1.0,
        unmet_penalty=10_000.0,
        max_supplier_share=max_supplier_share,
        composition_targets=composition_targets,
        composition_tolerance=composition_tolerance,
        composition_profiles=composition_profiles,
    )


# ---------------------------------------------------------------------------
# 1. Basic feasible — 2 suppliers, 2 markets, no composition
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_basic_feasible() -> None:
    kw = _base_kwargs()
    result = solve_with_ortools(**kw)

    assert result is not None
    assert result.solver_status in ("optimal", "optimal_mip_feasible")

    # All demand met
    total_unmet = sum(result.unmet.values())
    assert total_unmet < 1e-6

    # Allocations cover demand
    alloc_by_market: dict[str, float] = {}
    for (_, m), amt in result.allocation.items():
        alloc_by_market[m] = alloc_by_market.get(m, 0.0) + amt
    for m, d in kw["demand"].items():
        assert abs(alloc_by_market.get(m, 0.0) - d) < 1e-6


# ---------------------------------------------------------------------------
# 2. Composition feasible — la_rich + y_rich must blend 50/50 to hit targets
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_composition_feasible() -> None:
    suppliers = [
        {"name": "la_rich", "capacity": 100.0, "unit_cost": 1.0, "risk_score": 0.0},
        {"name": "y_rich", "capacity": 100.0, "unit_cost": 5.0, "risk_score": 0.0},
    ]
    profiles = {
        "la_rich": {"LA": 0.10, "Y": 0.00},
        "y_rich": {"LA": 0.00, "Y": 0.10},
    }
    kw = _base_kwargs(
        suppliers=suppliers,
        demand={"US": 100.0},
        composition_targets={"LA": 0.05, "Y": 0.05},
        composition_tolerance=0.001,
        composition_profiles=profiles,
    )
    result = solve_with_ortools(**kw)

    assert result is not None
    assert result.solver_status in ("optimal", "optimal_mip_feasible")
    assert sum(result.unmet.values()) < 1e-6

    # Both suppliers must be used ~50 each
    totals: dict[str, float] = {}
    for (s, _), amt in result.allocation.items():
        totals[s] = totals.get(s, 0.0) + amt
    assert abs(totals.get("la_rich", 0.0) - 50.0) < 2.0
    assert abs(totals.get("y_rich", 0.0) - 50.0) < 2.0


# ---------------------------------------------------------------------------
# 3. Infeasible capacity — total capacity < demand → unmet > 0
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_infeasible_capacity() -> None:
    suppliers = [
        {"name": "S1", "capacity": 30.0, "unit_cost": 5.0, "risk_score": 0.1},
        {"name": "S2", "capacity": 20.0, "unit_cost": 7.0, "risk_score": 0.2},
    ]
    kw = _base_kwargs(suppliers=suppliers, demand={"NA": 100.0})
    result = solve_with_ortools(**kw)

    assert result is not None
    assert sum(result.unmet.values()) > 0.0


# ---------------------------------------------------------------------------
# 4. Infeasible composition — suppliers can't achieve target
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_infeasible_composition() -> None:
    suppliers = [
        {"name": "A", "capacity": 100.0, "unit_cost": 1.0, "risk_score": 0.0},
        {"name": "B", "capacity": 100.0, "unit_cost": 1.0, "risk_score": 0.0},
    ]
    profiles = {
        "A": {"LA": 0.02, "Y": 0.01},
        "B": {"LA": 0.03, "Y": 0.02},
    }
    kw = _base_kwargs(
        suppliers=suppliers,
        demand={"US": 100.0},
        composition_targets={"LA": 0.10, "Y": 0.10},
        composition_tolerance=0.001,
        composition_profiles=profiles,
    )
    result = solve_with_ortools(**kw)

    # The composition constraints make a zero-unmet solution infeasible,
    # so either solver returns None (infeasible) or it uses unmet demand to
    # relax the blend constraints.
    if result is not None:
        assert sum(result.unmet.values()) > 0.0


# ---------------------------------------------------------------------------
# 5. MIP objective ≤ greedy objective
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_mip_leq_greedy() -> None:
    from ursa.tools.cmm_supply_chain_optimization_tool import (
        _greedy_fallback,
        _Supplier,
        _compute_costs,
    )

    suppliers_raw = [
        {"name": "S1", "capacity": 100.0, "unit_cost": 5.0, "risk_score": 0.1},
        {"name": "S2", "capacity": 80.0, "unit_cost": 7.0, "risk_score": 0.2},
        {"name": "S3", "capacity": 60.0, "unit_cost": 6.0, "risk_score": 0.05},
    ]
    demand = {"NA": 60.0, "EU": 40.0, "APAC": 30.0}
    shipping = {
        "S1": {"NA": 1.0, "EU": 3.0, "APAC": 4.0},
        "S2": {"NA": 2.0, "EU": 1.0, "APAC": 3.0},
        "S3": {"NA": 2.5, "EU": 2.0, "APAC": 1.0},
    }
    risk_weight = 2.0
    unmet_penalty = 10_000.0
    max_share = 0.6

    # --- OR-Tools ---
    names = [s["name"] for s in suppliers_raw]
    mip_result = solve_with_ortools(
        markets=sorted(demand.keys()),
        supplier_names=names,
        supplier_capacities={s["name"]: s["capacity"] for s in suppliers_raw},
        supplier_unit_costs={s["name"]: s["unit_cost"] for s in suppliers_raw},
        supplier_risk_scores={s["name"]: s["risk_score"] for s in suppliers_raw},
        demand=demand,
        shipping_cost=shipping,
        risk_weight=risk_weight,
        unmet_penalty=unmet_penalty,
        max_supplier_share=max_share,
        composition_targets={},
        composition_tolerance=0.0,
        composition_profiles={n: {} for n in names},
    )
    assert mip_result is not None

    # --- Greedy ---
    supplier_objs = sorted(
        [
            _Supplier(name=s["name"], capacity=s["capacity"],
                      unit_cost=s["unit_cost"], risk_score=s["risk_score"])
            for s in suppliers_raw
        ],
        key=lambda x: x.name,
    )
    alloc_g, unmet_g, _, _ = _greedy_fallback(
        markets=sorted(demand.keys()),
        suppliers=supplier_objs,
        demand=demand,
        shipping_cost=shipping,
        risk_weight=risk_weight,
        max_supplier_share=max_share,
    )
    greedy_costs = _compute_costs(
        allocation=alloc_g,
        suppliers=supplier_objs,
        shipping_cost=shipping,
        risk_weight=risk_weight,
        unmet=unmet_g,
        unmet_penalty=unmet_penalty,
    )
    greedy_obj = sum(greedy_costs.values())

    assert mip_result.objective_value <= greedy_obj + 1e-6


# ---------------------------------------------------------------------------
# 6. Max share enforced — one cheap supplier capped by share limit
# ---------------------------------------------------------------------------
@_skip_no_ortools
def test_max_share_enforced() -> None:
    suppliers = [
        {"name": "cheap", "capacity": 200.0, "unit_cost": 1.0, "risk_score": 0.0},
        {"name": "pricey", "capacity": 200.0, "unit_cost": 10.0, "risk_score": 0.0},
    ]
    demand = {"US": 100.0}
    kw = _base_kwargs(
        suppliers=suppliers,
        demand=demand,
        max_supplier_share=0.5,
    )
    result = solve_with_ortools(**kw)

    assert result is not None
    total_demand = sum(demand.values())
    share_cap = 0.5 * total_demand

    for s_name in ["cheap", "pricey"]:
        total_s = sum(
            amt for (s, _), amt in result.allocation.items() if s == s_name
        )
        assert total_s <= share_cap + 1e-6


# ---------------------------------------------------------------------------
# 7. Graceful None when unavailable
# ---------------------------------------------------------------------------
def test_graceful_none_when_unavailable() -> None:
    kw = _base_kwargs()
    with patch("ursa.tools.cmm_ortools_solver._HAS_ORTOOLS", False):
        result = solve_with_ortools(**kw)
    assert result is None
