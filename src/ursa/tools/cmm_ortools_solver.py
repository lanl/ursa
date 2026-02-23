"""OR-Tools MIP solver for CMM supply-chain allocation.

Falls back gracefully when ``ortools`` is not installed — callers should
check :func:`ortools_available` or catch the ``None`` return from
:func:`solve_with_ortools`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OR-Tools import
# ---------------------------------------------------------------------------
try:
    from ortools.linear_solver import pywraplp  # type: ignore[import-untyped]

    _HAS_ORTOOLS = True
except ImportError:
    _HAS_ORTOOLS = False

_EPS = 1e-9


def ortools_available() -> bool:
    """Return ``True`` when the OR-Tools package is importable."""
    return _HAS_ORTOOLS


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ORToolsResult:
    """Holds the raw solution extracted from the MIP."""

    allocation: dict[tuple[str, str], float]
    unmet: dict[str, float]
    solver_status: str
    objective_value: float = 0.0
    supplier_max: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MIP formulation
# ---------------------------------------------------------------------------
def solve_with_ortools(
    *,
    markets: list[str],
    supplier_names: list[str],
    supplier_capacities: dict[str, float],
    supplier_unit_costs: dict[str, float],
    supplier_risk_scores: dict[str, float],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet_penalty: float,
    max_supplier_share: float,
    composition_targets: dict[str, float],
    composition_tolerance: float,
    composition_profiles: dict[str, dict[str, float]],
) -> ORToolsResult | None:
    """Build and solve the CMM allocation MIP.

    Returns an :class:`ORToolsResult` on success, or ``None`` if OR-Tools is
    unavailable or the solver encounters an unexpected error.
    """
    if not _HAS_ORTOOLS:
        return None

    try:
        return _build_and_solve(
            markets=markets,
            supplier_names=supplier_names,
            supplier_capacities=supplier_capacities,
            supplier_unit_costs=supplier_unit_costs,
            supplier_risk_scores=supplier_risk_scores,
            demand=demand,
            shipping_cost=shipping_cost,
            risk_weight=risk_weight,
            unmet_penalty=unmet_penalty,
            max_supplier_share=max_supplier_share,
            composition_targets=composition_targets,
            composition_tolerance=composition_tolerance,
            composition_profiles=composition_profiles,
        )
    except Exception:
        logger.exception("OR-Tools solver failed unexpectedly")
        return None


def _build_and_solve(
    *,
    markets: list[str],
    supplier_names: list[str],
    supplier_capacities: dict[str, float],
    supplier_unit_costs: dict[str, float],
    supplier_risk_scores: dict[str, float],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet_penalty: float,
    max_supplier_share: float,
    composition_targets: dict[str, float],
    composition_tolerance: float,
    composition_profiles: dict[str, dict[str, float]],
) -> ORToolsResult | None:
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        logger.warning("CBC solver not available in OR-Tools installation")
        return None

    total_demand = sum(demand.values())

    # Supplier share caps
    supplier_max: dict[str, float] = {}
    for s in supplier_names:
        share_cap = max_supplier_share * total_demand
        supplier_max[s] = min(supplier_capacities[s], share_cap)

    # --- Decision variables -------------------------------------------------
    # x[s, m] — continuous allocation flow
    x: dict[tuple[str, str], Any] = {}
    for s in supplier_names:
        for m in markets:
            x[s, m] = solver.NumVar(0.0, solver.infinity(), f"x_{s}_{m}")

    # y[s] — binary supplier activation
    y: dict[str, Any] = {}
    for s in supplier_names:
        y[s] = solver.IntVar(0, 1, f"y_{s}")

    # u[m] — continuous unmet demand
    u: dict[str, Any] = {}
    for m in markets:
        u[m] = solver.NumVar(0.0, solver.infinity(), f"u_{m}")

    # --- Constraints --------------------------------------------------------
    # (1) Demand: sum_s x[s,m] + u[m] = demand[m]
    for m in markets:
        ct = solver.Constraint(demand[m], demand[m], f"demand_{m}")
        for s in supplier_names:
            ct.SetCoefficient(x[s, m], 1.0)
        ct.SetCoefficient(u[m], 1.0)

    # (2) Capacity: sum_m x[s,m] <= capacity[s]
    for s in supplier_names:
        ct = solver.Constraint(0.0, supplier_capacities[s], f"cap_{s}")
        for m in markets:
            ct.SetCoefficient(x[s, m], 1.0)

    # (3) Activation (big-M linking): x[s,m] <= capacity[s] * y[s]
    for s in supplier_names:
        big_m = supplier_capacities[s]
        for m in markets:
            ct = solver.Constraint(-solver.infinity(), 0.0, f"act_{s}_{m}")
            ct.SetCoefficient(x[s, m], 1.0)
            ct.SetCoefficient(y[s], -big_m)

    # (4) Max share: sum_m x[s,m] <= max_share * total_demand
    for s in supplier_names:
        ct = solver.Constraint(0.0, supplier_max[s], f"share_{s}")
        for m in markets:
            ct.SetCoefficient(x[s, m], 1.0)

    # (5) Composition constraints (linearized)
    for comp, target in composition_targets.items():
        # Upper bound: sum_{s,m} x[s,m] * (profile[s,c] - target - tol) <= 0
        ct_upper = solver.Constraint(
            -solver.infinity(), 0.0, f"comp_upper_{comp}"
        )
        for s in supplier_names:
            profile_val = composition_profiles.get(s, {}).get(comp, 0.0)
            coeff = profile_val - target - composition_tolerance
            for m in markets:
                ct_upper.SetCoefficient(x[s, m], coeff)

        # Lower bound: sum_{s,m} x[s,m] * (target - tol - profile[s,c]) <= 0
        ct_lower = solver.Constraint(
            -solver.infinity(), 0.0, f"comp_lower_{comp}"
        )
        for s in supplier_names:
            profile_val = composition_profiles.get(s, {}).get(comp, 0.0)
            coeff = target - composition_tolerance - profile_val
            for m in markets:
                ct_lower.SetCoefficient(x[s, m], coeff)

    # --- Objective ----------------------------------------------------------
    objective = solver.Objective()
    for s in supplier_names:
        for m in markets:
            unit_cost = supplier_unit_costs[s]
            ship = shipping_cost.get(s, {}).get(m, 0.0)
            risk = risk_weight * supplier_risk_scores[s]
            objective.SetCoefficient(x[s, m], unit_cost + ship + risk)
    for m in markets:
        objective.SetCoefficient(u[m], unmet_penalty)
    objective.SetMinimization()

    # --- Solve --------------------------------------------------------------
    solver.SetTimeLimit(60_000)  # 60 seconds
    status = solver.Solve()

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        logger.warning("OR-Tools solver status %s — no feasible solution", status)
        return None

    # --- Extract solution ---------------------------------------------------
    allocation: dict[tuple[str, str], float] = {}
    for s in supplier_names:
        for m in markets:
            val = x[s, m].solution_value()
            if val > _EPS:
                allocation[s, m] = val

    unmet: dict[str, float] = {}
    for m in markets:
        val = u[m].solution_value()
        unmet[m] = max(0.0, val)

    solver_status: str
    if status == pywraplp.Solver.OPTIMAL:
        solver_status = "optimal"
    else:
        solver_status = "optimal_mip_feasible"

    return ORToolsResult(
        allocation=allocation,
        unmet=unmet,
        solver_status=solver_status,
        objective_value=solver.Objective().Value(),
        supplier_max=supplier_max,
    )
