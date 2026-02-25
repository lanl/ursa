"""LP solver backend for CMM supply-chain optimization.

Uses ``scipy.optimize.linprog`` with the HiGHS method to produce
provably optimal allocations and dual (shadow) prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from scipy.optimize import linprog

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def scipy_available() -> bool:
    """Return ``True`` when scipy is importable."""
    return _HAS_SCIPY


@dataclass(frozen=True)
class LPResult:
    """Result container returned by :func:`solve_lp`."""

    success: bool
    status: str
    objective: float = 0.0
    allocation: dict[tuple[str, str], float] = field(
        default_factory=dict,
    )
    unmet: dict[str, float] = field(default_factory=dict)
    shadow_prices_demand: dict[str, float] = field(
        default_factory=dict,
    )
    shadow_prices_capacity: dict[str, float] = field(
        default_factory=dict,
    )
    shadow_prices_share: dict[str, float] = field(
        default_factory=dict,
    )
    shadow_prices_composition: dict[str, float] = field(
        default_factory=dict,
    )
    reduced_costs: dict[str, float] = field(
        default_factory=dict,
    )


def solve_lp(
    *,
    suppliers: list[Any],
    markets: list[str],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet_penalty: float,
    max_supplier_share: float,
    composition_targets: dict[str, float] | None = None,
    composition_tolerance: float = 0.0,
    composition_profiles: dict[str, dict[str, float]] | None = None,
) -> LPResult:
    """Solve the supply-chain allocation as a linear programme.

    Parameters
    ----------
    suppliers
        List of supplier objects (must have ``.name``, ``.capacity``,
        ``.unit_cost``, ``.risk_score`` attributes).
    markets
        Sorted list of market names.
    demand
        ``{market: required_quantity}``.
    shipping_cost
        ``{supplier_name: {market: cost_per_unit}}``.
    risk_weight
        Multiplier for supplier risk in the objective.
    unmet_penalty
        Per-unit penalty for unmet demand.
    max_supplier_share
        Maximum fraction of total demand from any single supplier.
    composition_targets
        Optional ``{component: target_fraction}``.
    composition_tolerance
        Allowed deviation from each composition target.
    composition_profiles
        ``{supplier_name: {component: fraction}}``.

    Returns
    -------
    LPResult
        Solver output including allocations, unmet demand, and
        shadow prices for all constraint groups.
    """
    if not _HAS_SCIPY:
        return LPResult(
            success=False,
            status="scipy_unavailable",
        )

    S = len(suppliers)
    M = len(markets)
    n_x = S * M  # allocation variables
    n_u = M  # unmet-demand slack variables
    n_vars = n_x + n_u

    total_demand = sum(demand.values())

    # --- Objective: c^T x -------------------------------------------
    c = [0.0] * n_vars
    for s_idx, sup in enumerate(suppliers):
        for m_idx, mkt in enumerate(markets):
            cost = (
                sup.unit_cost
                + shipping_cost.get(sup.name, {}).get(mkt, 0.0)
                + risk_weight * sup.risk_score
            )
            c[s_idx * M + m_idx] = cost
    for m_idx in range(M):
        c[n_x + m_idx] = unmet_penalty

    # --- Equality: demand balance -----------------------------------
    # Σ_s x[s,m] + u[m] = demand[m]   ∀ m
    A_eq: list[list[float]] = []
    b_eq: list[float] = []
    for m_idx, mkt in enumerate(markets):
        row = [0.0] * n_vars
        for s_idx in range(S):
            row[s_idx * M + m_idx] = 1.0
        row[n_x + m_idx] = 1.0
        A_eq.append(row)
        b_eq.append(demand[mkt])

    # --- Inequality: capacity + share + composition -----------------
    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    # (2) Capacity: Σ_m x[s,m] ≤ capacity[s]   ∀ s
    for s_idx, sup in enumerate(suppliers):
        row = [0.0] * n_vars
        for m_idx in range(M):
            row[s_idx * M + m_idx] = 1.0
        A_ub.append(row)
        b_ub.append(sup.capacity)

    # (3) Share: Σ_m x[s,m] ≤ max_share × total_demand   ∀ s
    share_cap = max_supplier_share * total_demand
    for s_idx in range(S):
        row = [0.0] * n_vars
        for m_idx in range(M):
            row[s_idx * M + m_idx] = 1.0
        A_ub.append(row)
        b_ub.append(share_cap)

    # (4-5) Composition constraints (optional)
    comp_targets = composition_targets or {}
    comp_profiles = composition_profiles or {}
    comp_components = sorted(comp_targets.keys())

    for component in comp_components:
        target = comp_targets[component]
        tol = composition_tolerance

        # Upper: Σ_{s,m} (profile[s,c] - target - tol) × x[s,m] ≤ 0
        row_upper = [0.0] * n_vars
        for s_idx, sup in enumerate(suppliers):
            coeff = (
                comp_profiles.get(sup.name, {}).get(component, 0.0)
                - target
                - tol
            )
            for m_idx in range(M):
                row_upper[s_idx * M + m_idx] = coeff
        A_ub.append(row_upper)
        b_ub.append(0.0)

        # Lower: Σ_{s,m} (target - tol - profile[s,c]) × x[s,m] ≤ 0
        row_lower = [0.0] * n_vars
        for s_idx, sup in enumerate(suppliers):
            coeff = (
                target
                - tol
                - comp_profiles.get(sup.name, {}).get(component, 0.0)
            )
            for m_idx in range(M):
                row_lower[s_idx * M + m_idx] = coeff
        A_ub.append(row_lower)
        b_ub.append(0.0)

    # --- Variable bounds: x ≥ 0, u ≥ 0 -----------------------------
    bounds = [(0.0, None)] * n_vars

    # --- Solve with HiGHS -------------------------------------------
    result = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return LPResult(
            success=False,
            status=f"lp_{result.status}",
            objective=0.0,
        )

    # --- Extract solution -------------------------------------------
    x = result.x

    allocation: dict[tuple[str, str], float] = {}
    for s_idx, sup in enumerate(suppliers):
        for m_idx, mkt in enumerate(markets):
            val = x[s_idx * M + m_idx]
            if val > 1e-9:
                allocation[(sup.name, mkt)] = round(val, 6)

    unmet: dict[str, float] = {}
    for m_idx, mkt in enumerate(markets):
        val = x[n_x + m_idx]
        unmet[mkt] = round(val, 6) if val > 1e-9 else 0.0

    # --- Extract shadow prices (duals) ------------------------------
    shadow_demand: dict[str, float] = {}
    shadow_capacity: dict[str, float] = {}
    shadow_share: dict[str, float] = {}
    shadow_composition: dict[str, float] = {}

    # Equality constraint duals (demand balance)
    if hasattr(result, "eqlin") and result.eqlin is not None:
        eq_duals = result.eqlin.marginals
        for m_idx, mkt in enumerate(markets):
            shadow_demand[mkt] = round(float(eq_duals[m_idx]), 6)

    # Inequality constraint duals
    if hasattr(result, "ineqlin") and result.ineqlin is not None:
        ineq_duals = result.ineqlin.marginals
        offset = 0

        # Capacity duals (S constraints)
        for s_idx, sup in enumerate(suppliers):
            shadow_capacity[sup.name] = round(
                float(ineq_duals[offset + s_idx]),
                6,
            )
        offset += S

        # Share duals (S constraints)
        for s_idx, sup in enumerate(suppliers):
            shadow_share[sup.name] = round(
                float(ineq_duals[offset + s_idx]),
                6,
            )
        offset += S

        # Composition duals (2 per component)
        for component in comp_components:
            upper_dual = float(ineq_duals[offset])
            lower_dual = float(ineq_duals[offset + 1])
            shadow_composition[component] = round(
                upper_dual - lower_dual,
                6,
            )
            offset += 2

    # --- Reduced costs ----------------------------------------------
    reduced: dict[str, float] = {}
    if hasattr(result, "x"):
        for s_idx, sup in enumerate(suppliers):
            for m_idx, mkt in enumerate(markets):
                idx = s_idx * M + m_idx
                # reduced cost from fun gradient minus dual
                rc = c[idx] - shadow_demand.get(mkt, 0.0)
                if abs(rc) > 1e-9:
                    reduced[f"{sup.name}->{mkt}"] = round(rc, 6)

    return LPResult(
        success=True,
        status="optimal",
        objective=round(float(result.fun), 6),
        allocation=allocation,
        unmet=unmet,
        shadow_prices_demand=shadow_demand,
        shadow_prices_capacity=shadow_capacity,
        shadow_prices_share=shadow_share,
        shadow_prices_composition=shadow_composition,
        reduced_costs=reduced,
    )
