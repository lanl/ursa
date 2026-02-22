from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool

_EPS = 1e-9


@dataclass(frozen=True)
class _Supplier:
    name: str
    capacity: float
    unit_cost: float
    risk_score: float


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_fraction(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _to_float(value, default)))


def _normalize_component_name(name: Any) -> str:
    return str(name).strip().upper()


def _sum_allocated_by_supplier(
    allocation: dict[tuple[str, str], float],
    suppliers: list[_Supplier],
) -> dict[str, float]:
    totals = {supplier.name: 0.0 for supplier in suppliers}
    for (supplier_name, _market), amount in allocation.items():
        totals[supplier_name] = totals.get(supplier_name, 0.0) + amount
    return totals


def _compute_costs(
    *,
    allocation: dict[tuple[str, str], float],
    suppliers: list[_Supplier],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet: dict[str, float],
    unmet_penalty: float,
) -> dict[str, float]:
    suppliers_by_name = {supplier.name: supplier for supplier in suppliers}
    procurement_cost = 0.0
    shipping_total = 0.0
    risk_total = 0.0

    for (supplier_name, market), amount in allocation.items():
        supplier = suppliers_by_name[supplier_name]
        procurement_cost += amount * supplier.unit_cost
        shipping_total += amount * shipping_cost.get(supplier_name, {}).get(
            market, 0.0
        )
        risk_total += amount * (risk_weight * supplier.risk_score)

    unmet_cost = sum(unmet.values()) * unmet_penalty
    return {
        "procurement": procurement_cost,
        "shipping": shipping_total,
        "risk_penalty": risk_total,
        "unmet_penalty": unmet_cost,
    }


def _compute_composition_metrics(
    *,
    totals_by_supplier: dict[str, float],
    composition_targets: dict[str, float],
    composition_profiles: dict[str, dict[str, float]],
    composition_tolerance: float,
) -> dict[str, Any] | None:
    if not composition_targets:
        return None

    total_allocated = sum(totals_by_supplier.values())
    actual: dict[str, float] = {}
    residuals: dict[str, float] = {}

    for component, target in composition_targets.items():
        if total_allocated <= _EPS:
            actual_value = 0.0
        else:
            weighted = 0.0
            for supplier_name, amount in totals_by_supplier.items():
                profile = composition_profiles.get(supplier_name, {})
                weighted += amount * profile.get(component, 0.0)
            actual_value = weighted / total_allocated
        actual[component] = actual_value
        residuals[component] = actual_value - target

    feasible = all(
        abs(residual) <= composition_tolerance + _EPS
        for residual in residuals.values()
    )

    return {
        "targets": composition_targets,
        "actual": actual,
        "residuals": residuals,
        "tolerance": composition_tolerance,
        "feasible": feasible,
    }


def _shift_between_suppliers(
    *,
    allocation: dict[tuple[str, str], float],
    supplier_remaining: dict[str, float],
    donor: str,
    receiver: str,
    markets: list[str],
    max_shift: float,
) -> float:
    if max_shift <= _EPS:
        return 0.0

    shifted_total = 0.0
    for market in markets:
        if shifted_total >= max_shift - _EPS:
            break

        donor_key = (donor, market)
        receiver_key = (receiver, market)
        donor_amount = allocation.get(donor_key, 0.0)
        receiver_cap = supplier_remaining.get(receiver, 0.0)

        shift = min(
            max_shift - shifted_total,
            donor_amount,
            receiver_cap,
        )
        if shift <= _EPS:
            continue

        new_donor_amount = donor_amount - shift
        if new_donor_amount <= _EPS:
            allocation.pop(donor_key, None)
        else:
            allocation[donor_key] = new_donor_amount

        allocation[receiver_key] = allocation.get(receiver_key, 0.0) + shift
        supplier_remaining[receiver] = receiver_cap - shift
        supplier_remaining[donor] = supplier_remaining.get(donor, 0.0) + shift
        shifted_total += shift

    return shifted_total


def _rebalance_for_composition(
    *,
    allocation: dict[tuple[str, str], float],
    supplier_remaining: dict[str, float],
    suppliers: list[_Supplier],
    markets: list[str],
    composition_targets: dict[str, float],
    composition_profiles: dict[str, dict[str, float]],
    composition_tolerance: float,
    max_iterations: int = 200,
) -> None:
    if not composition_targets:
        return

    for _ in range(max_iterations):
        totals = _sum_allocated_by_supplier(allocation, suppliers)
        metrics = _compute_composition_metrics(
            totals_by_supplier=totals,
            composition_targets=composition_targets,
            composition_profiles=composition_profiles,
            composition_tolerance=composition_tolerance,
        )
        if metrics is None or metrics["feasible"]:
            return

        total_allocated = sum(totals.values())
        if total_allocated <= _EPS:
            return

        residuals = metrics["residuals"]
        pending_components = [
            (component, float(residual))
            for component, residual in residuals.items()
            if abs(float(residual)) > composition_tolerance + _EPS
        ]
        pending_components.sort(key=lambda item: abs(item[1]), reverse=True)

        progressed = False
        for component, residual in pending_components:
            too_high = residual > 0.0
            if too_high:
                donor_candidates = sorted(
                    suppliers,
                    key=lambda supplier: (
                        -composition_profiles[supplier.name].get(component, 0.0),
                        supplier.name,
                    ),
                )
                receiver_candidates = sorted(
                    suppliers,
                    key=lambda supplier: (
                        composition_profiles[supplier.name].get(component, 0.0),
                        supplier.name,
                    ),
                )
            else:
                donor_candidates = sorted(
                    suppliers,
                    key=lambda supplier: (
                        composition_profiles[supplier.name].get(component, 0.0),
                        supplier.name,
                    ),
                )
                receiver_candidates = sorted(
                    suppliers,
                    key=lambda supplier: (
                        -composition_profiles[supplier.name].get(component, 0.0),
                        supplier.name,
                    ),
                )

            for donor in donor_candidates:
                donor_name = donor.name
                donor_amount = totals.get(donor_name, 0.0)
                if donor_amount <= _EPS:
                    continue

                donor_profile = composition_profiles[donor_name].get(component, 0.0)
                for receiver in receiver_candidates:
                    receiver_name = receiver.name
                    if receiver_name == donor_name:
                        continue
                    if supplier_remaining.get(receiver_name, 0.0) <= _EPS:
                        continue

                    receiver_profile = composition_profiles[receiver_name].get(
                        component, 0.0
                    )
                    profile_delta = (
                        donor_profile - receiver_profile
                        if too_high
                        else receiver_profile - donor_profile
                    )
                    if profile_delta <= _EPS:
                        continue

                    needed_shift = (
                        (abs(residual) - composition_tolerance)
                        * total_allocated
                        / profile_delta
                    )
                    if needed_shift <= _EPS:
                        continue

                    shifted = _shift_between_suppliers(
                        allocation=allocation,
                        supplier_remaining=supplier_remaining,
                        donor=donor_name,
                        receiver=receiver_name,
                        markets=markets,
                        max_shift=needed_shift,
                    )
                    if shifted > _EPS:
                        progressed = True
                        break
                if progressed:
                    break
            if progressed:
                break

        if not progressed:
            return


def _normalize_input(payload: dict[str, Any]) -> tuple[
    str,
    list[str],
    list[_Supplier],
    dict[str, float],
    dict[str, dict[str, float]],
    float,
    float,
    float,
    dict[str, float],
    float,
    dict[str, dict[str, float]],
]:
    commodity = str(payload.get("commodity", "CMM"))

    demand_raw = payload.get("demand", {})
    if not isinstance(demand_raw, dict) or not demand_raw:
        raise ValueError("optimization_input.demand must be a non-empty mapping")
    markets = sorted(str(k) for k in demand_raw.keys())
    demand = {str(k): _to_float(v, 0.0) for k, v in demand_raw.items()}

    suppliers_raw = payload.get("suppliers", [])
    if not isinstance(suppliers_raw, list) or not suppliers_raw:
        raise ValueError("optimization_input.suppliers must be a non-empty list")

    composition_targets_raw = payload.get("composition_targets", {})
    composition_targets: dict[str, float] = {}
    if isinstance(composition_targets_raw, dict):
        for component, target in composition_targets_raw.items():
            name = _normalize_component_name(component)
            if not name:
                continue
            composition_targets[name] = _to_fraction(target, 0.0)
    composition_tolerance = _to_fraction(
        payload.get("composition_tolerance"),
        0.0,
    )

    suppliers: list[_Supplier] = []
    composition_profiles: dict[str, dict[str, float]] = {}
    for idx, item in enumerate(suppliers_raw):
        if not isinstance(item, dict):
            raise ValueError("each supplier must be a mapping")
        name = str(item.get("name") or f"supplier_{idx + 1}")
        suppliers.append(
            _Supplier(
                name=name,
                capacity=max(0.0, _to_float(item.get("capacity"), 0.0)),
                unit_cost=max(0.0, _to_float(item.get("unit_cost"), 0.0)),
                risk_score=max(0.0, _to_float(item.get("risk_score"), 0.0)),
            )
        )

        raw_profile = item.get("composition_profile", {})
        profile: dict[str, float] = {}
        if isinstance(raw_profile, dict):
            for component, fraction in raw_profile.items():
                comp_name = _normalize_component_name(component)
                if not comp_name:
                    continue
                profile[comp_name] = _to_fraction(fraction, 0.0)
        composition_profiles[name] = profile

    suppliers = sorted(suppliers, key=lambda supplier: supplier.name)

    components = set(composition_targets.keys())
    for profile in composition_profiles.values():
        components.update(profile.keys())

    normalized_profiles: dict[str, dict[str, float]] = {}
    for supplier in suppliers:
        profile = composition_profiles.get(supplier.name, {})
        normalized_profiles[supplier.name] = {
            component: profile.get(component, 0.0)
            for component in sorted(components)
        }

    shipping_raw = payload.get("shipping_cost", {})
    shipping_cost: dict[str, dict[str, float]] = {}
    if isinstance(shipping_raw, dict):
        for supplier_name, market_costs in shipping_raw.items():
            if isinstance(market_costs, dict):
                shipping_cost[str(supplier_name)] = {
                    str(market): _to_float(cost, 0.0)
                    for market, cost in market_costs.items()
                }

    risk_weight = max(0.0, _to_float(payload.get("risk_weight"), 0.0))
    unmet_penalty = max(1.0, _to_float(payload.get("unmet_demand_penalty"), 10000.0))
    max_supplier_share = _to_fraction(payload.get("max_supplier_share"), 1.0)

    return (
        commodity,
        markets,
        suppliers,
        demand,
        shipping_cost,
        risk_weight,
        unmet_penalty,
        max_supplier_share,
        composition_targets,
        composition_tolerance,
        normalized_profiles,
    )


def _greedy_fallback(
    *,
    markets: list[str],
    suppliers: list[_Supplier],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    max_supplier_share: float,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    total_demand = sum(demand.values())
    supplier_max: dict[str, float] = {}
    for supplier in suppliers:
        share_cap = max_supplier_share * total_demand
        supplier_max[supplier.name] = min(supplier.capacity, share_cap)

    supplier_remaining = dict(supplier_max)
    demand_remaining = {market: float(demand[market]) for market in markets}
    allocation: dict[tuple[str, str], float] = {}

    for market in markets:
        candidates: list[tuple[float, _Supplier]] = []
        for supplier in suppliers:
            landed_cost = (
                supplier.unit_cost
                + shipping_cost.get(supplier.name, {}).get(market, 0.0)
                + risk_weight * supplier.risk_score
            )
            candidates.append((landed_cost, supplier))
        candidates.sort(key=lambda item: (item[0], item[1].name))

        for _landed_cost, supplier in candidates:
            if demand_remaining[market] <= _EPS:
                break
            available = supplier_remaining[supplier.name]
            if available <= _EPS:
                continue
            flow = min(available, demand_remaining[market])
            if flow <= _EPS:
                continue
            allocation[(supplier.name, market)] = flow
            supplier_remaining[supplier.name] -= flow
            demand_remaining[market] -= flow

    unmet = {market: max(0.0, demand_remaining[market]) for market in markets}
    return allocation, unmet, supplier_max, supplier_remaining


def solve_cmm_supply_chain_optimization(
    optimization_input: dict[str, Any],
) -> dict[str, Any]:
    (
        commodity,
        markets,
        suppliers,
        demand,
        shipping_cost,
        risk_weight,
        unmet_penalty,
        max_supplier_share,
        composition_targets,
        composition_tolerance,
        composition_profiles,
    ) = _normalize_input(optimization_input)

    allocation, unmet, supplier_max, supplier_remaining = _greedy_fallback(
        markets=markets,
        suppliers=suppliers,
        demand=demand,
        shipping_cost=shipping_cost,
        risk_weight=risk_weight,
        max_supplier_share=max_supplier_share,
    )

    _rebalance_for_composition(
        allocation=allocation,
        supplier_remaining=supplier_remaining,
        suppliers=suppliers,
        markets=markets,
        composition_targets=composition_targets,
        composition_profiles=composition_profiles,
        composition_tolerance=composition_tolerance,
    )

    totals_by_supplier = _sum_allocated_by_supplier(allocation, suppliers)
    costs = _compute_costs(
        allocation=allocation,
        suppliers=suppliers,
        shipping_cost=shipping_cost,
        risk_weight=risk_weight,
        unmet=unmet,
        unmet_penalty=unmet_penalty,
    )

    allocation_items: list[dict[str, Any]] = []
    suppliers_by_name = {supplier.name: supplier for supplier in suppliers}
    for (supplier_name, market), amount in sorted(
        allocation.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        supplier = suppliers_by_name[supplier_name]
        unit_total = (
            supplier.unit_cost
            + shipping_cost.get(supplier_name, {}).get(market, 0.0)
            + risk_weight * supplier.risk_score
        )
        allocation_items.append(
            {
                "supplier": supplier_name,
                "market": market,
                "amount": round(amount, 6),
                "unit_total_cost": round(unit_total, 6),
            }
        )

    demand_residual: dict[str, float] = {}
    for market in markets:
        allocated = sum(
            amount
            for (supplier_name, mkt), amount in allocation.items()
            if mkt == market and supplier_name
        )
        demand_residual[market] = round(allocated + unmet[market] - demand[market], 9)

    supplier_capacity_residual: dict[str, float] = {}
    supplier_share_residual: dict[str, float] = {}
    for supplier in suppliers:
        used = totals_by_supplier[supplier.name]
        supplier_capacity_residual[supplier.name] = round(
            supplier.capacity - used,
            9,
        )
        supplier_share_residual[supplier.name] = round(
            supplier_max[supplier.name] - used,
            9,
        )

    unmet_total = sum(unmet.values())
    objective_value = sum(costs.values())

    composition = _compute_composition_metrics(
        totals_by_supplier=totals_by_supplier,
        composition_targets=composition_targets,
        composition_profiles=composition_profiles,
        composition_tolerance=composition_tolerance,
    )
    composition_feasible = True
    composition_residuals: dict[str, float] = {}
    composition_binding: list[str] = []
    if composition is not None:
        composition_feasible = bool(composition["feasible"])
        composition_residuals = {
            component: round(float(residual), 9)
            for component, residual in composition["residuals"].items()
        }
        composition_binding = [
            component
            for component, residual in composition["residuals"].items()
            if abs(float(residual)) >= composition_tolerance - _EPS
        ]

    feasible = unmet_total <= _EPS and composition_feasible
    if unmet_total > _EPS and not composition_feasible:
        status = "infeasible_unmet_and_composition"
    elif unmet_total > _EPS:
        status = "infeasible_unmet_demand"
    elif not composition_feasible:
        status = "infeasible_composition_constraints"
    else:
        status = "optimal_greedy"

    active_capacity = [
        supplier.name
        for supplier in suppliers
        if abs(supplier_capacity_residual[supplier.name]) <= 1e-6
    ]
    bottleneck_markets = [market for market in markets if unmet[market] > _EPS]
    allocated_total = sum(totals_by_supplier.values())
    avg_unit = objective_value / allocated_total if allocated_total else 0.0

    composition_output = None
    if composition is not None:
        composition_output = {
            "targets": {
                component: round(float(target), 9)
                for component, target in composition["targets"].items()
            },
            "actual": {
                component: round(float(actual), 9)
                for component, actual in composition["actual"].items()
            },
            "residuals": composition_residuals,
            "tolerance": round(float(composition["tolerance"]), 9),
            "feasible": composition_feasible,
        }

    return {
        "commodity": commodity,
        "status": status,
        "feasible": feasible,
        "objective_value": round(objective_value, 6),
        "objective_breakdown": {k: round(v, 6) for k, v in costs.items()},
        "allocations": allocation_items,
        "unmet_demand": {k: round(v, 6) for k, v in unmet.items()},
        "constraint_residuals": {
            "demand_balance": demand_residual,
            "supplier_capacity": supplier_capacity_residual,
            "supplier_share_cap": supplier_share_residual,
            "composition": composition_residuals,
        },
        "composition": composition_output,
        "sensitivity_summary": {
            "active_capacity_constraints": active_capacity,
            "bottleneck_markets": bottleneck_markets,
            "average_unit_cost": round(avg_unit, 6),
            "unmet_demand_total": round(unmet_total, 6),
            "composition_binding_components": sorted(composition_binding),
            "composition_feasible": composition_feasible,
        },
    }


@tool
def run_cmm_supply_chain_optimization(
    optimization_input: dict[str, Any],
) -> dict[str, Any]:
    """Run a deterministic CMM supply allocation optimization.

    Expected optimization_input schema:
    - commodity: str
    - demand: mapping market -> required quantity
    - suppliers: list of {name, capacity, unit_cost, risk_score}
      - optional per-supplier composition_profile: mapping component -> fraction
    - shipping_cost: optional mapping supplier -> market -> cost
    - risk_weight: optional float
    - unmet_demand_penalty: optional float
    - max_supplier_share: optional float in [0, 1]
    - composition_targets: optional mapping component -> target fraction
    - composition_tolerance: optional fraction tolerance in [0, 1]
    """
    return solve_cmm_supply_chain_optimization(optimization_input)
