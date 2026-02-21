from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool


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


def _normalize_input(payload: dict[str, Any]) -> tuple[
    str,
    list[str],
    list[_Supplier],
    dict[str, float],
    dict[str, dict[str, float]],
    float,
    float,
    float,
]:
    commodity = str(payload.get("commodity", "CMM"))

    demand_raw = payload.get("demand", {})
    if not isinstance(demand_raw, dict) or not demand_raw:
        raise ValueError("optimization_input.demand must be a non-empty mapping")
    markets = sorted(str(k) for k in demand_raw.keys())
    demand = {str(k): _to_float(v, 0.0) for k, v in demand_raw.items()}

    suppliers_raw = payload.get("suppliers", [])
    if not isinstance(suppliers_raw, list) or not suppliers_raw:
        raise ValueError(
            "optimization_input.suppliers must be a non-empty list"
        )

    suppliers: list[_Supplier] = []
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
    suppliers = sorted(suppliers, key=lambda supplier: supplier.name)

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
    unmet_penalty = max(
        1.0, _to_float(payload.get("unmet_demand_penalty"), 10000.0)
    )
    max_supplier_share = _to_float(payload.get("max_supplier_share"), 1.0)
    max_supplier_share = max(0.0, min(1.0, max_supplier_share))

    return (
        commodity,
        markets,
        suppliers,
        demand,
        shipping_cost,
        risk_weight,
        unmet_penalty,
        max_supplier_share,
    )


def _greedy_fallback(
    *,
    markets: list[str],
    suppliers: list[_Supplier],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet_penalty: float,
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

        for _, supplier in candidates:
            if demand_remaining[market] <= 0:
                break
            avail = supplier_remaining[supplier.name]
            if avail <= 0:
                continue
            flow = min(avail, demand_remaining[market])
            if flow <= 0:
                continue
            allocation[(supplier.name, market)] = flow
            supplier_remaining[supplier.name] -= flow
            demand_remaining[market] -= flow

    procurement_cost = 0.0
    shipping_total = 0.0
    risk_total = 0.0
    for (supplier_name, market), amount in allocation.items():
        supplier = next(s for s in suppliers if s.name == supplier_name)
        procurement_cost += amount * supplier.unit_cost
        shipping_total += amount * shipping_cost.get(supplier_name, {}).get(
            market, 0.0
        )
        risk_total += amount * (risk_weight * supplier.risk_score)

    unmet = {market: max(0.0, demand_remaining[market]) for market in markets}
    unmet_cost = sum(unmet.values()) * unmet_penalty
    costs = {
        "procurement": procurement_cost,
        "shipping": shipping_total,
        "risk_penalty": risk_total,
        "unmet_penalty": unmet_cost,
    }

    return allocation, unmet, supplier_max, costs


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
    ) = _normalize_input(optimization_input)

    allocation, unmet, supplier_max, costs = _greedy_fallback(
        markets=markets,
        suppliers=suppliers,
        demand=demand,
        shipping_cost=shipping_cost,
        risk_weight=risk_weight,
        unmet_penalty=unmet_penalty,
        max_supplier_share=max_supplier_share,
    )

    total_allocated_by_supplier: dict[str, float] = {}
    allocation_items: list[dict[str, Any]] = []
    for supplier in suppliers:
        total_allocated_by_supplier[supplier.name] = 0.0

    for (supplier_name, market), amount in sorted(
        allocation.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        supplier = next(s for s in suppliers if s.name == supplier_name)
        total_allocated_by_supplier[supplier_name] += amount
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

    demand_residual = {}
    for market in markets:
        allocated = sum(
            amount
            for (supplier_name, mkt), amount in allocation.items()
            if mkt == market and supplier_name
        )
        demand_residual[market] = round(allocated + unmet[market] - demand[market], 9)

    supplier_capacity_residual = {}
    supplier_share_residual = {}
    for supplier in suppliers:
        used = total_allocated_by_supplier[supplier.name]
        supplier_capacity_residual[supplier.name] = round(
            supplier.capacity - used, 9
        )
        supplier_share_residual[supplier.name] = round(
            supplier_max[supplier.name] - used, 9
        )

    unmet_total = sum(unmet.values())
    feasible = unmet_total <= 1e-9
    objective_value = sum(costs.values())

    active_capacity = [
        supplier.name
        for supplier in suppliers
        if abs(supplier_capacity_residual[supplier.name]) <= 1e-6
    ]
    bottleneck_markets = [market for market in markets if unmet[market] > 1e-9]
    allocated_total = sum(total_allocated_by_supplier.values())
    avg_unit = objective_value / allocated_total if allocated_total else 0.0

    return {
        "commodity": commodity,
        "status": "optimal_greedy" if feasible else "infeasible_unmet_demand",
        "feasible": feasible,
        "objective_value": round(objective_value, 6),
        "objective_breakdown": {k: round(v, 6) for k, v in costs.items()},
        "allocations": allocation_items,
        "unmet_demand": {k: round(v, 6) for k, v in unmet.items()},
        "constraint_residuals": {
            "demand_balance": demand_residual,
            "supplier_capacity": supplier_capacity_residual,
            "supplier_share_cap": supplier_share_residual,
        },
        "sensitivity_summary": {
            "active_capacity_constraints": active_capacity,
            "bottleneck_markets": bottleneck_markets,
            "average_unit_cost": round(avg_unit, 6),
            "unmet_demand_total": round(unmet_total, 6),
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
    - shipping_cost: optional mapping supplier -> market -> cost
    - risk_weight: optional float
    - unmet_demand_penalty: optional float
    - max_supplier_share: optional float in [0, 1]
    """
    return solve_cmm_supply_chain_optimization(optimization_input)

