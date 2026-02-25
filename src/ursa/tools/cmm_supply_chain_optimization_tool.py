from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ursa.tools._lp_solver import LPResult, scipy_available, solve_lp

_log = logging.getLogger(__name__)

NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitFraction = Annotated[float, Field(ge=0.0, le=1.0)]

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------


class SupplierInput(BaseModel):
    """Validated input for a single supplier."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    capacity: NonNegativeFloat
    unit_cost: NonNegativeFloat
    risk_score: NonNegativeFloat = 0.0
    composition_profile: dict[str, UnitFraction] | None = None

    @field_validator("composition_profile", mode="before")
    @classmethod
    def _normalize_composition_keys(
        cls,
        v: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if v is None:
            return None
        return {_normalize_component_name(k): val for k, val in v.items()}


class OptimizationInput(BaseModel):
    """Validated top-level optimization input."""

    model_config = ConfigDict(extra="forbid")

    commodity: str = "CMM"
    demand: dict[str, float]
    suppliers: list[SupplierInput]
    shipping_cost: dict[str, dict[str, float]] | None = None
    risk_weight: NonNegativeFloat = 0.0
    unmet_demand_penalty: Annotated[float, Field(ge=1.0)] = 10000.0
    max_supplier_share: UnitFraction = 1.0
    composition_targets: dict[str, UnitFraction] | None = None
    composition_tolerance: UnitFraction = 0.0
    solver_backend: str | None = None

    @field_validator("demand", mode="after")
    @classmethod
    def _demand_non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            msg = "demand must be a non-empty mapping"
            raise ValueError(msg)
        return v

    @field_validator("suppliers", mode="after")
    @classmethod
    def _suppliers_non_empty(
        cls, v: list[SupplierInput]
    ) -> list[SupplierInput]:
        if not v:
            msg = "suppliers must be a non-empty list"
            raise ValueError(msg)
        return v

    @field_validator("composition_targets", mode="before")
    @classmethod
    def _normalize_target_keys(
        cls,
        v: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if v is None:
            return None
        return {_normalize_component_name(k): val for k, val in v.items()}

    @model_validator(mode="after")
    def _auto_name_suppliers(self) -> OptimizationInput:
        for idx, supplier in enumerate(self.suppliers):
            if supplier.name is None:
                supplier.name = f"supplier_{idx + 1}"
        return self


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class AllocationItem(BaseModel):
    """A single supplier-to-market allocation."""

    model_config = ConfigDict(extra="forbid")

    supplier: str
    market: str
    amount: float
    unit_total_cost: float


class ObjectiveBreakdown(BaseModel):
    """Cost breakdown of the objective function."""

    model_config = ConfigDict(extra="forbid")

    procurement: float
    shipping: float
    risk_penalty: float
    unmet_penalty: float


class CompositionResult(BaseModel):
    """Composition constraint evaluation results."""

    model_config = ConfigDict(extra="forbid")

    targets: dict[str, float]
    actual: dict[str, float]
    residuals: dict[str, float]
    tolerance: float
    feasible: bool


class ConstraintResiduals(BaseModel):
    """Residuals for all constraint groups."""

    model_config = ConfigDict(extra="forbid")

    demand_balance: dict[str, float]
    supplier_capacity: dict[str, float]
    supplier_share_cap: dict[str, float]
    composition: dict[str, float]


class SensitivitySummary(BaseModel):
    """Summary of binding constraints and bottlenecks."""

    model_config = ConfigDict(extra="forbid")

    active_capacity_constraints: list[str]
    bottleneck_markets: list[str]
    average_unit_cost: float
    unmet_demand_total: float
    composition_binding_components: list[str]
    composition_feasible: bool


class ShadowPrices(BaseModel):
    """Dual values from LP solver indicating marginal costs."""

    model_config = ConfigDict(extra="forbid")

    demand_balance: dict[str, float]
    supplier_capacity: dict[str, float]
    supplier_share_cap: dict[str, float]
    composition: dict[str, float]


class OptimizationOutput(BaseModel):
    """Validated optimization result."""

    model_config = ConfigDict(extra="forbid")

    commodity: str
    status: str
    feasible: bool
    objective_value: float
    objective_breakdown: ObjectiveBreakdown
    allocations: list[AllocationItem]
    unmet_demand: dict[str, float]
    constraint_residuals: ConstraintResiduals
    composition: CompositionResult | None
    sensitivity_summary: SensitivitySummary
    shadow_prices: ShadowPrices | None = None


# ---------------------------------------------------------------------------
# Validation error response
# ---------------------------------------------------------------------------


class ValidationErrorDetail(BaseModel):
    """A single validation error."""

    loc: list[str | int]
    msg: str
    type: str


class OptimizationErrorResponse(BaseModel):
    """Structured error response matching output schema conventions."""

    commodity: str = "CMM"
    status: str = "validation_error"
    feasible: bool = False
    errors: list[ValidationErrorDetail]


@dataclass(frozen=True)
class _Supplier:
    name: str
    capacity: float
    unit_cost: float
    risk_score: float


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
                        -composition_profiles[supplier.name].get(
                            component, 0.0
                        ),
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
                        -composition_profiles[supplier.name].get(
                            component, 0.0
                        ),
                        supplier.name,
                    ),
                )

            for donor in donor_candidates:
                donor_name = donor.name
                donor_amount = totals.get(donor_name, 0.0)
                if donor_amount <= _EPS:
                    continue

                donor_profile = composition_profiles[donor_name].get(
                    component, 0.0
                )
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


def _prepare_solver_input(
    inp: OptimizationInput,
) -> tuple[
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
    """Reshape a validated *OptimizationInput* into the tuple the solver expects."""
    commodity = inp.commodity
    markets = sorted(inp.demand.keys())
    demand = dict(inp.demand)

    suppliers: list[_Supplier] = []
    composition_profiles_raw: dict[str, dict[str, float]] = {}
    for supplier_inp in inp.suppliers:
        assert supplier_inp.name is not None  # guaranteed by model_validator
        suppliers.append(
            _Supplier(
                name=supplier_inp.name,
                capacity=supplier_inp.capacity,
                unit_cost=supplier_inp.unit_cost,
                risk_score=supplier_inp.risk_score,
            )
        )
        composition_profiles_raw[supplier_inp.name] = (
            dict(supplier_inp.composition_profile)
            if supplier_inp.composition_profile
            else {}
        )

    suppliers = sorted(suppliers, key=lambda supplier: supplier.name)

    composition_targets: dict[str, float] = (
        dict(inp.composition_targets) if inp.composition_targets else {}
    )

    components: set[str] = set(composition_targets.keys())
    for profile in composition_profiles_raw.values():
        components.update(profile.keys())

    normalized_profiles: dict[str, dict[str, float]] = {}
    for supplier in suppliers:
        profile = composition_profiles_raw.get(supplier.name, {})
        normalized_profiles[supplier.name] = {
            component: profile.get(component, 0.0)
            for component in sorted(components)
        }

    shipping_cost: dict[str, dict[str, float]] = (
        {
            supplier_name: dict(market_costs)
            for supplier_name, market_costs in inp.shipping_cost.items()
        }
        if inp.shipping_cost
        else {}
    )

    return (
        commodity,
        markets,
        suppliers,
        demand,
        shipping_cost,
        inp.risk_weight,
        inp.unmet_demand_penalty,
        inp.max_supplier_share,
        composition_targets,
        inp.composition_tolerance,
        normalized_profiles,
    )


def _validation_error_response(
    payload: dict[str, Any],
    exc: ValidationError,
) -> dict[str, Any]:
    """Convert a *ValidationError* into a dict matching the output schema conventions."""
    commodity = str(payload.get("commodity", "CMM"))
    return OptimizationErrorResponse(
        commodity=commodity,
        errors=[
            ValidationErrorDetail(
                loc=[str(x) for x in e["loc"]],
                msg=e["msg"],
                type=e["type"],
            )
            for e in exc.errors()
        ],
    ).model_dump()


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


def _build_output(
    *,
    commodity: str,
    allocation: dict[tuple[str, str], float],
    unmet: dict[str, float],
    suppliers: list[_Supplier],
    markets: list[str],
    demand: dict[str, float],
    shipping_cost: dict[str, dict[str, float]],
    risk_weight: float,
    unmet_penalty: float,
    supplier_max: dict[str, float],
    composition_targets: dict[str, float],
    composition_profiles: dict[str, dict[str, float]],
    composition_tolerance: float,
    shadow_prices: dict[str, Any] | None = None,
    status_override: str | None = None,
) -> dict[str, Any]:
    """Assemble the result dict and validate it through *OptimizationOutput*."""
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
        allocation_items.append({
            "supplier": supplier_name,
            "market": market,
            "amount": round(amount, 6),
            "unit_total_cost": round(unit_total, 6),
        })

    demand_residual: dict[str, float] = {}
    for market in markets:
        allocated = sum(
            amount
            for (supplier_name, mkt), amount in allocation.items()
            if mkt == market and supplier_name
        )
        demand_residual[market] = round(
            allocated + unmet[market] - demand[market], 9
        )

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
    if status_override is not None:
        status = status_override
    elif unmet_total > _EPS and not composition_feasible:
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
        composition_output = CompositionResult(
            targets={
                component: round(float(target), 9)
                for component, target in composition["targets"].items()
            },
            actual={
                component: round(float(actual), 9)
                for component, actual in composition["actual"].items()
            },
            residuals=composition_residuals,
            tolerance=round(float(composition["tolerance"]), 9),
            feasible=composition_feasible,
        )

    shadow_prices_model = None
    if shadow_prices is not None:
        shadow_prices_model = ShadowPrices(
            demand_balance=shadow_prices.get(
                "demand_balance",
                {},
            ),
            supplier_capacity=shadow_prices.get(
                "supplier_capacity",
                {},
            ),
            supplier_share_cap=shadow_prices.get(
                "supplier_share_cap",
                {},
            ),
            composition=shadow_prices.get("composition", {}),
        )

    return OptimizationOutput(
        commodity=commodity,
        status=status,
        feasible=feasible,
        objective_value=round(objective_value, 6),
        objective_breakdown=ObjectiveBreakdown(
            **{k: round(v, 6) for k, v in costs.items()},
        ),
        allocations=[AllocationItem(**item) for item in allocation_items],
        unmet_demand={k: round(v, 6) for k, v in unmet.items()},
        constraint_residuals=ConstraintResiduals(
            demand_balance=demand_residual,
            supplier_capacity=supplier_capacity_residual,
            supplier_share_cap=supplier_share_residual,
            composition=composition_residuals,
        ),
        composition=composition_output,
        sensitivity_summary=SensitivitySummary(
            active_capacity_constraints=active_capacity,
            bottleneck_markets=bottleneck_markets,
            average_unit_cost=round(avg_unit, 6),
            unmet_demand_total=round(unmet_total, 6),
            composition_binding_components=sorted(composition_binding),
            composition_feasible=composition_feasible,
        ),
        shadow_prices=shadow_prices_model,
    ).model_dump()


def solve_cmm_supply_chain_optimization(
    optimization_input: dict[str, Any],
) -> dict[str, Any]:
    try:
        inp = OptimizationInput.model_validate(optimization_input)
    except ValidationError as exc:
        return _validation_error_response(optimization_input, exc)

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
    ) = _prepare_solver_input(inp)

    backend = inp.solver_backend
    total_demand = sum(demand.values())

    # --- Try LP solver when appropriate ---
    if backend != "greedy":
        lp_ok = scipy_available()
        if not lp_ok and backend == "lp":
            return OptimizationErrorResponse(
                commodity=commodity,
                errors=[
                    ValidationErrorDetail(
                        loc=["solver_backend"],
                        msg=("LP backend requested but scipy is not installed"),
                        type="import_error",
                    ),
                ],
            ).model_dump()

        if lp_ok:
            lp_result: LPResult = solve_lp(
                suppliers=suppliers,
                markets=markets,
                demand=demand,
                shipping_cost=shipping_cost,
                risk_weight=risk_weight,
                unmet_penalty=unmet_penalty,
                max_supplier_share=max_supplier_share,
                composition_targets=(
                    composition_targets if composition_targets else None
                ),
                composition_tolerance=composition_tolerance,
                composition_profiles=(
                    composition_profiles if composition_profiles else None
                ),
            )

            if lp_result.success:
                # Compute supplier_max for _build_output
                share_cap = max_supplier_share * total_demand
                lp_supplier_max = {
                    sup.name: min(sup.capacity, share_cap) for sup in suppliers
                }

                lp_shadow = {
                    "demand_balance": (lp_result.shadow_prices_demand),
                    "supplier_capacity": (lp_result.shadow_prices_capacity),
                    "supplier_share_cap": (lp_result.shadow_prices_share),
                    "composition": (lp_result.shadow_prices_composition),
                }

                lp_status = "optimal_lp"
                if sum(lp_result.unmet.values()) > _EPS:
                    lp_status = "infeasible_unmet_demand"

                return _build_output(
                    commodity=commodity,
                    allocation=lp_result.allocation,
                    unmet=lp_result.unmet,
                    suppliers=suppliers,
                    markets=markets,
                    demand=demand,
                    shipping_cost=shipping_cost,
                    risk_weight=risk_weight,
                    unmet_penalty=unmet_penalty,
                    supplier_max=lp_supplier_max,
                    composition_targets=composition_targets,
                    composition_profiles=composition_profiles,
                    composition_tolerance=composition_tolerance,
                    shadow_prices=lp_shadow,
                    status_override=lp_status,
                )

            if backend == "lp":
                return OptimizationErrorResponse(
                    commodity=commodity,
                    errors=[
                        ValidationErrorDetail(
                            loc=["solver_backend"],
                            msg=(f"LP solver failed: {lp_result.status}"),
                            type="solver_error",
                        ),
                    ],
                ).model_dump()

            _log.info(
                "LP solver failed (%s), falling back to greedy",
                lp_result.status,
            )

    # --- Greedy fallback ---
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

    return _build_output(
        commodity=commodity,
        allocation=allocation,
        unmet=unmet,
        suppliers=suppliers,
        markets=markets,
        demand=demand,
        shipping_cost=shipping_cost,
        risk_weight=risk_weight,
        unmet_penalty=unmet_penalty,
        supplier_max=supplier_max,
        composition_targets=composition_targets,
        composition_profiles=composition_profiles,
        composition_tolerance=composition_tolerance,
    )


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
