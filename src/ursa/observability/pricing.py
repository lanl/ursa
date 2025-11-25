from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal, getcontext
from importlib import resources
from typing import Any, Optional, Tuple

getcontext().prec = 28  # robust money math


# ---------- Model pricing schema ----------


@dataclass
class ModelPricing:
    # Prices are USD per 1,000 tokens
    input_per_1k: Decimal
    output_per_1k: Decimal
    reasoning_per_1k: Optional[Decimal] = (
        None  # None --> charge 0 for reasoning tokens
    )
    cached_input_multiplier: Decimal = Decimal(
        "1"
    )  # e.g., 0.25 if your provider discounts cached prompt tokens

    def price_tokens(self, usage: dict[str, Any]) -> dict[str, Decimal]:
        """Compute cost components from a usage dict with keys like input_tokens, output_tokens, reasoning_tokens, cached_tokens."""

        def _to_dec(x) -> Decimal:
            if x is None:
                return Decimal("0")
            try:
                return Decimal(str(x))
            except Exception:
                return Decimal("0")

        in_t = _to_dec(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        out_t = _to_dec(
            usage.get("output_tokens", usage.get("completion_tokens", 0))
        )
        # total_t = _to_dec(usage.get("total_tokens", (in_t + out_t)))
        cached_t = _to_dec(usage.get("cached_tokens", 0))
        reasoning_t = _to_dec(usage.get("reasoning_tokens", 0))

        eff_in = (in_t - cached_t) if in_t > cached_t else Decimal("0")
        cached_eff_in = cached_t

        input_cost = (eff_in / Decimal(1000)) * self.input_per_1k
        cached_input_cost = (
            (cached_eff_in / Decimal(1000))
            * self.input_per_1k
            * self.cached_input_multiplier
        )
        output_cost = (out_t / Decimal(1000)) * self.output_per_1k
        reasoning_cost = Decimal("0")
        if self.reasoning_per_1k is not None and reasoning_t > 0:
            reasoning_cost = (
                reasoning_t / Decimal(1000)
            ) * self.reasoning_per_1k

        total_cost = (
            input_cost + cached_input_cost + output_cost + reasoning_cost
        )
        return {
            "input_cost": input_cost,
            "cached_input_cost": cached_input_cost,
            "output_cost": output_cost,
            "reasoning_cost": reasoning_cost,
            "total_cost": total_cost,
        }


# ---------- Registry & resolution ----------


def _dec(x: str | float | int) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal("0")


# DEFAULTS: keep $0.00 so you don’t accidentally attribute costs.
# Fill in values for your org as needed (USD per 1K tokens).
DEFAULT_REGISTRY: dict[str, ModelPricing] = {
    # Examples — edit to match your negotiated prices:
    # "openai:gpt-5-mini": ModelPricing(_dec("5.00"), _dec("15.00")),
    # "openai:gpt-5-mini": ModelPricing(_dec("2.50"), _dec("10.00"), reasoning_per_1k=_dec("5.00")),
    "openai:gpt-5-mini": ModelPricing(
        _dec("0.00"), _dec("0.00"), reasoning_per_1k=_dec("0.00")
    ),
    "local/*": ModelPricing(_dec("0.00"), _dec("0.00")),
}


def normalize_model_name(name: str) -> str:
    return (name or "").strip().lower()


def resolve_model_name(event: dict[str, Any]) -> str:
    m = (
        ((event.get("metadata") or {}).get("model"))
        or ((event.get("metadata") or {}).get("ls_model_name"))
        or (event.get("name") or "").replace("llm:", "")
    )
    return normalize_model_name(m)


def find_pricing(
    model: str, registry: dict[str, ModelPricing]
) -> Optional[ModelPricing]:
    if model in registry:
        return registry[model]
    # simple wildcard support like "local/*"
    for key, mp in registry.items():
        if key.endswith("/*") and model.startswith(key[:-2]):
            return mp
    # try provider/model normalization like "openai-o3" → "openai:gpt-5-mini"
    model2 = model.replace("-", "/")
    if model2 in registry:
        return registry[model2]
    return None


def default_registry_path() -> str:
    """Path to pricing file shipped with this package"""
    path = resources.files("ursa") / "observability" / "pricing.json"
    return str(path)


def load_registry(
    path: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
    use_default_if_missing: bool = True,
) -> dict[str, ModelPricing]:
    """
    Load pricing registry from:
      1) explicit `path` (if provided), else
      2) $URSA_PRICING_JSON (if set), else
      3) pricing.json next to pricing.py (if present, and use_default_if_missing)
      4) fall back to DEFAULT_REGISTRY
    """
    reg: dict[str, ModelPricing] = dict(DEFAULT_REGISTRY)

    # 1) explicit path from caller wins
    candidate = path

    # 2) else env var
    if not candidate:
        env_path = os.environ.get("URSA_PRICING_JSON")
        if env_path:
            candidate = env_path

    # 3) else module-local pricing.json
    if not candidate and use_default_if_missing:
        local_path = default_registry_path()
        if os.path.exists(local_path):
            candidate = local_path

    # Load if we have a candidate
    if candidate and os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in (data or {}).items():
            # Ignore non-model notes like "_note"
            if not isinstance(v, dict) or (
                "input_per_1k" not in v and "output_per_1k" not in v
            ):
                continue
            reg[normalize_model_name(k)] = ModelPricing(
                _dec(v.get("input_per_1k", 0)),
                _dec(v.get("output_per_1k", 0)),
                _dec(v["reasoning_per_1k"])
                if v.get("reasoning_per_1k") is not None
                else None,
                _dec(v.get("cached_input_multiplier", 1)),
            )

    # Apply programmatic overrides last
    if overrides:
        for k, v in overrides.items():
            reg[normalize_model_name(k)] = ModelPricing(
                _dec(v.get("input_per_1k", 0)),
                _dec(v.get("output_per_1k", 0)),
                _dec(v["reasoning_per_1k"])
                if v.get("reasoning_per_1k") is not None
                else None,
                _dec(v.get("cached_input_multiplier", 1)),
            )

    return reg


# ---------- Core pricing application ----------


def _has_provider_cost(roll: dict[str, Any]) -> bool:
    # Treat nonzero provider totals as authoritative
    try:
        return any([
            float(roll.get("total_cost", 0) or 0) > 0,
            float(roll.get("input_cost", 0) or 0) > 0,
            float(roll.get("output_cost", 0) or 0) > 0,
        ])
    except Exception:
        return False


def _round_money(x: Decimal) -> float:
    return float(x.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def price_event(
    event: dict[str, Any],
    registry: dict[str, ModelPricing],
    overwrite: bool = False,
) -> Tuple[dict[str, Any], Optional[Decimal], str]:
    """
    Returns (event, total_cost_decimal_or_None, cost_source)
      cost_source ∈ {"provider", "computed", "no_usage", "no_pricing"}
    """
    metrics = event.get("metrics") or {}
    roll = metrics.get("usage_rollup") or {}
    if not roll:
        return (event, None, "no_usage")

    if _has_provider_cost(roll) and not overwrite:
        # Respect provider-reported cost
        return (event, Decimal(str(roll.get("total_cost", 0) or 0)), "provider")

    model = resolve_model_name(event)
    mp = find_pricing(model, registry)
    if not mp:
        return (event, None, "no_pricing")

    # Compute costs from tokens
    costs = mp.price_tokens(roll)

    # Populate rollup fields (only fill or overwrite if asked)
    roll = dict(roll)  # copy to avoid mutating caller unexpectedly
    for key in ("input_cost", "output_cost", "total_cost"):
        if overwrite or not roll.get(key):
            roll[key] = _round_money(costs[key])
    # Optional: attach granular breakdown so you can inspect later
    metrics["cost_details"] = {
        "source": "computed",
        "model_resolved": model,
        "pricing_used": asdict(mp),
        "components_usd": {k: _round_money(v) for k, v in costs.items()},
    }
    metrics["cost_source"] = "computed"
    event["metrics"] = metrics
    event["metrics"]["usage_rollup"] = roll
    return (event, costs["total_cost"], "computed")


def price_payload(
    payload: dict[str, Any],
    registry: Optional[dict[str, ModelPricing]] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Enriches payload in-place with computed costs where missing.
    Adds a `costs` block with totals and by-model aggregation.
    """
    reg = registry or load_registry()
    llm_events = payload.get("llm_events") or []
    total = Decimal("0")
    by_model: dict[str, Decimal] = {}
    sources = {"provider": 0, "computed": 0, "no_usage": 0, "no_pricing": 0}

    for ev in llm_events:
        ev2, cost_dec, src = price_event(ev, reg, overwrite=overwrite)
        sources[src] = sources.get(src, 0) + 1
        model = resolve_model_name(ev2)
        if cost_dec is not None:
            total += cost_dec
            by_model[model] = by_model.get(model, Decimal("0")) + cost_dec

    payload.setdefault("costs", {})
    payload["costs"]["total_usd"] = _round_money(total)
    payload["costs"]["by_model_usd"] = {
        k: _round_money(v) for k, v in by_model.items()
    }
    payload["costs"]["event_sources"] = sources
    payload["costs"]["registry_note"] = (
        "Edit pricing via DEFAULT_REGISTRY, pricing.json, or overrides."
    )

    return payload


# ---------- Convenience file I/O ----------


def price_file(
    in_path: str,
    out_path: Optional[str] = None,
    registry_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Reads a metrics JSON file (from timing.py), enriches with costs, writes result.
    If out_path is None, writes alongside input as '<name>.priced.json'.
    Returns output path.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    reg = load_registry(path=registry_path)
    payload = price_payload(payload, registry=reg, overwrite=overwrite)

    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}.priced.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path
