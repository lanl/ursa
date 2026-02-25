"""Streamlit dashboard for CMM supply-chain optimization scenarios.

Launch with::

    uv run streamlit run scripts/cmm_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Ensure project src is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ursa.tools.cmm_supply_chain_optimization_tool import (  # noqa: E402
    solve_cmm_supply_chain_optimization,
)

# ---------------------------------------------------------------------------
# Constants (mirrored from notebook)
# ---------------------------------------------------------------------------

COLORS: dict[str, str] = {
    "domestic_recycled_blend": "#00d4aa",
    "allied_separated_oxide": "#636efa",
    "integrated_allied_metal": "#ffa15a",
    "US_DEFENSE": "#ef553b",
    "US_EV": "#ab63fa",
    "EU_OEM": "#19d3f3",
}

SUPPLIER_SHORT: dict[str, str] = {
    "domestic_recycled_blend": "Domestic Recycled",
    "allied_separated_oxide": "Allied Oxide",
    "integrated_allied_metal": "Allied Metal",
}

SCENARIO_LABELS: dict[str, str] = {
    "ndfeb_la_y_5pct_baseline": "Baseline",
    "ndfeb_la_y_5pct_quality_tightening": "Quality Tightening",
    "ndfeb_la_y_5pct_supply_shock": "Supply Shock",
}

SCENARIO_COLORS: dict[str, str] = {
    "Baseline": "#00d4aa",
    "Quality Tightening": "#636efa",
    "Supply Shock": "#ef553b",
}

LAYOUT_DEFAULTS: dict[str, object] = dict(
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(t=60, b=40, l=60, r=40),
)

_SCENARIOS_PATH = _PROJECT_ROOT / "configs" / "cmm_demo_scenarios.json"
_SUPPLIERS_LIST = list(SUPPLIER_SHORT.keys())
_MARKETS_LIST = ["US_DEFENSE", "US_EV", "EU_OEM"]

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading scenario definitions...")
def load_scenarios() -> dict[str, object]:
    """Load the demo scenario configuration JSON."""
    with open(_SCENARIOS_PATH) as fh:
        return json.load(fh)


@st.cache_data(show_spinner="Solving all scenarios...")
def run_all_scenarios(
    _scenarios_json: str,
) -> dict[str, dict[str, object]]:
    """Solve the optimisation for every scenario.

    Parameters
    ----------
    _scenarios_json
        JSON-serialised scenarios dict (used as cache key).

    Returns
    -------
    dict
        ``{scenario_key: result_dict}``
    """
    scenarios: dict[str, object] = json.loads(_scenarios_json)
    results: dict[str, dict[str, object]] = {}
    for key, cfg in scenarios.items():
        results[key] = solve_cmm_supply_chain_optimization(
            cfg["optimization_input"],
        )
    return results


# ---------------------------------------------------------------------------
# Chart builders — each returns a go.Figure (or pd.DataFrame)
# ---------------------------------------------------------------------------


def build_summary_table(
    scenarios: dict[str, object],
    results: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Summary table across all *non-errored* scenarios (notebook cell 28)."""
    rows: list[dict[str, object]] = []
    for key, label in SCENARIO_LABELS.items():
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        inp = scenarios[key]["optimization_input"]
        sens = r["sensitivity_summary"]
        rows.append(
            {
                "Scenario": label,
                "Status": r["status"],
                "Feasible": "\u2705" if r["feasible"] else "\u274c",
                "Objective ($)": f"${r['objective_value']:,.0f}",
                "Avg Unit Cost": f"${sens['average_unit_cost']:.1f}",
                "Unmet (t)": sens["unmet_demand_total"],
                "Comp. Feasible": (
                    "\u2705"
                    if sens["composition_feasible"]
                    else "\u274c"
                ),
                "Tolerance": inp.get("composition_tolerance", "\u2014"),
                "Active Capacity": ", ".join(
                    SUPPLIER_SHORT.get(s, s)
                    for s in sens["active_capacity_constraints"]
                )
                or "None",
            }
        )
    return pd.DataFrame(rows)


def build_sankey(result: dict[str, object]) -> go.Figure:
    """Supplier -> Market allocation Sankey (notebook cell 18)."""
    labels = [SUPPLIER_SHORT.get(s, s) for s in _SUPPLIERS_LIST] + (
        _MARKETS_LIST
    )
    node_colors = [COLORS[s] for s in _SUPPLIERS_LIST] + [
        COLORS[m] for m in _MARKETS_LIST
    ]

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    for alloc in result["allocations"]:
        s_idx = _SUPPLIERS_LIST.index(alloc["supplier"])
        t_idx = len(_SUPPLIERS_LIST) + _MARKETS_LIST.index(
            alloc["market"]
        )
        sources.append(s_idx)
        targets.append(t_idx)
        values.append(alloc["amount"])
        c = COLORS[alloc["supplier"]]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        link_colors.append(f"rgba({r},{g},{b},0.45)")

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=30,
                label=labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )
    fig.update_layout(
        title="Supplier \u2192 Market Flows (tonnes)",
        **LAYOUT_DEFAULTS,
        height=450,
    )
    return fig


def build_cost_waterfall(result: dict[str, object]) -> go.Figure:
    """Objective-function cost waterfall (notebook cell 19)."""
    bd = result["objective_breakdown"]
    cost_items = [
        ("Procurement", bd["procurement"]),
        ("Shipping", bd["shipping"]),
        ("Risk Penalty", bd["risk_penalty"]),
        ("Unmet Penalty", bd["unmet_penalty"]),
    ]

    fig = go.Figure(
        go.Waterfall(
            x=[item[0] for item in cost_items] + ["Total"],
            y=[item[1] for item in cost_items] + [0],
            measure=["relative"] * len(cost_items) + ["total"],
            text=[f"${v:,.0f}" for _, v in cost_items]
            + [f"${result['objective_value']:,.0f}"],
            textposition="outside",
            connector=dict(
                line=dict(color="rgba(63,63,63,0.6)")
            ),
            increasing=dict(marker=dict(color="#636efa")),
            decreasing=dict(marker=dict(color="#00d4aa")),
            totals=dict(marker=dict(color="#ffa15a")),
        )
    )
    fig.update_layout(
        title="Objective Function Breakdown",
        yaxis_title="Cost ($)",
        **LAYOUT_DEFAULTS,
    )
    return fig


def build_capacity_gauges(
    result: dict[str, object],
    scenario_input: dict[str, object],
) -> go.Figure:
    """Supplier capacity utilisation gauge charts (notebook cell 20)."""
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "indicator"}] * 3],
        subplot_titles=[SUPPLIER_SHORT[s] for s in _SUPPLIERS_LIST],
    )

    for i, supplier_cfg in enumerate(scenario_input["suppliers"]):
        name = supplier_cfg["name"]
        capacity = supplier_cfg["capacity"]
        used = sum(
            a["amount"]
            for a in result["allocations"]
            if a["supplier"] == name
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=used,
                number=dict(suffix=" t"),
                delta=dict(
                    reference=capacity, relative=False, suffix=" t"
                ),
                gauge=dict(
                    axis=dict(range=[0, capacity]),
                    bar=dict(color=COLORS[name]),
                    bgcolor="rgba(50,50,50,0.3)",
                    steps=[
                        dict(
                            range=[0, capacity * 0.7],
                            color="rgba(50,50,50,0.15)",
                        ),
                        dict(
                            range=[capacity * 0.7, capacity * 0.9],
                            color="rgba(255,161,90,0.15)",
                        ),
                        dict(
                            range=[capacity * 0.9, capacity],
                            color="rgba(239,85,59,0.15)",
                        ),
                    ],
                    threshold=dict(
                        line=dict(color="white", width=2),
                        value=capacity,
                    ),
                ),
            ),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title="Supplier Capacity Utilization",
        **LAYOUT_DEFAULTS,
        height=350,
    )
    return fig


def build_composition_bullets(
    result: dict[str, object],
) -> go.Figure | None:
    """Composition feasibility bullet chart (notebook cell 21).

    Returns ``None`` when no composition data is present.
    """
    comp = result.get("composition")
    if not comp:
        return None

    components = list(comp["targets"].keys())
    fig = make_subplots(
        rows=1,
        cols=len(components),
        specs=[[{"type": "indicator"}] * len(components)],
        subplot_titles=[f"{c} Fraction" for c in components],
    )

    for i, component in enumerate(components):
        target = comp["targets"][component]
        actual = comp["actual"][component]
        tol = comp["tolerance"]

        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=actual,
                number=dict(valueformat=".4f"),
                delta=dict(reference=target, valueformat=".4f"),
                gauge=dict(
                    shape="bullet",
                    axis=dict(
                        range=[target - 4 * tol, target + 4 * tol]
                    ),
                    bar=dict(
                        color=(
                            "#00d4aa"
                            if abs(actual - target) <= tol
                            else "#ef553b"
                        )
                    ),
                    steps=[
                        dict(
                            range=[target - tol, target + tol],
                            color="rgba(0,212,170,0.2)",
                        ),
                    ],
                    threshold=dict(
                        line=dict(color="white", width=3),
                        thickness=0.8,
                        value=target,
                    ),
                ),
            ),
            row=1,
            col=i + 1,
        )

    feas_label = "FEASIBLE" if comp["feasible"] else "INFEASIBLE"
    fig.update_layout(
        title=(
            f"Composition Constraints \u2014 {feas_label}"
            f" (tolerance \u00b1{comp['tolerance']})"
        ),
        **LAYOUT_DEFAULTS,
        height=250,
    )
    return fig


def build_multi_scenario_cost_bar(
    results: dict[str, dict[str, object]],
) -> go.Figure:
    """Stacked cost-breakdown bar across scenarios (notebook cell 24)."""
    rows: list[dict[str, object]] = []
    for key, label in SCENARIO_LABELS.items():
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        bd = r["objective_breakdown"]
        for cost_type, value in bd.items():
            rows.append(
                {
                    "Scenario": label,
                    "Cost Component": cost_type.replace("_", " ").title(),
                    "Value": value,
                }
            )

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Scenario",
        y="Value",
        color="Cost Component",
        barmode="stack",
        title="Objective Cost Breakdown \u2014 All Scenarios",
        color_discrete_sequence=[
            "#636efa",
            "#00d4aa",
            "#ffa15a",
            "#ef553b",
        ],
        template="plotly_dark",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, yaxis_title="Cost ($)")
    return fig


def build_allocation_comparison(
    results: dict[str, dict[str, object]],
) -> go.Figure:
    """Faceted stacked allocation bar (notebook cell 25)."""
    rows: list[dict[str, object]] = []
    for key, label in SCENARIO_LABELS.items():
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        for alloc in r["allocations"]:
            rows.append(
                {
                    "Scenario": label,
                    "Supplier": SUPPLIER_SHORT.get(
                        alloc["supplier"], alloc["supplier"]
                    ),
                    "Market": alloc["market"],
                    "Amount": alloc["amount"],
                }
            )

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Market",
        y="Amount",
        color="Supplier",
        facet_col="Scenario",
        barmode="stack",
        title="Allocation by Market & Supplier \u2014 All Scenarios",
        color_discrete_map={
            SUPPLIER_SHORT[k]: v
            for k, v in COLORS.items()
            if k in SUPPLIER_SHORT
        },
        template="plotly_dark",
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS, height=450, yaxis_title="Tonnes"
    )
    return fig


def build_risk_cost_scatter(
    scenarios: dict[str, object],
    results: dict[str, dict[str, object]],
) -> go.Figure:
    """Bubble scatter: risk vs cost by supplier (notebook cell 27)."""
    rows: list[dict[str, object]] = []
    for key, label in SCENARIO_LABELS.items():
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        inp = scenarios[key]["optimization_input"]
        for s in inp["suppliers"]:
            used = sum(
                a["amount"]
                for a in r["allocations"]
                if a["supplier"] == s["name"]
            )
            rows.append(
                {
                    "Scenario": label,
                    "Supplier": SUPPLIER_SHORT.get(
                        s["name"], s["name"]
                    ),
                    "Unit Cost": s["unit_cost"],
                    "Risk Score": s["risk_score"],
                    "Allocated (t)": used,
                    "Capacity": s["capacity"],
                }
            )

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="Unit Cost",
        y="Risk Score",
        size="Allocated (t)",
        color="Supplier",
        symbol="Scenario",
        hover_data=["Capacity", "Allocated (t)"],
        title="Supplier Risk vs. Cost \u2014 Bubble Size = Allocation",
        size_max=50,
        color_discrete_map={
            SUPPLIER_SHORT[k]: v
            for k, v in COLORS.items()
            if k in SUPPLIER_SHORT
        },
        template="plotly_dark",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=500)
    return fig


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="CMM Optimization Dashboard",
        page_icon="\U0001f9f2",
        layout="wide",
    )

    # --- Load & solve -------------------------------------------------
    scenarios = load_scenarios()
    scenarios_json = json.dumps(scenarios)
    results = run_all_scenarios(scenarios_json)

    # --- Sidebar ------------------------------------------------------
    st.sidebar.title("\U0001f9f2 CMM Dashboard")
    scenario_key = st.sidebar.radio(
        "Select Scenario",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda k: SCENARIO_LABELS[k],
    )
    selected_label = SCENARIO_LABELS[scenario_key]

    with st.sidebar.expander("Scenario Parameters", expanded=False):
        st.json(scenarios[scenario_key]["optimization_input"])

    st.sidebar.markdown("---")
    st.sidebar.caption("URSA \u2014 LANL / PNNL")

    # --- Title --------------------------------------------------------
    st.title("Critical Minerals & Materials \u2014 Optimization Dashboard")

    # --- Current result / validation error check ----------------------
    result = results[scenario_key]
    is_error = result.get("status") == "validation_error"

    # --- KPI metrics row ----------------------------------------------
    if is_error:
        st.error(
            f"**{selected_label}**: Validation error \u2014 "
            f"{result.get('errors', [])}"
        )
    else:
        sens = result["sensitivity_summary"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Status",
            result["status"].replace("_", " ").title(),
        )
        c2.metric(
            "Objective",
            f"${result['objective_value']:,.0f}",
        )
        c3.metric(
            "Avg Unit Cost",
            f"${sens['average_unit_cost']:.1f}",
        )
        c4.metric(
            "Unmet Demand",
            f"{sens['unmet_demand_total']:.1f} t",
        )
        c5.metric(
            "Feasible",
            "\u2705 Yes" if result["feasible"] else "\u274c No",
        )

    # --- Tabs ---------------------------------------------------------
    tab_single, tab_multi = st.tabs(
        [
            "\U0001f50d Single Scenario",
            "\U0001f4ca Multi-Scenario Compare",
        ]
    )

    # ===== Single-scenario tab ========================================
    with tab_single:
        if is_error:
            st.warning(
                "Charts unavailable for this scenario due to "
                "validation errors."
            )
        else:
            scenario_input = scenarios[scenario_key][
                "optimization_input"
            ]

            # Row 1: Sankey + Waterfall
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(
                    build_sankey(result),
                    use_container_width=True,
                )
            with col_right:
                st.plotly_chart(
                    build_cost_waterfall(result),
                    use_container_width=True,
                )

            # Row 2: Capacity gauges (full width)
            st.plotly_chart(
                build_capacity_gauges(result, scenario_input),
                use_container_width=True,
            )

            # Row 3: Composition bullets (full width)
            comp_fig = build_composition_bullets(result)
            if comp_fig is not None:
                st.plotly_chart(
                    comp_fig, use_container_width=True
                )

    # ===== Multi-scenario tab =========================================
    with tab_multi:
        # Filter out errored scenarios
        valid_keys = [
            k
            for k in SCENARIO_LABELS
            if results[k].get("status") != "validation_error"
        ]
        if not valid_keys:
            st.warning("No valid scenario results to compare.")
        else:
            # Summary table
            st.subheader("Scenario Summary")
            summary_df = build_summary_table(scenarios, results)
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            # Row 1: Cost bar + Risk scatter
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(
                    build_multi_scenario_cost_bar(results),
                    use_container_width=True,
                )
            with col_right:
                st.plotly_chart(
                    build_risk_cost_scatter(scenarios, results),
                    use_container_width=True,
                )

            # Row 2: Allocation comparison (full width)
            st.plotly_chart(
                build_allocation_comparison(results),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
