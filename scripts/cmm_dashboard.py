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
# Display configuration per scenario file
# ---------------------------------------------------------------------------

_CMM_COLORS: dict[str, str] = {
    "domestic_recycled_blend": "#00d4aa",
    "allied_separated_oxide": "#636efa",
    "integrated_allied_metal": "#ffa15a",
    "US_DEFENSE": "#ef553b",
    "US_EV": "#ab63fa",
    "EU_OEM": "#19d3f3",
}

_CMM_SUPPLIER_SHORT: dict[str, str] = {
    "domestic_recycled_blend": "Domestic Recycled",
    "allied_separated_oxide": "Allied Oxide",
    "integrated_allied_metal": "Allied Metal",
}

_CMM_SCENARIO_LABELS: dict[str, str] = {
    "ndfeb_la_y_5pct_baseline": "Baseline",
    "ndfeb_la_y_5pct_quality_tightening": "Quality Tightening",
    "ndfeb_la_y_5pct_supply_shock": "Supply Shock",
}

_ND_COLORS: dict[str, str] = {
    "China_consolidated": "#ef553b",
    "Lynas_Australia": "#636efa",
    "MP_Materials_USA": "#00d4aa",
    "Neo_Performance_EU": "#ffa15a",
    "Recycled_domestic_USA": "#ab63fa",
    "US_DEFENSE": "#ff6692",
    "US_COMMERCIAL": "#19d3f3",
    "EU_AUTOMOTIVE": "#b6e880",
    "EU_INDUSTRIAL": "#ff97ff",
    "JP_AUTOMOTIVE": "#fecb52",
    "KR_ELECTRONICS": "#00cc96",
}

_ND_SUPPLIER_SHORT: dict[str, str] = {
    "China_consolidated": "China",
    "Lynas_Australia": "Lynas (AU)",
    "MP_Materials_USA": "MP Materials (US)",
    "Neo_Performance_EU": "Neo Perf. (EU)",
    "Recycled_domestic_USA": "Recycled (US)",
}

_ND_SCENARIO_LABELS: dict[str, str] = {
    "nd_preshock_baseline": "Pre-Shock Baseline",
    "nd_post_april_2025": "Post-April 2025",
    "nd_post_december_2025": "Post-December 2025",
}

LAYOUT_DEFAULTS: dict[str, object] = dict(
    template="plotly_dark",
    font=dict(size=14),
    margin=dict(t=60, b=40, l=60, r=40),
)

_CONFIGS_DIR = _PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Display config selection helpers
# ---------------------------------------------------------------------------


def _detect_display_config(
    file_stem: str,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Return (colors, supplier_short, scenario_labels) for a file.

    Falls back to auto-generated mappings when no preset
    matches.
    """
    if "nd_china" in file_stem:
        return _ND_COLORS, _ND_SUPPLIER_SHORT, _ND_SCENARIO_LABELS
    if "cmm_demo" in file_stem:
        return _CMM_COLORS, _CMM_SUPPLIER_SHORT, _CMM_SCENARIO_LABELS
    return {}, {}, {}


def _auto_display_config(
    scenarios: dict[str, object],
    colors: dict[str, str],
    supplier_short: dict[str, str],
    scenario_labels: dict[str, str],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    list[str],
    list[str],
]:
    """Augment display dicts with auto-generated entries.

    Returns ``(colors, supplier_short, scenario_labels,
    suppliers_list, markets_list)``.
    """
    palette = [
        "#636efa",
        "#ef553b",
        "#00d4aa",
        "#ffa15a",
        "#ab63fa",
        "#19d3f3",
        "#ff6692",
        "#b6e880",
        "#ff97ff",
        "#fecb52",
        "#00cc96",
    ]

    # Auto-generate scenario labels for unknown keys
    labels = dict(scenario_labels)
    for key in scenarios:
        if key not in labels:
            labels[key] = key.replace("_", " ").title()

    # Collect all suppliers and markets from all scenarios
    all_suppliers: list[str] = []
    all_markets: list[str] = []
    seen_sup: set[str] = set()
    seen_mkt: set[str] = set()
    for cfg in scenarios.values():
        inp = cfg["optimization_input"]
        for s in inp["suppliers"]:
            name = s["name"]
            if name not in seen_sup:
                all_suppliers.append(name)
                seen_sup.add(name)
        for m in inp["demand"]:
            if m not in seen_mkt:
                all_markets.append(m)
                seen_mkt.add(m)

    all_suppliers.sort()
    all_markets.sort()

    # Auto-fill colors and short names
    c = dict(colors)
    s_short = dict(supplier_short)
    pi = 0
    for name in all_suppliers + all_markets:
        if name not in c:
            c[name] = palette[pi % len(palette)]
            pi += 1
        if name not in s_short and name in seen_sup:
            s_short[name] = name.replace("_", " ")

    return c, s_short, labels, all_suppliers, all_markets


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading scenario definitions...")
def load_scenarios(path_str: str) -> dict[str, object]:
    """Load a scenario configuration JSON."""
    with open(path_str) as fh:
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
    scenario_labels: dict[str, str],
    supplier_short: dict[str, str],
) -> pd.DataFrame:
    """Summary table across all non-errored scenarios."""
    rows: list[dict[str, object]] = []
    for key in scenarios:
        label = scenario_labels.get(key, key)
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        inp = scenarios[key]["optimization_input"]
        sens = r["sensitivity_summary"]
        rows.append({
            "Scenario": label,
            "Status": r["status"],
            "Feasible": ("\u2705" if r["feasible"] else "\u274c"),
            "Objective ($)": (f"${r['objective_value']:,.0f}"),
            "Avg Unit Cost": (f"${sens['average_unit_cost']:.1f}"),
            "Unmet (t)": sens["unmet_demand_total"],
            "Comp. Feasible": (
                "\u2705" if sens["composition_feasible"] else "\u274c"
            ),
            "Tolerance": inp.get("composition_tolerance", "\u2014"),
            "Active Capacity": ", ".join(
                supplier_short.get(s, s)
                for s in sens["active_capacity_constraints"]
            )
            or "None",
        })
    return pd.DataFrame(rows)


def build_sankey(
    result: dict[str, object],
    suppliers_list: list[str],
    markets_list: list[str],
    colors: dict[str, str],
    supplier_short: dict[str, str],
) -> go.Figure:
    """Supplier -> Market allocation Sankey."""
    labels = [supplier_short.get(s, s) for s in suppliers_list] + markets_list
    node_colors = [colors.get(s, "#888") for s in suppliers_list] + [
        colors.get(m, "#888") for m in markets_list
    ]

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []
    for alloc in result["allocations"]:
        sup_name = alloc["supplier"]
        mkt_name = alloc["market"]
        if sup_name not in suppliers_list:
            continue
        if mkt_name not in markets_list:
            continue
        s_idx = suppliers_list.index(sup_name)
        t_idx = len(suppliers_list) + markets_list.index(mkt_name)
        sources.append(s_idx)
        targets.append(t_idx)
        values.append(alloc["amount"])
        c = colors.get(sup_name, "#888888")
        r, g, b = (
            int(c[1:3], 16),
            int(c[3:5], 16),
            int(c[5:7], 16),
        )
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


def build_cost_waterfall(
    result: dict[str, object],
) -> go.Figure:
    """Objective-function cost waterfall."""
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
            connector=dict(line=dict(color="rgba(63,63,63,0.6)")),
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
    colors: dict[str, str],
    supplier_short: dict[str, str],
) -> go.Figure:
    """Supplier capacity utilisation gauge charts."""
    n_suppliers = len(scenario_input["suppliers"])
    fig = make_subplots(
        rows=1,
        cols=n_suppliers,
        specs=[[{"type": "indicator"}] * n_suppliers],
        subplot_titles=[
            supplier_short.get(s["name"], s["name"])
            for s in scenario_input["suppliers"]
        ],
    )

    for i, supplier_cfg in enumerate(scenario_input["suppliers"]):
        name = supplier_cfg["name"]
        capacity = supplier_cfg["capacity"]
        used = sum(
            a["amount"] for a in result["allocations"] if a["supplier"] == name
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=used,
                number=dict(suffix=" t"),
                delta=dict(
                    reference=capacity,
                    relative=False,
                    suffix=" t",
                ),
                gauge=dict(
                    axis=dict(range=[0, capacity]),
                    bar=dict(color=colors.get(name, "#888")),
                    bgcolor="rgba(50,50,50,0.3)",
                    steps=[
                        dict(
                            range=[0, capacity * 0.7],
                            color="rgba(50,50,50,0.15)",
                        ),
                        dict(
                            range=[
                                capacity * 0.7,
                                capacity * 0.9,
                            ],
                            color=("rgba(255,161,90,0.15)"),
                        ),
                        dict(
                            range=[
                                capacity * 0.9,
                                capacity,
                            ],
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
    """Composition feasibility bullet chart.

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
                        range=[
                            target - 4 * tol,
                            target + 4 * tol,
                        ]
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
                            range=[
                                target - tol,
                                target + tol,
                            ],
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
    scenario_labels: dict[str, str],
) -> go.Figure:
    """Stacked cost-breakdown bar across scenarios."""
    rows: list[dict[str, object]] = []
    for key in results:
        label = scenario_labels.get(key, key)
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        bd = r["objective_breakdown"]
        for cost_type, value in bd.items():
            rows.append({
                "Scenario": label,
                "Cost Component": cost_type.replace("_", " ").title(),
                "Value": value,
            })

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Scenario",
        y="Value",
        color="Cost Component",
        barmode="stack",
        title=("Objective Cost Breakdown \u2014 All Scenarios"),
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
    scenario_labels: dict[str, str],
    supplier_short: dict[str, str],
    colors: dict[str, str],
) -> go.Figure:
    """Faceted stacked allocation bar."""
    rows: list[dict[str, object]] = []
    for key in results:
        label = scenario_labels.get(key, key)
        r = results[key]
        if r.get("status") == "validation_error":
            continue
        for alloc in r["allocations"]:
            rows.append({
                "Scenario": label,
                "Supplier": supplier_short.get(
                    alloc["supplier"],
                    alloc["supplier"],
                ),
                "Market": alloc["market"],
                "Amount": alloc["amount"],
            })

    df = pd.DataFrame(rows)
    color_map = {
        supplier_short.get(k, k): v
        for k, v in colors.items()
        if k in supplier_short
    }
    fig = px.bar(
        df,
        x="Market",
        y="Amount",
        color="Supplier",
        facet_col="Scenario",
        barmode="stack",
        title=("Allocation by Market & Supplier \u2014 All Scenarios"),
        color_discrete_map=color_map,
        template="plotly_dark",
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=450,
        yaxis_title="Tonnes",
    )
    return fig


def build_risk_cost_scatter(
    scenarios: dict[str, object],
    results: dict[str, dict[str, object]],
    scenario_labels: dict[str, str],
    supplier_short: dict[str, str],
    colors: dict[str, str],
) -> go.Figure:
    """Bubble scatter: risk vs cost by supplier."""
    rows: list[dict[str, object]] = []
    for key in results:
        label = scenario_labels.get(key, key)
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
            rows.append({
                "Scenario": label,
                "Supplier": supplier_short.get(s["name"], s["name"]),
                "Unit Cost": s["unit_cost"],
                "Risk Score": s["risk_score"],
                "Allocated (t)": used,
                "Capacity": s["capacity"],
            })

    df = pd.DataFrame(rows)
    color_map = {
        supplier_short.get(k, k): v
        for k, v in colors.items()
        if k in supplier_short
    }
    fig = px.scatter(
        df,
        x="Unit Cost",
        y="Risk Score",
        size="Allocated (t)",
        color="Supplier",
        symbol="Scenario",
        hover_data=["Capacity", "Allocated (t)"],
        title=("Supplier Risk vs. Cost \u2014 Bubble Size = Allocation"),
        size_max=50,
        color_discrete_map=color_map,
        template="plotly_dark",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=500)
    return fig


def build_shadow_price_bars(
    result: dict[str, object],
    supplier_short: dict[str, str],
) -> tuple[go.Figure, go.Figure] | None:
    """Build demand + capacity shadow price bar charts.

    Returns ``None`` when no shadow prices are present.
    """
    sp = result.get("shadow_prices")
    if not sp:
        return None

    # Demand shadow prices
    demand_sp = sp.get("demand_balance", {})
    if demand_sp:
        df_d = pd.DataFrame([
            {"Market": k, "Shadow Price ($/t)": v} for k, v in demand_sp.items()
        ])
        fig_d = px.bar(
            df_d,
            x="Market",
            y="Shadow Price ($/t)",
            title="Demand Shadow Prices (Marginal Cost)",
            template="plotly_dark",
            color_discrete_sequence=["#636efa"],
        )
        fig_d.update_layout(**LAYOUT_DEFAULTS)
    else:
        fig_d = go.Figure()
        fig_d.update_layout(
            title="No demand shadow prices",
            **LAYOUT_DEFAULTS,
        )

    # Capacity shadow prices
    cap_sp = sp.get("supplier_capacity", {})
    share_sp = sp.get("supplier_share_cap", {})
    combined_cap: dict[str, float] = {}
    for name in set(list(cap_sp.keys()) + list(share_sp.keys())):
        combined_cap[supplier_short.get(name, name)] = cap_sp.get(
            name, 0.0
        ) + share_sp.get(name, 0.0)

    if combined_cap:
        df_c = pd.DataFrame([
            {"Supplier": k, "Shadow Price ($/t)": v}
            for k, v in combined_cap.items()
        ])
        fig_c = px.bar(
            df_c,
            x="Supplier",
            y="Shadow Price ($/t)",
            title=("Capacity Shadow Prices (Marginal Value of +1t)"),
            template="plotly_dark",
            color_discrete_sequence=["#ef553b"],
        )
        fig_c.update_layout(**LAYOUT_DEFAULTS)
    else:
        fig_c = go.Figure()
        fig_c.update_layout(
            title="No capacity shadow prices",
            **LAYOUT_DEFAULTS,
        )

    return fig_d, fig_c


def build_multi_scenario_shadow_comparison(
    results: dict[str, dict[str, object]],
    scenario_labels: dict[str, str],
    supplier_short: dict[str, str],
) -> go.Figure | None:
    """Bar comparing capacity shadow prices across scenarios."""
    rows: list[dict[str, object]] = []
    any_shadow = False
    for key in results:
        label = scenario_labels.get(key, key)
        r = results[key]
        sp = r.get("shadow_prices")
        if not sp:
            continue
        any_shadow = True
        cap_sp = sp.get("supplier_capacity", {})
        share_sp = sp.get("supplier_share_cap", {})
        all_names = set(list(cap_sp.keys()) + list(share_sp.keys()))
        for name in all_names:
            combined = cap_sp.get(name, 0.0) + share_sp.get(name, 0.0)
            rows.append({
                "Scenario": label,
                "Supplier": supplier_short.get(name, name),
                "Shadow Price ($/t)": abs(combined),
            })

    if not any_shadow:
        return None

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Supplier",
        y="Shadow Price ($/t)",
        color="Scenario",
        barmode="group",
        title=(
            "Capacity Shadow Prices Across Scenarios (|marginal value| of +1t)"
        ),
        template="plotly_dark",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=450)
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

    # --- Scenario file selector ---
    scenario_files = sorted(_CONFIGS_DIR.glob("*_scenarios.json"))
    if not scenario_files:
        st.error(
            "No scenario files found in configs/. "
            "Expected *_scenarios.json files."
        )
        return

    file_labels = {f.stem: f for f in scenario_files}

    st.sidebar.title("\U0001f9f2 CMM Dashboard")
    selected_file_stem = st.sidebar.selectbox(
        "Scenario File",
        options=list(file_labels.keys()),
        format_func=lambda s: s.replace("_", " ").title(),
    )
    selected_path = file_labels[selected_file_stem]

    # --- Load & solve ---
    scenarios = load_scenarios(str(selected_path))
    scenarios_json = json.dumps(scenarios)
    results = run_all_scenarios(scenarios_json)

    # --- Resolve display config ---
    (
        base_colors,
        base_supplier_short,
        base_scenario_labels,
    ) = _detect_display_config(selected_file_stem)

    (
        colors,
        supplier_short,
        scenario_labels,
        suppliers_list,
        markets_list,
    ) = _auto_display_config(
        scenarios,
        base_colors,
        base_supplier_short,
        base_scenario_labels,
    )

    # --- Sidebar scenario selector ---
    scenario_key = st.sidebar.radio(
        "Select Scenario",
        options=list(scenarios.keys()),
        format_func=lambda k: scenario_labels.get(k, k),
    )
    selected_label = scenario_labels.get(scenario_key, scenario_key)

    with st.sidebar.expander("Scenario Parameters", expanded=False):
        st.json(scenarios[scenario_key]["optimization_input"])

    st.sidebar.markdown("---")
    st.sidebar.caption("URSA \u2014 LANL / PNNL")

    # --- Title ---
    st.title("Critical Minerals & Materials \u2014 Optimization Dashboard")

    # --- Current result / validation error check ---
    result = results[scenario_key]
    is_error = result.get("status") == "validation_error"

    # --- KPI metrics row ---
    if is_error:
        st.error(
            f"**{selected_label}**: Validation error"
            f" \u2014 {result.get('errors', [])}"
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

    # --- Tabs ---
    tab_single, tab_multi = st.tabs([
        "\U0001f50d Single Scenario",
        "\U0001f4ca Multi-Scenario Compare",
    ])

    # ===== Single-scenario tab ====================================
    with tab_single:
        if is_error:
            st.warning(
                "Charts unavailable for this scenario due to validation errors."
            )
        else:
            scenario_input = scenarios[scenario_key]["optimization_input"]

            # Row 1: Sankey + Waterfall
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(
                    build_sankey(
                        result,
                        suppliers_list,
                        markets_list,
                        colors,
                        supplier_short,
                    ),
                    use_container_width=True,
                )
            with col_right:
                st.plotly_chart(
                    build_cost_waterfall(result),
                    use_container_width=True,
                )

            # Row 2: Capacity gauges (full width)
            st.plotly_chart(
                build_capacity_gauges(
                    result,
                    scenario_input,
                    colors,
                    supplier_short,
                ),
                use_container_width=True,
            )

            # Row 3: Composition bullets (full width)
            comp_fig = build_composition_bullets(result)
            if comp_fig is not None:
                st.plotly_chart(
                    comp_fig,
                    use_container_width=True,
                )

            # Row 4: Shadow prices (LP only)
            sp_figs = build_shadow_price_bars(
                result,
                supplier_short,
            )
            if sp_figs is not None:
                st.subheader("Shadow Prices (LP Duals)")
                sp_left, sp_right = st.columns(2)
                with sp_left:
                    st.plotly_chart(
                        sp_figs[0],
                        use_container_width=True,
                    )
                with sp_right:
                    st.plotly_chart(
                        sp_figs[1],
                        use_container_width=True,
                    )
                st.caption(
                    "**Demand shadow prices** show the"
                    " marginal cost of supplying one"
                    " additional tonne to each market."
                    " **Capacity shadow prices** show"
                    " the cost reduction from adding"
                    " one tonne of supplier capacity."
                )

    # ===== Multi-scenario tab =====================================
    with tab_multi:
        valid_keys = [
            k
            for k in scenarios
            if results[k].get("status") != "validation_error"
        ]
        if not valid_keys:
            st.warning("No valid scenario results to compare.")
        else:
            # Summary table
            st.subheader("Scenario Summary")
            summary_df = build_summary_table(
                scenarios,
                results,
                scenario_labels,
                supplier_short,
            )
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            # Row 1: Cost bar + Risk scatter
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(
                    build_multi_scenario_cost_bar(
                        results,
                        scenario_labels,
                    ),
                    use_container_width=True,
                )
            with col_right:
                st.plotly_chart(
                    build_risk_cost_scatter(
                        scenarios,
                        results,
                        scenario_labels,
                        supplier_short,
                        colors,
                    ),
                    use_container_width=True,
                )

            # Row 2: Allocation comparison (full width)
            st.plotly_chart(
                build_allocation_comparison(
                    results,
                    scenario_labels,
                    supplier_short,
                    colors,
                ),
                use_container_width=True,
            )

            # Row 3: Shadow price comparison
            shadow_fig = build_multi_scenario_shadow_comparison(
                results,
                scenario_labels,
                supplier_short,
            )
            if shadow_fig is not None:
                st.subheader("Shadow Price Comparison Across Scenarios")
                st.plotly_chart(
                    shadow_fig,
                    use_container_width=True,
                )
                st.caption(
                    "Compares the marginal value of"
                    " additional capacity from each"
                    " supplier across scenarios. Higher"
                    " values indicate tighter capacity"
                    " constraints."
                )


if __name__ == "__main__":
    main()
