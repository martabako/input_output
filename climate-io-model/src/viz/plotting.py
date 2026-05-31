"""
Visualization helpers for the Czech IO electrification analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

FAKTA_BLUE = "#3B6DB3"
FAKTA_GREEN = "#2EAC76"
FAKTA_RED = "#E05A4B"
FAKTA_GREY = "#7F8C8D"
FAKTA_ORANGE = "#E87A2E"


def _short(name: str, max_len: int = 28) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


# ---------------------------------------------------------------------------
# 1. Czech energy trends (OWID data)
# ---------------------------------------------------------------------------

def plot_czech_energy_trends(df: pd.DataFrame, start_year: int = 2000) -> plt.Figure:
    """
    Four-panel chart of Czech primary energy trends from OWID data.
    Shows oil, coal, electricity demand, and renewables share.
    """
    df = df[df["year"] >= start_year].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle("Czech Energy Landscape — OWID Data", fontsize=14, fontweight="bold", y=1.01)

    panels = [
        ("oil_consumption",        "Oil Consumption (TWh)",            FAKTA_RED,    axes[0, 0]),
        ("coal_consumption",       "Coal Consumption (TWh)",           FAKTA_GREY,   axes[0, 1]),
        ("electricity_demand",     "Electricity Demand (TWh)",         FAKTA_BLUE,   axes[1, 0]),
        ("renewables_share_energy","Renewables Share of Energy (%)",   FAKTA_GREEN,  axes[1, 1]),
    ]

    for col, label, color, ax in panels:
        if col in df.columns:
            data = df.dropna(subset=[col])
            ax.plot(data["year"], data[col], color=color, linewidth=2)
            ax.fill_between(data["year"], data[col], alpha=0.15, color=color)
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Year", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.tick_params(labelsize=8)
            # Annotate last value
            last = data.iloc[-1]
            ax.annotate(f"{last[col]:.0f}", xy=(last["year"], last[col]),
                        fontsize=8, color=color, fontweight="bold",
                        xytext=(3, 3), textcoords="offset points")
        else:
            ax.set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Sector backward-linkage multipliers
# ---------------------------------------------------------------------------

def plot_multipliers(multipliers: pd.Series, title: str = "Backward Linkage Index (Rasmussen)") -> plt.Figure:
    """
    Horizontal bar chart of Rasmussen backward linkage indices.
    Bars > 1 (dashed line) are key upstream-demand sectors.
    """
    data = multipliers.sort_values()
    colors = [FAKTA_BLUE if v >= 1.0 else FAKTA_GREY for v in data.values]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh([_short(s) for s in data.index], data.values, color=colors, edgecolor="none")
    ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6, label="Average (= 1.0)")
    ax.set_xlabel("Backward Linkage Index", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, data.values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Sector impact chart (main result)
# ---------------------------------------------------------------------------

def plot_sector_impacts(
    delta_x: pd.Series,
    title: str = "Sector Output Change — Transport Electrification Scenario",
    pct: bool = False,
) -> plt.Figure:
    """
    Horizontal diverging bar chart showing sector gains (green) and losses (red).

    Parameters
    ----------
    delta_x : pd.Series
        Output change per sector (billion CZK absolute, or % if pct=True).
    pct : bool
        If True, label axis as percentage change.
    """
    data = delta_x.sort_values()
    colors = [FAKTA_GREEN if v >= 0 else FAKTA_RED for v in data.values]
    unit = "%" if pct else "billion CZK"

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([_short(s) for s in data.index], data.values, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel(f"Change in Gross Output ({unit})", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, data.values):
        xpos = val + (0.2 if val >= 0 else -0.2)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center", ha=ha, fontsize=8)

    gain_patch = mpatches.Patch(color=FAKTA_GREEN, label="Gains")
    loss_patch = mpatches.Patch(color=FAKTA_RED, label="Losses")
    ax.legend(handles=[gain_patch, loss_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Leontief heatmap
# ---------------------------------------------------------------------------

def plot_leontief_heatmap(model, top_n: int = 12) -> plt.Figure:
    """
    Heatmap of the Leontief inverse for the top `top_n` most-connected sectors
    (sorted by column sum = output multiplier).
    """
    L_df = model.leontief_df()
    top = model.output_multipliers().nlargest(top_n).index.tolist()
    sub = L_df.loc[top, top]

    short_names = [_short(s, 20) for s in top]
    sub.index = short_names
    sub.columns = short_names

    size = max(10, top_n * 0.75)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        sub, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
        linewidths=0.4, annot_kws={"size": max(5, 9 - top_n // 4)},
        cbar_kws={"label": "L[i,j] — units of i needed per unit of j final demand"},
    )
    ax.set_title(
        f"Leontief Inverse — Top {top_n} Sectors by Output Multiplier",
        fontsize=11, fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Scenario comparison (multiple ev_share values)
# ---------------------------------------------------------------------------

def plot_scenario_sweep(results: dict, sectors_of_interest: list) -> plt.Figure:
    """
    Line chart showing how output changes for selected sectors as ev_share varies.

    Parameters
    ----------
    results : dict  {ev_share (float) -> delta_x (pd.Series)}
    sectors_of_interest : list of sector name strings
    """
    shares = sorted(results.keys())
    palette = [FAKTA_RED, FAKTA_GREY, FAKTA_ORANGE, FAKTA_GREEN, FAKTA_BLUE,
               "#8E44AD", "#1ABC9C", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, sector in enumerate(sectors_of_interest):
        vals = [results[s][sector] for s in shares]
        color = palette[i % len(palette)]
        ax.plot([s * 100 for s in shares], vals, marker="o", label=_short(sector, 22),
                color=color, linewidth=2)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("EV Fleet Share (%)", fontsize=10)
    ax.set_ylabel("Output Change (billion CZK)", fontsize=10)
    ax.set_title("Sector Sensitivity to EV Penetration", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Passive vs. captured transition comparison
# ---------------------------------------------------------------------------

def plot_transition_comparison(
    delta_x_passive: "pd.Series",
    delta_x_captured: "pd.Series",
    title: str = "Passive vs. Captured Transition — Sector Output Change (billion CZK)",
) -> "plt.Figure":
    """
    Side-by-side horizontal bars comparing the passive (baseline) and
    captured (policy-active) electrification scenarios.
    """
    import pandas as pd

    sectors = delta_x_passive.index.tolist()
    order = (delta_x_captured - delta_x_passive).sort_values().index.tolist()

    passive  = delta_x_passive.loc[order]
    captured = delta_x_captured.loc[order]

    y = range(len(order))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11, 8))

    for i, (s, pv, cv) in enumerate(zip(order, passive.values, captured.values)):
        ax.barh(i - height / 2, pv, height=height,
                color=FAKTA_RED if pv < 0 else FAKTA_GREEN, alpha=0.75, label="_")
        ax.barh(i + height / 2, cv, height=height,
                color="#1A5C8A" if cv < 0 else "#0D6E3A", alpha=0.90, label="_")
        ax.text(max(pv, cv) + 0.4, i, f"{cv - pv:+.1f}", va="center", fontsize=7.5,
                color="#0D6E3A" if cv > pv else FAKTA_RED, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([_short(s, 30) for s in order], fontsize=8.5)
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Change in Gross Output (billion CZK)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    passive_patch  = mpatches.Patch(color=FAKTA_GREEN,  alpha=0.75, label="Passive transition")
    captured_patch = mpatches.Patch(color="#0D6E3A", alpha=0.90, label="Captured transition (with policy levers)")
    ax.legend(handles=[passive_patch, captured_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    return fig


def plot_lever_contributions(contributions: dict, levers_meta: dict) -> "plt.Figure":
    """
    Stacked horizontal bar showing each lever's incremental economy-wide output gain.
    contributions: {lever_key -> pd.Series}  (from lever_contributions())
    levers_meta:   LEVERS dict from scenarios.py
    """
    import pandas as pd

    totals = {k: v.sum() for k, v in contributions.items()}
    labels = [levers_meta[k]["label"] for k in totals]
    values = list(totals.values())
    colors = [FAKTA_BLUE, FAKTA_GREEN, FAKTA_ORANGE, "#8E44AD"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors[:len(values)], edgecolor="none", height=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Incremental Economy-wide Output Gain (billion CZK)", fontsize=10)
    ax.set_title("How Much Does Each Policy Lever Add to Czech GDP?", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"+{val:.1f} bCZK", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig
