"""
Transport electrification scenario for the Czech economy.

The scenario models EV penetration reaching `ev_share` of the total Czech
vehicle fleet by a target year (default: 40% by ~2030-35, consistent with
Czech NEK targets).

Two types of shocks are applied:

1. Technology shock (delta_A) — changes in how sectors source their inputs:
   - Transport & Storage: buys less road fuel, more electricity per unit of output.
   - Motor Vehicles: battery packs shift the input mix toward Electronics and
     away from pure-ICE metals and rubber.

2. Final demand shock (delta_f) — changes in what households buy:
   - Households spend less on motor fuel (Oil & Fuel Refining).
   - Households spend more on home/public charging (Electricity & Heat).
   - A share of fuel savings flows into new EV purchases (Motor Vehicles).

Calibration notes
-----------------
* Czech road transport consumes ~67 TWh of oil equivalent per year (62%
  of ~108 TWh total oil consumption). Household motor fuel ≈ 27% of the
  domestic oil sector's gross output. This fraction is applied directly
  to model.x[Oil] so the final demand shock scales consistently whether
  the model uses the synthetic A matrix (~220 bCZK) or the Eurostat DOM
  IOT (~26 bCZK domestic refinery output).
* EVs are ~4× more energy-efficient: equal distance covered with ¼ the
  energy, but at electricity rather than fuel price.
* The household fuel switch is scaled by ev_share × (model.x[Oil] × 0.27).
"""

import numpy as np
from src.data.czech_sectors import IDX


def build_electrification_scenario(
    model,
    ev_share: float = 0.40,
) -> tuple:
    """
    Build delta_A and delta_f for a transport electrification scenario.

    Parameters
    ----------
    model : LeontiefModel
    ev_share : float
        Share of the Czech vehicle fleet assumed to be electric (0–1).
        0.40 corresponds to ~2030-35 ambition.

    Returns
    -------
    delta_A : np.ndarray (n, n)
    delta_f : np.ndarray (n,)
    """
    n = model.n
    delta_A = np.zeros((n, n))
    delta_f = np.zeros(n)

    T = IDX["Transport & Storage"]
    E = IDX["Electricity & Heat"]
    O = IDX["Oil & Fuel Refining"]
    V = IDX["Motor Vehicles"]
    X = IDX["Electronics & Electrical Equip"]
    M = IDX["Basic & Fabricated Metals"]
    R = IDX["Rubber, Plastics & Minerals"]

    # ----------------------------------------------------------------
    # 1. Technology shock: Transport sector input mix
    # ----------------------------------------------------------------
    # Road fuel coefficient in transport: currently 0.15
    # EVs use ~4x less energy; electricity unit cost ~0.4x that of diesel.
    # Net effect per km: energy cost ≈ 0.1x of ICE fuel cost.
    # Weighted by ev_share of fleet:
    #   new_fuel_coeff  = old × (1 − ev_share × 0.85)
    #   new_elec_coeff  = old + old_fuel × ev_share × 0.22
    fuel_coeff = model.A[O, T]
    elec_coeff = model.A[E, T]
    delta_A[O, T] = -fuel_coeff * ev_share * 0.85
    delta_A[E, T] = +fuel_coeff * ev_share * 0.22

    # ----------------------------------------------------------------
    # 2. Technology shock: Motor Vehicles sector input mix
    # ----------------------------------------------------------------
    # Battery packs raise electronics content by ~8 pp of vehicle output value
    # at full EV share; scaled by ev_share.
    # Simpler ICE (no gearbox, fewer metal parts) reduces metals slightly.
    delta_A[X, V] = +0.08 * ev_share
    delta_A[M, V] = -0.02 * ev_share
    delta_A[R, V] = -0.01 * ev_share   # fewer ICE-specific rubber seals

    # ----------------------------------------------------------------
    # 3. Final demand shock: household spending reallocation
    # ----------------------------------------------------------------
    # Household motor fuel spend, scaled to the model's domestic Oil sector size.
    # Real Czech data: household road fuel ≈ 27% of domestic refinery output.
    # This ensures the shock is proportional regardless of whether the model
    # uses the synthetic A matrix (x[Oil]~220 bCZK) or real Eurostat IOT
    # (x[Oil]~26 bCZK, DOM flows only) — giving ~60 or ~7 bCZK respectively.
    household_fuel = model.x[O] * 0.27
    switched = household_fuel * ev_share

    delta_f[O] = -switched                     # less petrol/diesel at pump
    delta_f[E] = +switched * 0.25              # home/public charging (cheaper per km)
    delta_f[V] = +switched * 0.10              # partial reinvestment in EV purchases

    return delta_A, delta_f


LEVERS = {
    "gigafactory": {
        "label": "Battery gigafactory",
        "label_cs": "Domácí výroba baterií (gigafactory)",
        "description": (
            "CZ attracts a battery cell manufacturer (e.g. CATL/Samsung SDI scale). "
            "Adds ~30 bCZK to domestic Electronics output for export + domestic supply."
        ),
    },
    "skoda_pivot": {
        "label": "Škoda full EV pivot",
        "label_cs": "Plný přechod Škody na elektromobily v ČR",
        "description": (
            "VW Group commits EV production to Czech plants. "
            "Adds ~20 bCZK Motor Vehicles final demand and deepens domestic electronics sourcing."
        ),
    },
    "renewables_buildout": {
        "label": "Renewable energy buildout",
        "label_cs": "Výstavba obnovitelných zdrojů energie",
        "description": (
            "Accelerated solar/wind deployment. Adds ~15 bCZK Construction (installation) "
            "and +5 bCZK Electricity investment demand."
        ),
    },
    "charging_network": {
        "label": "Public charging network",
        "label_cs": "Veřejná nabíjecí infrastruktura",
        "description": (
            "State-funded EV charging rollout. Adds ~8 bCZK Construction + "
            "~3 bCZK Electronics (charging hardware)."
        ),
    },
}


def build_lever_scenario(model, ev_share: float = 0.40, levers: list = None) -> tuple:
    """
    'Captured transition' scenario: baseline electrification + selected policy levers.

    Each lever represents a concrete Czech policy action that redirects transition
    gains into the domestic economy.

    Parameters
    ----------
    model : LeontiefModel
    ev_share : float
    levers : list of str, subset of LEVERS keys.
        Default: all four levers active.

    Returns
    -------
    delta_A, delta_f — same format as build_electrification_scenario.
    """
    if levers is None:
        levers = list(LEVERS.keys())

    delta_A, delta_f = build_electrification_scenario(model, ev_share)

    X = IDX["Electronics & Electrical Equip"]
    V = IDX["Motor Vehicles"]
    C = IDX["Construction"]
    E = IDX["Electricity & Heat"]

    if "gigafactory" in levers:
        delta_f[X] += 30.0

    if "skoda_pivot" in levers:
        delta_f[V] += 20.0
        delta_A[X, V] += 0.03  # more domestic electronics sourced per vehicle

    if "renewables_buildout" in levers:
        delta_f[C] += 15.0
        delta_f[E] += 5.0

    if "charging_network" in levers:
        delta_f[C] += 8.0
        delta_f[X] += 3.0

    return delta_A, delta_f


def lever_contributions(model, ev_share: float = 0.40) -> dict:
    """
    Compute the incremental output impact of each lever individually.

    Returns
    -------
    dict: {lever_key -> pd.Series of output changes (net of baseline)}
    """
    import pandas as pd
    baseline_A, baseline_f = build_electrification_scenario(model, ev_share)
    baseline_dx = model.combined_shock(baseline_A, baseline_f)

    contributions = {}
    for key in LEVERS:
        dA, df = build_lever_scenario(model, ev_share, levers=[key])
        dx = model.combined_shock(dA, df)
        contributions[key] = dx - baseline_dx  # incremental effect of this lever alone

    return contributions


def scenario_description(ev_share: float = 0.40) -> dict:
    """Return a human-readable summary of the scenario parameters."""
    fuel_saved_pct = ev_share * 85
    return {
        "ev_fleet_share": f"{ev_share*100:.0f}%",
        "fuel_demand_reduction": f"−{fuel_saved_pct:.0f}% of road fuel intermediate use",
        "electricity_demand_increase": f"+{ev_share*22:.0f}% relative to current transport electricity use",
        "household_fuel_switch_bCZK": f"−{50*ev_share:.0f} bCZK fuel → +{50*ev_share*0.25:.0f} bCZK electricity",
        "motor_vehicle_electronics_shift": f"+{8*ev_share:.1f} pp electronics content in vehicle production",
    }
