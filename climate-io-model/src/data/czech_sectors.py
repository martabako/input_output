"""
Synthetic 16-sector Czech Input-Output table (approx. 2023).

Values are in billion CZK. The technical coefficient matrix A is constructed
from stylised Czech economic structure; gross outputs are calibrated against
Czech Statistical Office / Eurostat aggregates and can be refined with
OWID energy data (see build_czech_io_model).

Sector ordering follows a simplified NACE Rev. 2 aggregation:
  0  Agriculture
  1  Mining & Quarrying
  2  Food & Beverages
  3  Chemicals & Pharma
  4  Rubber, Plastics & Minerals
  5  Basic & Fabricated Metals
  6  Motor Vehicles           ← key: Skoda + Tier-1/2 suppliers
  7  Electronics & Electrical Equip
  8  Other Manufacturing
  9  Electricity & Heat       ← key: generation + distribution
  10 Oil & Fuel Refining      ← key: Kralupy/Litvinov refineries + fuel dist.
  11 Construction
  12 Trade & Vehicle Repair
  13 Transport & Storage      ← key: road/rail/air transport services
  14 Business & Financial Services
  15 Public & Social Services
"""

import numpy as np

SECTOR_NAMES = [
    "Agriculture",
    "Mining & Quarrying",
    "Food & Beverages",
    "Chemicals & Pharma",
    "Rubber, Plastics & Minerals",
    "Basic & Fabricated Metals",
    "Motor Vehicles",
    "Electronics & Electrical Equip",
    "Other Manufacturing",
    "Electricity & Heat",
    "Oil & Fuel Refining",
    "Construction",
    "Trade & Vehicle Repair",
    "Transport & Storage",
    "Business & Financial Services",
    "Public & Social Services",
]

SECTOR_NAMES_CS = [
    "Zemědělství",
    "Těžba",
    "Potraviny a nápoje",
    "Chemie a farmacie",
    "Pryž, plasty a minerály",
    "Základní kovy a kovové výrobky",
    "Motorová vozidla",
    "Elektronika a el. zařízení",
    "Ostatní výroba",
    "Elektřina a teplo",
    "Zpracování ropy a paliv",
    "Stavebnictví",
    "Obchod a opravy vozidel",
    "Doprava a skladování",
    "Obchodní a finanční služby",
    "Veřejné a sociální služby",
]

# Indices for easy reference in scenario code
IDX = {name: i for i, name in enumerate(SECTOR_NAMES)}

# Default gross output vector (billion CZK, 2023 approx.)
# Source: Czech Statistical Office aggregates / Eurostat supply tables
# 2023 values scaled ~+17% vs 2019 for nominal growth + Czech CPI (2020-2023)
# Oil & Electricity overridden by OWID calibration in build_czech_io_model.
_DEFAULT_OUTPUT = np.array([
    175,   # Agriculture
    115,   # Mining (incl. quarrying, sand/gravel, lignite)
    470,   # Food & Bev
    235,   # Chemicals
    410,   # Rubber/Plastics
    585,   # Metals
    850,   # Motor Vehicles   (Skoda + Tier-1/2; Skoda record output 2022-23)
    525,   # Electronics
    295,   # Other Manufacturing
    320,   # Electricity & Heat  (→ overridden: 66.6 TWh × 4.8 ≈ 320 bCZK)
    220,   # Oil & Fuel Refining (→ overridden: 107.8 TWh × 2.05 ≈ 221 bCZK)
    645,   # Construction
    820,   # Trade
    470,   # Transport
    700,   # Business Services
    645,   # Public Services
], dtype=float)


def build_technical_coefficients() -> np.ndarray:
    """
    Return the 16x16 technical coefficient matrix A.
    A[i,j] = fraction of sector j's gross output sourced from sector i.
    """
    n = len(SECTOR_NAMES)
    A = np.zeros((n, n))

    # --- Agriculture (col 0) ---
    A[IDX["Agriculture"], 0] = 0.08      # seeds, feed, breeding stock
    A[IDX["Chemicals & Pharma"], 0] = 0.05  # fertilisers, pesticides
    A[IDX["Electricity & Heat"], 0] = 0.02
    A[IDX["Transport & Storage"], 0] = 0.02
    A[IDX["Business & Financial Services"], 0] = 0.02

    # --- Mining & Quarrying (col 1) ---
    A[IDX["Electricity & Heat"], 1] = 0.08   # pumps, ventilation
    A[IDX["Transport & Storage"], 1] = 0.03
    A[IDX["Business & Financial Services"], 1] = 0.03
    A[IDX["Construction"], 1] = 0.02         # mine infrastructure

    # --- Food & Beverages (col 2) ---
    A[IDX["Agriculture"], 2] = 0.25
    A[IDX["Chemicals & Pharma"], 2] = 0.02   # additives, packaging chemicals
    A[IDX["Rubber, Plastics & Minerals"], 2] = 0.02   # packaging
    A[IDX["Electricity & Heat"], 2] = 0.02
    A[IDX["Transport & Storage"], 2] = 0.03
    A[IDX["Trade & Vehicle Repair"], 2] = 0.03
    A[IDX["Business & Financial Services"], 2] = 0.02

    # --- Chemicals & Pharma (col 3) ---
    A[IDX["Mining & Quarrying"], 3] = 0.05
    A[IDX["Chemicals & Pharma"], 3] = 0.08   # intra-sector (feedstocks)
    A[IDX["Oil & Fuel Refining"], 3] = 0.10  # naphtha, refinery products
    A[IDX["Electricity & Heat"], 3] = 0.04
    A[IDX["Transport & Storage"], 3] = 0.02
    A[IDX["Business & Financial Services"], 3] = 0.03

    # --- Rubber, Plastics & Minerals (col 4) ---
    A[IDX["Mining & Quarrying"], 4] = 0.03
    A[IDX["Chemicals & Pharma"], 4] = 0.06
    A[IDX["Oil & Fuel Refining"], 4] = 0.05  # polymer feedstocks
    A[IDX["Electricity & Heat"], 4] = 0.04
    A[IDX["Transport & Storage"], 4] = 0.02
    A[IDX["Business & Financial Services"], 4] = 0.02

    # --- Basic & Fabricated Metals (col 5) ---
    A[IDX["Mining & Quarrying"], 5] = 0.08   # iron ore, scrap
    A[IDX["Basic & Fabricated Metals"], 5] = 0.10  # scrap recycling
    A[IDX["Oil & Fuel Refining"], 5] = 0.03
    A[IDX["Electricity & Heat"], 5] = 0.06   # arc furnaces
    A[IDX["Transport & Storage"], 5] = 0.02
    A[IDX["Business & Financial Services"], 5] = 0.02

    # --- Motor Vehicles (col 6) — KEY SECTOR ---
    A[IDX["Basic & Fabricated Metals"], 6] = 0.15   # body, chassis, powertrain
    A[IDX["Rubber, Plastics & Minerals"], 6] = 0.08 # tyres, interior
    A[IDX["Electronics & Electrical Equip"], 6] = 0.10  # ICE: sensors, BCU, infotainment
    A[IDX["Chemicals & Pharma"], 6] = 0.03          # fluids, coatings
    A[IDX["Other Manufacturing"], 6] = 0.05         # glass, textiles
    A[IDX["Electricity & Heat"], 6] = 0.02
    A[IDX["Transport & Storage"], 6] = 0.02         # inbound/outbound logistics
    A[IDX["Business & Financial Services"], 6] = 0.05  # R&D, finance leases
    A[IDX["Trade & Vehicle Repair"], 6] = 0.02      # dealer network

    # --- Electronics & Electrical Equip (col 7) ---
    A[IDX["Basic & Fabricated Metals"], 7] = 0.08   # copper, aluminium
    A[IDX["Electronics & Electrical Equip"], 7] = 0.05  # intra-sector
    A[IDX["Chemicals & Pharma"], 7] = 0.04          # PCB chemicals
    A[IDX["Other Manufacturing"], 7] = 0.04
    A[IDX["Electricity & Heat"], 7] = 0.02
    A[IDX["Transport & Storage"], 7] = 0.02
    A[IDX["Business & Financial Services"], 7] = 0.04  # R&D

    # --- Other Manufacturing (col 8) ---
    A[IDX["Agriculture"], 8] = 0.03
    A[IDX["Basic & Fabricated Metals"], 8] = 0.05
    A[IDX["Rubber, Plastics & Minerals"], 8] = 0.04
    A[IDX["Electricity & Heat"], 8] = 0.02
    A[IDX["Transport & Storage"], 8] = 0.02
    A[IDX["Business & Financial Services"], 8] = 0.02

    # --- Electricity & Heat (col 9) — KEY SECTOR ---
    A[IDX["Mining & Quarrying"], 9] = 0.05          # coal (Czech power mix ~40% coal)
    A[IDX["Oil & Fuel Refining"], 9] = 0.06         # gas-fired plants
    A[IDX["Electricity & Heat"], 9] = 0.04          # transmission losses
    A[IDX["Business & Financial Services"], 9] = 0.03
    A[IDX["Construction"], 9] = 0.02                # grid maintenance

    # --- Oil & Fuel Refining (col 10) — KEY SECTOR ---
    A[IDX["Mining & Quarrying"], 10] = 0.02
    A[IDX["Oil & Fuel Refining"], 10] = 0.05        # refinery own consumption
    A[IDX["Chemicals & Pharma"], 10] = 0.04
    A[IDX["Transport & Storage"], 10] = 0.04        # pipeline transport
    A[IDX["Business & Financial Services"], 10] = 0.02

    # --- Construction (col 11) ---
    A[IDX["Basic & Fabricated Metals"], 11] = 0.08
    A[IDX["Rubber, Plastics & Minerals"], 11] = 0.04
    A[IDX["Chemicals & Pharma"], 11] = 0.03
    A[IDX["Other Manufacturing"], 11] = 0.03
    A[IDX["Electricity & Heat"], 11] = 0.02
    A[IDX["Transport & Storage"], 11] = 0.03
    A[IDX["Business & Financial Services"], 11] = 0.05  # architecture, finance

    # --- Trade & Vehicle Repair (col 12) ---
    A[IDX["Oil & Fuel Refining"], 12] = 0.05        # fuel for logistics fleet
    A[IDX["Motor Vehicles"], 12] = 0.02             # fleet renewal
    A[IDX["Electricity & Heat"], 12] = 0.01
    A[IDX["Transport & Storage"], 12] = 0.03
    A[IDX["Business & Financial Services"], 12] = 0.04

    # --- Transport & Storage (col 13) — KEY SECTOR ---
    A[IDX["Oil & Fuel Refining"], 13] = 0.15        # dominant cost: road fuel
    A[IDX["Motor Vehicles"], 13] = 0.04             # fleet renewal
    A[IDX["Electricity & Heat"], 13] = 0.02         # rail traction, warehouses
    A[IDX["Transport & Storage"], 13] = 0.02        # intra-sector (haulage)
    A[IDX["Business & Financial Services"], 13] = 0.03

    # --- Business & Financial Services (col 14) ---
    A[IDX["Electricity & Heat"], 14] = 0.01
    A[IDX["Transport & Storage"], 14] = 0.01
    A[IDX["Business & Financial Services"], 14] = 0.03

    # --- Public & Social Services (col 15) ---
    A[IDX["Electricity & Heat"], 15] = 0.02
    A[IDX["Transport & Storage"], 15] = 0.01
    A[IDX["Business & Financial Services"], 15] = 0.03
    A[IDX["Construction"], 15] = 0.02               # public investment

    return A


def build_czech_io_model(owid_data=None, year: int = 2023, eurostat_iot: dict = None):
    """
    Construct the Czech IO model inputs: (Z, f, x, sector_names).

    Parameters
    ----------
    owid_data : pd.DataFrame, optional
        From load_owid_czech_energy(). Overrides oil/electricity sector sizes
        using real OWID energy consumption for `year`.
    year : int
        Reference year for OWID calibration (default 2023).
    eurostat_iot : dict, optional
        From load_eurostat_czech_iot(). When provided, uses the real Eurostat
        A matrix and sector gross outputs instead of synthetic approximations.
        OWID calibration (if owid_data given) is applied on top to update
        oil/electricity sizes to the most recent available year.

    Returns
    -------
    Z : np.ndarray (16, 16)  transaction matrix (billion CZK)
    f : np.ndarray (16,)     final demand vector (billion CZK)
    x : np.ndarray (16,)     gross output vector (billion CZK)
    sector_names : list[str]
    """
    if eurostat_iot is not None:
        A = eurostat_iot["A"].copy()
        x = eurostat_iot["x"].copy()
    else:
        A = build_technical_coefficients()
        x = _DEFAULT_OUTPUT.copy()

    if owid_data is not None:
        row = owid_data[owid_data["year"] == year]
        if not row.empty:
            # Rescale Oil/Fuel sector: Czech oil consumption (TWh) → billion CZK
            # ~2.05 bCZK/TWh: refinery gate + distribution margin + non-fuel products
            # (107.8 TWh × 2.05 ≈ 221 bCZK; higher than 2019 factor of 1.72 because
            # Czech pump prices in CZK were ~55% higher in 2023 vs 2019)
            oil_twh = row["oil_consumption"].values[0]
            if not np.isnan(oil_twh):
                x[IDX["Oil & Fuel Refining"]] = oil_twh * 2.05

            # Rescale Electricity sector: electricity demand (TWh) → billion CZK
            # ~4.8 bCZK/TWh at ~2.0 CZK/kWh avg tariff (2023)
            elec_twh = row["electricity_demand"].values[0]
            if not np.isnan(elec_twh):
                x[IDX["Electricity & Heat"]] = elec_twh * 4.8

    # Transaction matrix: Z[i,j] = A[i,j] * x[j]
    Z = A * x[np.newaxis, :]

    # Final demand: f = x - row_sums(Z)  →  f = (I - A) @ x
    f = x - Z.sum(axis=1)

    return Z, f, x, SECTOR_NAMES
