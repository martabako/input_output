import numpy as np
import pandas as pd

OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

OWID_COLUMNS = [
    "country", "year", "iso_code",
    "oil_consumption", "coal_consumption", "gas_consumption",
    "electricity_demand", "electricity_generation",
    "renewables_consumption", "solar_consumption", "wind_consumption",
    "carbon_intensity_elec", "energy_per_capita",
    "fossil_fuel_consumption", "fossil_share_energy",
    "low_carbon_share_energy", "renewables_share_energy",
]

# CPA product codes → 16-sector index
# Non-overlapping leaf codes only (excludes CPA_B, CPA_F, CPA_C10-12 and other aggregates).
# Source: Eurostat NAIO_10_CP1700, product-by-product symmetric IOT.
CPA_TO_SECTOR = {
    # 0  Agriculture
    "CPA_A01": 0, "CPA_A02": 0, "CPA_A03": 0,
    # 1  Mining & Quarrying
    "CPA_B05": 1, "CPA_B06": 1, "CPA_B07": 1, "CPA_B08": 1, "CPA_B09": 1,
    # 2  Food & Beverages
    "CPA_C10": 2, "CPA_C11": 2, "CPA_C12": 2,
    # 3  Chemicals & Pharma
    "CPA_C20": 3, "CPA_C21": 3,
    # 4  Rubber, Plastics & Minerals
    "CPA_C22": 4, "CPA_C23": 4,
    # 5  Basic & Fabricated Metals
    "CPA_C24": 5, "CPA_C25": 5,
    # 6  Motor Vehicles
    "CPA_C29": 6, "CPA_C30": 6,
    # 7  Electronics & Electrical Equip
    "CPA_C26": 7, "CPA_C27": 7, "CPA_C28": 7,
    # 8  Other Manufacturing
    "CPA_C13": 8, "CPA_C14": 8, "CPA_C15": 8,
    "CPA_C16": 8, "CPA_C17": 8, "CPA_C18": 8,
    "CPA_C31": 8, "CPA_C32": 8, "CPA_C33": 8,
    # 9  Electricity & Heat
    "CPA_D": 9,
    # 10 Oil & Fuel Refining
    "CPA_C19": 10,
    # 11 Construction
    "CPA_F41": 11, "CPA_F42": 11, "CPA_F43": 11,
    # 12 Trade & Vehicle Repair (G47 retail not submitted by CZ to Eurostat IOT)
    "CPA_G45": 12, "CPA_G46": 12,
    # 13 Transport & Storage
    "CPA_H49": 13, "CPA_H50": 13, "CPA_H51": 13, "CPA_H52": 13, "CPA_H53": 13,
    # 14 Business & Financial Services
    "CPA_I55": 14, "CPA_I56": 14,
    "CPA_J58": 14, "CPA_J59": 14, "CPA_J60": 14, "CPA_J61": 14, "CPA_J62": 14, "CPA_J63": 14,
    "CPA_K64": 14, "CPA_K65": 14, "CPA_K66": 14,
    "CPA_L68B": 14,
    "CPA_M69": 14, "CPA_M70": 14, "CPA_M71": 14, "CPA_M72": 14,
    "CPA_M73": 14, "CPA_M74": 14, "CPA_M75": 14,
    "CPA_N77": 14, "CPA_N78": 14, "CPA_N79": 14,
    "CPA_N80": 14, "CPA_N81": 14, "CPA_N82": 14,
    # 15 Public & Social Services
    "CPA_E36": 15, "CPA_E37-39": 15,
    "CPA_O": 15, "CPA_P": 15,
    "CPA_Q86": 15, "CPA_Q87": 15, "CPA_Q88": 15,
    "CPA_R90": 15, "CPA_R91": 15, "CPA_R92": 15, "CPA_R93": 15,
    "CPA_S94": 15, "CPA_S95": 15, "CPA_S96": 15,
}

# Annual average EUR/CZK exchange rates
_EUR_CZK = {2019: 25.67, 2020: 26.45, 2021: 25.64, 2022: 24.57, 2023: 24.00}


def load_owid_czech_energy(country: str = "Czechia") -> pd.DataFrame:
    """
    Load Czech energy data from OWID GitHub repository.
    Returns a DataFrame with annual energy metrics (TWh unless noted)
    filtered to the specified country, sorted by year.
    """
    df = pd.read_csv(OWID_URL, usecols=lambda c: c in OWID_COLUMNS)
    cz = df[df["country"] == country].copy()
    return cz.sort_values("year").reset_index(drop=True)


def load_eurostat_czech_iot(year: int = 2020) -> dict:
    """
    Fetch Czech symmetric IO table from Eurostat (dataset NAIO_10_CP1700,
    product × product at basic prices, total flows) and aggregate from
    ~60 CPA product codes to the 16 sectors defined in czech_sectors.py.

    Parameters
    ----------
    year : int
        Reference year. 2020 is the default (last stable pre/post-COVID year
        with full CZ coverage). Available range: 2010–2023.

    Returns
    -------
    dict with keys:
      Z     np.ndarray (16, 16)  transaction matrix (billion CZK)
      x     np.ndarray (16,)     gross output (billion CZK)
      A     np.ndarray (16, 16)  technical coefficient matrix
      year  int                  data year used
    """
    try:
        import eurostat
    except ImportError:
        raise ImportError("Run: pip install eurostat")

    # stk_flow='DOM': domestic flows only. Using 'TOTAL' (domestic + imported) causes
    # negative final demand in import-heavy sectors (oil, chemicals, metals) because
    # imported intermediates appear in Z rows but not in domestic x.
    df = eurostat.get_data_df(
        "NAIO_10_CP1700",
        filter_pars={"geo": "CZ", "freq": "A", "unit": "MIO_EUR", "stk_flow": "DOM"},
    )

    year_col = str(year)
    available = sorted(c for c in df.columns if str(c).isdigit() or (isinstance(c, str) and c.isdigit()))
    if year_col not in df.columns:
        raise ValueError(f"Year {year} not in dataset. Available: {available}")

    leaf_codes = list(CPA_TO_SECTOR.keys())
    n = 16

    # --- Build Z matrix (intermediate flows) ---
    Z_eur = np.zeros((n, n))
    sub = df[
        df["prd_ava"].isin(leaf_codes) & df["prd_use"].isin(leaf_codes)
    ][["prd_ava", "prd_use", year_col]].dropna()

    for _, row in sub.iterrows():
        i = CPA_TO_SECTOR[row["prd_ava"]]
        j = CPA_TO_SECTOR[row["prd_use"]]
        Z_eur[i, j] += float(row[year_col])

    # --- Gross value added (B1G row) → gross output = Z column sum + GVA ---
    va_sub = df[
        (df["prd_ava"] == "B1G") & df["prd_use"].isin(leaf_codes)
    ][["prd_use", year_col]].dropna()

    va_eur = np.zeros(n)
    for _, row in va_sub.iterrows():
        j = CPA_TO_SECTOR[row["prd_use"]]
        va_eur[j] += float(row[year_col])

    x_eur = Z_eur.sum(axis=0) + va_eur

    # --- Convert MIO EUR → billion CZK ---
    rate = _EUR_CZK.get(year, 25.0)
    Z = Z_eur * rate / 1000.0
    x = x_eur * rate / 1000.0

    # --- Derive A matrix: A[i,j] = Z[i,j] / x[j] ---
    x_safe = np.where(x > 0, x, 1.0)
    A = Z / x_safe[np.newaxis, :]

    return {"Z": Z, "x": x, "A": A, "year": year}
