"""
Leontief Input-Output Model.

Core mechanics
--------------
Given:
  Z  (n×n)  transaction matrix  –  Z[i,j] = output of sector i used by sector j
  f  (n,)   final demand vector –  household + government + export demand per sector
  x  (n,)   gross output        –  x = Z.sum(axis=1) + f

Technical coefficients:
  A[i,j] = Z[i,j] / x[j]   (share of sector j's output sourced from sector i)

Leontief inverse:
  L = (I - A)^{-1}          (total direct + indirect requirements)

Demand multiplier:
  x = L @ f                 (total output to satisfy final demand f)
"""

import numpy as np
import pandas as pd


class LeontiefModel:
    def __init__(self, Z: np.ndarray, f: np.ndarray, sector_names: list):
        self.sector_names = list(sector_names)
        self.n = len(sector_names)
        self.Z = Z
        self.f = f
        self.x = Z.sum(axis=1) + f
        self.A = Z / self.x[np.newaxis, :]
        I = np.eye(self.n)
        self.L = np.linalg.inv(I - self.A)

    # ------------------------------------------------------------------
    # Descriptive properties
    # ------------------------------------------------------------------

    def output_multipliers(self) -> pd.Series:
        """Column sums of L: total economy-wide output triggered by 1 unit of final demand."""
        return pd.Series(self.L.sum(axis=0), index=self.sector_names, name="output_multiplier")

    def backward_linkages(self) -> pd.Series:
        """
        Rasmussen backward linkage index.
        Values > 1 indicate above-average upstream pull (key buying sectors).
        """
        mult = self.L.sum(axis=0)
        return pd.Series(mult / mult.mean(), index=self.sector_names, name="backward_linkage")

    def forward_linkages(self) -> pd.Series:
        """
        Rasmussen forward linkage index.
        Values > 1 indicate above-average downstream push (key supplying sectors).
        """
        row_sums = self.L.sum(axis=1)
        return pd.Series(row_sums / row_sums.mean(), index=self.sector_names, name="forward_linkage")

    def value_added_share(self) -> pd.Series:
        """Value-added as share of gross output (1 - column sum of A)."""
        va = 1.0 - self.A.sum(axis=0)
        return pd.Series(va, index=self.sector_names, name="value_added_share")

    def summary_df(self) -> pd.DataFrame:
        """Combined sector metrics in one DataFrame."""
        return pd.DataFrame({
            "gross_output_bCZK": self.x,
            "final_demand_bCZK": self.f,
            "backward_linkage": self.backward_linkages().values,
            "forward_linkage": self.forward_linkages().values,
            "value_added_share": self.value_added_share().values,
        }, index=self.sector_names)

    # ------------------------------------------------------------------
    # Shock analysis
    # ------------------------------------------------------------------

    def demand_shock(self, delta_f: np.ndarray) -> pd.Series:
        """
        Output change from a shift in final demand.
        Δx = L · Δf
        """
        delta_x = self.L @ delta_f
        return pd.Series(delta_x, index=self.sector_names, name="output_change_bCZK")

    def technology_shock(self, delta_A: np.ndarray) -> pd.Series:
        """
        Output change from a change in technical coefficients (same final demand).
        Builds new Leontief inverse from A + delta_A, then Δx = x_new - x_old.
        """
        A_new = self.A + delta_A
        L_new = np.linalg.inv(np.eye(self.n) - A_new)
        x_new = L_new @ self.f
        return pd.Series(x_new - self.x, index=self.sector_names, name="output_change_bCZK")

    def combined_shock(self, delta_A: np.ndarray, delta_f: np.ndarray) -> pd.Series:
        """
        Combined technology + demand shock.
        New equilibrium: x_new = L_new · (f + Δf)
        """
        A_new = self.A + delta_A
        L_new = np.linalg.inv(np.eye(self.n) - A_new)
        x_new = L_new @ (self.f + delta_f)
        return pd.Series(x_new - self.x, index=self.sector_names, name="output_change_bCZK")

    def pct_change(self, delta_x: pd.Series) -> pd.Series:
        """Convert absolute output changes to percent of baseline gross output."""
        return (delta_x / self.x * 100).rename("output_change_pct")

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    def transaction_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.Z, index=self.sector_names, columns=self.sector_names)

    def leontief_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.L, index=self.sector_names, columns=self.sector_names)
