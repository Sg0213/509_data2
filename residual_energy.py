# residual_energy.py
"""
Residual energy utilities for MHD turbulence.

Core public functions
---------------------
- compute_residual_energy(df, window="4h", min_points=2, ...)
    Compute kinetic energy Ev, magnetic energy Eb (in Alfvén units),
    residual energy ED = Ev - Eb, total energy ET = Ev + Eb,
    and normalized residual energy sigma_D = ED / ET for each time window.

- plot_sigma_vs_distance(res_df, ...)
    Quick plot of sigma_D vs heliocentric distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ physical constants ------------------------ #
mu0: float = 4.0 * np.pi * 1e-7        # vacuum permeability [H/m]
mp:  float = 1.6726219e-27            # proton mass [kg]


# ------------------------ internal helpers -------------------------- #
def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the DataFrame has a DatetimeIndex called 'time'.

    - If the index is already a DatetimeIndex, just sort and return.
    - If there is a 'time' column, parse it and set it as the index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    if "time" not in df.columns:
        raise ValueError(
            "DataFrame must either have a DatetimeIndex or a 'time' column."
        )

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.set_index("time").sort_index()
    return out


def _ensure_rho(
    df: pd.DataFrame,
    rho_col: str = "rho",
    density_col: str | None = "n",
    n_in_cm3: bool = True,
) -> pd.DataFrame:
    """
    Ensure a mass density column rho [kg/m^3] exists.

    If `rho_col` is missing and `density_col` is given, build:
        rho = n * m_p

    where n is a number density (cm^-3 if n_in_cm3=True, else m^-3).
    """
    if rho_col in df.columns:
        return df.copy()

    if density_col is None or density_col not in df.columns:
        raise ValueError(
            f"Column '{rho_col}' not found and density_col='{density_col}' "
            "is not available to build it."
        )

    out = df.copy()
    n = out[density_col].astype(float)

    if n_in_cm3:
        n = n * 1e6      # cm^-3 -> m^-3

    out[rho_col] = n * mp
    return out


# --------------------------- public API ----------------------------- #
def compute_residual_energy(
    df: pd.DataFrame,
    *,
    window: str = "4h",
    min_points: int = 2,
    r_col: str = "R_AU",
    vr_col: str = "Vr",
    vt_col: str = "Vt",
    vn_col: str = "Vn",
    br_col: str = "Br",
    bt_col: str = "Bt",
    bn_col: str = "Bn",
    rho_col: str = "rho",
    density_col: str | None = "n",
    n_in_cm3: bool = True,
    clip_sigma: bool = True,
) -> pd.DataFrame:
    """
    Compute residual energy for time–series data.

    Assumes:
        v-components in km/s,
        B-components in nT,
        rho in kg/m^3 (or n in cm^-3 / m^-3 from which rho is built).

    For each window:
        δv_i  = v_i - <v_i>
        b_i'  = B_i / sqrt(mu0 * <rho>)
        δb_i' = b_i' - <b_i'>

        E_v   = Var(δv_r) + Var(δv_t) + Var(δv_n)           [km^2/s^2]
        E_b   = (Var(δb_r') + Var(δb_t') + Var(δb_n'))/1e6  [km^2/s^2]
        E_D   = E_v - E_b
        E_T   = E_v + E_b
        σ_D   = E_D / E_T

    Variances are unbiased (ddof=1).
    """
    # 1. time index & density
    df = _ensure_time_index(df)
    df = _ensure_rho(df, rho_col=rho_col, density_col=density_col,
                     n_in_cm3=n_in_cm3)

    # 2. keep only necessary columns, drop NaNs
    needed = [
        r_col,
        vr_col, vt_col, vn_col,
        br_col, bt_col, bn_col,
        rho_col,
    ]
    data = df[needed].apply(pd.to_numeric, errors="coerce").dropna()

    if data.empty:
        raise ValueError("No valid rows after dropping NaNs in required columns.")

    rows: list[tuple] = []

    # 3. loop over windows
    for t0, w in data.resample(window):
        N = len(w)
        if N < min_points:
            continue

        # --- kinetic energy Ev ---
        vr = w[vr_col] - w[vr_col].mean()
        vt = w[vt_col] - w[vt_col].mean()
        vn = w[vn_col] - w[vn_col].mean()

        var_vr = np.var(vr.to_numpy(), ddof=1)
        var_vt = np.var(vt.to_numpy(), ddof=1)
        var_vn = np.var(vn.to_numpy(), ddof=1)
        Ev = var_vr + var_vt + var_vn   # km^2/s^2

        # --- magnetic energy Eb (Alfvén units) ---
        rho_mean = float(w[rho_col].mean())
        if not np.isfinite(rho_mean) or rho_mean <= 0:
            continue

        factor = 1.0 / np.sqrt(mu0 * rho_mean)  # convert B (T) -> v_A (m/s)

        br_si = w[br_col].to_numpy() * 1e-9 * factor
        bt_si = w[bt_col].to_numpy() * 1e-9 * factor
        bn_si = w[bn_col].to_numpy() * 1e-9 * factor

        br_fl = br_si - br_si.mean()
        bt_fl = bt_si - bt_si.mean()
        bn_fl = bn_si - bn_si.mean()

        var_brn = np.var(br_fl, ddof=1)  # m^2/s^2
        var_btn = np.var(bt_fl, ddof=1)
        var_bnn = np.var(bn_fl, ddof=1)

        Eb_m2 = var_brn + var_btn + var_bnn       # m^2/s^2
        Eb = Eb_m2 / 1e6                          # -> km^2/s^2

        if not np.isfinite(Eb) or Eb <= 0:
            continue

        # --- residual / total ---
        ED = Ev - Eb
        ET = Ev + Eb
        if ET <= 0:
            continue

        sigma_D = ED / ET

        rows.append(
            (
                t0,
                float(w[r_col].mean()),
                Ev,
                Eb,
                ED,
                ET,
                sigma_D,
            )
        )

    if not rows:
        raise RuntimeError("No windows produced valid residual energy estimates.")

    res = pd.DataFrame(
        rows,
        columns=["time", r_col, "E_v", "E_b", "E_D", "E_T", "sigma_D"],
    ).set_index("time")

    if clip_sigma:
        res["sigma_D"] = res["sigma_D"].clip(-1.0, 1.0)

    return res


def plot_sigma_vs_distance(
    res_df: pd.DataFrame,
    *,
    r_col: str = "R_AU",
    sigma_col: str = "sigma_D",
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    Quick scatter plot: sigma_D vs heliocentric distance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    r = res_df[r_col].to_numpy()
    s = res_df[sigma_col].to_numpy()

    ax.plot(r, s, "*", ms=7, alpha=0.85, label=label)
    ax.axhline(0.0, color="k", ls=":", lw=1.0)

    ax.set_xlabel("Distance [AU]")
    ax.set_ylabel(r"$\sigma_D$")
    if label is not None:
        ax.legend()

    ax.grid(alpha=0.3)
    return ax


# Optional quick demo if run directly
if __name__ == "__main__":
    print(
        "This is a library module. Run example_residual_energy_psp.py\n"
        "to see a full example with real data."
    )
