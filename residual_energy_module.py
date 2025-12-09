
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Physical constants

mu0: float = 4.0 * np.pi * 1e-7      # vacuum permeability [H/m]
mp: float = 1.6726219e-27            # proton mass [kg]


def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:

    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    if "time" not in df.columns:
        raise ValueError(
        )

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.set_index("time").sort_index()
    return out


def _ensure_rho(
    df: pd.DataFrame,
    rho_col: str = "rho",
    n_col: str = "n",
    n_in_cm3: bool = True,
) -> pd.DataFrame:

    if rho_col in df.columns:
        return df.copy()

    if n_col not in df.columns:
        raise ValueError(
            f"Column '{rho_col}' not found and '{n_col}' is not available "
            "to build it."
        )

    out = df.copy()
    n = out[n_col].astype(float)
    if n_in_cm3:
        n = n * 1e6  # cm^-3 -> m^-3

    out[rho_col] = n * mp
    return out

# Public API

def compute_residual_energy(
    df: pd.DataFrame,
    *,
    window: str = "4h",
    min_points: int = 10,
    r_col: str = "R_AU",
    vr_col: str = "Vr",
    vt_col: str = "Vt",
    vn_col: str = "Vn",
    br_col: str = "Br",
    bt_col: str = "Bt",
    bn_col: str = "Bn",
    rho_col: str = "rho",
    n_col: str = "n",
    n_in_cm3: bool = True,
    clip_sigma: bool = True,
) -> pd.DataFrame:

    # 1. Time index & density preparation
    df = _ensure_time_index(df)
    df = _ensure_rho(df, rho_col=rho_col, n_col=n_col, n_in_cm3=n_in_cm3)

    # Dropping rows with missing required values
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

    for t0, w in data.resample(window):
        N = len(w)
        if N < min_points:
            continue

        # velocities: fluctuations & energy Ev (km^2/s^2)
        vr = w[vr_col] - w[vr_col].mean()
        vt = w[vt_col] - w[vt_col].mean()
        vn = w[vn_col] - w[vn_col].mean()

        var_vr = np.var(vr.to_numpy(), ddof=1)
        var_vt = np.var(vt.to_numpy(), ddof=1)
        var_vn = np.var(vn.to_numpy(), ddof=1)
        Ev = var_vr + var_vt + var_vn

        # --- magnetic: normalize to Alfvén units, fluctuations, Eb ---
        rho_mean = float(w[rho_col].mean())
        if not np.isfinite(rho_mean) or rho_mean <= 0:
            continue

        factor = 1.0 / np.sqrt(mu0 * rho_mean)  # converts nT -> m/s when multiplied by 1e-9

        br_si = w[br_col].to_numpy() * 1e-9 * factor
        bt_si = w[bt_col].to_numpy() * 1e-9 * factor
        bn_si = w[bn_col].to_numpy() * 1e-9 * factor

        br_fl = br_si - br_si.mean()
        bt_fl = bt_si - bt_si.mean()
        bn_fl = bn_si - bn_si.mean()

        var_brn = np.var(br_fl, ddof=1)  # m^2/s^2
        var_btn = np.var(bt_fl, ddof=1)
        var_bnn = np.var(bn_fl, ddof=1)

        Eb_m2 = var_brn + var_btn + var_bnn        # m^2/s^2
        Eb = Eb_m2 / 1e6                           # -> km^2/s^2

        if not np.isfinite(Eb) or Eb <= 0:
            continue

        # --- residual / total energy & sigma_D ---
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

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

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
