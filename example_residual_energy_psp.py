# example_residual_energy_psp.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from residual_energy import compute_residual_energy, plot_sigma_vs_distance


def pick_csv_path(argv: list[str]) -> Path:
    """
    Decide which CSV file to use, in this order:

    1) Command-line argument (if given),
    2) data/psp_enc22_sample.csv in the repo,
    3) PSP_MAG_SPC_POS_1min_merged.csv next to this script,
    4) Your personal full-path file (works only on your machine).
    """
    # 1) User-supplied path from command line
    if len(argv) > 1:
        path = Path(argv[1]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Command-line file not found:\n  {path}\n"
                "Check the path and try again."
            )
        return path

    # 2) Auto-detect data file
    repo_root = Path(__file__).resolve().parent

    candidates = [
        repo_root / "data" / "psp_enc22_sample.csv",         # committed sample
        repo_root / "PSP_MAG_SPC_POS_1min_merged.csv",      # full file in repo
        Path("/home/sagar-ghimire/space/PSP_All/22enc/PSP_MAG_SPC_POS_1min_merged.csv"),
    ]

    for c in candidates:
        if c.is_file():
            print(f"Using data file: {c}")
            return c

    msg_lines = [
        "Could not find a PSP CSV file.",
        "",
        "Tried these locations:",
    ] + [f"  - {c}" for c in candidates] + [
        "",
        "Fix options:",
        "  * Put a small sample file at:  data/psp_enc22_sample.csv  (recommended),",
        "  * OR copy your full file here as: PSP_MAG_SPC_POS_1min_merged.csv,",
        "  * OR run with an explicit path, e.g.:",
        "      python example_residual_energy_psp.py /path/to/file.csv",
    ]
    raise FileNotFoundError("\n".join(msg_lines))


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv

    # --------------------------------------------------------------
    # 1. Choose data file
    # --------------------------------------------------------------
    csv_path = pick_csv_path(argv)

    print(f"Reading CSV: {csv_path}")
    raw = pd.read_csv(csv_path, parse_dates=["time"])

    # --------------------------------------------------------------
    # 2. Compute residual energy
    # --------------------------------------------------------------
    res = compute_residual_energy(
        raw,
        window="4h",      # 4-hour windows
        min_points=10,    # require at least 10 points per window
    )

    print(f"Number of windows: {len(res)}")
    print(f"<sigma_D> = {res['sigma_D'].mean(): .3f}")
    print(
        "sigma_D range = "
        f"[{res['sigma_D'].min(): .3f}, {res['sigma_D'].max(): .3f}]"
    )

    # --------------------------------------------------------------
    # 3. Figure: σ_D vs distance
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sigma_vs_distance(res, ax=ax, label="PSP example")
    ax.set_title(r"Normalized residual energy $\sigma_D$")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
