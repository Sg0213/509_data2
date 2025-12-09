
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from residual_energy_module import compute_residual_energy, plot_sigma_vs_distance

# default relative path inside the repo
DEFAULT_CSV = Path("data/psp_enc22_sample.csv")


def main():

    # 1. Choosing which file to use

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = DEFAULT_CSV

    if not csv_path.is_file():
        raise FileNotFoundError(
        )

    print(f"Reading: {csv_path}")
    raw = pd.read_csv(csv_path, parse_dates=["time"])


    # 2. Computing residual energy

    res = compute_residual_energy(
        raw,
        window="4h",      # 4-hour windows
        min_points=10,    # require at least 10 points per window
    )

    print(f"Number of windows: {len(res)}")
    print(f"<sigma_D> = {res['sigma_D'].mean(): .3f}")
    print(f"sigma_D range = [{res['sigma_D'].min(): .3f}, {res['sigma_D'].max(): .3f}]")

    # 3. Figure: sigma_D vs distance only

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sigma_vs_distance(res, ax=ax, label="PSP example")
    ax.set_title(r"Normalized residual energy $\sigma_D$")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
