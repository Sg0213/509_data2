# 509_data2
# A Python Module for Computing Residual Energy in Solar Wind Turbulence from Parker Solar Probe Encounter 22

This repo contains a small, self-contained Python module that computes the
**normalized residual energy**  
\[
\sigma_D = \frac{E_v - E_b}{E_v + E_b}
\]
from time–series of velocity, magnetic field, and density, and plots
\(\sigma_D\) vs heliocentric distance.

---

## Contents

- **`residual_energy.py`**  
  Core module with:
  - `compute_residual_energy(...)` – computes \(E_v\), \(E_b\), \(E_D\), \(E_T\), and \(\sigma_D\) in non-overlapping time windows.
  - `plot_sigma_vs_distance(...)` – convenience plot of \(\sigma_D\) vs distance.
- **`example_residual_energy_psp.py`**  
  Example script that reads a PSP Encounter 22 CSV file, calls the module, and
  makes two figures:
  1. \(\sigma_D\) vs distance  
  2. \(E_v\) and \(E_b\) vs distance
- **`make_sample_data.py`**  
  Optional helper to carve out a smaller sample CSV from the full dataset
  for quick tests.

---

## Data

The full merged Encounter 22 PSP dataset used in this project is available on
Google Drive:

> **Full data (CSV):**  
> `PSP_MAG_SPC_POS_1min_merged.csv`  
> [https://drive.google.com/drive/u/0/folders/1Z_tSAjyrBBH8PQ4dZdYBw-1N5hSPJsUd](https://drive.google.com/drive/u/0/folders/1Z_tSAjyrBBH8PQ4dZdYBw-1N5hSPJsUd)

Download this CSV to any location on your machine and pass its path to the
example script (see below).

The code assumes:
- velocities in **km/s**
- magnetic field components in **nT**
- number density `n` in **cm⁻³**

## Requirements

- Python 3.10+  
- `numpy`, `pandas`, `matplotlib`

You can install the Python packages with:

bash
pip install numpy pandas matplotlib



