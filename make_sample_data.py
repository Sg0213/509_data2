
from pathlib import Path
import pandas as pd

# Full file on my computer
FULL = Path("/home/sagar-ghimire/space/PSP_All/22enc/PSP_MAG_SPC_POS_1min_merged.csv")

# Sample file
OUT = Path("data/psp_enc22_sample.csv")

def main():
    cols = ["time", "R_AU", "Br", "Bt", "Bn", "Vr", "Vt", "Vn", "n"]

    print(f"Reading full file: {FULL}")
    raw = pd.read_csv(FULL, usecols=cols, parse_dates=["time"])

    # keep only rows with all fields present and n > 0
    mask = raw[["R_AU", "Br", "Bt", "Bn", "Vr", "Vt", "Vn", "n"]].notna().all(axis=1)
    mask &= raw["n"] > 0
    good = raw.loc[mask].copy()
    print("Good rows with full plasma + field:", len(good))

    # taking a compact sample
    N = 5000
    sample = good.iloc[:N].copy()

    OUT.parent.mkdir(exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"Wrote sample: {OUT}  (rows={len(sample)})")

if __name__ == "__main__":
    main()
