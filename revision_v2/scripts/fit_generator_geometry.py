"""
fit_generator_geometry.py — Find a closed-form solar geometry that reproduces the IWEC2 generator's cos z.

Target: cosz_gen = etr_h / etr_n for hours with cosz_gen >= 0.2 (integer rounding of etr_h then perturbs cos z
by < 0.002). Candidates combine a declination formula, an equation-of-time formula and a time reference
(instantaneous at mid-interval, or the mean of max(cos z, 0) over the hour). Ranking metric: mean |diff|.
If one candidate reaches rounding level (~1e-3) it can be used for ALL hours, including low sun, where the
integer ratio cannot be trusted.

Usage: python fit_generator_geometry.py --epw-dir <folder> [--stations MENDOZA,USHUAIA,TARTAGAL,POSADAS]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epw_io import epw_files, read_epw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_v2" / "data" / "processed"


def decl_cooper(n):   # degrees
    return 23.45 * np.sin(np.radians(360.0 * (284.0 + n) / 365.0))


def decl_spencer(n):  # degrees
    g = 2 * np.pi * (n - 1) / 365.0
    d = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
         + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    return np.degrees(d)


def decl_simple(n):   # degrees, common 'declination = 23.45 sin(360/365 (n-81))'
    return 23.45 * np.sin(np.radians(360.0 / 365.0 * (n - 81.0)))


def eot_spencer(n):   # minutes
    g = 2 * np.pi * (n - 1) / 365.0
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.04089 * np.sin(2 * g))


def eot_whitman(n):   # minutes (formula used in solar_geometry.py)
    b = np.radians(360.0 * (n - 81.0) / 364.0)
    return 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)


def eot_zero(n):
    return np.zeros_like(n, dtype=float)


DECL = {"cooper": decl_cooper, "spencer": decl_spencer, "simple81": decl_simple}
EOT = {"spencer": eot_spencer, "whitman": eot_whitman, "none": eot_zero}


def cosz(lat, lon, tz, doy, hour_clock, decl_f, eot_f):
    """cos z at local-standard clock time `hour_clock` (hours, fractional)."""
    d = np.radians(decl_f(doy))
    ast = hour_clock + eot_f(doy) / 60.0 + (lon - 15.0 * tz) * 4.0 / 60.0
    h = np.radians(15.0 * (ast - 12.0))
    L = np.radians(lat)
    return np.sin(L) * np.sin(d) + np.cos(L) * np.cos(d) * np.cos(h)


def candidate(meta, df, decl_name, eot_name, ref, shift_min=0.0):
    doy = df["doy"].to_numpy(float)
    t0 = (df["hour"].to_numpy(float) - 1.0) + shift_min / 60.0
    df_, ef_ = DECL[decl_name], EOT[eot_name]
    if ref == "mid":
        return np.clip(cosz(meta["lat"], meta["lon"], meta["tz"], doy, t0 + 0.5, df_, ef_), 0, None)
    sub = (np.arange(12) + 0.5) / 12.0
    c = np.stack([np.clip(cosz(meta["lat"], meta["lon"], meta["tz"], doy, t0 + s, df_, ef_), 0, None) for s in sub])
    return c.mean(axis=0)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    ap.add_argument("--stations", default="MENDOZA,USHUAIA,TARTAGAL,POSADAS")
    a = ap.parse_args()
    want = a.stations.split(",")
    rows = []
    for p in epw_files(a.epw_dir):
        if not any(w in p.name.upper() for w in want):
            continue
        meta, df = read_epw(p)
        etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
        gen = np.where(etr_n > 0, etr_h / np.where(etr_n > 0, etr_n, 1.0), 0.0)
        m = gen >= 0.2
        for dn, en, ref in itertools.product(DECL, EOT, ("mid", "avg")):
            c = candidate(meta, df, dn, en, ref)
            d = np.abs(c[m] - gen[m])
            rows.append(dict(city=meta["city"], decl=dn, eot=en, ref=ref, mean_abs=d.mean(), max_abs=d.max()))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 160, "display.max_rows", 200)
    agg = res.groupby(["decl", "eot", "ref"])[["mean_abs", "max_abs"]].mean().sort_values("mean_abs")
    print("Candidates ranked by mean |cosz_cand - cosz_gen| (mean over stations):")
    print(agg.round(5).head(12).to_string())
    best = agg.index[0]
    print("\nbest:", best)

    # refine the best candidate with a small clock shift
    print("\nclock-shift refinement of the best candidate (minutes):")
    shifts = np.arange(-6, 6.1, 1.0)
    out = []
    for p in epw_files(a.epw_dir):
        if not any(w in p.name.upper() for w in want):
            continue
        meta, df = read_epw(p)
        etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
        gen = np.where(etr_n > 0, etr_h / np.where(etr_n > 0, etr_n, 1.0), 0.0)
        m = gen >= 0.2
        for s in shifts:
            c = candidate(meta, df, best[0], best[1], best[2], shift_min=s)
            out.append(dict(shift=s, mean_abs=np.abs(c[m] - gen[m]).mean()))
    print(pd.DataFrame(out).groupby("shift")["mean_abs"].mean().round(5).to_string())
    res.to_csv(OUT / "fit_generator_geometry.csv", index=False)


if __name__ == "__main__":
    main()
