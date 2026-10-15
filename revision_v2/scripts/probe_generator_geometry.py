"""
probe_generator_geometry.py — Recover the weather-file generator's own geometry from the EPW itself.

The EPW carries extraterrestrial horizontal (etr_h) and normal (etr_n) radiation, both computed by the generator
from geometry alone. Hence:
    cosz_gen = etr_h / etr_n          (the generator's cos z, per hour)
    I0 * e0  = etr_n                  (its solar constant times eccentricity factor)
Using cosz_gen removes any doubt about my own solar-position algorithm or time convention in Test 0.

Steps per file:
  1. implied solar constant I0 = etr_n / e0 for two common e0 formulas (Spencer 1971; 1+0.033cos(2 pi n/365)).
  2. compare cosz_gen with pvlib SPA cos z at a scan of time offsets (minutes from the interval centre) and with
     the interval average: which convention does the generator follow?
  3. Test 0 again, with cosz_gen. Two Kt definitions:
       (a) constant I0 (as in the Fortran): Kt = GHI / (I0 cosz), DNI = KN * I0
       (b) ET from the file:               Kt = GHI / etr_h,      DNI = KN * etr_n
  4. closure DHI = GHI - DNI cosz_gen on the EPW fields themselves (no Gompertz).

Usage: python probe_generator_geometry.py --epw <file> [--epw <file> ...]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epw_io import read_epw  # noqa: E402
from gompertz import gompertz_kn  # noqa: E402
from test0_gompertz import MIN_GHI, spencer_e0  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_v2" / "data" / "processed"


def e0_simple(doy: np.ndarray) -> np.ndarray:
    return 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)


def cosz_at(meta: dict, t_start: pd.DatetimeIndex, offset_min: float) -> np.ndarray:
    tz = timezone(timedelta(hours=meta["tz"]))
    t = (t_start + pd.Timedelta(minutes=30 + offset_min)).tz_localize(tz)
    z = pvlib.solarposition.get_solarposition(t, meta["lat"], meta["lon"], altitude=meta["elev"], method="nrel_numpy")["zenith"].to_numpy()
    return np.clip(np.cos(np.radians(z)), 0.0, None)


def summ(obs: np.ndarray, pred: np.ndarray) -> dict:
    d = pred - obs
    ss = ((obs - obs.mean()) ** 2).sum()
    return dict(r2=1 - (d ** 2).sum() / ss, max_abs=np.abs(d).max(), within2=(np.abs(d) <= 2).mean(),
                within10=(np.abs(d) <= 10).mean(), mbe=d.mean())


def probe(path: Path) -> dict:
    meta, df = read_epw(path)
    t0 = pd.DatetimeIndex(df["t_start"])
    doy = df["doy"].to_numpy()
    etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
    ghi, dni, dhi = df["ghi"].to_numpy(), df["dni"].to_numpy(), df["dhi"].to_numpy()
    print(f"\n=== {meta['city']}  ({meta['source']})")

    up = etr_h > 20  # hours with a well-defined ratio
    sun = etr_n > 0  # etr_n is stored as 0 at night; those hours carry no information on I0
    for name, e0 in (("Spencer1971", spencer_e0(doy)), ("1+0.033cos", e0_simple(doy))):
        i0 = (etr_n / e0)[sun]
        print(f"[1] implied I0 with e0={name:<11}: median {np.median(i0):8.3f}   IQR {np.quantile(i0,.75)-np.quantile(i0,.25):6.3f}   min {i0.min():8.2f} max {i0.max():8.2f}")

    cosz_gen = np.where(etr_n > 0, etr_h / np.where(etr_n > 0, etr_n, 1), 0.0)
    print("[2] generator cos z vs pvlib SPA, hours with etr_h>20 (mean |diff|, max |diff|):")
    best = None
    for off in (-60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60):
        c = cosz_at(meta, t0, off)
        d = np.abs(cosz_gen[up] - c[up])
        print(f"      offset {off:+4d} min: mean {d.mean():.5f}  max {d.max():.5f}")
        if best is None or d.mean() < best[1]:
            best = (off, d.mean())
    print(f"      -> closest offset on this grid: {best[0]:+d} min (mean |diff| {best[1]:.5f})")

    # interval average of cos z over [start, start+60 min]
    sub = np.tile((np.arange(12) + 0.5) * 5.0, len(t0))
    tz = timezone(timedelta(hours=meta["tz"]))
    ts = (pd.DatetimeIndex(np.repeat(t0.to_numpy(), 12)) + pd.to_timedelta(sub, unit="min")).tz_localize(tz)
    zs = pvlib.solarposition.get_solarposition(ts, meta["lat"], meta["lon"], altitude=meta["elev"], method="nrel_numpy")["zenith"].to_numpy()
    c_avg = np.clip(np.cos(np.radians(zs)), 0, None).reshape(len(t0), 12).mean(axis=1)
    d = np.abs(cosz_gen[up] - c_avg[up])
    print(f"      interval average       : mean {d.mean():.5f}  max {d.max():.5f}")

    m = (ghi >= MIN_GHI) & (cosz_gen > 0.0174) & np.isfinite(dni) & np.isfinite(dhi)
    print(f"[3] Gompertz vs EPW DNI with cosz_gen  (n={int(m.sum())} hours)")
    rows = []
    for i0 in (1355.0, 1367.0):
        kt = ghi[m] / (i0 * cosz_gen[m])
        pred = gompertz_kn(kt, cosz_gen[m]) * i0
        s = summ(dni[m], pred); rows.append(("(a) const I0=%.0f" % i0, s))
    kt = ghi[m] / etr_h[m]
    pred = gompertz_kn(kt, cosz_gen[m]) * etr_n[m]
    rows.append(("(b) ET from file", summ(dni[m], pred)))
    for lab, s in rows:
        print(f"      {lab:<20} R2={s['r2']:.5f}  max|err|={s['max_abs']:8.1f}  within±2={s['within2']:.3f}  within±10={s['within10']:.3f}  MBE={s['mbe']:+.2f}")

    res = ghi - dni * cosz_gen - dhi
    mm = m & (cosz_gen > 0)
    q = np.quantile(np.abs(res[mm]), [0.5, 0.9, 0.99])
    print(f"[4] closure GHI - DNI*cosz_gen - DHI: |res| median {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  within±2={np.mean(np.abs(res[mm])<=2):.3f}  within±10={np.mean(np.abs(res[mm])<=10):.3f}")
    return dict(city=meta["city"], i0_spencer_median=float(np.median(etr_n / spencer_e0(doy))),
                i0_simple_median=float(np.median(etr_n / e0_simple(doy))), closest_offset_min=best[0])


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw", action="append", required=True)
    a = ap.parse_args()
    rows = [probe(Path(p)) for p in a.epw]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "probe_generator_geometry.csv", index=False)


if __name__ == "__main__":
    main()
