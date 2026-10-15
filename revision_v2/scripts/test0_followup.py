"""
test0_followup.py — Follow-ups to Test 0.

 A. Variant b restricted to cos z >= COS_MIN (where integer rounding of etr_h/etr_n no longer distorts cos z).
 B. Which eccentricity formula does the generator use? implied I0 = etr_n/e0, IQR for three formulas
    (Spencer 1971; 1+0.033cos(2 pi n/365); ASHRAE 1+0.033cos(2 pi (n-3)/365)).
 C. The two non-IWEC2 files: what do their ETR/closure fields look like, and how do they compare with
    Gompertz when cos z comes from an independent solar-position algorithm (pvlib SPA, interval average)?

Usage: python test0_followup.py --epw-dir <folder>
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
from epw_io import epw_files, read_epw  # noqa: E402
from gompertz import gompertz_kn  # noqa: E402
from test0_gompertz import MIN_GHI, cos_z_variants, spencer_e0  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_v2" / "data" / "processed"
COS_MIN = 0.2   # provisional; chosen where etr_h >= ~280 Wh/m2 so integer rounding is < 0.2 %


def e0_ashrae(doy):
    return 1.0 + 0.033 * np.cos(2 * np.pi * (doy - 3) / 365.0)


def e0_simple(doy):
    return 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.0)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    a = ap.parse_args()
    rows = []
    for p in epw_files(a.epw_dir):
        meta, df = read_epw(p)
        doy = df["doy"].to_numpy()
        etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
        ghi, dni, dhi = df["ghi"].to_numpy(), df["dni"].to_numpy(), df["dhi"].to_numpy()
        sun = etr_n > 0
        cosz = np.where(sun, etr_h / np.where(sun, etr_n, 1.0), 0.0)
        m = (ghi >= MIN_GHI) & (cosz >= COS_MIN) & np.isfinite(dni) & np.isfinite(dhi)
        pred = gompertz_kn(ghi[m] / etr_h[m], cosz[m]) * etr_n[m]
        d = pred - dni[m]
        res = (ghi - dni * cosz - dhi)[m]
        r = dict(city=meta["city"], source=meta["source"], n=int(m.sum()),
                 within2=float((np.abs(d) <= 2).mean()), within5=float((np.abs(d) <= 5).mean()),
                 within10=float((np.abs(d) <= 10).mean()), max_abs=float(np.abs(d).max()),
                 mbe=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                 r2=float(1 - (d ** 2).sum() / ((dni[m] - dni[m].mean()) ** 2).sum()),
                 closure_within2=float((np.abs(res) <= 2).mean()))
        for nm, f in (("spencer", spencer_e0), ("simple", e0_simple), ("ashrae", e0_ashrae)):
            i0 = (etr_n / f(doy))[sun]
            r[f"i0_{nm}_med"] = float(np.median(i0)); r[f"i0_{nm}_iqr"] = float(np.quantile(i0, .75) - np.quantile(i0, .25))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test0_followup.csv", index=False)
    iw = res["source"] == "IWEC2"
    pd.set_option("display.width", 200)

    print(f"A. Variant b, cos z >= {COS_MIN}, IWEC2 stations (n={iw.sum()}): median [min, max]")
    for c in ["within2", "within5", "within10", "max_abs", "mbe", "rmse", "r2", "closure_within2"]:
        v = res.loc[iw, c]; print(f"   {c:<16} {v.median():10.4f}  [{v.min():10.4f}, {v.max():10.4f}]")
    print("\nB. implied solar constant, IWEC2 stations: median of station medians / median IQR")
    for nm in ("spencer", "simple", "ashrae"):
        print(f"   e0={nm:<8} I0 = {res.loc[iw, f'i0_{nm}_med'].median():9.3f}   IQR = {res.loc[iw, f'i0_{nm}_iqr'].median():6.3f}")

    print("\nC. non-IWEC2 files")
    for p in epw_files(a.epw_dir):
        meta, df = read_epw(p)
        if meta["source"] == "IWEC2":
            continue
        etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
        ghi, dni, dhi = df["ghi"].to_numpy(), df["dni"].to_numpy(), df["dhi"].to_numpy()
        print(f"  --- {meta['city']} ({meta['source']})")
        print(f"      hours with ghi>=10: {int((ghi>=10).sum())};  of these etr_n==0: {int(((ghi>=10)&(etr_n==0)).sum())};  etr_h==0: {int(((ghi>=10)&(etr_h==0)).sum())}")
        cz = cos_z_variants(meta, pd.DatetimeIndex(df["t_start"]))
        for kind in ("inst_mid", "avg_interval"):
            c = cz[kind]
            mk = (ghi >= MIN_GHI) & (c >= COS_MIN) & np.isfinite(dni) & np.isfinite(dhi)
            e0 = spencer_e0(df["doy"].to_numpy())
            for i0 in (1361.0, 1367.0):
                et_h = i0 * e0 * c
                pred = gompertz_kn(ghi[mk] / et_h[mk], c[mk]) * i0 * e0[mk]
                dd = pred - dni[mk]
                clos = np.abs((ghi - dni * c - dhi)[mk])
                print(f"      SPA {kind:<12} I0={i0:.0f}: Gompertz within±2={np.mean(np.abs(dd)<=2):.3f} within±10={np.mean(np.abs(dd)<=10):.3f} "
                      f"R2={1-(dd**2).sum()/((dni[mk]-dni[mk].mean())**2).sum():.4f} MBE={dd.mean():+.1f} | closure within±2={np.mean(clos<=2):.3f}")
        noon = df[(df["month"] == 1) & (df["hour"].isin([12, 13, 14]))].iloc[:6]
        print("      sample January midday rows (ghi dni dhi etr_h etr_n):")
        print(noon[["day", "hour", "ghi", "dni", "dhi", "etr_h", "etr_n"]].to_string(index=False))


if __name__ == "__main__":
    main()
