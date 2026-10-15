"""
test0_final.py — Test 0 on all EPW files, using the generator's own geometry recovered from the file.

For each EPW:
  cosz_gen = etr_h / etr_n                 (generator's cos z; hours with etr_n > 0)
  implied I0 = etr_n / e0                  (two e0 formulas; IQR tells which one the generator used)
  Gompertz DNI, three variants:
     a1355 / a1367 : Kt = GHI/(I0 cosz_gen), DNI = KN * I0           (constant I0, as in the Fortran source)
     b             : Kt = GHI/etr_h,         DNI = KN * etr_n        (extraterrestrial radiation from the file)
  closure residual GHI - DNI cosz_gen - DHI (no Gompertz involved)
  mismatch anatomy for variant b: share of hours with |error| > 10 W/m2 by Kt bin and by cos z bin.

Hour selection replicates the submitted batch (GHI >= 10 W/m2, cos z > 0.0174) so counts are comparable.
Kt bins below are the handoff's PROVISIONAL proposal [0, .2, .35, .5, .65, .75, inf].

Usage: python test0_final.py --epw-dir <folder>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epw_io import epw_files, read_epw  # noqa: E402
from gompertz import gompertz_kn  # noqa: E402
from probe_generator_geometry import e0_simple  # noqa: E402
from test0_gompertz import MIN_COSZ, MIN_GHI, spencer_e0  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_v2" / "data" / "processed"
KT_BINS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.75, np.inf]   # provisional (handoff 3.1)
COS_BINS = [0.0174, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]


def stats(obs, pred):
    d = pred - obs
    ss = ((obs - obs.mean()) ** 2).sum()
    return dict(r2=1 - (d ** 2).sum() / ss, mbe=d.mean(), rmse=np.sqrt((d ** 2).mean()), max_abs=np.abs(d).max(),
                within2=(np.abs(d) <= 2).mean(), within10=(np.abs(d) <= 10).mean())


def station(path: Path) -> tuple[dict, pd.DataFrame]:
    meta, df = read_epw(path)
    doy = df["doy"].to_numpy()
    etr_h, etr_n = df["etr_h"].to_numpy(), df["etr_n"].to_numpy()
    ghi, dni, dhi = df["ghi"].to_numpy(), df["dni"].to_numpy(), df["dhi"].to_numpy()
    sun = etr_n > 0
    cosz = np.where(sun, etr_h / np.where(sun, etr_n, 1.0), 0.0)

    row = dict(file=meta["file"], city=meta["city"], source=meta["source"], lat=meta["lat"])
    for nm, e0 in (("spencer", spencer_e0(doy)), ("simple", e0_simple(doy))):
        i0 = (etr_n / e0)[sun]
        row[f"i0_{nm}_median"], row[f"i0_{nm}_iqr"] = float(np.median(i0)), float(np.quantile(i0, .75) - np.quantile(i0, .25))

    m = (ghi >= MIN_GHI) & (cosz > MIN_COSZ) & np.isfinite(dni) & np.isfinite(dhi) & (etr_h > 0)
    row["n_hours"] = int(m.sum())
    for i0 in (1355.0, 1367.0):
        pred = gompertz_kn(ghi[m] / (i0 * cosz[m]), cosz[m]) * i0
        for k, v in stats(dni[m], pred).items():
            row[f"a{int(i0)}_{k}"] = float(v)
    pred_b = gompertz_kn(ghi[m] / etr_h[m], cosz[m]) * etr_n[m]
    for k, v in stats(dni[m], pred_b).items():
        row[f"b_{k}"] = float(v)
    res = (ghi - dni * cosz - dhi)[m]
    row["closure_within2"] = float((np.abs(res) <= 2).mean())
    row["closure_within10"] = float((np.abs(res) <= 10).mean())
    row["closure_p99"] = float(np.quantile(np.abs(res), 0.99))

    err_b = np.abs(pred_b - dni[m])
    kt = ghi[m] / etr_h[m]
    anat = pd.DataFrame({"kt": kt, "cosz": cosz[m], "bad": err_b > 10.0, "city": meta["city"], "month": df["month"].to_numpy()[m]})
    return row, anat


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    a = ap.parse_args()
    rows, anats = [], []
    for p in epw_files(a.epw_dir):
        r, an = station(p)
        rows.append(r); anats.append(an)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "test0_final_per_station.csv", index=False)
    anat = pd.concat(anats)
    anat.to_parquet(OUT / "test0_final_hours.parquet") if False else None

    pd.set_option("display.width", 230, "display.max_rows", 100)
    iw = res["source"] == "IWEC2"
    cols = ["city", "i0_spencer_median", "i0_spencer_iqr", "i0_simple_iqr", "b_r2", "b_within2", "b_within10", "b_mbe", "b_max_abs", "closure_within2"]
    print(res.loc[iw, cols].round(3).to_string(index=False))
    print("\nNON-IWEC2 files:")
    print(res.loc[~iw, cols].round(3).to_string(index=False))
    print("\nIWEC2 (n=%d) — median [min, max] across stations" % iw.sum())
    for c in ["i0_spencer_median", "i0_spencer_iqr", "i0_simple_iqr", "a1355_within2", "a1367_within2", "b_within2", "b_within10", "b_r2", "b_mbe", "closure_within2", "closure_within10"]:
        v = res.loc[iw, c]
        print(f"  {c:<20} {v.median():9.4f}   [{v.min():9.4f}, {v.max():9.4f}]")

    print("\nMismatch anatomy, variant b, IWEC2 stations pooled: share of hours with |DNI error| > 10 W/m2")
    an = anat[anat["city"].isin(res.loc[iw, "city"])]
    an = an.assign(ktbin=pd.cut(an["kt"], KT_BINS, right=False), cbin=pd.cut(an["cosz"], COS_BINS, right=False))
    print("  by Kt bin:")
    print(an.groupby("ktbin", observed=True)["bad"].agg(["mean", "size"]).round(3).to_string())
    print("  by cos z bin:")
    print(an.groupby("cbin", observed=True)["bad"].agg(["mean", "size"]).round(3).to_string())
    print("  by month:")
    print(an.groupby("month")["bad"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
