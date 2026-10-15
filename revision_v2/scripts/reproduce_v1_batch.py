"""
reproduce_v1_batch.py — Re-run the SUBMITTED analysis (legacy parser + solar_decomp v1.0.0 models + the batch's
filters and metric definitions) on the real EPW files, and compare with Table 2 and Table S2 of the manuscript.

Logic copied from Paper/validate_epw_batch.py (git 7db2498^): GHI >= 10 W/m2 rows only, MIN_COSZ = 0.0174,
metrics RMSE / MBE / R2 / nRMSE on hourly values, run_decomposition(primary='DIRINT').

Target numbers (manuscript Table 2, mean +- SD over 41 stations):
  DNI  Reindl-2 RMSE 69.5+-11.0  MBE -19.5+-6.1  nRMSE 21.5+-3.5  R2 0.923+-0.026
  DNI  Erbs     RMSE 77.2+-10.9  MBE -18.0+-7.9  nRMSE 24.0+-4.4  R2 0.904+-0.035
  DNI  DIRINT   RMSE 186.8+-18.9 MBE +97.0+-15.2 nRMSE 58.5+-10.0 R2 0.436+-0.174
  DHI  Erbs     RMSE 16.3+-4.3   MBE +2.7+-2.8   nRMSE 9.1+-2.8   R2 0.978+-0.015
  DHI  Reindl-2 RMSE 21.7+-6.1   MBE +12.0+-3.4  nRMSE 12.1+-3.9  R2 0.960+-0.027
  DHI  DIRINT   RMSE 51.7+-6.7   MBE -30.0+-6.4  nRMSE 28.6+-3.7  R2 0.787+-0.063

Usage: python reproduce_v1_batch.py --epw-dir <folder>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "legacy"))
sys.path.insert(0, str(HERE))

from decomposition import run_decomposition  # noqa: E402
from epw_io import epw_files  # noqa: E402
from legacy_epw_parser import parse_epw  # noqa: E402
from solar_geometry import compute_geometry_df  # noqa: E402

OUT = ROOT / "revision_v2" / "data" / "processed"
MIN_GHI = 10.0
MIN_COSZ = 0.0174

TABLE2 = {  # (component, model): (RMSE, MBE, nRMSE, R2) mean values from the manuscript
    ("DNI", "Reindl2"): (69.5, -19.5, 21.5, 0.923), ("DNI", "Erbs"): (77.2, -18.0, 24.0, 0.904),
    ("DNI", "DIRINT"): (186.8, 97.0, 58.5, 0.436), ("DHI", "Erbs"): (16.3, 2.7, 9.1, 0.978),
    ("DHI", "Reindl2"): (21.7, 12.0, 12.1, 0.960), ("DHI", "DIRINT"): (51.7, -30.0, 28.6, 0.787),
}


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return dict(n=len(obs), RMSE=rmse, MBE=float(err.mean()), R2=float(1 - np.sum(err ** 2) / ss_tot),
                nRMSE=rmse / obs.mean() * 100)


def process(path: Path) -> tuple[dict, pd.DataFrame]:
    meta, raw = parse_epw(path)
    day = raw[raw["GHI"] >= MIN_GHI].copy().reset_index(drop=True)
    hour = pd.DataFrame({
        "timestamp_start": day["timestamp"], "timestamp_center": day["timestamp"] + pd.Timedelta(minutes=30),
        "GHI_h": day["GHI"], "coverage": 1.0, "temp_C": day["temp_C"], "pressure_kPa": day["pressure_kPa"]})
    geo = compute_geometry_df(hour, meta["lat"], meta["lon"], meta["tz"])
    for k in ("cos_Z", "G0n", "G0h", "zenith_deg", "AM"):
        hour[k] = geo[k]
    hour["Kt"] = np.where(geo["G0h"] > 0, day["GHI"].to_numpy() / geo["G0h"], np.nan)
    dec = run_decomposition(hour, primary="DIRINT", min_cosz=MIN_COSZ)
    dec["DNI_ref"], dec["DHI_ref"] = day["DNI_ref"].to_numpy(), day["DHI_ref"].to_numpy()
    dec["month"] = day["timestamp"].dt.month.to_numpy()
    return meta, dec


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    a = ap.parse_args()
    rows = []
    for p in epw_files(a.epw_dir):
        meta, dec = process(p)
        for model, (dni_c, dhi_c) in {"DIRINT": ("DNI_dirint", "DHI_dirint"), "Erbs": ("DNI_erbs", "DHI_erbs"),
                                      "Reindl2": ("DNI_reindl2", "DHI_reindl2")}.items():
            md, mh = metrics(dec["DNI_ref"].to_numpy(), dec[dni_c].to_numpy()), metrics(dec["DHI_ref"].to_numpy(), dec[dhi_c].to_numpy())
            rows.append(dict(city=meta["city"], source=meta["source"], model=model, n=md["n"],
                             **{f"DNI_{k}": v for k, v in md.items() if k != "n"}, **{f"DHI_{k}": v for k, v in mh.items() if k != "n"}))
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "reproduce_v1_per_station.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"stations: {res['city'].nunique()}; hours per station (min/median/max): "
          f"{res.groupby('city')['n'].first().min()} / {int(res.groupby('city')['n'].first().median())} / {res.groupby('city')['n'].first().max()}")
    print("\nTable 2 reproduction — mean over stations (manuscript value in brackets):")
    print(f"{'comp':<4} {'model':<8} {'RMSE':>18} {'MBE':>18} {'nRMSE':>16} {'R2':>16}")
    for (comp, model), tgt in TABLE2.items():
        g = res[res["model"] == model]
        vals = [g[f"{comp}_{k}"].mean() for k in ("RMSE", "MBE", "nRMSE", "R2")]
        fmt = lambda v, t, f: f"{v:{f}} [{t:{f}}]"
        print(f"{comp:<4} {model:<8} {fmt(vals[0], tgt[0], '.1f'):>18} {fmt(vals[1], tgt[1], '+.1f'):>18} {fmt(vals[2], tgt[2], '.1f'):>16} {fmt(vals[3], tgt[3], '.3f'):>16}")
    print("\nMendoza vs Table S2 (DNI R2 / RMSE / MBE | DHI R2 / RMSE / MBE):")
    ref = {"DIRINT": "0.520 / 185.9 / +92.9 | 0.794 / 50.7 / -27.3", "Erbs": "0.909 / 81.1 / -19.0 | 0.978 / 16.5 / +2.9",
           "Reindl2": "0.925 / 73.4 / -24.8 | 0.946 / 26.0 / +15.5"}
    for model in ("DIRINT", "Erbs", "Reindl2"):
        r = res[(res["city"] == "MENDOZA-AERO") & (res["model"] == model)].iloc[0]
        print(f"  {model:<8} ours: {r.DNI_R2:.3f} / {r.DNI_RMSE:.1f} / {r.DNI_MBE:+.1f} | {r.DHI_R2:.3f} / {r.DHI_RMSE:.1f} / {r.DHI_MBE:+.1f}   S2: {ref[model]}")
    print("  Mendoza hours (ours):", int(res[(res['city'] == 'MENDOZA-AERO')]['n'].iloc[0]), " (Table S1: 4,428)")


if __name__ == "__main__":
    main()
