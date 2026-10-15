"""anchor_pilot_bootstrap.py — day-block bootstrap for the anchor pilot (T1/T2), domain c20.

Hours of the same day are strongly dependent, so the unit of resampling is the UTC calendar day (not the hour). For each station the
script rebuilds the same hourly series as bsrn_anchor.anchor (same functions, same parameters), then draws B bootstrap samples of days and
reports: 95 % percentile CI of the DNI RMSE of every model and of the Gompertz split (T1), and of the paired RMSE differences versus Erbs,
the rank of each model in every draw (share of draws in which it is first, and mean rank).

Seeded (numpy default_rng, SEED below). Run from revision_v2:  python scripts/anchor_pilot_bootstrap.py
Writes data/processed/anchor_pilot_bootstrap_<STN>.csv and prints a summary.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bsrn_anchor as ba  # noqa: E402
import run_anchor_pilot as rp  # noqa: E402
from benchmark_v2 import DOMAINS, MIN_COSZ, MIN_GHI, prepare, run_models  # noqa: E402
from gompertz import gompertz_dni  # noqa: E402

B = 2000
SEED = 20261005
DOMAIN = "c20"


def hourly_arrays(stn: str):
    kind, files = {
        "PTR": ("pangaea", sorted((rp.RAW / "PTR").glob("PTR_radiation_2016-*.tab"))),
        "SMS": ("pangaea", sorted((rp.RAW / "SMS").glob("SMS_radiation_2016-*.tab"))),
        "DRA": ("surfrad", sorted((rp.RAW / "DRA").glob("dra16???.dat"))),
    }[stn]
    raw, meta = rp.read_pangaea_station(files) if kind == "pangaea" else rp.read_surfrad_year(files)
    minute = rp.regularise(raw, "2016-01-01 00:00", "2016-12-31 23:59")
    geo = ba.geometry(minute.index, meta["lat"], meta["lon"], meta["elev"], "start")
    qc = ba.bsrn_qc(minute, geo)
    h = ba.to_hourly(minute, geo, qc, "start")
    epwlike = ba.to_epw_like(h, "avg")
    d, _ = prepare({}, epwlike)
    models = run_models(d, epwlike)
    cosz = d["cos_Z"].to_numpy()
    i0 = ba.PARAMS["generator_i0"] * ba.e0_ashrae(epwlike["doy"].to_numpy())
    gomp = gompertz_dni(np.nan_to_num(d["GHI_h"].to_numpy()), cosz, i0)
    gomp = np.where(np.isfinite(d["GHI_h"].to_numpy()), gomp, np.nan)
    dni = h["dni"].to_numpy()
    mask = (np.isfinite(d["GHI_h"].to_numpy()) & np.isfinite(dni) & (d["GHI_h"].to_numpy() >= MIN_GHI)
            & (cosz >= DOMAINS[DOMAIN]) & (cosz > MIN_COSZ))
    series = {"gompertz (T1)": gomp}
    series.update({k: v[0] for k, v in models.items()})
    day = h.index.floor("D").to_numpy()
    return dni, series, mask, day


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)
    summary = []
    for stn in ("PTR", "SMS", "DRA"):
        dni, series, mask, day = hourly_arrays(stn)
        # one common sample per model: hours where the measurement, the model and the domain mask are all valid
        names = list(series)
        ok = mask.copy()
        for k in names:
            ok &= np.isfinite(series[k])
        days = np.unique(day[ok])
        idx_by_day = {d: np.flatnonzero(ok & (day == d)) for d in days}
        err = {k: (series[k] - dni) for k in names}
        # per-day sufficient statistics: n, sum of squared errors, sum of errors
        n_d = np.array([len(idx_by_day[d]) for d in days], float)
        sse = {k: np.array([np.sum(err[k][idx_by_day[d]] ** 2) for d in days]) for k in names}
        se = {k: np.array([np.sum(err[k][idx_by_day[d]]) for d in days]) for k in names}
        point = {k: float(np.sqrt(sse[k].sum() / n_d.sum())) for k in names}
        boot = {k: np.empty(B) for k in names}
        bias = {k: np.empty(B) for k in names}
        ranks = np.empty((B, len(names)))
        for b in range(B):
            pick = rng.integers(0, len(days), len(days))
            nn = n_d[pick].sum()
            for k in names:
                boot[k][b] = np.sqrt(sse[k][pick].sum() / nn)
                bias[k][b] = se[k][pick].sum() / nn
            ranks[b] = pd.Series([boot[k][b] for k in names]).rank().to_numpy()
        rows = []
        for j, k in enumerate(names):
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            diff = boot[k] - boot["erbs"]
            dlo, dhi = np.percentile(diff, [2.5, 97.5])
            blo, bhi = np.percentile(bias[k], [2.5, 97.5])
            rows.append(dict(station=stn, model=k, n_hours=int(n_d.sum()), n_days=len(days), RMSE=point[k], RMSE_lo=lo, RMSE_hi=hi,
                             MBE=float(se[k].sum() / n_d.sum()), MBE_lo=blo, MBE_hi=bhi,
                             dRMSE_vs_erbs=point[k] - point["erbs"], dRMSE_lo=dlo, dRMSE_hi=dhi,
                             mean_rank=float(ranks[:, j].mean()), share_first=float((ranks[:, j] == 1).mean())))
        out = pd.DataFrame(rows)
        out.to_csv(rp.OUT / f"anchor_pilot_bootstrap_{stn}.csv", index=False)
        summary.append(out)
        pd.set_option("display.width", 230)
        print(f"\n== {stn}: common sample {int(n_d.sum())} hours in {len(days)} days (c20), B = {B}")
        print(out.drop(columns=["station", "n_hours", "n_days"]).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
