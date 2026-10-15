"""
benchmark_v2.py — Geometry-consistent benchmark of GHI decomposition models against the EPW DNI/DHI fields.

Geometry "G2" (v2, primary): cos z and extraterrestrial radiation are taken from the file itself
  cosz = etr_h / etr_n, G0h = etr_h, G0n = etr_n  (the generator's own values; exact for IWEC2 for cos z >= 0.2),
  so model-vs-reference differences are not contaminated by a different solar-position algorithm.
  If a file has etr_h > etr_n for most daytime hours the two columns are swapped (General Pico) and are swapped back.

Models (all run on the full 8760 h series; NaN where a model is undefined):
  erbs            solar_decomp.decomposition.erbs_series            (Erbs et al. 1982)
  reindl2         solar_decomp.decomposition.reindl2_series         (as implemented in v1.0.0; coefficients NOT yet
                                                                     verified against the primary paper)
  dirint_v1       solar_decomp.decomposition.dirint_series          (v1.0.0 'DIRINT': own table, one-sided dKt',
                                                                     W from T,P at RH=70 %), continuous series
  dirint_std      pvlib.irradiance.dirint (Perez et al. 1992), dew point from the EPW, EPW pressure,
                  two-sided dKt', continuous series
  dirint_std_noTd same as dirint_std but WITHOUT dew point (Perez's «unknown precipitable water» bin; added 2026-10-05: it is the default of v1.1.0)
  erbs_driesse, orgill_hollands, boland, disc : pvlib reference implementations (extra context; DHI by closure for disc)
The submitted analysis is reproduced separately by reproduce_v1_batch.py (geometry "G1").

Domains for metrics (all with GHI >= 10 W/m2):
  all   : cos z > 0.0174  (as the submitted batch)
  c065  : cos z >= 0.065  (pvlib default domain)
  c20   : cos z >= 0.20   (generator geometry reliable)

Outputs (revision_v2/data/processed): benchmark_v2_per_station.csv, hourly_v2.pkl.gz

Usage: python benchmark_v2.py --epw-dir <folder>
"""
from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import decomposition as dec  # noqa: E402
from epw_io import epw_files, read_epw  # noqa: E402
from solar_geometry import air_mass_kasten  # noqa: E402

OUT = ROOT / "revision_v2" / "data" / "processed"
MIN_GHI, MIN_COSZ = 10.0, 0.0174
DOMAINS = {"all": 0.0174, "c065": 0.065, "c20": 0.20}

# Koppen class used in the SUBMITTED Table S1 (by WMO code; the two non-WMO files by city name)
CLASS_S1 = {
    "870220": "Aw", "870460": "BSh", "870470": "BSh", "870970": "Cfa", "871620": "Aw", "871490": "Cfa", "871210": "Cfa",
    "871780": "Cfa", "871550": "Cfa", "871660": "Cfa", "871290": "BSk", "872220": "BWk", "872700": "Cfa", "872170": "BWh",
    "872890": "Cfa", "873950": "Cfa", "873440": "BSk", "873110": "BWk", "873710": "Cfa", "873740": "Cfa", "873280": "BSk",
    "874180": "BWk", "874800": "Cfa", "874970": "Cfa", "874360": "BSk", "874480": "BSk", "875480": "BSk", "875820": "Cfa",
    "875090": "BWk", "875060": "BWk", "876230": "BSk", "877150": "BSk", "877910": "BSk", "877650": "Cfb", "878030": "BSk",
    "878280": "BWk", "878600": "BSk", "879250": "BSk", "879380": "ET",
}
CLASS_BY_CITY = {"General Pico Airp.": "BSk", "Andalgalá": "BWk"}   # Andalgala: BWk in Table S1, BSk in Table S2 (!)


def arr(x) -> np.ndarray:
    """pvlib returns Series, DataFrame columns or ndarrays depending on the function and input types."""
    return np.asarray(x, dtype=float)


def call(fn, **kw):
    """Call a pvlib function passing only the keyword arguments its installed version accepts."""
    ok = set(inspect.signature(fn).parameters)
    return fn(**{k: v for k, v in kw.items() if k in ok})


def prepare(meta: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    etr_h, etr_n = df["etr_h"].to_numpy().copy(), df["etr_n"].to_numpy().copy()
    day = (df["ghi"].to_numpy() >= MIN_GHI)
    swapped = bool(np.mean(etr_h[day] > etr_n[day] * 1.0001) > 0.5)
    if swapped:
        etr_h, etr_n = etr_n, etr_h
    sun = (etr_n > 0) & (etr_h > 0)
    cosz = np.where(sun, np.clip(etr_h / np.where(sun, etr_n, 1.0), 0, 1), 0.0)
    zen = np.degrees(np.arccos(np.clip(cosz, -1, 1)))
    am = np.full(len(df), np.nan)
    for i in np.flatnonzero(sun & (zen < 90)):
        am[i] = air_mass_kasten(float(zen[i]))
    ghi = df["ghi"].to_numpy()
    d = pd.DataFrame({
        "GHI_h": ghi, "G0h": etr_h.astype(float), "G0n": etr_n.astype(float), "cos_Z": cosz, "zenith_deg": zen, "AM": am,
        "Kt": np.where(etr_h > 0, ghi / np.where(etr_h > 0, etr_h, 1.0), np.nan),
        "temp_C": df["t_db"].to_numpy(), "pressure_kPa": df["p_pa"].to_numpy() / 1000.0})
    return d, swapped


def run_models(d: pd.DataFrame, df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ghi, cosz = d["GHI_h"].to_numpy(), d["cos_Z"].to_numpy()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["erbs"] = (lambda r: (r["DNI_erbs"].to_numpy(), r["DHI_erbs"].to_numpy()))(dec.erbs_series(d, min_cosz=MIN_COSZ))
    out["reindl2"] = (lambda r: (r["DNI_reindl2"].to_numpy(), r["DHI_reindl2"].to_numpy()))(dec.reindl2_series(d, min_cosz=MIN_COSZ))
    out["dirint_v1"] = (lambda r: (r["DNI_dirint"].to_numpy(), r["DHI_dirint"].to_numpy()))(dec.dirint_series(d, min_cosz=MIN_COSZ))

    times = pd.DatetimeIndex(df["t_start"])
    doy = df["doy"].to_numpy()
    zen = pd.Series(d["zenith_deg"].to_numpy(), index=times)
    g = pd.Series(ghi, index=times)
    kw = dict(min_cos_zenith=MIN_COSZ, max_zenith=89.0)

    pv = call(pvlib.irradiance.dirint, ghi=g, solar_zenith=zen, times=times, pressure=df["p_pa"].to_numpy(),
              use_delta_kt_prime=True, temp_dew=pd.Series(df["t_dew"].to_numpy(), index=times), **kw)
    pv = arr(pv)
    out["dirint_std"] = (pv, ghi - pv * cosz)
    # same call without dew point: Perez's «unknown precipitable water» bin = the DEFAULT of Solar Decomp v1.1.0 (dew point is opt-in there)
    pv0 = arr(call(pvlib.irradiance.dirint, ghi=g, solar_zenith=zen, times=times, pressure=df["p_pa"].to_numpy(),
                   use_delta_kt_prime=True, temp_dew=None, **kw))
    out["dirint_std_noTd"] = (pv0, ghi - pv0 * cosz)
    r = call(pvlib.irradiance.erbs_driesse, ghi=g, zenith=zen, datetime_or_doy=doy, **kw)
    out["erbs_driesse"] = (arr(r["dni"]), arr(r["dhi"]))
    r = call(pvlib.irradiance.orgill_hollands, ghi=g, zenith=zen, datetime_or_doy=doy, **kw)
    out["orgill_hollands"] = (arr(r["dni"]), arr(r["dhi"]))
    r = call(pvlib.irradiance.boland, ghi=g, solar_zenith=zen, datetime_or_doy=doy, **kw)
    out["boland"] = (arr(r["dni"]), arr(r["dhi"]))
    r = call(pvlib.irradiance.disc, ghi=g, solar_zenith=zen, datetime_or_doy=doy, pressure=df["p_pa"].to_numpy(), **kw)
    dn = arr(r["dni"])
    out["disc"] = (dn, ghi - dn * cosz)
    return out


def metr(obs, pred) -> dict:
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    e = p - o
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return dict(n=int(m.sum()), RMSE=rmse, MBE=float(e.mean()), nRMSE=rmse / float(o.mean()) * 100.0,
                R2=float(1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum()))


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    a = ap.parse_args()
    rows, hourly = [], []
    for p in epw_files(a.epw_dir):
        meta, df = read_epw(p)
        d, swapped = prepare(meta, df)
        mods = run_models(d, df)
        dni, dhi = df["dni"].to_numpy(), df["dhi"].to_numpy()
        klass = CLASS_S1.get(meta["wmo"]) or CLASS_BY_CITY[meta["city"]]
        base = dict(city=meta["city"], source=meta["source"], koppen_s1=klass, etr_swapped=swapped, lat=meta["lat"])
        for dom, c0 in DOMAINS.items():
            mask = (d["GHI_h"].to_numpy() >= MIN_GHI) & (d["cos_Z"].to_numpy() >= c0) & (d["cos_Z"].to_numpy() > MIN_COSZ)
            for name, (mdni, mdhi) in mods.items():
                md, mh = metr(dni[mask], mdni[mask]), metr(dhi[mask], mdhi[mask])
                rows.append({**base, "model": name, "domain": dom, "n": md["n"],
                             **{f"DNI_{k}": v for k, v in md.items() if k != "n"}, **{f"DHI_{k}": v for k, v in mh.items() if k != "n"}})
        h = pd.DataFrame({"city": meta["city"], "source": meta["source"], "koppen_s1": klass, "month": df["month"].to_numpy(),
                          "hour": df["hour"].to_numpy(), "ghi": d["GHI_h"].to_numpy(), "kt": d["Kt"].to_numpy(),
                          "cosz": d["cos_Z"].to_numpy(), "am": d["AM"].to_numpy(), "dni_ref": dni, "dhi_ref": dhi,
                          "t_dew": df["t_dew"].to_numpy(), "doy": df["doy"].to_numpy()})
        for name, (mdni, mdhi) in mods.items():
            h[f"dni_{name}"], h[f"dhi_{name}"] = mdni, mdhi
        hourly.append(h[(h["ghi"] >= MIN_GHI) & (h["cosz"] > MIN_COSZ)])
        print(f"{meta['city']:<22} swapped={swapped!s:<5} hours={int(((d['GHI_h']>=MIN_GHI)&(d['cos_Z']>MIN_COSZ)).sum())}")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "benchmark_v2_per_station.csv", index=False)
    pd.concat(hourly).to_pickle(OUT / "hourly_v2.pkl.gz")

    pd.set_option("display.width", 220)
    iw = res[res["source"] == "IWEC2"]
    for dom in ("all", "c20"):
        g = iw[iw["domain"] == dom].groupby("model")[["DNI_R2", "DNI_RMSE", "DNI_MBE", "DNI_nRMSE", "DHI_R2", "DHI_RMSE", "DHI_MBE"]].mean().round(3)
        print(f"\nIWEC2 stations (n={iw['city'].nunique()}), domain={dom}: mean over stations")
        print(g.to_string())


if __name__ == "__main__":
    main()
