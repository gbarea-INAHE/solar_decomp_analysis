"""
bsrn_anchor.py — External anchor with MEASURED data (Task 2). Prepared and tested on SYNTHETIC data only: the session that wrote it
could not reach PANGAEA/BSRN, so nothing here has been run on a real BSRN file.

What it answers (the three questions that Test 0 makes necessary, because IWEC2 DNI is the Gompertz split of the file GHI):
  T1  How far is the Gompertz split (generator chain of IWEC2) from MEASURED DNI at a station?
  T2  What is the model ranking against measured DNI, and does it agree with the ranking obtained against the EPW reference?
  T3  How close is each model to the Gompertz curve itself (RMSD model-Gompertz) compared with its error against measurements?

Pipeline (every threshold is a provisional parameter in PARAMS, documented in DECISIONS_PROVISIONAL.md once real data is used):
  1-min GHI/DNI/DHI (UTC) -> geometry at mid-minute (pvlib NREL SPA) -> BSRN-style QC -> hourly means over valid minutes with a
  coverage rule -> EPW-like hourly frame -> the SAME model code as benchmark_v2.py (repo erbs/reindl2/dirint_v1, pvlib dirint & others)
  -> metrics over the same domains (all / c065 / c20) -> T1-T3 tables.

QC limits: Long & Dutton (2002), "BSRN Global Network recommended QC tests, V2.0" (physically possible / extremely rare limits) and the
closure test of QCRad (Long & Shi 2008). The constants below were written from memory of those documents and are NOT verified: check them
against the source before any number goes in the paper (LIMITS_VERIFIED = False).

Time convention: minute stamps are taken as the START of the averaging minute (parameter stamp="start"); BSRN station-to-archive files
and PANGAEA tables must be checked for this before use.

Usage:
  python bsrn_anchor.py --selftest
  python bsrn_anchor.py --minute-csv file.csv --lat -31.6 --lon -68.3 --elev 700 --out data/processed/anchor_<station>.csv
      (CSV columns: time_utc, ghi, dni, dhi [W/m2]; optional t_c, rh, p_hpa)
  python bsrn_anchor.py --bsrn-files PAY0116.dat.gz PAY0216.dat.gz ... --out data/processed/anchor_PAY.csv
      (BSRN station-to-archive monthly files, LR0100, read with pvlib.iotools.read_bsrn; lat/lon/elevation come from the file's LR0004
       unless given; consecutive months only; BSRN data are free after registering, see the BSRN data release guidelines)
"""
from __future__ import annotations

import argparse
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

from benchmark_v2 import DOMAINS, MIN_COSZ, MIN_GHI, metr, prepare, run_models  # noqa: E402
from gompertz import gompertz_dni  # noqa: E402

LIMITS_VERIFIED = False
PARAMS = dict(
    min_valid_fraction=0.75,     # hour kept if >= 45 of 60 minutes pass QC (provisional)
    closure_tol_lowsza=0.08,     # |GHI/(DHI+DNI mu) - 1| for SZA <= 75 deg and GHI > 50 W/m2
    closure_tol_highsza=0.15,    # 75 < SZA < 93 deg
    closure_min_ghi=50.0,
    generator_i0=1365.8,         # effective I0 of the IWEC2 generator (Test 0), ASHRAE eccentricity form
)


def e0_ashrae(doy: np.ndarray) -> np.ndarray:
    return 1.0 + 0.033 * np.cos(2 * np.pi * (np.asarray(doy, dtype=float) - 3) / 365.0)


# ----------------------------------------------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------------------------------------------
def read_minute_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time_utc", "ghi", "dni", "dhi"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(df.columns))}; found {list(df.columns)}")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    if df["time_utc"].duplicated().any():
        raise ValueError("duplicate timestamps")
    return df.set_index("time_utc").sort_index()


def read_bsrn_station_files(paths) -> tuple[pd.DataFrame, dict]:
    """BSRN station-to-archive files (LR0100) via pvlib.iotools.read_bsrn. Timestamps are UTC, start of the minute in pvlib's parser.
    Returns the minute frame (ghi, dni, dhi, t_c, rh, p_hpa) and {lat, lon, elev, station}. Fails if the files disagree on the station
    coordinates or contain no LR0100 data. -999 / -99.9 become NaN inside pvlib."""
    frames, metas = [], []
    for p in paths:
        data, meta = pvlib.iotools.read_bsrn(p, logical_records=("0100",))
        if data.empty:
            raise ValueError(f"{p}: no LR0100 records found")
        frames.append(data[["ghi", "dni", "dhi", "temp_air", "relative_humidity", "pressure"]]
                      .rename(columns={"temp_air": "t_c", "relative_humidity": "rh", "pressure": "p_hpa"}))
        metas.append(meta)
    keys = ("latitude", "longitude", "altitude", "station identification number")
    if any(tuple(m[k] for k in keys) != tuple(metas[0][k] for k in keys) for m in metas[1:]):
        raise ValueError("files disagree on station coordinates/identification; do not mix stations or description changes silently")
    df = pd.concat(frames).sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps across files")
    df.index.name = "time_utc"
    m0 = metas[0]
    return df, dict(lat=float(m0["latitude"]), lon=float(m0["longitude"]), elev=float(m0["altitude"]), station=int(m0["station identification number"]))


def read_pangaea_tab(path: str | Path) -> pd.DataFrame:
    """PANGAEA tab-delimited BSRN table (metadata block between '/*' and '*/', then a header row).
    FORMAT ASSUMPTION from memory (columns 'Date/Time', 'SWD', 'DIR', 'DIF', optional 'T2', 'RH', 'PoPoPo'); fails loudly if the header differs."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == "*/"), 0)
    header = lines[start].split("\t")
    def col(prefix):
        hit = [i for i, h in enumerate(header) if h.startswith(prefix)]
        if len(hit) != 1:
            raise ValueError(f"{path}: expected exactly one column starting with {prefix!r}; header = {header}")
        return hit[0]
    ic = {"time_utc": col("Date/Time"), "ghi": col("SWD"), "dni": col("DIR"), "dhi": col("DIF")}
    for name, pre in (("t_c", "T2"), ("rh", "RH"), ("p_hpa", "PoPoPo")):
        hit = [i for i, h in enumerate(header) if h.startswith(pre)]
        if len(hit) == 1:
            ic[name] = hit[0]
    rows = [ln.split("\t") for ln in lines[start + 1:] if ln.strip()]
    df = pd.DataFrame({k: [r[i] if i < len(r) else "" for r in rows] for k, i in ic.items()})
    for k in df.columns[1:]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df.set_index("time_utc").sort_index()


# ----------------------------------------------------------------------------------------------------------------------
# Geometry and QC
# ----------------------------------------------------------------------------------------------------------------------
def geometry(index_utc: pd.DatetimeIndex, lat: float, lon: float, elev: float, stamp: str = "start") -> pd.DataFrame:
    if stamp not in ("start", "end"):
        raise ValueError("stamp must be 'start' or 'end'")
    shift = pd.Timedelta(seconds=30) if stamp == "start" else pd.Timedelta(seconds=-30)
    mid = index_utc + shift
    sp = pvlib.solarposition.get_solarposition(mid, lat, lon, altitude=elev, method="nrel_numpy")
    zen = sp["zenith"].to_numpy()
    cosz = np.cos(np.radians(zen))
    s0 = pvlib.irradiance.get_extra_radiation(mid, method="spencer").to_numpy()   # S0 with Earth-Sun distance, for the QC limits
    doy = mid.dayofyear.to_numpy()
    return pd.DataFrame({"zenith": zen, "cosz": cosz, "s0": s0, "doy": doy}, index=index_utc)


def bsrn_qc(df: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame of failed tests. A minute is valid only if ghi, dni and dhi are all present and no test fails."""
    mu = np.clip(geo["cosz"].to_numpy(), 0, None)
    s0 = geo["s0"].to_numpy()
    g, n, d = (df[c].to_numpy(float) for c in ("ghi", "dni", "dhi"))
    sun = geo["cosz"].to_numpy() > 0
    f = pd.DataFrame(index=df.index)
    f["missing"] = ~(np.isfinite(g) & np.isfinite(n) & np.isfinite(d))
    with np.errstate(invalid="ignore"):
        f["phys"] = ((g < -4) | (g > 1.5 * s0 * mu ** 1.2 + 100) | (n < -4) | (n > s0) | (d < -4) | (d > 0.95 * s0 * mu ** 1.2 + 50))
        f["rare"] = ((g < -2) | (g > 1.2 * s0 * mu ** 1.2 + 50) | (n < -2) | (n > 0.95 * s0 * mu ** 0.2 + 10) | (d < -2) | (d > 0.75 * s0 * mu ** 1.2 + 30))
        tot = d + n * mu
        ratio = np.where(tot > 0, g / np.where(tot > 0, tot, 1.0), np.nan)
        sza = geo["zenith"].to_numpy()
        tol = np.where(sza <= 75, PARAMS["closure_tol_lowsza"], PARAMS["closure_tol_highsza"])
        f["closure"] = sun & (g > PARAMS["closure_min_ghi"]) & (sza < 93) & (np.abs(ratio - 1) > tol)
    f["missing_any_fail"] = False
    f["invalid"] = f[["missing", "phys", "rare", "closure"]].any(axis=1)
    return f.drop(columns="missing_any_fail")


# ----------------------------------------------------------------------------------------------------------------------
# Hourly aggregation
# ----------------------------------------------------------------------------------------------------------------------
def to_hourly(df: pd.DataFrame, geo: pd.DataFrame, qc: pd.DataFrame, stamp: str = "start") -> pd.DataFrame:
    """Hourly means over VALID minutes; hours below the coverage rule are NaN. Sun-below-horizon minutes are set to 0 (not dropped), as in an
    hour-average convention. Output index = hour start (UTC), complete and regular (night rows and gaps kept)."""
    d = df[["ghi", "dni", "dhi"]].copy()
    sun = geo["cosz"] > 0
    d.loc[~sun, ["ghi", "dni", "dhi"]] = 0.0
    valid = ~qc["invalid"] | ~sun   # night minutes count as valid zeros unless values are missing
    valid &= ~qc["missing"]
    d = d.where(valid)
    hour = d.index.floor("h")
    g = d.groupby(hour)
    out = pd.DataFrame({
        "ghi": g["ghi"].mean(), "dni": g["dni"].mean(), "dhi": g["dhi"].mean(),
        "n_valid": valid.groupby(hour).sum(), "n_min": valid.groupby(hour).size(),
    })
    out["cov"] = out["n_valid"] / 60.0
    ok = out["cov"] >= PARAMS["min_valid_fraction"]
    out.loc[~ok, ["ghi", "dni", "dhi"]] = np.nan
    gh = geo.groupby(hour)
    out["cosz_avg"] = gh["cosz"].apply(lambda s: float(np.clip(s.to_numpy(), 0, None).mean()))
    out["s0_mean"] = gh["s0"].mean()
    out["doy"] = gh["doy"].first()
    for c in ("t_c", "rh", "p_hpa"):
        if c in df.columns:
            out[c] = df[c].groupby(hour).mean()
    full = pd.date_range(out.index[0], out.index[-1], freq="h")
    out = out.reindex(full)
    if out["cosz_avg"].isna().any():
        raise ValueError("hours without any minute in the record inside the span; split the record into continuous blocks first")
    return out


def dew_point_c(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Magnus (a=17.62, b=243.12 C), valid for -45..60 C; used only to feed the dew-point input of DIRINT."""
    a, b = 17.62, 243.12
    gam = np.log(np.clip(rh, 1.0, 100.0) / 100.0) + a * t_c / (b + t_c)
    return b * gam / (a - gam)


def to_epw_like(h: pd.DataFrame, cosz_variant: str) -> pd.DataFrame:
    for c in ("t_c", "rh", "p_hpa"):
        if c not in h.columns or h[c].isna().all():
            raise ValueError(f"column {c!r} is required by the dew-point DIRINT run; no silent default is used")
    t = h.index
    # extraterrestrial normal irradiance in the generator convention is irrelevant for models; use the hour-mean S0 (Spencer)
    cz = h["cosz_avg"].to_numpy() if cosz_variant == "avg" else None
    if cz is None:
        raise ValueError("cosz_variant must be 'avg' (interval-average cos z); the mid-hour variant needs the geometry frame")
    etr_n = h["s0_mean"].to_numpy()
    return pd.DataFrame({
        "t_start": t.tz_localize(None), "doy": h["doy"].ffill().bfill().to_numpy().astype(int), "ghi": h["ghi"].to_numpy(),
        "etr_n": etr_n, "etr_h": etr_n * cz, "t_db": h["t_c"].to_numpy(), "t_dew": dew_point_c(h["t_c"].to_numpy(), h["rh"].to_numpy()),
        "p_pa": h["p_hpa"].to_numpy() * 100.0,
    })


# ----------------------------------------------------------------------------------------------------------------------
# T1-T3
# ----------------------------------------------------------------------------------------------------------------------
def anchor(minute: pd.DataFrame, lat: float, lon: float, elev: float, stamp: str = "start") -> tuple[pd.DataFrame, pd.DataFrame]:
    geo = geometry(minute.index, lat, lon, elev, stamp)
    qc = bsrn_qc(minute, geo)
    h = to_hourly(minute, geo, qc, stamp)
    epwlike = to_epw_like(h, "avg")
    d, _ = prepare({}, epwlike)
    # Models need a gap-free series; hours with NaN GHI are passed through as NaN and the result is checked below.
    models = run_models(d, epwlike)
    cosz = d["cos_Z"].to_numpy()
    i0 = PARAMS["generator_i0"] * e0_ashrae(epwlike["doy"].to_numpy())
    gomp = gompertz_dni(np.nan_to_num(d["GHI_h"].to_numpy()), cosz, i0)
    gomp = np.where(np.isfinite(d["GHI_h"].to_numpy()), gomp, np.nan)
    dni = h["dni"].to_numpy()
    rows, sim = [], []
    for dom, c0 in DOMAINS.items():
        mask = np.isfinite(d["GHI_h"].to_numpy()) & np.isfinite(dni) & (d["GHI_h"].to_numpy() >= MIN_GHI) & (cosz >= c0) & (cosz > MIN_COSZ)
        t1 = metr(dni[mask], gomp[mask])
        rows.append(dict(domain=dom, model="gompertz (T1)", **{f"DNI_{k}": v for k, v in t1.items()}))
        for name, (mdni, _) in models.items():
            m = metr(dni[mask], mdni[mask])
            rows.append(dict(domain=dom, model=name, **{f"DNI_{k}": v for k, v in m.items()}))
            mm = mask & np.isfinite(mdni)
            sim.append(dict(domain=dom, model=name, n=int(mm.sum()),
                            RMSD_vs_measured=float(np.sqrt(np.mean((mdni[mm] - dni[mm]) ** 2))),
                            RMSD_vs_gompertz=float(np.sqrt(np.mean((mdni[mm] - gomp[mm]) ** 2)))))
    res, simdf = pd.DataFrame(rows), pd.DataFrame(sim)
    simdf["ratio_gomp_over_meas"] = simdf["RMSD_vs_gompertz"] / simdf["RMSD_vs_measured"]
    return res, simdf


def rank_agreement(res: pd.DataFrame, epw_rank_csv: str | Path | None, domain: str = "c20") -> pd.DataFrame:
    """T2: ranking by DNI RMSE against measurements, optionally with Spearman agreement against the EPW-reference ranking
    (benchmark_v2_per_station.csv, mean over IWEC2 stations)."""
    r = res[(res["domain"] == domain) & (~res["model"].str.contains("T1"))][["model", "DNI_RMSE"]].copy()
    r["rank_measured"] = r["DNI_RMSE"].rank()
    if epw_rank_csv is not None:
        e = pd.read_csv(epw_rank_csv)
        e = e[(e["source"] == "IWEC2") & (e["domain"] == domain)].groupby("model")["DNI_RMSE"].mean().rename("DNI_RMSE_epw").reset_index()
        r = r.merge(e, on="model", how="inner")
        r["rank_epw"] = r["DNI_RMSE_epw"].rank()
        r.attrs["spearman"] = float(r["rank_measured"].corr(r["rank_epw"], method="pearson"))   # Pearson on ranks == Spearman
    return r.sort_values("rank_measured")


# ----------------------------------------------------------------------------------------------------------------------
# Self-test on synthetic data
# ----------------------------------------------------------------------------------------------------------------------
LAT, LON, ELEV = -31.65, -68.30, 700.0   # a Mendoza-like point, only to place the synthetic sun


def make_synthetic_minute(days: int, truth: str, seed: int = 11, start: str = "2001-03-01") -> pd.DataFrame:
    """truth: 'gompertz' (DNI = Gompertz of the 1-min GHI; closure for DHI), 'disc' (pvlib DISC on the 1-min GHI) or 'physical'
    (DNI = clear-sky DNI x cloud factor tied to the clear-sky index). All are SYNTHETIC; they only exercise the plumbing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="min", tz="UTC")
    geo = geometry(idx, LAT, LON, ELEV)
    cosz = np.clip(geo["cosz"].to_numpy(), 0, None)
    loc = pvlib.location.Location(LAT, LON, altitude=ELEV)
    cs = loc.get_clearsky(idx + pd.Timedelta(seconds=30), model="ineichen")
    cs_ghi, cs_dni = cs["ghi"].to_numpy(), cs["dni"].to_numpy()
    # clear-sky index: slow AR(1) in logit space per ~hour block plus minute noise
    blocks = np.repeat(rng.normal(0, 1.2, size=days * 24 + 1), 60)[: len(idx)]
    kc = np.clip(1 / (1 + np.exp(-(0.8 + blocks))) * 1.05 + rng.normal(0, 0.03, len(idx)), 0.05, 1.12)
    ghi = np.where(cosz > 0, cs_ghi * kc, 0.0)
    if truth == "gompertz":
        i0 = PARAMS["generator_i0"] * e0_ashrae(geo["doy"].to_numpy())
        dni = gompertz_dni(ghi, cosz, i0)
    elif truth == "disc":
        r = pvlib.irradiance.disc(pd.Series(ghi, index=idx), pd.Series(geo["zenith"].to_numpy(), index=idx), idx, pressure=93000.0)
        dni = r["dni"].to_numpy()
    elif truth == "physical":
        dni = cs_dni * np.clip((kc - 0.35) / 0.65, 0, 1) ** 1.3
    else:
        raise ValueError(truth)
    dni = np.where(cosz > 0, np.minimum(dni, 0.95 * geo["s0"].to_numpy()), 0.0)
    dhi = np.clip(ghi - dni * cosz, 0, None)
    ghi = dhi + dni * cosz   # exact closure before noise
    noise = lambda x: x + rng.normal(0, 1.0, len(x)) * (x > 0)
    t_c = 18 + 8 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 9) / 24)
    return pd.DataFrame({"ghi": noise(ghi), "dni": noise(dni), "dhi": noise(dhi), "t_c": t_c, "rh": 45.0, "p_hpa": 930.0}, index=idx)


def write_bsrn_lr0100(df: pd.DataFrame, path: Path, station: int, lat: float, lon: float, alt: int) -> None:
    """Write ONE month of minute data as a minimal station-to-archive file with the layout pvlib's parser expects (LR0001/LR0004 header,
    then LR0100 with two lines per minute and the fixed-width columns of pvlib.iotools.bsrn.BSRN_LR0100_COL_SPECS). Used only to test the
    reader path; the layout comes from pvlib's parser, NOT from the official BSRN manual."""
    first = df.index[0]
    month_idx = df.index[(df.index.year == first.year) & (df.index.month == first.month)]
    d = df.loc[month_idx]
    lines = ["*U0001", f"{station:3d}{first.month:3d} {first.year:4d}  1", "*C0004", "20010101", f"{1:3d}{2:3d}", "Synthetic address",
             "000000", "000000", "0.0.0.0", "none@example.org", f"{lat + 90:8.3f}{lon + 180:8.3f}{alt:5d} 00000", "20010101", "  -1   0", "*U0100"]
    f = lambda v, w, nd: " " * w if not np.isfinite(v) else f"{v:{w}.{nd}f}"
    for t, r in d.iterrows():
        day, minute = t.day, t.hour * 60 + t.minute
        l1 = f"{day:3d} {minute:5d} " + f(r["ghi"], 6, 1) + f(0.0, 6, 1) + f(r["ghi"], 5, 0) + f(r["ghi"], 5, 0) + f(r["dni"], 7, 1) + f(0.0, 6, 1) \
            + f(r["dni"], 5, 0) + f(r["dni"], 5, 0) + " " * 9 + " " * 6 + " " * 5
        l2 = " " * 3 + " " + " " * 5 + " " + f(r["dhi"], 6, 1) + f(0.0, 6, 1) + f(r["dhi"], 5, 0) + f(r["dhi"], 5, 0) + f(-999.0, 7, 1) + f(0.0, 6, 1) \
            + f(-999.0, 5, 0) + f(-999.0, 5, 0) + f(r["t_c"], 9, 1) + f(r["rh"], 6, 0) + f(r["p_hpa"], 5, 0)
        lines += [l1, l2]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def selftest() -> None:
    warnings.filterwarnings("ignore")
    # 1) hourly aggregation on a hand-checkable case
    idx = pd.date_range("2001-03-01 12:00", periods=120, freq="min", tz="UTC")     # two noon hours
    m = pd.DataFrame({"ghi": 600.0, "dni": 800.0, "dhi": 100.0, "t_c": 20.0, "rh": 50.0, "p_hpa": 930.0}, index=idx)
    geo = geometry(idx, LAT, LON, ELEV)
    qc = bsrn_qc(m, geo)
    mu = geo["cosz"].to_numpy()
    m["ghi"] = m["dhi"] + m["dni"] * np.clip(mu, 0, None)           # exact closure
    qc = bsrn_qc(m, geo)
    m.loc[m.index[60:90], "dni"] = np.nan                            # half of the second hour missing -> coverage 0.5
    qc = bsrn_qc(m, geo)
    h = to_hourly(m, geo, qc)
    assert abs(h["ghi"].iloc[0] - m["ghi"].iloc[:60].mean()) < 1e-9, "hourly mean wrong"
    assert np.isnan(h["ghi"].iloc[1]), "coverage rule not applied"
    print("  hourly aggregation and coverage rule: OK")
    # 2) QC catches injected faults and keeps clean data
    s = make_synthetic_minute(2, "physical")
    g = geometry(s.index, LAT, LON, ELEV)
    q0 = bsrn_qc(s, g)
    noon = np.flatnonzero((g["cosz"].to_numpy() > 0.8))[:5]
    bad = s.copy()
    bad.iloc[noon[0], bad.columns.get_loc("ghi")] = 3000.0           # beyond physical limit
    bad.iloc[noon[1], bad.columns.get_loc("dni")] = -50.0
    bad.iloc[noon[2], bad.columns.get_loc("ghi")] = bad.iloc[noon[2]]["ghi"] * 0.6    # closure violation
    q1 = bsrn_qc(bad, g)
    assert q1["invalid"].to_numpy()[noon[:3]].all(), "injected faults not flagged"
    assert q0["invalid"].mean() < 0.01, f"clean synthetic data flagged too often: {q0['invalid'].mean():.3f}"
    print(f"  QC: 3/3 injected faults flagged; clean data flagged {100*q0['invalid'].mean():.2f} %")
    # 3) controls through the whole anchor: truth = Gompertz must give a small T1 error; truth = DISC / physical a larger one
    out = {}
    for truth in ("gompertz", "disc", "physical"):
        s = make_synthetic_minute(25, truth)
        res, sim = anchor(s, LAT, LON, ELEV)
        out[truth] = (res, sim)
    t1 = {k: float(v[0].query("domain=='c20' and model=='gompertz (T1)'")["DNI_RMSE"].iloc[0]) for k, v in out.items()}
    print("  T1 RMSE of Gompertz vs synthetic truth, domain c20 [W/m2]:", {k: round(v, 1) for k, v in t1.items()})
    assert t1["gompertz"] < 0.5 * min(t1["disc"], t1["physical"]), "positive control not clearly smaller than negative controls"
    # T3 sanity: with Gompertz truth, Erbs/Reindl2 are about as far from Gompertz as from the truth (ratio near 1)
    sim = out["gompertz"][1].query("domain=='c20'")
    assert (np.abs(np.log(sim["ratio_gomp_over_meas"])) < 0.35).all(), "T3 ratio far from 1 with Gompertz truth"
    sim_d = out["disc"][1].query("domain=='c20'")
    print("  T3 ratio (RMSD to Gompertz / RMSD to truth), truth=Gompertz:", sim.set_index("model")["ratio_gomp_over_meas"].round(2).to_dict())
    print("  T3 ratio, truth=DISC:", sim_d.set_index("model")["ratio_gomp_over_meas"].round(2).to_dict())
    r = rank_agreement(out["gompertz"][0], None)
    assert r["rank_measured"].min() == 1
    # 4) reader failures are loud
    tmp = Path(ROOT / "revision_v2" / "data" / "synthetic"); tmp.mkdir(parents=True, exist_ok=True)
    bad_csv = tmp / "bad_minute.csv"; bad_csv.write_text("time_utc,ghi\n2001-03-01T00:00:00Z,1\n")
    try:
        read_minute_csv(bad_csv)
    except ValueError as e:
        print("  reader fails loudly:", str(e)[:60])
    else:
        raise AssertionError("bad CSV accepted")
    # 5) the real-file reader path on a synthetic station-to-archive file (one month of 'physical' truth)
    sm = make_synthetic_minute(10, "physical", start="2001-03-01")
    fpath = Path(ROOT / "revision_v2" / "data" / "synthetic" / "SYN0301.dat")
    write_bsrn_lr0100(sm, fpath, station=99, lat=LAT, lon=LON, alt=int(ELEV))
    back, meta = read_bsrn_station_files([fpath])
    assert abs(meta["lat"] - LAT) < 1e-2 and abs(meta["lon"] - LON) < 1e-2 and meta["elev"] == int(ELEV), meta
    common = sm.index.intersection(back.index)
    assert len(common) == len(sm) and (np.abs(back.loc[common, "ghi"] - sm["ghi"].round(1)) < 0.06).all(), "round trip changed the data"
    assert np.allclose(back["dni"].to_numpy(), sm["dni"].round(1).to_numpy(), atol=0.06)
    res_a, _ = anchor(sm.assign(ghi=sm["ghi"].round(1), dni=sm["dni"].round(1), dhi=sm["dhi"].round(1), rh=sm["rh"].round(0), t_c=sm["t_c"].round(1), p_hpa=sm["p_hpa"].round(0)), LAT, LON, ELEV)
    res_b, _ = anchor(back, meta["lat"], meta["lon"], meta["elev"])
    ta, tb = (r.query("domain=='c20'").set_index("model")["DNI_RMSE"] for r in (res_a, res_b))
    assert np.allclose(ta, tb, rtol=0.02), (ta.round(2).to_dict(), tb.round(2).to_dict())
    print("  BSRN-format reader round trip: OK (data identical to 0.05 W/m2; anchor results within 2 %)")
    print("PASS (synthetic only; LIMITS_VERIFIED =", LIMITS_VERIFIED, ")")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--minute-csv"); ap.add_argument("--pangaea-tab"); ap.add_argument("--bsrn-files", nargs="+")
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float); ap.add_argument("--elev", type=float)
    ap.add_argument("--stamp", default="start"); ap.add_argument("--out")
    ap.add_argument("--epw-benchmark", help="data/processed/benchmark_v2_per_station.csv for the T2 rank agreement")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.bsrn_files:
        minute, meta = read_bsrn_station_files(a.bsrn_files)
        a.lat, a.lon, a.elev = (v if v is not None else meta[k] for v, k in ((a.lat, "lat"), (a.lon, "lon"), (a.elev, "elev")))
        print(f"BSRN station {meta['station']}: lat {a.lat}, lon {a.lon}, elev {a.elev}", file=sys.stderr)
    elif not (a.minute_csv or a.pangaea_tab) or None in (a.lat, a.lon, a.elev):
        ap.error("need --bsrn-files, or --minute-csv/--pangaea-tab plus --lat --lon --elev")
    if not LIMITS_VERIFIED:
        print("WARNING: QC limits are unverified (see module docstring). Do not quote results before checking them.", file=sys.stderr)
    if not a.bsrn_files:
        minute = read_minute_csv(a.minute_csv) if a.minute_csv else read_pangaea_tab(a.pangaea_tab)
    res, sim = anchor(minute, a.lat, a.lon, a.elev, a.stamp)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False)); print(sim.to_string(index=False))
    print(rank_agreement(res, a.epw_benchmark).to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False); sim.to_csv(Path(a.out).with_suffix(".t3.csv"), index=False)


if __name__ == "__main__":
    main()
