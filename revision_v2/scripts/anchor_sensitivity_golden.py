"""anchor_sensitivity_golden.py — (1) sensitivity of the anchor pilot to the QC and coverage choices, and (2) the SRRL Golden 2021 hourly
station (semi-arid cold steppe, a possible BSk analogue) analysed with the same models.

(1) Minute stations PTR, SMS, DRA. Variants (all reproduce the base run when no option is changed; this is asserted):
      base            : bsrn_anchor.bsrn_qc as is, coverage 75 %
      sum50           : closure test applied where Sum SW = DHI + DNI*mu > 50 W/m2 (document: Long & Dutton V2.0) instead of GHI > 50
      sum50+difratio  : sum50 plus the diffuse-ratio test (Dif/Global < 1.05 for SZA < 75, < 1.10 for 75-93, Global > 50)
      rare_warn       : 'extremely rare' limits are warnings, not rejections (document: only 'physically possible' means bad data)
      doc_style       : sum50 + diffuse ratio + rare as warning (closest to the recommended tests)
      cov60 / cov90   : coverage rule 60 % / 90 % of valid minutes per hour
    Also reported: Reindl-2 with the kd bounds usually quoted for the original correlation (kd <= 1 for kt <= 0.3; 0.1 <= kd <= 0.97
    for 0.3 < kt < 0.78; kd >= 0.1 for kt >= 0.78), as a sensitivity of the code's clipping to [0, 1]. These bounds are quoted from
    memory of the original paper and are NOT verified; the coefficients and kt ranges were checked against the Sandia PVPMC page.
(2) Golden (hourly, end-of-hour stamps in fixed MST = UTC-7): hour start = stamp - 1 h; interval-average cos z from 1-min geometry;
    Tdew and pressure are taken from the file (NaN kept, no silent defaults). QC variants: none / hourly physical limits /
    exclude the provider's 'review_uncertainty' hours / both.

Run from revision_v2:  python scripts/anchor_sensitivity_golden.py
Writes data/processed/anchor_pilot_sensitivity.csv, anchor_pilot_GOLDEN_*.csv, anchor_pilot_bootstrap_GOLDEN.csv, logs/anchor_sensitivity_<ts>.txt
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bsrn_anchor as ba  # noqa: E402
import benchmark_v2 as bm  # noqa: E402
import run_anchor_pilot as rp  # noqa: E402
from benchmark_v2 import DOMAINS, MIN_COSZ, MIN_GHI, metr, prepare, run_models  # noqa: E402
from gompertz import gompertz_dni  # noqa: E402

OUT = rp.OUT
LOGS = rp.LOGS
GOLDEN_CSV = rp.RAW / "GOLDEN" / "SRRL_Golden_2021_horario.csv"
G_LAT, G_LON, G_ELEV = 39.7423, -105.1785, 1828.8      # from the file's own 'Ficha' sheet
B, SEED = 2000, 20261005


# ----------------------------------------------------------------------------------------------------------------------
def qc_variant(df, geo, closure_on_sum=False, diffuse_ratio=False, rare_invalid=True) -> pd.DataFrame:
    """Same formulas as bsrn_anchor.bsrn_qc (Long & Dutton V2.0), with switches for the documented differences."""
    mu = np.clip(geo["cosz"].to_numpy(), 0, None)
    s0 = geo["s0"].to_numpy()
    g, n, d = (df[c].to_numpy(float) for c in ("ghi", "dni", "dhi"))
    sun = geo["cosz"].to_numpy() > 0
    sza = geo["zenith"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        missing = ~(np.isfinite(g) & np.isfinite(n) & np.isfinite(d))
        phys = ((g < -4) | (g > 1.5 * s0 * mu ** 1.2 + 100) | (n < -4) | (n > s0) | (d < -4) | (d > 0.95 * s0 * mu ** 1.2 + 50))
        rare = ((g < -2) | (g > 1.2 * s0 * mu ** 1.2 + 50) | (n < -2) | (n > 0.95 * s0 * mu ** 0.2 + 10) | (d < -2) | (d > 0.75 * s0 * mu ** 1.2 + 30))
        tot = d + n * mu
        ratio = np.where(tot > 0, g / np.where(tot > 0, tot, 1.0), np.nan)
        tol = np.where(sza <= 75, ba.PARAMS["closure_tol_lowsza"], ba.PARAMS["closure_tol_highsza"])
        basis = tot if closure_on_sum else g
        closure = sun & (basis > ba.PARAMS["closure_min_ghi"]) & (sza < 93) & (np.abs(ratio - 1) > tol)
        dr = np.zeros(len(g), bool)
        if diffuse_ratio:
            lim = np.where(sza < 75, 1.05, 1.10)
            dr = sun & (g > 50) & (sza < 93) & (d / np.where(g > 0, g, np.nan) > lim)
    inv = missing | phys | closure | dr | (rare if rare_invalid else False)
    return pd.DataFrame({"missing": missing, "phys": phys, "rare": rare, "closure": closure, "difratio": dr, "invalid": inv}, index=df.index)


def reindl2_bounded(ghi, kt, cosz):
    sa = np.clip(cosz, 0, 1)
    k = np.clip(kt, 0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kd = np.where(k <= 0.30, np.minimum(1.020 - 0.254 * k + 0.0123 * sa, 1.0),
                      np.where(k < 0.78, np.clip(1.400 - 1.749 * k + 0.177 * sa, 0.1, 0.97), np.maximum(0.486 * k - 0.182 * sa, 0.1)))
        dni = np.where(cosz > 0, np.clip((ghi - kd * ghi) / np.where(cosz > 0, cosz, 1), 0, bm.dec.DNI_MAX_W_M2), np.nan)
    return dni


def evaluate(epwlike: pd.DataFrame, dni_meas: np.ndarray):
    """Same chain as bsrn_anchor.anchor after the hourly step; also returns the arrays for the bootstrap."""
    d, _ = prepare({}, epwlike)
    models = run_models(d, epwlike)
    ghi, cosz = d["GHI_h"].to_numpy(), d["cos_Z"].to_numpy()
    i0 = ba.PARAMS["generator_i0"] * ba.e0_ashrae(epwlike["doy"].to_numpy())
    gomp = gompertz_dni(np.nan_to_num(ghi), cosz, i0)
    gomp = np.where(np.isfinite(ghi), gomp, np.nan)
    series = {"gompertz (T1)": gomp}
    series.update({k: v[0] for k, v in models.items()})
    series["reindl2_bounded (sens.)"] = reindl2_bounded(ghi, d["Kt"].to_numpy(), cosz)
    rows, sim = [], []
    for dom, c0 in DOMAINS.items():
        mask = np.isfinite(ghi) & np.isfinite(dni_meas) & (ghi >= MIN_GHI) & (cosz >= c0) & (cosz > MIN_COSZ)
        for name, s in series.items():
            m = metr(dni_meas[mask], s[mask])
            rows.append(dict(domain=dom, model=name, **{f"DNI_{k}": v for k, v in m.items()}))
            if name != "gompertz (T1)":
                mm = mask & np.isfinite(s)
                sim.append(dict(domain=dom, model=name, n=int(mm.sum()),
                                RMSD_vs_measured=float(np.sqrt(np.mean((s[mm] - dni_meas[mm]) ** 2))),
                                RMSD_vs_gompertz=float(np.sqrt(np.mean((s[mm] - gomp[mm]) ** 2)))))
    mask_c20 = np.isfinite(ghi) & np.isfinite(dni_meas) & (ghi >= MIN_GHI) & (cosz >= DOMAINS["c20"]) & (cosz > MIN_COSZ)
    simdf = pd.DataFrame(sim)
    simdf["ratio_gomp_over_meas"] = simdf["RMSD_vs_gompertz"] / simdf["RMSD_vs_measured"]
    return pd.DataFrame(rows), simdf, series, mask_c20


def summarise(res: pd.DataFrame) -> dict:
    r = res[res["domain"] == "c20"].set_index("model")
    out = {"n_erbs": int(r.loc["erbs", "DNI_n"])}
    for k, lab in (("gompertz (T1)", "gomp"), ("erbs", "erbs"), ("reindl2", "reindl2"), ("reindl2_bounded (sens.)", "reindl2_b"),
                   ("dirint_v1", "dirint_v1"), ("dirint_std", "dirint_std"), ("disc", "disc")):
        out[f"RMSE_{lab}"] = float(r.loc[k, "DNI_RMSE"])
    out["MBE_dirint_v1"] = float(r.loc["dirint_v1", "DNI_MBE"])
    out["MBE_reindl2"] = float(r.loc["reindl2", "DNI_MBE"])
    out["d_reindl2_erbs"] = out["RMSE_reindl2"] - out["RMSE_erbs"]
    out["d_dirint_v1_erbs"] = out["RMSE_dirint_v1"] - out["RMSE_erbs"]
    rank = r.drop(index=["gompertz (T1)", "reindl2_bounded (sens.)"])["DNI_RMSE"].rank()
    out["best_model"] = str(rank.idxmin())
    return out


# ----------------------------------------------------------------------------------------------------------------------
def minute_station(stn: str):
    kind, files = {
        "PTR": ("pangaea", sorted((rp.RAW / "PTR").glob("PTR_radiation_2016-*.tab"))),
        "SMS": ("pangaea", sorted((rp.RAW / "SMS").glob("SMS_radiation_2016-*.tab"))),
        "DRA": ("surfrad", sorted((rp.RAW / "DRA").glob("dra16???.dat"))),
    }[stn]
    raw, meta = rp.read_pangaea_station(files) if kind == "pangaea" else rp.read_surfrad_year(files)
    minute = rp.regularise(raw, "2016-01-01 00:00", "2016-12-31 23:59")
    geo = ba.geometry(minute.index, meta["lat"], meta["lon"], meta["elev"], "start")
    return minute, geo, meta


MIN_VARIANTS = {
    "base": dict(qc=dict(), cov=0.75), "sum50": dict(qc=dict(closure_on_sum=True), cov=0.75),
    "sum50+difratio": dict(qc=dict(closure_on_sum=True, diffuse_ratio=True), cov=0.75),
    "rare_warn": dict(qc=dict(rare_invalid=False), cov=0.75),
    "doc_style": dict(qc=dict(closure_on_sum=True, diffuse_ratio=True, rare_invalid=False), cov=0.75),
    "cov60": dict(qc=dict(), cov=0.60), "cov90": dict(qc=dict(), cov=0.90),
}


def run_minute_variant(minute, geo, meta, spec):
    qc = qc_variant(minute, geo, **spec["qc"])
    old = ba.PARAMS["min_valid_fraction"]
    ba.PARAMS["min_valid_fraction"] = spec["cov"]
    try:
        h = ba.to_hourly(minute, geo, qc, "start")
    finally:
        ba.PARAMS["min_valid_fraction"] = old
    epw = ba.to_epw_like(h, "avg")
    res, sim, series, mask = evaluate(epw, h["dni"].to_numpy())
    day_h = h["cosz_avg"] > 0
    cov = dict(daytime_hours_kept_pct=100 * float(h.loc[day_h, "ghi"].notna().mean()),
               sun_minutes_invalid_pct=100 * float(qc.loc[geo["cosz"] > 0, "invalid"].mean()))
    return res, sim, series, mask, h, cov


# ----------------------------------------------------------------------------------------------------------------------
def golden_base_frame():
    df = pd.read_csv(GOLDEN_CSV, encoding="utf-8-sig")
    t_end = pd.to_datetime(df["timestamp"], utc=True)          # '...-07:00' offsets -> UTC
    start = pd.DatetimeIndex(t_end - pd.Timedelta(hours=1))      # end-of-hour stamp -> hour start
    mi = pd.date_range(start[0], start[-1] + pd.Timedelta(minutes=59), freq="min")
    geo = ba.geometry(mi, G_LAT, G_LON, G_ELEV, "start")
    hr = geo.index.floor("h")
    cz = geo.groupby(hr)["cosz"].apply(lambda s: float(np.clip(s.to_numpy(), 0, None).mean()))
    s0m = geo.groupby(hr)["s0"].mean()
    doy = geo.groupby(hr)["doy"].first()
    h = pd.DataFrame({"ghi": df["GHI"].to_numpy(float), "dni": df["DNI"].to_numpy(float), "dhi": df["DHI"].to_numpy(float),
                      "t_c": df["Tair_C"].to_numpy(float), "t_dew": df["Tdew_C"].to_numpy(float),
                      "p_pa": df["pressure_Pa"].to_numpy(float), "review": df["review_uncertainty"].to_numpy(int)}, index=start)
    h["cosz_avg"] = cz.reindex(start).to_numpy()
    h["s0_mean"] = s0m.reindex(start).to_numpy()
    h["doy"] = doy.reindex(start).to_numpy()
    return h


def golden_epw(h: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"t_start": h.index.tz_localize(None), "doy": h["doy"].to_numpy().astype(int), "ghi": h["ghi"].to_numpy(),
                         "etr_n": h["s0_mean"].to_numpy(), "etr_h": (h["s0_mean"] * h["cosz_avg"]).to_numpy(),
                         "t_db": h["t_c"].to_numpy(), "t_dew": h["t_dew"].to_numpy(), "p_pa": h["p_pa"].to_numpy()})


def golden_variants(h0: pd.DataFrame):
    mu, s0 = h0["cosz_avg"].to_numpy(), h0["s0_mean"].to_numpy()
    g, n, d = h0["ghi"].to_numpy(), h0["dni"].to_numpy(), h0["dhi"].to_numpy()
    phys = ((g < -4) | (g > 1.5 * s0 * mu ** 1.2 + 100) | (n < -4) | (n > s0) | (d < -4) | (d > 0.95 * s0 * mu ** 1.2 + 50))
    rev = h0["review"].to_numpy() == 1
    out = {}
    for name, bad in (("none", np.zeros(len(h0), bool)), ("hourly_phys", phys), ("excl_review", rev), ("hourly_phys+excl_review", phys | rev)):
        h = h0.copy()
        h.loc[bad, ["ghi", "dni", "dhi"]] = np.nan
        out[name] = (h, int(bad.sum()))
    return out


def day_bootstrap(dni, series, mask, day, rng):
    names = list(series)
    ok = mask.copy()
    for k in names:
        ok &= np.isfinite(series[k])
    days = np.unique(day[ok])
    idx = {d: np.flatnonzero(ok & (day == d)) for d in days}
    nd = np.array([len(idx[d]) for d in days], float)
    err = {k: series[k] - dni for k in names}
    sse = {k: np.array([np.sum(err[k][idx[d]] ** 2) for d in days]) for k in names}
    se = {k: np.array([np.sum(err[k][idx[d]]) for d in days]) for k in names}
    point = {k: float(np.sqrt(sse[k].sum() / nd.sum())) for k in names}
    boot = {k: np.empty(B) for k in names}
    bias = {k: np.empty(B) for k in names}
    ranks = np.empty((B, len(names)))
    for b in range(B):
        p = rng.integers(0, len(days), len(days))
        nn = nd[p].sum()
        for k in names:
            boot[k][b] = np.sqrt(sse[k][p].sum() / nn)
            bias[k][b] = se[k][p].sum() / nn
        ranks[b] = pd.Series([boot[k][b] for k in names]).rank().to_numpy()
    rows = []
    for j, k in enumerate(names):
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        dl, dh = np.percentile(boot[k] - boot["erbs"], [2.5, 97.5])
        bl, bh = np.percentile(bias[k], [2.5, 97.5])
        rows.append(dict(station="GOLDEN", model=k, n_hours=int(nd.sum()), n_days=len(days), RMSE=point[k], RMSE_lo=lo, RMSE_hi=hi,
                         MBE=float(se[k].sum() / nd.sum()), MBE_lo=bl, MBE_hi=bh, dRMSE_vs_erbs=point[k] - point["erbs"],
                         dRMSE_lo=dl, dRMSE_hi=dh, mean_rank=float(ranks[:, j].mean()), share_first=float((ranks[:, j] == 1).mean()),
                         RMSE_erbs_same_hours=point["erbs"], mean_obs=float(np.mean(dni[ok]))))
    return pd.DataFrame(rows)


def main() -> int:
    warnings.filterwarnings("ignore")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    pd.set_option("display.width", 250)
    log(f"anchor sensitivity + Golden {ts} | python {sys.version.split()[0]} pandas {pd.__version__} numpy {np.__version__} pvlib {pvlib.__version__}")
    sens = []
    for stn in ("PTR", "SMS", "DRA"):
        minute, geo, meta = minute_station(stn)
        base_csv = pd.read_csv(OUT / f"anchor_pilot_{stn}_base_metrics.csv")
        for vname, spec in MIN_VARIANTS.items():
            res, sim, series, mask, h, cov = run_minute_variant(minute, geo, meta, spec)
            if vname == "base":
                a = base_csv.set_index(["domain", "model"])["DNI_RMSE"]
                b = res.set_index(["domain", "model"])["DNI_RMSE"].reindex(a.index)
                assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), f"{stn}: base variant does not reproduce the saved base run"
                log(f"[{stn}] base variant reproduces anchor_pilot_{stn}_base_metrics.csv (max |dRMSE| = {np.nanmax(np.abs(a.to_numpy() - b.to_numpy())):.2e})")
            sens.append(dict(station=stn, variant=vname, **cov, **summarise(res)))
    sdf = pd.DataFrame(sens)

    # Golden
    h0 = golden_base_frame()
    log(f"\nGOLDEN: {len(h0)} hourly records; hour starts {h0.index[0]} .. {h0.index[-1]} (UTC)")
    mu = h0["cosz_avg"].to_numpy()
    tot = h0["dhi"].to_numpy() + h0["dni"].to_numpy() * mu
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(tot > 0, h0["ghi"].to_numpy() / np.where(tot > 0, tot, 1), np.nan)
    sel = (np.degrees(np.arccos(np.clip(mu, 0, 1))) < 75) & (tot > 50)
    log(f"  hourly closure GHI/(DHI + DNI*cosz_avg), SZA<75, sum>50: {int(sel.sum())} h; within +-8 %: {100*np.mean(np.abs(ratio[sel]-1) <= 0.08):.1f} %; "
        f"quantiles 5/50/95 %: {np.nanpercentile(ratio[sel], [5, 50, 95]).round(3).tolist()} (diagnostic only; hourly means do not satisfy the 1-min test exactly)")
    gres = {}
    for vname, (h, nbad) in golden_variants(h0).items():
        epw = golden_epw(h)
        res, sim, series, mask = evaluate(epw, h["dni"].to_numpy())
        gres[vname] = (res, sim, series, mask, h)
        day_h = h["cosz_avg"] > 0
        sens.append(dict(station="GOLDEN", variant=vname, daytime_hours_kept_pct=100 * float(h.loc[day_h, "ghi"].notna().mean()),
                         sun_minutes_invalid_pct=np.nan, hours_removed=nbad, **summarise(res)))
    sdf = pd.DataFrame(sens)
    sdf.to_csv(OUT / "anchor_pilot_sensitivity.csv", index=False)
    cols = ["station", "variant", "n_erbs", "daytime_hours_kept_pct", "RMSE_gomp", "RMSE_erbs", "RMSE_reindl2", "RMSE_reindl2_b", "RMSE_dirint_v1",
            "RMSE_dirint_std", "RMSE_disc", "MBE_dirint_v1", "d_reindl2_erbs", "d_dirint_v1_erbs", "best_model"]
    log("\n=== SENSIBILIDAD (DNI horario medido, c20; RMSE y MBE en W/m2) ===")
    log(sdf[cols].round(2).to_string(index=False))

    # Golden base outputs
    res, sim, series, mask, h = gres["none"]
    res.to_csv(OUT / "anchor_pilot_GOLDEN_base_metrics.csv", index=False)
    sim.to_csv(OUT / "anchor_pilot_GOLDEN_base_t3.csv", index=False)
    bench = OUT / "benchmark_v2_per_station.csv"
    rk = ba.rank_agreement(res[~res["model"].str.contains("sens")], bench, "c20")
    rk.to_csv(OUT / "anchor_pilot_GOLDEN_base_t2rank.csv", index=False)
    log("\nGOLDEN base (sin QC adicional), c20:")
    log(res[res["domain"] == "c20"].round(2).to_string(index=False))
    log("\nGOLDEN T2 ranking vs IWEC2 ranking: Spearman = %s" % rk.attrs.get("spearman"))
    log(rk.round(2).to_string(index=False))
    log("\nGOLDEN T3 (c20):")
    log(sim[sim["domain"] == "c20"].round(2).to_string(index=False))
    # bootstrap for Golden base (days in MST)
    day = (h.index - pd.Timedelta(hours=7)).floor("D").to_numpy()
    bt = day_bootstrap(h["dni"].to_numpy(), series, mask, day, np.random.default_rng(SEED))
    bt.to_csv(OUT / "anchor_pilot_bootstrap_GOLDEN.csv", index=False)
    log(f"\nGOLDEN bootstrap por dias (B = {B}), muestra comun {int(bt['n_hours'].iloc[0])} h en {int(bt['n_days'].iloc[0])} dias:")
    log(bt.drop(columns=["station", "n_hours", "n_days"]).round(2).to_string(index=False))
    (LOGS / f"anchor_sensitivity_{ts}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nlog: {LOGS / ('anchor_sensitivity_' + ts + '.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
