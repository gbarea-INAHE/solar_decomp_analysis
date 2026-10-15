"""run_modern_models_anchor.py — BRL y Engerer2 contra el DNI medido (cuatro estaciones), junto a los modelos ya evaluados.

Entradas: las mismas series y el mismo control de calidad del piloto (run_anchor_pilot / anchor_sensitivity_golden, variante 'base').
  * Engerer2 1 min  : aplicado a los minutos válidos de PTR, SMS y DRA y luego promediado a hora (como se promedia el DNI medido).
  * Engerer2 60 min : aplicado a las medias horarias (EXPLORATORIO: parámetros de fuente única; convención de marca de tiempo del
                      entrenamiento desconocida). Corre en las 4 estaciones.
  * BRL horario     : aplicado a las medias horarias (Kt, AST, altura solar, Kt diario, persistencia). Corre en las 4 estaciones.
Verificaciones incluidas: (1) los modelos ya existentes reproducen los CSV del piloto; (2) cada métrica de los modelos nuevos se recalcula con
pandas (camino independiente) y debe coincidir con la de numpy; (3) sensibilidades: Engerer2 con la definición de Kde de splitting_models,
BRL con persistencia que cuenta la noche como 0. Dominio c20. Bootstrap por días (2000 réplicas, semilla fija).
Uso (desde revision_v2, venv activo): python scripts/run_modern_models_anchor.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
import modern_models as mm  # noqa: E402
from benchmark_v2 import DOMAINS, MIN_COSZ, MIN_GHI, metr  # noqa: E402

OUT, LOGS = asg.OUT, asg.LOGS
NEW = ["engerer2_1min", "engerer2_60 (expl.)", "brl_hourly"]


def hourly_geometry(h: pd.DataFrame, lat: float, lon: float):
    mid = pd.DatetimeIndex(h.index) + pd.Timedelta(minutes=30)
    _, ast, _, _, _ = mm.geometry_bright(mid.tz_convert("UTC").tz_localize(None), lat, lon)
    e_n = h["s0_mean"].to_numpy(float)
    cosz = h["cosz_avg"].to_numpy(float)
    day_id = (pd.DatetimeIndex(h.index) + pd.Timedelta(hours=lon / 15.0)).floor("D").asi8
    return mid, ast, e_n, e_n * cosz, cosz, day_id


def hourly_models(h, lat, lon, ghi_h, kde_offset=0.015, psi_night_zero=False):
    mid, ast, e_n, e_h, cosz, day_id = hourly_geometry(h, lat, lon)
    z = np.degrees(np.arccos(np.clip(cosz, 0, 1)))
    t_naive = mid.tz_convert("UTC").tz_localize(None)
    e2, _ = mm.engerer2_dni(ghi_h, t_naive, lat, lon, 60, kde_offset, e_h_override=e_h, z_override=z)
    brl, kd, kt, psi, ktd = mm.brl_dni_hourly(ghi_h, e_n, e_h, cosz, ast, day_id, psi_night_zero)
    return e2, brl


def engerer2_minute_to_hour(minute, geo, qc, h, lat, lon, kde_offset=0.015):
    """Engerer2 (1 min) sobre los minutos válidos, promediado a hora con la misma regla de cobertura del DNI medido."""
    t_mid = pd.DatetimeIndex(minute.index + pd.Timedelta(seconds=30)).tz_convert("UTC").tz_localize(None)
    sun = (geo["cosz"] > 0).to_numpy()
    valid = ((~qc["invalid"].to_numpy()) | ~sun) & ~qc["missing"].to_numpy()
    ghi = np.where(valid & sun, minute["ghi"].to_numpy(float), np.nan)
    dni, _ = mm.engerer2_dni(ghi, t_mid, lat, lon, 1, kde_offset)
    ok = valid & (np.isfinite(dni) | ~sun)
    val = np.where(sun, dni, 0.0)
    df = pd.DataFrame({"v": np.where(ok, val, np.nan), "ok": ok.astype(float)}, index=minute.index)
    g = df.groupby(df.index.floor("h"))
    mean = g["v"].mean()
    cov = g["ok"].sum() / 60.0
    out = mean.where(cov >= ba.PARAMS["min_valid_fraction"])
    return out.reindex(h.index).to_numpy(float)


def independent_rmse(dni, pred, mask):
    """Camino independiente (pandas) para RMSE y MBE sobre las horas con máscara."""
    d = pd.DataFrame({"o": dni, "p": pred})[mask].dropna()
    e = d["p"] - d["o"]
    return float(np.sqrt((e ** 2).mean())), float(e.mean()), len(d)


def main() -> int:
    warnings.filterwarnings("ignore")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    pd.set_option("display.width", 250)
    log(f"modern models vs measured {ts}")
    mm.selftest(verbose=False)
    log("selftest de modern_models: OK (vectorizado == escalar, propiedades, punto a mano, geometría vs pvlib)")
    summary, tables = [], []
    for stn in ("PTR", "SMS", "DRA", "GOLDEN"):
        log("\n" + "=" * 90 + f"\n{stn}")
        if stn == "GOLDEN":
            h0 = asg.golden_base_frame()
            lat, lon = asg.G_LAT, asg.G_LON
            epw = asg.golden_epw(h0)
            res, sim, series, mask = asg.evaluate(epw, h0["dni"].to_numpy())
            h = h0
            e2m = None
            ghi_h = h["ghi"].to_numpy(float)
            minute = geo = qc = None
        else:
            minute, geo, meta = asg.minute_station(stn)
            lat, lon = meta["lat"], meta["lon"]
            res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
            qc = asg.qc_variant(minute, geo)
            ghi_h = h["ghi"].to_numpy(float)
            e2m = engerer2_minute_to_hour(minute, geo, qc, h, lat, lon)
        # (1) los modelos existentes reproducen el piloto
        base = pd.read_csv(OUT / f"anchor_pilot_{stn}_base_metrics.csv")
        a = base[base["domain"] == "c20"].set_index("model")["DNI_RMSE"]
        b = res[res["domain"] == "c20"].set_index("model")["DNI_RMSE"].reindex(a.index)
        assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True), f"{stn}: los modelos existentes no reproducen el piloto"
        log(f"  [verif 1] modelos existentes reproducen anchor_pilot_{stn}_base_metrics.csv (max |dRMSE| = {np.nanmax(np.abs(a.to_numpy()-b.to_numpy())):.1e})")
        dni = h["dni"].to_numpy(float)
        e2h, brl = hourly_models(h, lat, lon, ghi_h)
        e2h_alt, brl_alt = hourly_models(h, lat, lon, ghi_h, kde_offset=0.0, psi_night_zero=True)
        new = {"engerer2_60 (expl.)": e2h, "brl_hourly": brl}
        if e2m is not None:
            new["engerer2_1min"] = e2m
        allser = dict(series)
        allser.pop("reindl2_bounded (sens.)", None)
        allser.update(new)
        # (2) métricas nuevas: numpy (metr) vs pandas independiente
        for k, v in new.items():
            m1 = metr(dni[mask], v[mask])
            r2, b2, n2 = independent_rmse(dni, v, mask)
            assert abs(m1["RMSE"] - r2) < 1e-9 and abs(m1["MBE"] - b2) < 1e-9 and m1["n"] == n2, f"{stn} {k}: métrica no coincide"
            summary.append(dict(station=stn, model=k, n=m1["n"], RMSE=m1["RMSE"], MBE=m1["MBE"], nRMSE=m1["nRMSE"], R2=m1["R2"]))
            log(f"  [verif 2] {k:22s} n={m1['n']:5d} RMSE={m1['RMSE']:.2f} MBE={m1['MBE']:+.2f}  (recalculo pandas: {r2:.2f}/{b2:+.2f}/{n2})")
        # sensibilidades
        sens = {"engerer2_60 Kde sin offset (splitting_models)": e2h_alt, "brl psi con noche=0": brl_alt}
        for k, v in sens.items():
            m1 = metr(dni[mask], v[mask])
            summary.append(dict(station=stn, model=k, n=m1["n"], RMSE=m1["RMSE"], MBE=m1["MBE"], nRMSE=m1["nRMSE"], R2=m1["R2"]))
            log(f"  [sens]    {k:46s} RMSE={m1['RMSE']:.2f} MBE={m1['MBE']:+.2f}")
        # bootstrap por días con muestra común
        day = (pd.DatetimeIndex(h.index) + pd.Timedelta(hours=lon / 15.0)).floor("D").to_numpy()
        keep = ["gompertz (T1)", "erbs", "reindl2", "dirint_v1", "dirint_std"] + list(new)
        bt = asg.day_bootstrap(dni, {k: allser[k] for k in keep}, mask, day, np.random.default_rng(asg.SEED))
        bt["station"] = stn
        bt.to_csv(OUT / f"modern_models_bootstrap_{stn}.csv", index=False)
        tables.append(bt)
        log(f"  muestra común: {int(bt['n_hours'].iloc[0])} h en {int(bt['n_days'].iloc[0])} días")
        log(bt.drop(columns=["station", "n_hours", "n_days"]).round(2).to_string(index=False))
    pd.DataFrame(summary).to_csv(OUT / "modern_models_summary.csv", index=False)
    (LOGS / f"modern_models_{ts}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nlog: {LOGS / ('modern_models_' + ts + '.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
