"""run_mendoza_provisional.py — Corrida PROVISORIA de los modelos de descomposición contra los datos medidos de Mendoza (INAHE, 2022-2025, 1 min).

Referencia: difusa horizontal MEDIDA (corregida por el técnico: banda de sombra LeBaron + filtros de Nollas 2023) y DNI DERIVADO por cierre,
DNI = (GHI − DHI) / cos z (cos z = media del intervalo). No hay DNI medido. Con modelos que cierran (DHI = GHI − DNI·cos z) el error de DNI es el error de
difusa dividido por cos z: el DNI derivado NO aporta información independiente de la difusa; la métrica directa es la de DHI.

CONFIRMADO por el técnico (2026-10-06): latitud −32.89, longitud −68.87; hora de reloj (UTC−3, sin horario de verano en 2022-2025); la marca de tiempo
corresponde al FINAL de cada minuto (por eso `--stamp end`; `start` queda solo como variante de sensibilidad). NO confirmada: la altitud (el técnico no la
dio; se pasa por línea de comandos y es un supuesto). Coordenadas, marca de tiempo y altitud se pasan por línea de comandos, sin valores por defecto.
Sin temperatura, humedad ni presión medidas (presión = atmósfera estándar a la altitud dada, y el DIRINT de Perez usa el bin de «agua precipitable
desconocida»; el DIRINT de la v1.0.0 usa su temperatura por defecto de 20 °C, como la herramienta sin columna de temperatura).

Verificación incluida: el mismo código aplicado a Desert Rock con el DNI oculto (solo GHI y DHI) debe dar errores cercanos a los del piloto con DNI medido.

Uso (desde revision_v2, venv activo):
  python scripts/run_mendoza_provisional.py --lat <°N> --lon <°E> --elev <m> --stamp {start,end}
Escribe data/processed/mendoza_prov_*.csv, response/MENDOZA_corrida_provisoria.md y logs/mendoza_prov_<ts>.txt. No modifica los datos.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
import dirint_dewpoint_anchor as dd  # noqa: E402
import mendoza_profile as mp  # noqa: E402
import modern_models as mm  # noqa: E402
import run_modern_models_anchor as rm  # noqa: E402
from benchmark_v2 import DOMAINS, MIN_COSZ, MIN_GHI, metr  # noqa: E402

REV = mp.REV
OUT, LOGS = asg.OUT, asg.LOGS
RESP = REV / "response"
MAIN = ["gompertz (T1)", "erbs", "reindl2", "dirint_v1", "dirint_perez", "brl_hourly", "engerer2_60 (expl.)", "engerer2_1min"]
SEASONS = {12: "DEF", 1: "DEF", 2: "DEF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


# ---------------------------------------------------------------------------------------------------------------------- datos
def mendoza_minute(stamp: str) -> pd.DataFrame:
    df = pd.concat([mp.read_year(y) for y in mp.YEARS])
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("los años no concatenan en una serie continua")
    if (df.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).any():
        raise ValueError("hay saltos entre años")
    shift = pd.Timedelta(hours=3) - (pd.Timedelta(minutes=1) if stamp == "end" else pd.Timedelta(0))
    idx = (df.index + shift).tz_localize("UTC")                      # hora local UTC-3 -> inicio de minuto UTC
    return pd.DataFrame({"ghi": df["IRGLO"].to_numpy(float), "dni": np.nan, "dhi": df["IRDIF_C"].to_numpy(float)}, index=idx)


def qc_ghi_dhi(minute: pd.DataFrame, geo: pd.DataFrame, ratio: bool = True) -> pd.DataFrame:
    """Controles mínimos sobre datos que YA pasaron el control del técnico: faltantes, límites físicos de BSRN para GHI y DHI, y (opcional) cociente
    difusa/global (Long y Dutton: < 1.05 para SZA < 75°, < 1.10 para 75–93°, con GHI > 50 W/m²). No se usa el cierre (no hay DNI)."""
    mu = np.clip(geo["cosz"].to_numpy(), 0, None)
    s0 = geo["s0"].to_numpy()
    sza = geo["zenith"].to_numpy()
    g, d = minute["ghi"].to_numpy(float), minute["dhi"].to_numpy(float)
    sun = geo["cosz"].to_numpy() > 0
    missing = ~(np.isfinite(g) & np.isfinite(d))
    with np.errstate(invalid="ignore", divide="ignore"):
        phys = (g < -4) | (g > 1.5 * s0 * mu ** 1.2 + 100) | (d < -4) | (d > 0.95 * s0 * mu ** 1.2 + 50)
        r = np.where(g > 50, d / np.where(g > 50, g, 1.0), np.nan)
        rfail = sun & (g > 50) & (((sza < 75) & (r > 1.05)) | ((sza >= 75) & (sza < 93) & (r > 1.10)))
    invalid = missing | phys | (rfail if ratio else False)
    return pd.DataFrame({"missing": missing, "invalid": invalid}, index=minute.index)


# ---------------------------------------------------------------------------------------------------------------------- proceso
def process(minute: pd.DataFrame, lat: float, lon: float, elev: float, ratio: bool = True, cov: float = 0.75):
    geo = ba.geometry(minute.index, lat, lon, elev, "start")
    qc = qc_ghi_dhi(minute, geo, ratio)
    old = ba.PARAMS["min_valid_fraction"]
    ba.PARAMS["min_valid_fraction"] = cov
    try:
        h = ba.to_hourly(minute, geo, qc, "start")
    finally:
        ba.PARAMS["min_valid_fraction"] = old
    cosz = h["cosz_avg"].to_numpy()
    ghi, dhi_m = h["ghi"].to_numpy(float), h["dhi"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        dni_der = np.where(cosz > 0, (ghi - dhi_m) / np.where(cosz > 0, cosz, 1.0), np.nan)
    p_pa = float(pvlib.atmosphere.alt2pres(elev))                      # atmósfera estándar a la altitud dada (supuesto declarado)
    epw = pd.DataFrame({"t_start": h.index.tz_localize(None), "doy": h["doy"].ffill().bfill().to_numpy().astype(int), "ghi": ghi,
                        "etr_n": h["s0_mean"].to_numpy(), "etr_h": (h["s0_mean"] * h["cosz_avg"]).to_numpy(),
                        "t_db": np.nan, "t_dew": np.nan, "p_pa": p_pa})
    res, sim, series, mask = asg.evaluate(epw, dni_der)
    _, without, _ = dd.dirint_variants(epw)                            # DIRINT de Perez con «agua precipitable desconocida»
    series = {k: v for k, v in series.items() if k not in ("reindl2_bounded (sens.)", "dirint_std")}
    series["dirint_perez"] = without
    e2h, brl = rm.hourly_models(h, lat, lon, ghi)
    series["engerer2_60 (expl.)"], series["brl_hourly"] = e2h, brl
    series["engerer2_1min"] = rm.engerer2_minute_to_hour(minute, geo, qc, h, lat, lon)
    return dict(h=h, geo=geo, qc=qc, ghi=ghi, cosz=cosz, dhi_m=dhi_m, dni_der=dni_der, series=series, mask=mask, sim=sim, lon=lon)


def dhi_from_dni(ghi, dni, cosz):
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(dni), np.clip(ghi - dni * cosz, 0.0, None), np.nan)


CLASSIC = ["gompertz (T1)", "erbs", "reindl2", "dirint_v1", "dirint_perez"]


def metrics_by_domain(P: dict, kind: str, models: list | None = None) -> pd.DataFrame:
    """RMSE/MBE/nRMSE por dominio y modelo, para DNI derivado o DHI medido. La muestra es común a `models` (por defecto MAIN) en cada dominio."""
    models = models or MAIN
    ref = P["dni_der"] if kind == "DNI" else P["dhi_m"]
    rows = []
    for dom, c0 in DOMAINS.items():
        mk = np.isfinite(P["ghi"]) & np.isfinite(ref) & np.isfinite(P["dni_der"]) & (P["ghi"] >= MIN_GHI) & (P["cosz"] >= c0) & (P["cosz"] > MIN_COSZ)
        for k in models:                                 # muestra común a los modelos pedidos en cada dominio
            mk &= np.isfinite(P["series"][k])
        for k in models:
            s = P["series"][k]
            pred = s if kind == "DNI" else dhi_from_dni(P["ghi"], s, P["cosz"])
            m = metr(ref[mk], pred[mk])
            rows.append(dict(domain=dom, model=k, ref=kind, **m))
    return pd.DataFrame(rows)


def common_mask(P: dict) -> np.ndarray:
    mk = P["mask"].copy()
    for k in MAIN:
        mk &= np.isfinite(P["series"][k])
    return mk


def boot(P: dict, kind: str, rng) -> pd.DataFrame:
    ref = P["dni_der"] if kind == "DNI" else P["dhi_m"]
    ser = {k: (P["series"][k] if kind == "DNI" else dhi_from_dni(P["ghi"], P["series"][k], P["cosz"])) for k in MAIN}
    day = (pd.DatetimeIndex(P["h"].index) + pd.Timedelta(hours=P["lon"] / 15.0)).floor("D").to_numpy()
    bt = asg.day_bootstrap(ref, ser, P["mask"], day, rng)
    bt["ref"] = kind
    return bt.drop(columns="station")


def by_group(P: dict, kind: str, group: str) -> pd.DataFrame:
    ref = P["dni_der"] if kind == "DNI" else P["dhi_m"]
    mk = common_mask(P)
    idx = pd.DatetimeIndex(P["h"].index) - pd.Timedelta(hours=3)                       # fecha civil local (UTC-3) para año y estación
    key = idx.year if group == "year" else np.array([SEASONS[m] for m in idx.month])
    rows = []
    for g in pd.unique(key):
        sel = mk & (key == g)
        for k in MAIN:
            pred = P["series"][k] if kind == "DNI" else dhi_from_dni(P["ghi"], P["series"][k], P["cosz"])
            e = pred[sel] - ref[sel]
            rows.append({group: g, "model": k, "ref": kind, "n": int(sel.sum()), "RMSE": float(np.sqrt(np.mean(e ** 2))), "MBE": float(e.mean())})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.get(c, lambda x: f"{x:.1f}" if isinstance(x, (float, np.floating)) else str(x))(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, required=True)
    ap.add_argument("--stamp", choices=("start", "end"), required=True, help="la marca de tiempo del archivo es el inicio o el final de cada minuto")
    a = ap.parse_args()
    other = "start" if a.stamp == "end" else "end"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    log(f"Mendoza {ts} (lat={a.lat}, lon={a.lon}: confirmadas por el técnico; elev={a.elev}: SUPUESTO, no informada; marca de tiempo = {a.stamp} de minuto)")
    mm.selftest(verbose=False)

    # ---- base
    minute = mendoza_minute(a.stamp)
    P = process(minute, a.lat, a.lon, a.elev)
    n_c20 = int(common_mask(P).sum())
    log(f"base: {len(P['h'])} horas; muestra común c20: {n_c20} h; cobertura diurna (cos z>0): {100*np.isfinite(P['ghi'][P['cosz']>0]).mean():.1f} % de las horas con sol")
    # verificación 1: cierre del DNI derivado y que DNI y DHI comparten máscara
    ok = np.isfinite(P["dni_der"])
    assert np.allclose(P["dni_der"][ok] * P["cosz"][ok] + P["dhi_m"][ok], P["ghi"][ok], atol=1e-6), "el DNI derivado no cierra"
    # verificación 2: la GHI horaria reproduce la suma de minutos válidos (independiente, pandas)
    m_ok = (~P["qc"]["invalid"]) & (~P["qc"]["missing"])
    g_min = minute["ghi"].where(m_ok & (P["geo"]["cosz"] > 0), 0.0).where(~P["qc"]["missing"])
    chk = g_min.groupby(minute.index.floor("h")).mean().reindex(P["h"].index)
    cov = (m_ok | (P["geo"]["cosz"] <= 0)).groupby(minute.index.floor("h")).mean().reindex(P["h"].index)
    sel = (cov >= 0.75).to_numpy() & np.isfinite(P["ghi"])
    assert np.nanmax(np.abs(chk.to_numpy()[sel] - P["ghi"][sel])) < 1e-9, "GHI horaria distinta de la media independiente"
    log("[verif] el DNI derivado cierra con GHI y DHI; la GHI horaria coincide con la media independiente de los minutos válidos")
    n_neg = int(((P["dni_der"] < 0) & P["mask"]).sum())
    log(f"[info] horas c20 con DNI derivado negativo (difusa > global en la hora): {n_neg}")

    met_dni, met_dhi = metrics_by_domain(P, "DNI"), metrics_by_domain(P, "DHI")
    met = pd.concat([met_dni, met_dhi])
    met.to_csv(OUT / "mendoza_prov_metrics.csv", index=False)
    met_c = pd.concat([metrics_by_domain(P, "DNI", CLASSIC), metrics_by_domain(P, "DHI", CLASSIC)])
    met_c.to_csv(OUT / "mendoza_prov_metrics_classic.csv", index=False)
    rng = np.random.default_rng(asg.SEED)
    bt = pd.concat([boot(P, "DHI", rng), boot(P, "DNI", np.random.default_rng(asg.SEED))])
    bt.to_csv(OUT / "mendoza_prov_bootstrap.csv", index=False)
    by_year = pd.concat([by_group(P, "DHI", "year"), by_group(P, "DNI", "year")])
    by_season = pd.concat([by_group(P, "DHI", "season"), by_group(P, "DNI", "season")])
    by_year.to_csv(OUT / "mendoza_prov_by_year.csv", index=False)
    by_season.to_csv(OUT / "mendoza_prov_by_season.csv", index=False)
    # T3 (c20, muestra común): parecido a Gompertz frente a acuerdo con el DNI derivado, para todos los modelos principales
    mkc = common_mask(P)
    gom = P["series"]["gompertz (T1)"]
    t3 = []
    for k in MAIN:
        if k == "gompertz (T1)":
            continue
        s = P["series"][k]
        a_m, a_g = float(np.sqrt(np.mean((s[mkc] - P["dni_der"][mkc]) ** 2))), float(np.sqrt(np.mean((s[mkc] - gom[mkc]) ** 2)))
        t3.append(dict(model=k, n=int(mkc.sum()), RMSD_vs_measured=a_m, RMSD_vs_gompertz=a_g, ratio_gomp_over_meas=a_g / a_m))
    sim = pd.DataFrame(t3)
    sim.to_csv(OUT / "mendoza_prov_t3.csv", index=False)

    # ---- variantes
    var_rows = []
    variants = {"base": dict(stamp=a.stamp, lat=a.lat, lon=a.lon, ratio=True, cov=0.75),
                f"marca_{'inicio' if other == 'start' else 'fin'}_de_minuto": dict(stamp=other, lat=a.lat, lon=a.lon, ratio=True, cov=0.75),
                "coordenadas_0.1°_S_O": dict(stamp=a.stamp, lat=a.lat - 0.1, lon=a.lon - 0.1, ratio=True, cov=0.75),
                "sin_filtro_cociente": dict(stamp=a.stamp, lat=a.lat, lon=a.lon, ratio=False, cov=0.75),
                "cobertura_60": dict(stamp=a.stamp, lat=a.lat, lon=a.lon, ratio=True, cov=0.60),
                "cobertura_90": dict(stamp=a.stamp, lat=a.lat, lon=a.lon, ratio=True, cov=0.90)}
    for name, v in variants.items():
        Pv = P if name == "base" else process(mendoza_minute(v["stamp"]), v["lat"], v["lon"], a.elev, v["ratio"], v["cov"])
        mk = common_mask(Pv)
        for kind in ("DHI", "DNI"):
            ref = Pv["dhi_m"] if kind == "DHI" else Pv["dni_der"]
            for k in MAIN:
                pred = Pv["series"][k] if kind == "DNI" else dhi_from_dni(Pv["ghi"], Pv["series"][k], Pv["cosz"])
                e = pred[mk] - ref[mk]
                var_rows.append(dict(variant=name, ref=kind, model=k, n=int(mk.sum()), RMSE=float(np.sqrt(np.mean(e ** 2))), MBE=float(e.mean())))
    var = pd.DataFrame(var_rows)
    var.to_csv(OUT / "mendoza_prov_variants.csv", index=False)
    piv = var[var["ref"] == "DHI"].pivot(index="model", columns="variant", values="RMSE").reindex(MAIN)[list(variants)]
    log("\nSENSIBILIDAD — RMSE de DHI (c20, muestra común de cada variante):\n" + piv.round(2).to_string())
    d_max = float((piv.sub(piv["base"], axis=0)).abs().max().max())
    log(f"máxima diferencia de RMSE de DHI respecto de la base: {d_max:.2f} W/m²")

    # ---- verificación del código con Desert Rock (DNI oculto) contra el piloto con DNI medido, EN LAS MISMAS HORAS
    mdra, geo_d, meta = asg.minute_station("DRA")
    hid = mdra[["ghi", "dhi"]].copy()
    hid["dni"] = np.nan
    Pd = process(hid, meta["lat"], meta["lon"], meta["elev"])
    res_p, sim_p, series_p, mask_p, h_p, cov_p = asg.run_minute_variant(mdra, geo_d, meta, asg.MIN_VARIANTS["base"])
    assert Pd["h"].index.equals(h_p.index), "las rejillas horarias de la verificación no coinciden"
    dni_meas = h_p["dni"].to_numpy(float)
    qc_p = asg.qc_variant(mdra, geo_d)
    e2h_p, brl_p = rm.hourly_models(h_p, meta["lat"], meta["lon"], h_p["ghi"].to_numpy(float))
    pilot_series = {"gompertz (T1)": series_p["gompertz (T1)"], "erbs": series_p["erbs"], "reindl2": series_p["reindl2"],
                    "brl_hourly": brl_p, "engerer2_60 (expl.)": e2h_p,
                    "engerer2_1min": rm.engerer2_minute_to_hour(mdra, geo_d, qc_p, h_p, meta["lat"], meta["lon"])}
    S = Pd["mask"] & mask_p & np.isfinite(dni_meas)
    for k in pilot_series:
        S &= np.isfinite(pilot_series[k]) & np.isfinite(Pd["series"][k])
    comp = []
    dhi_meas_p = h_p["dhi"].to_numpy(float)
    for k in pilot_series:
        r_pil = float(np.sqrt(np.mean((pilot_series[k][S] - dni_meas[S]) ** 2)))
        r_mine = float(np.sqrt(np.mean((Pd["series"][k][S] - Pd["dni_der"][S]) ** 2)))
        dh_pil = dhi_from_dni(h_p["ghi"].to_numpy(float), pilot_series[k], h_p["cosz_avg"].to_numpy(float))
        dh_mine = dhi_from_dni(Pd["ghi"], Pd["series"][k], Pd["cosz"])
        rd_pil = float(np.sqrt(np.mean((dh_pil[S] - dhi_meas_p[S]) ** 2)))
        rd_mine = float(np.sqrt(np.mean((dh_mine[S] - Pd["dhi_m"][S]) ** 2)))
        comp.append(dict(model=k, RMSE_DHI_piloto=rd_pil, RMSE_DHI_este_codigo=rd_mine, dif_DHI=rd_mine - rd_pil,
                         RMSE_DNI_medido_piloto=r_pil, RMSE_DNI_derivado_este_codigo=r_mine, dif_DNI=r_mine - r_pil))
    comp = pd.DataFrame(comp)
    dd_err = Pd["dni_der"][S] - dni_meas[S]
    der_rmse, der_mbe = float(np.sqrt(np.mean(dd_err ** 2))), float(dd_err.mean())
    comp.to_csv(OUT / "mendoza_prov_verif_desert_rock.csv", index=False)
    log(f"\nVERIFICACIÓN con Desert Rock, DNI oculto, MISMAS {int(S.sum())} horas c20 (DHI medida en ambos; DNI medido vs derivado)\n" + comp.round(2).to_string(index=False))
    log(f"DNI derivado vs DNI medido en esas horas: RMSE {der_rmse:.2f} W/m², MBE {der_mbe:+.2f} W/m² (OJO: el piloto solo conservó horas que cumplen el cierre ±8 %, así que esto subestima el error del DNI derivado en general)")
    log("(dirint_v1 y dirint_perez no se comparan: aquí no hay T ni presión medidas)")
    verif_n, verif_der = int(S.sum()), (der_rmse, der_mbe)

    # ---- informe
    g = lambda df, dom, kind, k, col: float(df[(df["domain"] == dom) & (df["model"] == k) & (df["ref"] == kind)][col].iloc[0])
    b = bt.set_index(["ref", "model"])
    rows_main = []
    for k in MAIN:
        rows_main.append({"modelo": k,
                          "DHI RMSE c20 [IC 95 %]": f"{b.loc[('DHI', k), 'RMSE']:.1f} [{b.loc[('DHI', k), 'RMSE_lo']:.1f}, {b.loc[('DHI', k), 'RMSE_hi']:.1f}]",
                          "DHI MBE": f"{b.loc[('DHI', k), 'MBE']:+.1f}",
                          "dif. vs Erbs [IC]": ("—" if k == "erbs" else f"{b.loc[('DHI', k), 'dRMSE_vs_erbs']:+.1f} [{b.loc[('DHI', k), 'dRMSE_lo']:+.1f}, {b.loc[('DHI', k), 'dRMSE_hi']:+.1f}]"
                                                + ("" if not (b.loc[('DHI', k), 'dRMSE_lo'] <= 0 <= b.loc[('DHI', k), 'dRMSE_hi']) else " (IC incluye 0)")),
                          "DNI derivado RMSE c20 [IC]": f"{b.loc[('DNI', k), 'RMSE']:.1f} [{b.loc[('DNI', k), 'RMSE_lo']:.1f}, {b.loc[('DNI', k), 'RMSE_hi']:.1f}]",
                          "DNI MBE": f"{b.loc[('DNI', k), 'MBE']:+.1f}"})
    t_main = pd.DataFrame(rows_main)
    dom_rows = []
    for k in MAIN:
        dom_rows.append({"modelo": k, **{f"DHI {dom}": g(met, dom, "DHI", k, "RMSE") for dom in DOMAINS}, **{f"DNI {dom}": g(met, dom, "DNI", k, "RMSE") for dom in DOMAINS},
                         "n DHI c20": int(g(met, "c20", "DHI", k, "n"))})
    t_dom = pd.DataFrame(dom_rows)
    cl_rows = []
    for k in CLASSIC:
        cl_rows.append({"modelo": k, **{f"DHI {dom}": g(met_c, dom, "DHI", k, "RMSE") for dom in DOMAINS}, **{f"DNI {dom}": g(met_c, dom, "DNI", k, "RMSE") for dom in DOMAINS}})
    t_cl = pd.DataFrame(cl_rows)
    n_cl = {dom: int(g(met_c, dom, "DHI", "erbs", "n")) for dom in DOMAINS}
    ty = by_year[by_year["ref"] == "DHI"].pivot(index="model", columns="year", values="RMSE").reindex(MAIN)
    ty.columns = [str(c) for c in ty.columns]
    ts_ = by_season[by_season["ref"] == "DHI"].pivot(index="model", columns="season", values="RMSE").reindex(MAIN)[["DEF", "MAM", "JJA", "SON"]]
    nyr = by_year[(by_year["ref"] == "DHI") & (by_year["model"] == "erbs")].set_index("year")["n"]
    ns = by_season[(by_season["ref"] == "DHI") & (by_season["model"] == "erbs")].set_index("season")["n"]
    t_sim = sim.copy()
    t_sim = t_sim.rename(columns={"model": "modelo", "RMSD_vs_measured": "RMSD vs DNI derivado", "RMSD_vs_gompertz": "RMSD vs Gompertz", "ratio_gomp_over_meas": "cociente Gompertz/derivado"})
    best_dhi = t_main.iloc[[int(np.argmin([b.loc[("DHI", k), "RMSE"] for k in MAIN]))]]["modelo"].iloc[0]
    L = ["# Mendoza (INAHE) — corrida PROVISORIA de modelos de descomposición contra datos medidos",
         "",
         f"Generado por `scripts/run_mendoza_provisional.py` el {ts} desde `data/Mendoza_Datos_Medidos` (solo lectura). **Todo sigue siendo provisorio**, pero ahora con los datos que confirmó el técnico "
         f"el 2026-10-06: latitud {a.lat}, longitud {a.lon}, hora de reloj, marca de tiempo = {'final' if a.stamp == 'end' else 'inicio'} de cada minuto. **Sigue sin confirmar la altitud** "
         f"(se usa {a.elev:.0f} m como supuesto), y no hay temperatura, humedad ni presión medidas.",
         "",
         "## Qué se hizo",
         f"- **Ubicación (confirmada por el técnico):** latitud {a.lat}, longitud {a.lon}. **La altitud no fue informada**: se usa {a.elev:.0f} m (supuesto); ver `mendoza_location_sensitivity.csv`: con la altitud entre 650 y 850 m el RMSE de DHI del DIRINT de Perez cambia entre −1.2 y +1.3 W/m² (usa la presión) y con 1000 m, +3.8 W/m²; los demás modelos no cambian (≤ 0.01 W/m²); un desplazamiento horizontal de 2–3 km cambia el RMSE de DHI de cualquier modelo en no más de 0.1 W/m².",
         "- **Instrumentos (según el técnico):** global y difusa con piranómetros Eppley 8-48; banda de sombra de construcción propia, ancho 7.6 cm, radio 31.15 cm (diámetro 62.3 cm). No se informó la calibración ni su incertidumbre.",
         f"- 4 años (2022–2025) de datos de 1 min, **hora de reloj** (UTC−3) y **marca de tiempo = {'final' if a.stamp == 'end' else 'inicio'} de cada minuto**, ambas confirmadas por el técnico (antes eran un supuesto inferido del perfil).",
         "- Los datos ya vienen con control del técnico: corrección de banda de sombra de LeBaron et al. (1990) y filtros de Nollas et al. (2023) (referencias recibidas del técnico). Los huecos son **mezcla de filtro y de falta de dato** (no se sabe en qué proporción; ver `MENDOZA_huecos.md`); el técnico conserva los datos crudos, que solo tienen huecos por falta de dato. Se agregaron solo controles mínimos: faltantes, límites físicos de BSRN para GHI y DHI, y cociente difusa/global (< 1.05 para SZA < 75°, < 1.10 para 75–93°).",
         "- Medias horarias con la regla del piloto (≥ 75 % de minutos válidos por hora). Dominio principal **c20** (cos z ≥ 0.2, GHI ≥ 10 W/m²), según la decisión del autor; `all` y `c065` en la tabla de dominios. Bootstrap por días (2000 réplicas, semilla 20261005), muestra común a todos los modelos.",
         "- **Referencia:** DHI **medida**. El DNI es **derivado** por cierre, (GHI − DHI)/cos z: no es medido y no aporta información independiente de la difusa (el error de DNI de un modelo que cierra es el error de DHI dividido por cos z). Por eso el resultado principal es el de DHI.",
         "- Modelos: Gompertz (la partición de la referencia IWEC2), Erbs, Reindl-2, DIRINT de la v1.0.0 (con su temperatura por defecto de 20 °C, porque no hay temperatura), DIRINT de Perez con «agua precipitable desconocida» y presión estándar, BRL, Engerer2 a 1 min y a 60 min (exploratorio).",
         "",
         f"Muestra común c20: **{n_c20} horas**. Horas c20 con DNI derivado negativo: {n_neg}.",
         "",
         "## Resultado principal (dominio c20)",
         "",
         md_table(t_main),
         "",
         f"Menor RMSE de DHI: `{best_dhi}`. Unidades W/m². Diferencia vs Erbs: negativa = mejor que Erbs.",
         "",
         "## RMSE por dominio (W/m²), los 8 modelos",
         "",
         md_table(t_dom, {"n DHI c20": lambda x: str(int(x))}),
         "",
         "BRL y Engerer2 no están definidos con el sol muy bajo (Z ≥ 85°), así que la muestra común de los 8 modelos deja fuera esas horas y `all` coincide con `c065`. Para ver `all` hay que mirar solo los modelos clásicos:",
         "",
         "## RMSE por dominio (W/m²), solo modelos clásicos (muestra común de esos 5 en cada dominio)",
         "",
         md_table(t_cl),
         "",
         f"n de horas: all = {n_cl['all']}, c065 = {n_cl['c065']}, c20 = {n_cl['c20']}. `all` incluye horas con el sol muy bajo, donde el DNI derivado se vuelve inestable (cos z pequeño); por eso el dominio principal es c20, con `all` aquí como tabla aparte.",
         "",
         "## ¿La referencia Gompertz reproduce lo medido? (T3, c20, DNI derivado)",
         "",
         md_table(t_sim, {"n": lambda x: str(int(x)), "cociente Gompertz/derivado": lambda x: f"{x:.2f}"}),
         "",
         "RMSD vs Gompertz mide parecido a la partición Gompertz; RMSD vs DNI derivado mide acuerdo con lo medido. Un cociente > 1 indica que el modelo se parece más a la medición que a Gompertz.",
         "",
         "## Por año (RMSE de DHI c20, W/m²; muestra común)",
         "",
         md_table(ty.reset_index().rename(columns={"model": "modelo"})),
         "",
         "n por año: " + ", ".join(f"{int(i)}: {int(v)}" for i, v in nyr.items()),
         "",
         "## Por estación del año (RMSE de DHI c20, W/m²; DEF = dic–feb, verano austral)",
         "",
         md_table(ts_.reset_index().rename(columns={"model": "modelo"})),
         "",
         "n por estación: " + ", ".join(f"{i}: {int(v)}" for i, v in ns.items()),
         "",
         "## Sensibilidad (RMSE de DHI c20, W/m²)",
         "",
         md_table(piv.reset_index().rename(columns={"model": "modelo"}), {c: (lambda x: f"{x:.2f}") for c in piv.columns}),
         "",
         f"Máxima diferencia respecto de la base: {d_max:.2f} W/m². Variantes: marca de tiempo = {'inicio' if a.stamp == 'end' else 'final'} de minuto (la contraria a la confirmada); coordenadas desplazadas 0.1° al sur y al oeste; sin el filtro de cociente difusa/global; cobertura horaria mínima 60 % y 90 %.",
         "",
         "## Verificación del código (Desert Rock con el DNI oculto)",
         "",
         f"El mismo código, aplicado a Desert Rock usando solo GHI y DHI, se compara con el piloto que usa el DNI medido, en las **mismas {verif_n} horas c20**. **Comparación principal: DHI** (la difusa medida es la misma en las dos corridas, así que la diferencia solo viene de la agregación horaria y del control de calidad; debe ser pequeña). La comparación de DNI (medido en el piloto, derivado aquí) además mezcla la diferencia de referencia.",
         "",
         md_table(comp, {c: (lambda x: f"{x:+.2f}") if c.startswith("dif") else (lambda x: f"{x:.2f}") for c in comp.columns if c != "model"}),
         "",
         f"DNI derivado contra DNI medido en esas horas: RMSE {verif_der[0]:.2f} W/m², sesgo {verif_der[1]:+.2f} W/m². **Cuidado:** el piloto solo conservó horas que cumplen el cierre ±8 %, de modo que esa cifra es un piso y subestima el error del DNI derivado en general.",
         "",
         f"En las columnas de DNI la diferencia va de {comp['dif_DNI'].min():+.1f} a {comp['dif_DNI'].max():+.1f} W/m² según el modelo, con distinto signo: **el DNI derivado no sirve para ordenar modelos con confianza; el resultado principal es el de DHI**, que reproduce el piloto con diferencias de a lo sumo {comp['dif_DHI'].abs().max():.2f} W/m².",
         "",
         "Los DIRINT no se comparan (aquí no hay T ni presión medidas).",
         "",
         "## Qué NO está verificado (no usar como hecho)",
         "1. **Altitud** de la estación: no informada. Tampoco la calibración ni la incertidumbre de los Eppley 8-48.",
         "2. **Independencia de los datos respecto de los modelos evaluados.** El filtro 8 de Nollas et al. (2023) acepta cada minuto solo si la fracción difusa medida queda dentro de ±0.1 (cielo despejado, modelo Engerer3) o ±0.35 (resto, modelo Engerer2) de la que predice Engerer3/Engerer2. Si el técnico lo aplicó tal cual (falta confirmarlo), los datos de Mendoza fueron seleccionados con un modelo de la familia Engerer, y la comparación de Engerer2 (y, en menor medida, de BRL) con estos datos no es independiente. El filtrado además deja huecos que no son aleatorios (Nollas et al.: la fracción filtrada crece con el ángulo cenital).",
         "3. El DNI es derivado; la incertidumbre de la difusa medida con banda de sombra no está cuantificada acá. La corrección de LeBaron et al. (1990) se desarrolló con datos horarios de dos sitios de EE. UU. (RMSE 6.9 W/m² sobre un tercio de esos datos) y los autores no la validaron en otros climas; acá se aplica a datos de 1 min en un sitio árido de altura.",
         "4. Sin temperatura, humedad ni presión: el DIRINT se evaluó sin esos datos (la v1.0.0 con 20 °C por defecto).",
         "5. Posible solapamiento entre estas mediciones y los datos con que se ajustaron BRL/Engerer2: no verificado.",
         "6. Es una estación; no representa a todo el país.",
         "",
         "## Reproducir",
         "```powershell",
         "cd revision_v2",
         f"python scripts/run_mendoza_provisional.py --lat {a.lat} --lon {a.lon} --elev {a.elev:g} --stamp {a.stamp}",
         "```",
         ""]
    (RESP / "MENDOZA_corrida_provisoria.md").write_text("\n".join(L), encoding="utf-8")
    (LOGS / f"mendoza_prov_{ts}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nescrito {RESP / 'MENDOZA_corrida_provisoria.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
