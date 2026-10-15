"""controlled_reference_effect.py — experimento controlado del efecto de la referencia (observación 1 de la revisión externa).

Pregunta: ¿el orden de los modelos cambia por la REFERENCIA (partición de Gompertz) o por las ENTRADAS (la mezcla de cielos de la GHI)?
Diseño: en cada estación medida se usan exactamente las mismas horas (c20, muestra común), la misma GHI, la misma geometría y los mismos modelos;
las predicciones se evalúan (a) contra el DNI medido y (b) contra la partición de Gompertz calculada con esa misma GHI y geometría.
ΔRMSE = RMSE(modelo) − RMSE(Erbs) (negativo = más cerca que Erbs). IC 95 % por bootstrap de días (2000 réplicas, semilla de asg.SEED), pareado.
Se informa en total y por clase de Kt (<0.3, 0.3–0.7, 0.7–0.9, >0.9: los archivos IWEC2 no superan 0.9), con reponderación por la mezcla de cielos
de IWEC2 y, en Mendoza, el análogo con la difusa medida.
Salidas: data/processed/controlled_ref_effect.csv, controlled_ref_effect_reweight.csv, controlled_ref_effect_mendoza.csv, response/EXPERIMENTO_CONTROLADO.md
Uso (desde revision_v2, venv): python scripts/controlled_reference_effect.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
import dirint_dewpoint_anchor as dd  # noqa: E402
import run_mendoza_provisional as rmp  # noqa: E402

ROOT = HERE.parent
PROC = ROOT / "data" / "processed"
B = 2000
MODELS = ["erbs", "reindl2", "dirint", "sd100"]
LAB = {"erbs": "Erbs", "reindl2": "Reindl-2", "dirint": "DIRINT", "sd100": "SD-1.0.0 table"}
CLASSES = [("all", None, None), ("<0.3", -np.inf, 0.3), ("0.3-0.7", 0.3, 0.7), (">=0.7", 0.7, np.inf), ("0.7-0.9", 0.7, 0.9), (">0.9", 0.9, np.inf)]
STN = ["PTR", "SMS", "DRA", "GOLDEN"]


def load_station(stn: str) -> dict:
    if stn == "GOLDEN":
        h = asg.golden_base_frame()
        epw = asg.golden_epw(h)
        res, sim, series, mask = asg.evaluate(epw, h["dni"].to_numpy())
        lon = asg.G_LON
    else:
        minute, geo, meta = asg.minute_station(stn)
        res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
        epw = ba.to_epw_like(h, "avg")
        lon = meta["lon"]
    _, noTd, _ = dd.dirint_variants(epw)
    ser = {"gompertz": np.asarray(series["gompertz (T1)"], float), "erbs": np.asarray(series["erbs"], float), "reindl2": np.asarray(series["reindl2"], float),
           "sd100": np.asarray(series["dirint_v1"], float), "dirint": np.asarray(noTd, float)}
    kt = np.asarray(epw["ghi"], float) / np.asarray(epw["etr_h"], float)
    day = (pd.DatetimeIndex(h.index) + pd.Timedelta(hours=lon / 15.0)).floor("D").to_numpy()
    return dict(ref=h["dni"].to_numpy(float), ser=ser, mask=np.asarray(mask, bool), kt=kt, day=day, unit="DNI")


def load_mendoza_dhi() -> dict:
    P = rmp.process(rmp.mendoza_minute("end"), -32.89, -68.87, 750.0)
    ser_dni = {"gompertz": P["series"]["gompertz (T1)"], "erbs": P["series"]["erbs"], "reindl2": P["series"]["reindl2"], "sd100": P["series"]["dirint_v1"], "dirint": P["series"]["dirint_perez"]}
    dhi = {k: np.asarray(rmp.dhi_from_dni(P["ghi"], v, P["cosz"]), float) for k, v in ser_dni.items()}
    idx = pd.DatetimeIndex(P["h"].index)
    doy = idx.dayofyear.to_numpy()
    g0h = 1367.0 * (1 + 0.033 * np.cos(np.radians(360.0 * doy / 365.0))) * np.asarray(P["cosz"], float)
    kt = np.asarray(P["ghi"], float) / g0h
    day = (idx + pd.Timedelta(hours=-68.87 / 15.0)).floor("D").to_numpy()
    mask = np.asarray(rmp.common_mask(P), bool)
    return dict(ref=np.asarray(P["dhi_m"], float), ser=dhi, mask=mask, kt=kt, day=day, unit="DHI")


def class_stats(d: dict, ref_kind: str, rng) -> list[dict]:
    """ref_kind: 'medido' o 'Gompertz'. Misma muestra para todos los modelos y para las dos referencias."""
    ok = d["mask"] & np.isfinite(d["ref"]) & np.isfinite(d["kt"])
    for k in d["ser"]:
        ok &= np.isfinite(d["ser"][k])
    ref = d["ref"] if ref_kind == "medido" else d["ser"]["gompertz"]
    days = np.unique(d["day"][ok])
    rows = []
    for cname, lo, hi in CLASSES:
        sel = ok if lo is None else ok & (d["kt"] >= lo) & (d["kt"] < hi)
        if sel.sum() < 30:
            continue
        di = {x: np.flatnonzero(sel & (d["day"] == x)) for x in days}
        nd = np.array([len(di[x]) for x in days], float)
        sse = {m: np.array([np.sum((d["ser"][m][di[x]] - ref[di[x]]) ** 2) for x in days]) for m in MODELS}
        se = {m: np.array([np.sum(d["ser"][m][di[x]] - ref[di[x]]) for x in days]) for m in MODELS}
        pt = {m: np.sqrt(sse[m].sum() / nd.sum()) for m in MODELS}
        draws = rng.integers(0, len(days), size=(B, len(days)))
        den = nd[draws].sum(axis=1)
        bs = {m: np.sqrt(sse[m][draws].sum(axis=1) / den) for m in MODELS}
        for m in MODELS:
            dl, dh = np.percentile(bs[m] - bs["erbs"], [2.5, 97.5])
            rows.append(dict(ref=ref_kind, clase=cname, n=int(nd.sum()), pct=100 * nd.sum() / ok.sum(), modelo=m, RMSE=pt[m], MBE=se[m].sum() / nd.sum(),
                             d_vs_erbs=pt[m] - pt["erbs"], d_lo=dl, d_hi=dh))
    return rows


def main() -> int:
    warnings.filterwarnings("ignore")
    rows = []
    data = {}
    for stn in STN:
        print("estación", stn, flush=True)
        data[stn] = load_station(stn)
        for kind in ("medido", "Gompertz"):
            r = class_stats(data[stn], kind, np.random.default_rng(asg.SEED))
            rows += [dict(estacion=stn, **x) for x in r]
    print("Mendoza (difusa)", flush=True)
    men = load_mendoza_dhi()
    men_rows = []
    for kind in ("medido", "Gompertz"):
        men_rows += [dict(estacion="MENDOZA", **x) for x in class_stats(men, kind, np.random.default_rng(asg.SEED))]
    df = pd.DataFrame(rows)
    dm = pd.DataFrame(men_rows)
    df.to_csv(PROC / "controlled_ref_effect.csv", index=False)
    dm.to_csv(PROC / "controlled_ref_effect_mendoza.csv", index=False)

    # reponderación con la mezcla de cielos de IWEC2 (horas de c20, 39 archivos): 16.0 / 73.3 / 10.7 %
    w = {"<0.3": 0.160, "0.3-0.7": 0.733, ">=0.7": 0.107}
    rw = []
    for (stn, kind), g in df.groupby(["estacion", "ref"]):
        rec = {"estacion": stn, "ref": kind}
        for m in MODELS:
            v = np.sqrt(sum(w[c] * g[(g.clase == c) & (g.modelo == m)].RMSE.iloc[0] ** 2 for c in w))
            rec[m] = v
        rec["d_reindl_erbs_mezcla_IWEC2"] = rec["reindl2"] - rec["erbs"]
        rec["d_reindl_erbs_mezcla_propia"] = g[(g.clase == "all") & (g.modelo == "reindl2")].d_vs_erbs.iloc[0]
        rec["pct_Kt>=0.7_propio"] = g[(g.clase == ">=0.7") & (g.modelo == "erbs")].pct.iloc[0]
        rw.append(rec)
    rw = pd.DataFrame(rw)
    rw.to_csv(PROC / "controlled_ref_effect_reweight.csv", index=False)

    pd.set_option("display.width", 250)
    p = df[(df.modelo != "erbs")].copy()
    print(p.round(1).to_string())
    print(rw.round(1).to_string())
    print(dm[dm.modelo != "erbs"].round(1).to_string())

    # experimento natural dentro de IWEC2: misma referencia, solo cambia la mezcla de cielos entre estaciones
    from scipy import stats
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    h = h[(h.cosz >= 0.2) & (h.source == "IWEC2")]
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    bm = bm[(bm.domain == "c20") & (bm.source == "IWEC2")]
    rmse = bm.pivot(index="city", columns="model", values="DNI_RMSE")
    share = h.groupby("city").apply(lambda g: (g.kt >= 0.7).mean() * 100)
    prov = pd.read_csv(PROC / "epw_provenance.csv").set_index("city")
    cross = pd.DataFrame({"share_kt07": share, "d_reindl_erbs": (rmse.reindl2 - rmse.erbs).loc[share.index], "d_dirint_erbs": (rmse.dirint_std_noTd - rmse.erbs).loc[share.index],
                          "abs_lat": prov.loc[share.index, "lat"].abs(), "elev_m": prov.loc[share.index, "elev_m"]})
    cross.to_csv(PROC / "controlled_ref_effect_cross_station.csv")
    rho, pv = stats.spearmanr(cross.share_kt07, cross.d_reindl_erbs)
    lr = stats.linregress(cross.share_kt07, cross.d_reindl_erbs)

    def rk(x):
        return stats.rankdata(x)

    def resid(y, X):
        X = np.column_stack([np.ones(len(y))] + X)
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    pr = np.corrcoef(resid(rk(cross.d_reindl_erbs), [rk(cross.abs_lat), rk(cross.elev_m)]), resid(rk(cross.share_kt07), [rk(cross.abs_lat), rk(cross.elev_m)]))[0, 1]
    ter = pd.qcut(cross.share_kt07, 3, labels=["bajo", "medio", "alto"])
    tt = cross.groupby(ter, observed=True).agg(share=("share_kt07", "mean"), d=("d_reindl_erbs", "mean"), n=("d_reindl_erbs", "count"))
    rho_d, p_d = stats.spearmanr(cross.share_kt07, cross.d_dirint_erbs)
    hi = cross[cross.share_kt07 > -lr.intercept / lr.slope]
    print(f"IWEC2: Spearman(share Kt>=0.7, dRMSE Reindl-Erbs) = {rho:.3f} (p = {pv:.2g}); pendiente {lr.slope:.3f} W/m2 por punto; cruce en {-lr.intercept / lr.slope:.1f} %; parcial|lat,elev = {pr:.3f}")
    print(tt.round(1))

    # informe breve
    L = ["# Experimento controlado del efecto de la referencia (provisorio, generado por `scripts/controlled_reference_effect.py`)", "",
         "Mismas horas (c20, muestra común), misma GHI y geometría; predicciones evaluadas contra el DNI medido y contra Gompertz calculado con la misma GHI. "
         "ΔRMSE = RMSE(modelo) − RMSE(Erbs), IC 95 % por bootstrap de días (2000 réplicas). Kt = GHI horaria / extraterrestre horizontal.", "",
         "## RMSE de DNI (W/m²) y ΔRMSE respecto de Erbs [IC 95 %] por estación, referencia y clase de Kt", "",
         "| estación | referencia | clase Kt | % horas | n | Erbs | Reindl-2 | Δ Reindl-2 | DIRINT | Δ DIRINT | SD-1.0.0 | Δ SD-1.0.0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (stn, kind, cl), g in df.groupby(["estacion", "ref", "clase"], sort=False):
        e = g[g.modelo == "erbs"].iloc[0]
        def f(m):
            x = g[g.modelo == m].iloc[0]
            return f"{x.RMSE:.1f} | {x.d_vs_erbs:+.1f} [{x.d_lo:+.1f}, {x.d_hi:+.1f}]"
        L.append(f"| {stn} | {kind} | {cl} | {e.pct:.1f} | {e.n} | {e.RMSE:.1f} | {f('reindl2')} | {f('dirint')} | {f('sd100')} |")
    L += ["", "## Reponderación: ΔRMSE Reindl-2 − Erbs si cada estación tuviera la mezcla de cielos de IWEC2 (16.0 / 73.3 / 10.7 %)", "",
          "```", rw.round(2).to_string(index=False), "```", "",
          "## Mendoza: análogo con la difusa medida (RMSE de DHI y ΔRMSE respecto de Erbs)", ""]
    for (kind, cl), g in dm.groupby(["ref", "clase"], sort=False):
        e = g[g.modelo == "erbs"].iloc[0]
        r = g[g.modelo == "reindl2"].iloc[0]
        L.append(f"- {kind}, Kt {cl} ({e.pct:.1f} % de las horas, n = {e.n}): Erbs {e.RMSE:.1f}; Reindl-2 {r.RMSE:.1f} (Δ {r.d_vs_erbs:+.1f} [{r.d_lo:+.1f}, {r.d_hi:+.1f}])")
    L += ["", "## Análisis entre estaciones dentro de IWEC2 (asociación; misma referencia, cambia la mezcla de cielos entre las 39 estaciones; no demuestra causalidad)", "",
          f"- Spearman entre el % de horas con Kt ≥ 0.7 y ΔRMSE (Reindl-2 − Erbs): ρ = {rho:.3f} (p = {pv:.2g}); parcial controlando |latitud| y elevación: {pr:.3f}.",
          f"- Regresión: pendiente {lr.slope:.3f} W/m² por punto porcentual; ΔRMSE = 0 con {-lr.intercept / lr.slope:.1f} % de horas con Kt ≥ 0.7; las {len(hi)} estaciones por encima de ese valor tienen Erbs más cerca ({int((hi.d_reindl_erbs > 0).sum())} de {len(hi)}).",
          f"- Terciles de la fracción de horas con Kt ≥ 0.7: " + "; ".join(f"{i}: {r.share:.1f} % → ΔRMSE {r.d:+.1f} W/m² (n = {int(r.n)})" for i, r in tt.iterrows()) + ".",
          f"- ΔRMSE (DIRINT − Erbs) contra la fracción: ρ = {rho_d:.3f} (p = {p_d:.2g}).", ""]
    (ROOT / "response" / "EXPERIMENTO_CONTROLADO.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito response/EXPERIMENTO_CONTROLADO.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
