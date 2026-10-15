"""mendoza_gaps_profile.py — ¿Dónde están los huecos de los datos de Mendoza? SOLO LECTURA.

El técnico informó (2026-10-06) que los huecos son «tanto por filtro de datos como por falta de datos»; los datos crudos (que no se nos pasaron) solo
tienen huecos por falta de dato. Sin los crudos NO se puede separar filtro de falta de dato; acá se describe lo que sí se ve en los datos entregados:
(1) cobertura de minutos por ángulo cenital y por año; (2) largo de los huecos (los huecos de muchos minutos seguidos son compatibles con cortes del
equipo, los de pocos minutos con filtrado; es una pista, no una clasificación); (3) qué parte de las horas del dominio c20 se pierde por la regla de
cobertura horaria (≥ 75 % de minutos válidos) y (4) si en las horas con huecos parciales los minutos faltantes se concentran en cierto índice de claridad.

Uso (desde revision_v2, venv activo): python scripts/mendoza_gaps_profile.py --lat -32.89 --lon -68.87 --elev 750 --stamp end
Escribe data/processed/mendoza_gaps_*.csv y response/MENDOZA_huecos.md. No modifica los datos.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bsrn_anchor as ba  # noqa: E402
import run_mendoza_provisional as rmp  # noqa: E402

REV = rmp.REV
OUT = rmp.OUT
Z_BINS = [0, 30, 45, 60, 70, float(np.degrees(np.arccos(0.2))), 85, 90.0001]
Z_LAB = ["0–30°", "30–45°", "45–60°", "60–70°", "70–78.5° (límite c20)", "78.5–85°", "85–90°"]
RUN_BINS = [0, 1, 9, 59, 1439, np.inf]
RUN_LAB = ["1 min", "2–9 min", "10–59 min", "1 h – 1 día", "más de 1 día"]


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.1f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, required=True)
    ap.add_argument("--stamp", choices=("start", "end"), required=True)
    a = ap.parse_args()

    minute = rmp.mendoza_minute(a.stamp)                                   # índice = inicio de minuto, UTC
    geo = ba.geometry(minute.index, a.lat, a.lon, a.elev, "start")
    g, d = minute["ghi"].to_numpy(float), minute["dhi"].to_numpy(float)
    assert np.array_equal(np.isnan(g), np.isnan(d)), "huecos de global y difusa no coinciden: el análisis asume que se pierden juntas"
    valid = np.isfinite(g)
    zen, cosz = geo["zenith"].to_numpy(), geo["cosz"].to_numpy()
    day = cosz > 0
    local_year = (minute.index - pd.Timedelta(hours=3)).year.to_numpy()

    # ---- (1) cobertura de minutos de día por ángulo cenital y por año
    zb = pd.cut(pd.Series(zen[day]), Z_BINS, right=False, labels=Z_LAB)
    t1 = pd.DataFrame({"zbin": zb.to_numpy(), "valid": valid[day], "year": local_year[day]})
    by_z = t1.groupby("zbin", observed=True)["valid"].agg(minutos="size", validos_pct=lambda x: 100 * x.mean()).reset_index()
    by_zy = t1.pivot_table(index="zbin", columns="year", values="valid", aggfunc=lambda x: 100 * x.mean(), observed=True).reset_index()
    by_zy.columns = [str(c) for c in by_zy.columns]
    by_z.to_csv(OUT / "mendoza_gaps_by_sza.csv", index=False)
    by_zy.to_csv(OUT / "mendoza_gaps_by_sza_year.csv", index=False)
    tot_day_valid = 100 * float(valid[day].mean())

    # ---- (2) largo de los huecos (corridas de minutos seguidos sin dato, sobre toda la serie)
    nan = ~valid
    edge = np.diff(np.r_[False, nan, False].astype(int))
    starts, ends = np.where(edge == 1)[0], np.where(edge == -1)[0]
    lens = ends - starts
    run_len = np.zeros(len(nan), int)
    for s0, e0 in zip(starts, ends):
        run_len[s0:e0] = e0 - s0
    dn = nan & day
    cls = pd.cut(pd.Series(run_len[dn]), RUN_BINS, labels=RUN_LAB, right=True)
    t2 = cls.value_counts(sort=False).rename("minutos_de_dia_sin_dato").reset_index().rename(columns={"index": "largo_del_hueco", "Series": "largo_del_hueco"})
    t2.columns = ["largo_del_hueco", "minutos_de_dia_sin_dato"]
    t2["pct_de_los_minutos_de_dia_sin_dato"] = 100 * t2["minutos_de_dia_sin_dato"] / t2["minutos_de_dia_sin_dato"].sum()
    t2["huecos"] = [int(((lens > lo) & (lens <= hi)).sum()) for lo, hi in zip(RUN_BINS[:-1], RUN_BINS[1:])]
    t2.to_csv(OUT / "mendoza_gaps_runs.csv", index=False)

    # ---- (3) horas del dominio c20 (coseno medio del cenit ≥ 0.2) y cobertura horaria
    h = pd.DataFrame({"valid": valid, "day": day, "cosz": np.clip(cosz, 0, None), "s0": geo["s0"].to_numpy(), "g": np.where(valid, g, 0.0), "zen": zen}, index=minute.index)
    key = minute.index.floor("h")
    n_day = h.groupby(key)["day"].sum()
    hh = pd.DataFrame({"cosz_avg": h.groupby(key)["cosz"].mean(), "zen_avg": h.groupby(key)["zen"].mean(), "n_day": n_day,
                       "n_valid_day": h[h["day"]]["valid"].groupby(h.index[h["day"]].floor("h")).sum().reindex(n_day.index).fillna(0)})
    hh["cov"] = np.where(hh["n_day"] > 0, hh["n_valid_day"] / hh["n_day"].where(hh["n_day"] > 0, 1), np.nan)
    hv = h[h["valid"] & h["day"]]
    etr = (hv["s0"] * hv["cosz"]).groupby(hv.index.floor("h")).mean()
    gm = hv["g"].groupby(hv.index.floor("h")).mean()
    hh["kt"] = (gm / etr).reindex(hh.index)
    el = hh[(hh["cosz_avg"] >= 0.2) & (hh["n_day"] == 60)].copy()
    el["estado"] = np.select([el["cov"] >= 0.75, el["cov"] > 0], ["pasa (≥ 75 %)", "se pierde: cobertura parcial (< 75 %)"], default="se pierde: hora entera sin dato")
    t3 = el["estado"].value_counts().reindex(["pasa (≥ 75 %)", "se pierde: cobertura parcial (< 75 %)", "se pierde: hora entera sin dato"]).rename("horas").reset_index()
    t3.columns = ["estado", "horas"]
    t3["pct"] = 100 * t3["horas"] / t3["horas"].sum()
    t3.to_csv(OUT / "mendoza_gaps_hours_c20.csv", index=False)
    el["zbin"] = pd.cut(el["zen_avg"], Z_BINS, right=False, labels=Z_LAB)
    t3z = el.groupby("zbin", observed=True).apply(lambda x: pd.Series({"horas": len(x), "pasan_pct": 100 * (x["cov"] >= 0.75).mean(),
                                                                       "hora_entera_sin_dato_pct": 100 * (x["cov"] == 0).mean()}), include_groups=False).reset_index()
    t3z.to_csv(OUT / "mendoza_gaps_hours_c20_by_sza.csv", index=False)

    # ---- (4) en horas con huecos PARCIALES (algún minuto válido y alguno faltante): ¿se concentra la falta en cierto índice de claridad?
    kb = pd.cut(el["kt"], [-np.inf, 0.3, 0.7, np.inf], labels=["Kt < 0.3", "0.3 ≤ Kt < 0.7", "Kt ≥ 0.7"], right=False)
    el["kt_class"] = kb
    el["zgrp"] = np.where(el["zen_avg"] < 60, "cenit medio < 60°", "cenit medio 60–78.5°")
    has = el[el["cov"] > 0]
    t4 = has.groupby("kt_class", observed=True).apply(lambda x: pd.Series({"horas_con_algun_dato": len(x), "minutos_faltantes_pct": 100 * (1 - x["cov"]).mean(),
                                                                          "horas_con_algun_hueco_pct": 100 * (x["cov"] < 1).mean()}), include_groups=False).reset_index()
    t4.to_csv(OUT / "mendoza_gaps_by_kt.csv", index=False)
    t4b = has.groupby(["zgrp", "kt_class"], observed=True).apply(lambda x: pd.Series({"horas_con_algun_dato": len(x), "minutos_faltantes_pct": 100 * (1 - x["cov"]).mean(),
                                                                                       "horas_con_algun_hueco_pct": 100 * (x["cov"] < 1).mean()}), include_groups=False).reset_index()
    t4b.to_csv(OUT / "mendoza_gaps_by_kt_and_sza.csv", index=False)
    for t_, c_ in ((t3z, "horas"), (t4, "horas_con_algun_dato"), (t4b, "horas_con_algun_dato")):
        t_[c_] = t_[c_].astype(int)

    # ---- informe
    part = el[(el["cov"] > 0) & (el["cov"] < 1)]
    L = ["# Mendoza (INAHE) — dónde están los huecos de los datos entregados",
         "",
         f"Generado por `scripts/mendoza_gaps_profile.py` (lat {a.lat}, lon {a.lon}, altitud {a.elev:g} m [supuesto], marca de tiempo = {'final' if a.stamp == 'end' else 'inicio'} de minuto). Solo lectura de los datos.",
         "",
         "**Lo que dijo el técnico (2026-10-06):** los huecos son «tanto por filtro de datos como por falta de datos»; los datos crudos, que no se nos entregaron, solo tienen huecos por falta de dato. "
         "**Con lo que tenemos no se puede separar filtro de falta de dato**: lo que sigue describe los huecos, no su causa. Si se consiguen los datos crudos, se puede medir cuánto saca el filtro y si saca de manera selectiva.",
         "",
         f"Global y difusa faltan siempre juntas (verificado). En los minutos con el sol sobre el horizonte hay dato válido en el **{tot_day_valid:.1f} %** (2022–2025).",
         "",
         "## 1. Cobertura de minutos de día por ángulo cenital (% con dato)",
         "",
         md_table(by_z.rename(columns={"zbin": "ángulo cenital", "minutos": "minutos", "validos_pct": "% con dato"})),
         "",
         "Por año:",
         "",
         md_table(by_zy.rename(columns={"zbin": "ángulo cenital"})),
         "",
         "## 2. Largo de los huecos (minutos de día sin dato, según el largo del hueco que los contiene)",
         "",
         md_table(t2.rename(columns={"largo_del_hueco": "largo del hueco", "minutos_de_dia_sin_dato": "minutos de día sin dato", "pct_de_los_minutos_de_dia_sin_dato": "% de los minutos sin dato", "huecos": "huecos (cantidad)"})),
         "",
         "Pista, no clasificación: los huecos de más de un día son compatibles con cortes del equipo; los de 1 a pocos minutos, con filtrado (o con fallas breves del registro).",
         "",
         "## 3. Horas del dominio c20 (coseno medio del cenit ≥ 0.2) y regla de cobertura horaria (≥ 75 % de los minutos)",
         "",
         md_table(t3.rename(columns={"estado": "estado de la hora", "pct": "%"})),
         "",
         "Por ángulo cenital medio de la hora:",
         "",
         md_table(t3z.rename(columns={"zbin": "ángulo cenital", "pasan_pct": "% de horas que pasan", "hora_entera_sin_dato_pct": "% de horas enteras sin dato"})),
         "",
         "## 4. Horas con algún dato: ¿cuánto falta según el índice de claridad de la hora?",
         "",
         "Índice de claridad horario calculado con los minutos que sí tienen dato. Solo horas del dominio c20 con al menos un minuto válido.",
         "",
         md_table(t4.rename(columns={"kt_class": "clase", "horas_con_algun_dato": "horas con algún dato", "minutos_faltantes_pct": "% de minutos faltantes", "horas_con_algun_hueco_pct": "% de horas con algún hueco"})),
         "",
         "Lo mismo separando por ángulo cenital medio de la hora (para que la diferencia no sea solo efecto del sol bajo):",
         "",
         md_table(t4b.rename(columns={"zgrp": "ángulo cenital", "kt_class": "clase", "horas_con_algun_dato": "horas con algún dato", "minutos_faltantes_pct": "% de minutos faltantes", "horas_con_algun_hueco_pct": "% de horas con algún hueco"})),
         "",
         f"De las horas c20 con algún dato, {len(part)} tienen huecos parciales (hay minutos válidos y faltantes). **Lectura:** faltan más minutos en las horas no despejadas que en las despejadas; eso es compatible con un filtrado que actúa más sobre cielos con nubes (el filtro 8 y el K–Kt de Nollas et al.), pero también con cualquier otra causa que se parezca a eso; sin el dato crudo no se puede afirmar. Si fuera filtrado selectivo, la muestra horaria que usamos sobrerrepresentaría los cielos despejados, donde los modelos erran menos (ver el análisis por clases de Kt), y el RMSE de los modelos podría ser algo menor que con todos los datos; no se puede cuantificar con lo que tenemos.",
         "",
         "## Qué se puede decir y qué no",
         "- Se puede decir cuánto de los datos entregados falta, dónde (ángulo cenital) y en qué forma (largo de los huecos).",
         "- No se puede decir qué parte es filtro y qué parte es falta de dato, ni si el filtrado favorece o perjudica a algún modelo. El filtro 8 de Nollas et al. (2023) usa Engerer2/Engerer3, y si el técnico lo aplicó, la comparación de esos modelos con estos datos no es independiente (ver `MENDOZA_corrida_provisoria.md`).",
         "- Para medirlo hace falta el dato crudo: con él se repite el análisis con y sin el filtro 8 y se ve si cambia el orden de los modelos.",
         ""]
    (rmp.RESP / "MENDOZA_huecos.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito", rmp.RESP / "MENDOZA_huecos.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
