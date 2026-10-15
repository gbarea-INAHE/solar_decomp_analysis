"""sample_representativeness.py — ¿Qué tan representativas son las 39 estaciones IWEC2 analizadas frente a las 54 de Argentina que tiene la lista de IWEC2?

Contexto (autor, 2026-10-06): las 39 fueron los EPW que se tenían descargados, sin criterio de selección. La lista de IWEC2 (ASHRAE RP-1477) tiene 54 estaciones argentinas;
las 15 que faltan (no hay archivos en disco; son de licencia) no se pueden analizar. Este script NO las analiza: documenta la composición de la muestra con los datos de la
lista (latitud, altitud, clase, calidad) y pone a prueba cuánto podría cambiar la conclusión si se hubieran incluido, de dos maneras:
  (1) estabilidad por subconjuntos: ¿el orden de los modelos y su diferencia se mantienen en las estaciones de baja altitud, en las Cfa, etc.? (las 15 ausentes son de ese tipo);
  (2) punto de quiebre: ¿qué diferencia media tendrían que tener las 15 ausentes para que la diferencia media de las 54 cambiara de signo, comparada con el rango observado en las 39?
Dominio c20 (principal). RMSE de DNI contra IWEC2; modelos: Reindl-2, Erbs, DIRINT de Perez et al. (1992) sin punto de rocío (`dirint_std_noTd`) y DIRINT v1.0.0 (`dirint_v1`).
Bootstrap de estaciones: 2000 réplicas, semilla 20261005. Solo lectura de data/processed y data/ashrae_list.
Uso (desde revision_v2, venv activo): python scripts/sample_representativeness.py
Escribe data/processed/sample_rep_*.csv y response/MUESTRA_39_de_54.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
SEED, B = 20261005, 2000
LAB = {"reindl2": "Reindl-2", "erbs": "Erbs", "dirint_std_noTd": "DIRINT (Perez et al., 1992)", "dirint_v1": "DIRINT v1.0.0"}
MODELS = list(LAB)
N_ALL, N_USED = 54, 39


def md_table(df: pd.DataFrame, nd: int = 1) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def sg(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}".replace("-", "−")


def main() -> int:
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    k = pd.read_csv(PROC / "koppen_ashrae_vs_s1.csv")
    a = pd.read_csv(REV / "data" / "ashrae_list" / "ashrae_arg_stations.csv")
    assert len(a) == N_ALL and len(k) == N_USED, (len(a), len(k))
    a["en_analisis"] = a["wmo"].isin(k["wmo"])
    assert int(a["en_analisis"].sum()) == N_USED
    meta = k.merge(a[["wmo", "lat", "elev_m"]], on="wmo", how="left")
    assert meta["elev_m"].notna().all() and set(meta["city"]) <= set(bm["city"])
    c20 = bm[(bm["domain"] == "c20") & bm["city"].isin(meta["city"])]
    rmse = c20.pivot(index="city", columns="model", values="DNI_RMSE").loc[meta["city"], MODELS]
    mbe = c20.pivot(index="city", columns="model", values="DNI_MBE").loc[meta["city"], MODELS]
    assert rmse.notna().all().all() and len(rmse) == N_USED
    meta = meta.set_index("city")

    # ---- A. composición
    comp_rows = []
    for name, sub in (("54 de la lista de IWEC2", a), ("39 analizadas", a[a["en_analisis"]]), ("15 no analizadas", a[~a["en_analisis"]])):
        comp_rows.append({"grupo": name, "n": len(sub), "latitud mín–máx (°S)": f"{-sub['lat'].max():.1f}–{-sub['lat'].min():.1f}".replace("-", ""),
                          "altitud media (m)": float(sub["elev_m"].mean()), "altitud máx (m)": int(sub["elev_m"].max()), "con más de 500 m": int((sub["elev_m"] > 500).sum()),
                          "calidad A / B / C": " / ".join(str(int((sub["dq"] == q).sum())) for q in "ABC")})
    comp = pd.DataFrame(comp_rows)
    cls = pd.crosstab(a["koppen_ashrae"], a["en_analisis"]).rename(columns={True: "analizadas", False: "no analizadas"})
    cls["total"] = cls.sum(axis=1)
    cls = cls.reset_index().rename(columns={"koppen_ashrae": "clase (ASHRAE)"})
    miss = a[~a["en_analisis"]][["ashrae_station", "lat", "elev_m", "koppen_ashrae", "dq"]].rename(columns={"ashrae_station": "estación", "lat": "lat", "elev_m": "altitud (m)", "koppen_ashrae": "clase", "dq": "calidad"})

    # ---- B. estabilidad por subconjuntos
    subsets = {"las 39": np.ones(N_USED, bool),
               "baja altitud (≤ 500 m)": (meta["elev_m"] <= 500).to_numpy(),
               "más de 500 m": (meta["elev_m"] > 500).to_numpy(),
               "clase Cfa (todas son de baja altitud)": (meta["koppen_ashrae"] == "Cfa").to_numpy()}
    rng = np.random.default_rng(SEED)
    rows, rowsd = [], []
    for name, m in subsets.items():
        r = rmse[m]
        n = int(m.sum())
        means = r.mean()
        order = " < ".join(LAB[x] for x in means.sort_values().index)
        d = (r["reindl2"] - r["erbs"]).to_numpy()
        idx = rng.integers(0, n, (B, n))
        bs = d[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"subconjunto": name, "n": n, **{LAB[x] + " (RMSE medio de DNI, W/m²)": float(means[x]) for x in MODELS}, "orden (menor a mayor RMSE)": order})
        rowsd.append({"subconjunto": name, "n": n, "Reindl-2 menos Erbs, media (W/m²)": float(d.mean()), "IC 95 % (bootstrap de estaciones)": f"[{lo:.1f}, {hi:.1f}]",
                      "estaciones con Reindl-2 < Erbs": f"{int((d < 0).sum())} de {n}",
                      "sesgo medio DIRINT v1.0.0 (W/m²)": float(mbe.loc[m, 'dirint_v1'].mean()), "estaciones con sesgo > 0 (DIRINT v1.0.0)": f"{int((mbe.loc[m, 'dirint_v1'] > 0).sum())} de {n}"})
    # las 39 reponderadas por altitud a la composición de las 54 (44 de ≤ 500 m y 10 de más de 500 m): remuestreo estratificado por altitud
    low = (meta["elev_m"] <= 500).to_numpy()
    hi = ~low
    n_low54, n_hi54 = int((a["elev_m"] <= 500).sum()), int((a["elev_m"] > 500).sum())
    assert n_low54 + n_hi54 == N_ALL

    def post(v: np.ndarray) -> float:
        return (n_low54 * v[low].mean() + n_hi54 * v[hi].mean()) / N_ALL

    means_w = {x: post(rmse[x].to_numpy()) for x in MODELS}
    dw = (rmse["reindl2"] - rmse["erbs"]).to_numpy()
    rng2 = np.random.default_rng(SEED)
    il, ih = np.where(low)[0], np.where(hi)[0]
    bsw = np.empty(B)
    for i in range(B):
        sl, sh = rng2.choice(il, len(il)), rng2.choice(ih, len(ih))
        bsw[i] = (n_low54 * dw[sl].mean() + n_hi54 * dw[sh].mean()) / N_ALL
    lo_w, hi_w = np.percentile(bsw, [2.5, 97.5])
    rows.append({"subconjunto": f"las 39 reponderadas por altitud a la composición de las 54 ({n_low54} de ≤ 500 m, {n_hi54} de más)", "n": N_USED,
                 **{LAB[x] + " (RMSE medio de DNI, W/m²)": float(means_w[x]) for x in MODELS}, "orden (menor a mayor RMSE)": " < ".join(LAB[x] for x in sorted(MODELS, key=means_w.get))})
    rowsd.append({"subconjunto": "las 39 reponderadas por altitud a la composición de las 54", "n": N_USED, "Reindl-2 menos Erbs, media (W/m²)": post(dw),
                  "IC 95 % (bootstrap de estaciones)": f"[{lo_w:.1f}, {hi_w:.1f}]", "estaciones con Reindl-2 < Erbs": "—",
                  "sesgo medio DIRINT v1.0.0 (W/m²)": post(mbe["dirint_v1"].to_numpy()), "estaciones con sesgo > 0 (DIRINT v1.0.0)": "—"})
    tb, tbd = pd.DataFrame(rows), pd.DataFrame(rowsd)

    # ---- C. punto de quiebre: ¿qué media tendrían que tener las 15 ausentes para anular la diferencia media de las 54?
    n_out = N_ALL - N_USED
    tip = []
    pairs = [("Reindl-2 menos Erbs", rmse["reindl2"] - rmse["erbs"]),
             ("Erbs menos DIRINT (Perez et al., 1992)", rmse["erbs"] - rmse["dirint_std_noTd"]),
             ("DIRINT (Perez et al., 1992) menos DIRINT v1.0.0", rmse["dirint_std_noTd"] - rmse["dirint_v1"]),
             ("Reindl-2 menos DIRINT v1.0.0", rmse["reindl2"] - rmse["dirint_v1"]),
             ("sesgo medio de DIRINT v1.0.0 (MBE)", mbe["dirint_v1"])]
    for name, d in pairs:
        m = float(d.mean())
        xstar = -N_USED * m / n_out                          # media de las 15 que haría 0 la media de las 54
        wrong = d.max() if m < 0 else d.min()                # extremo observado en el sentido contrario al de la media
        beyond = (xstar > wrong) if m < 0 else (xstar < wrong)
        tip.append({"comparación": name, "media en las 39 (W/m²)": m, "media que tendrían que tener las 15 para anularla (W/m²)": xstar,
                    "valor extremo observado en una estación de las 39 (en sentido contrario)": float(wrong), "¿el valor de quiebre queda fuera del rango observado?": "sí" if beyond else "no",
                    "estaciones con el signo contrario al de la media": f"{int(((d > 0) if m < 0 else (d < 0)).sum())} de {N_USED}"})
    tp = pd.DataFrame(tip)

    comp.to_csv(PROC / "sample_rep_composition.csv", index=False)
    cls.to_csv(PROC / "sample_rep_classes.csv", index=False)
    tb.to_csv(PROC / "sample_rep_subsets_rmse.csv", index=False)
    tbd.to_csv(PROC / "sample_rep_subsets_diff.csv", index=False)
    tp.to_csv(PROC / "sample_rep_tipping.csv", index=False)

    L = ["# Las 39 estaciones frente a las 54 de IWEC2 en Argentina: justificación por si un revisor pregunta", "",
         "Generado por `scripts/sample_representativeness.py`. **Hecho de base (autor, 6/10):** las 39 son los archivos EPW que se tenían descargados; no hubo un criterio de selección. "
         "La lista de IWEC2 (ASHRAE RP-1477) tiene 54 estaciones argentinas (Huang et al., 2014, Tabla 1). Las 15 restantes **no están en disco** (IWEC2 es de licencia ASHRAE/White Box Technologies) y no se analizan. "
         "Dominio c20, RMSE de DNI contra IWEC2; bootstrap de estaciones (2000 réplicas, semilla 20261005). Lo que sigue **no analiza las 15**: documenta cómo son y prueba cuánto podrían cambiar las conclusiones.", "",
         "## 1. Composición (datos de la lista de IWEC2)", "", md_table(comp), "", "Por clase Köppen–Geiger de la lista:", "", md_table(cls, 0), "",
         "Las 15 no analizadas:", "", md_table(miss, 1), "",
         "## 2. Estabilidad por subconjuntos (las 15 ausentes son de baja altitud y de la llanura húmeda o la costa)", "", md_table(tb, 1), "", md_table(tbd, 1), "",
         "## 3. Punto de quiebre: qué tendrían que valer las 15 para cambiar la conclusión", "",
         "Si a las 39 se sumaran 15 estaciones con una diferencia media distinta, la media de las 54 sería (39·m₃₉ + 15·m₁₅)/54. Se anula cuando m₁₅ = −39·m₃₉/15.", "", md_table(tp, 1), "",
         "## Cómo leerlo",
         "- Es un análisis de **sensibilidad**, no una prueba: no sabemos cómo se comportan las 15 estaciones. Lo que muestra es que una conclusión que se sostiene en todos los subconjuntos y que exigiría que las 15 se comportaran de un modo que ninguna de las 39 se comporta es poco probable que se revierta.",
         "- La muestra de 39 tiene más estaciones altas (10 de 39, el 26 %) que la lista completa (10 de 54, el 19 %). Donde Reindl-2 se distingue menos de Erbs es en las altas; por eso, reponderar por altitud a la composición de las 54 refuerza y no debilita la diferencia a favor de Reindl-2 (tabla 2).",
         "- Las 15 son de baja altitud (hasta 421 m) y están en clases ya presentes (Cfa, BSk) o ausentes (Cfb, Cfc); el análisis **no** puede decir nada sobre Cfb ni Cfc, y el texto no debe generalizar a esas clases.",
         "- Como el DNI de IWEC2 es una función determinista de la GHI de cada archivo (Huang et al., 2014; Test 0), lo que cambia entre estaciones es la distribución de Kt y la geometría solar, y no una calidad de medición; eso es lo que las 39 recorren (latitud de 22.6° a 54.8° S, altitud hasta 1425 m).",
         ""]
    (REV / "response" / "MUESTRA_39_de_54.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito response/MUESTRA_39_de_54.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
