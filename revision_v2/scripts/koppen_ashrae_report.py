"""koppen_ashrae_report.py — compara las clases Köppen de la Tabla S1 enviada con las de la lista de ASHRAE de IWEC2 (decisión D7 del autor, 2026-10-05: usar la lista de ASHRAE para las 39 estaciones IWEC2)
y rehace con las clases de ASHRAE las tablas por clase (R1-6, R2-2). Aditivo: no toca las salidas existentes (que siguen con las clases de S1).
Entradas: data/processed/koppen_ashrae_vs_s1.csv (scripts/extract_ashrae_koppen.py), benchmark_v2_per_station.csv, r14_station_diag_c20.csv.
Salidas : response/KOPPEN_ashrae_vs_S1.md; data/processed/koppen_class_tables_ashrae_{c20,all}.csv.
Bootstrap: 2000 remuestreos de estaciones dentro de cada clase, semilla 20261005, solo clases con n >= 3 (como en reanalysis_v2.py). Sin texto para el manuscrito. Provisorio.
Uso (desde revision_v2, venv activo): python scripts/koppen_ashrae_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
SEED = 20261005
B = 2000
MODELS = [("reindl2", "Reindl-2"), ("erbs", "Erbs"), ("dirint_std_noTd", "DIRINT Perez (sin Td)"), ("dirint_v1", "DIRINT v1.0.0")]


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in df.columns) + " |")
    return "\n".join(out)


def class_table(bm: pd.DataFrame, cmap: pd.Series, dom: str) -> pd.DataFrame:
    d = bm[(bm["source"] == "IWEC2") & (bm["domain"] == dom)].copy()
    d["klass"] = d["city"].map(cmap)
    rng = np.random.default_rng(SEED)
    rows = []
    for k, g in d.groupby("klass"):
        ns = g["city"].nunique()
        for m, lab in MODELS:
            v = g[g["model"] == m]["DNI_RMSE"].to_numpy()
            r = dict(domain=dom, klass=k, n_stations=ns, model=lab, DNI_RMSE_mean=float(v.mean()))
            if ns >= 3:
                bs = v[rng.integers(0, ns, size=(B, ns))].mean(axis=1)
                r["lo"], r["hi"] = [float(x) for x in np.quantile(bs, [0.025, 0.975])]
            else:
                r["lo"] = r["hi"] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    c = pd.read_csv(PROC / "koppen_ashrae_vs_s1.csv")
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    kt = pd.read_csv(PROC / "r14_station_diag_c20.csv").set_index("city")["kt_mean"]
    c["kt_mean_c20"] = c["city"].map(kt)
    s1, ash = c.set_index("city")["koppen_s1"], c.set_index("city")["koppen_ashrae"]
    L = ["# Köppen: clases de la Tabla S1 enviada frente a la lista de ASHRAE de IWEC2 (provisorio)", "",
         "Generado por `scripts/koppen_ashrae_report.py` desde `data/processed/koppen_ashrae_vs_s1.csv`. **Decisión D7 del autor (5/10): usar las clases de la lista de ASHRAE** para las 39 estaciones IWEC2 (las 2 de otra procedencia no están en la lista y conservan su clase de S1). "
         "Fuente: PDF «IWEC2 Station List» (ASHRAE RP-1477, Apéndice A), descargado el 5/10 con autorización del autor (465,455 bytes; SHA256 en `data/ashrae_list/MANIFEST_ashrae_list.csv`). "
         "El emparejamiento es por número WMO del nombre del EPW; 39 de 39 estaciones se emparejaron y las 54 filas de Argentina del PDF se leyeron sin pérdidas (el script falla si no).", ""]
    L += [f"## 1. Resultado: {int((~c['same']).sum())} de {len(c)} estaciones cambian de clase", "",
          "**Clases (39 IWEC2)**", "",
          md_table(pd.DataFrame({"Clase": sorted(set(s1) | set(ash)), "S1 enviada (n)": [int((s1 == k).sum()) for k in sorted(set(s1) | set(ash))], "ASHRAE (n)": [int((ash == k).sum()) for k in sorted(set(s1) | set(ash))]})), "",
          f"Estaciones de clase B (áridas): S1 {int(s1.str.startswith('B').sum())}, ASHRAE {int(ash.str.startswith('B').sum())}. Clases distintas: S1 {s1.nunique()}, ASHRAE {ash.nunique()} "
          f"(el manuscrito dice ocho clases). Clases con n ≤ 2: S1 {int((s1.value_counts() <= 2).sum())}, ASHRAE {int((ash.value_counts() <= 2).sum())}. "
          f"Grados de calidad de datos (DQ) de ASHRAE en estas 39: {c['dq'].value_counts().sort_index().to_dict()}.", "",
          "**De S1 a ASHRAE (filas = clase en S1; columnas = clase en ASHRAE)**", "",
          md_table(pd.crosstab(s1, ash).reset_index().rename(columns={"koppen_s1": "S1 \\ ASHRAE"})), "",
          "**Las estaciones que cambian**", "",
          md_table(c[~c["same"]][["city", "wmo", "koppen_s1", "koppen_ashrae", "dq", "kt_mean_c20"]].assign(kt_mean_c20=lambda d: d["kt_mean_c20"].round(3)).rename(columns={"koppen_s1": "S1", "koppen_ashrae": "ASHRAE", "kt_mean_c20": "Kt medio (c20)"})), "",
          "Lectura (hechos): la lista de ASHRAE pone 13 estaciones en clases C con invierno seco o verano templado (Cwa 9, Csb 3, Cwb 1), clases que la Tabla S1 no tiene; de las 6 BWk de S1 solo San Juan sigue siendo BWk (Mendoza, San Rafael y Trelew pasan a BSk; Catamarca, Malargüe pasan a Cwa y Csb). "
          "El manuscrito (p. 5) dice que BWk y BSk dominan el Cuyo (Mendoza, San Juan, La Rioja, San Luis); en ASHRAE La Rioja y San Luis son Cwa. "
          "Las dos listas son clasificaciones distintas del mismo sitio; este análisis no establece cuál refleja mejor el clima real, solo que la de ASHRAE es la que usó la construcción de IWEC2 (según el handoff, sin verificar en el texto completo de Huang et al. 2014).", ""]
    # Kt medio por clase
    rows = []
    for nm, mp in (("S1", s1), ("ASHRAE", ash)):
        for k, g in c.assign(k=c["city"].map(mp)).groupby("k"):
            rows.append({"Fuente": nm, "Clase": k, "n": len(g), "Kt medio por estación, media (mín.–máx.)": f"{g['kt_mean_c20'].mean():.3f} ({g['kt_mean_c20'].min():.3f}–{g['kt_mean_c20'].max():.3f})"})
    L += ["## 2. Kt medio (c20) por clase, según cada fuente", "", md_table(pd.DataFrame(rows)), "",
          "La Tabla S1 enviada (p. 5) da un Kt típico por clase; esta tabla recalcula el Kt medio con los EPW (dominio c20).", ""]
    # tablas por clase
    for dom in ("c20", "all"):
        t = class_table(bm, ash, dom)
        t.to_csv(PROC / f"koppen_class_tables_ashrae_{dom}.csv", index=False)
        rows = []
        for k, g in t.groupby("klass"):
            r = {"Clase ASHRAE": k, "n estaciones": int(g["n_stations"].iloc[0])}
            for _, lab in MODELS:
                x = g[g["model"] == lab].iloc[0]
                r[lab] = f"{x.DNI_RMSE_mean:.1f}" + (f" [{x.lo:.1f}, {x.hi:.1f}]" if x.n_stations >= 3 else " (n ≤ 2)")
            rows.append(r)
        L += [f"## 3. DNI RMSE medio por clase de ASHRAE, dominio {dom} (W/m²; IC 95 % por bootstrap de estaciones solo con n ≥ 3)", "", md_table(pd.DataFrame(rows)), ""]
    L += ["## 4. Qué cambia y qué falta", "",
          "- Las tablas por clase (R1-6, R2-2) y la Fig. 10 pasan a las clases de ASHRAE; las salidas con S1 siguen en `RECALCULO_IWEC2_dirint_corregido.md` §8 y `figures/revised_c20/` (no se tocaron).",
          "- La descripción del manuscrito («ocho clases Köppen–Geiger», la tabla de clases de la p. 5, «Cuyo BWk/BSk», la advertencia sobre BWh, Cfb y ET) deja de coincidir; la redacción es del autor.",
          "- No se estableció cuál clasificación es climáticamente más fiel; si interesa, hay que comparar con un mapa Köppen–Geiger publicado (descarga aparte, sin hacer).",
          "- Las 2 estaciones de otra procedencia (Andalgalá, General Pico) conservan la clase de S1 y no entran en las tablas por clase.", ""]
    (REV / "response" / "KOPPEN_ashrae_vs_S1.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito response/KOPPEN_ashrae_vs_S1.md")


if __name__ == "__main__":
    main()
