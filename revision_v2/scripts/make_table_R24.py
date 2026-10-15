"""make_table_R24.py — tabla para R2-4 (los 2 archivos que no son IWEC2: Andalgalá y General Pico), decisión del autor 5/10 (opción 1): análisis principal con las 39 IWEC2 y los 2 archivos aparte
(sensibilidad con y sin ellos, resultados individuales y su procedencia), marcados con † en las figuras por sitio.
Entradas: data/processed/benchmark_v2_per_station.csv, test0_followup.csv (identidad con Gompertz y cierre), epw_provenance.csv (procedencia declarada en el encabezado del EPW).
Salidas : response/R2-4_tabla.md y data/processed/r24_*.csv. Sin texto para el manuscrito. Provisorio (n = 2: indicativo, no concluyente).
Uso (desde revision_v2, venv activo): python scripts/make_table_R24.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
MODELS = [("reindl2", "Reindl-2"), ("erbs", "Erbs"), ("dirint_std_noTd", "DIRINT Perez"), ("dirint_v1", "DIRINT v1.0.0")]


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in df.columns) + " |")
    return "\n".join(out)


def main() -> None:
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    t0 = pd.read_csv(PROC / "test0_followup.csv").set_index("city")
    pv = pd.read_csv(PROC / "epw_provenance.csv").set_index("city")
    other = sorted(bm.loc[bm["source"] != "IWEC2", "city"].unique())
    if len(other) != 2:
        raise SystemExit(f"se esperaban 2 archivos no IWEC2, hay {len(other)}: {other}")
    L = ["# R2-4: los 2 archivos que no son IWEC2 (provisorio)", "",
         "Generado por `scripts/make_table_R24.py`. **Decisión del autor (5/10): análisis principal con las 39 IWEC2; Andalgalá y General Pico aparte, con tabla de sensibilidad, y marcados con † en las figuras por sitio.** "
         "Sin texto para el manuscrito. Con n = 2 los resultados individuales son indicativos, no concluyentes.", ""]
    # 1. procedencia e identidad con Gompertz
    rows = []
    for c in other:
        t, p = t0.loc[c], pv.loc[c]
        rows.append({"Archivo": c, "Procedencia (encabezado EPW)": str(p["comments_1"]).strip(), "Año de los datos": int(p["years_in_data"]),
                     "Horas dentro de ±2 W/m² de Gompertz": f"{100 * t['within2']:.1f} %", "R² contra Gompertz": f"{t['r2']:.2f}", "RMSE contra Gompertz (W/m²)": f"{t['rmse']:.0f}",
                     "Cierre GHI = DNI·cos z + DHI dentro de ±2 W/m²": f"{100 * t['closure_within2']:.1f} %"})
    L += ["## 1. Procedencia e identidad con la partición Gompertz (Test 0, `test0_followup.csv`)", "", md_table(pd.DataFrame(rows)), "",
          "Lectura (hechos): en las 39 IWEC2 el DNI es la partición Gompertz de la GHI del archivo (100 % de las horas dentro de ±10 W/m²) y el DHI sale por cierre. Andalgalá declara Meteonorm 7.x y General Pico TMY2: ninguno sigue a Gompertz; en General Pico, además, el cierre GHI = DNI·cos z + DHI casi no se cumple "
          "(las columnas de radiación extraterrestre vienen intercambiadas y se corrigen al leer). Son procedencias distintas entre sí y de IWEC2.", ""]
    # 2. sensibilidad 39 vs 41
    csv_rows = []
    for dom in ("c20", "all"):
        d = bm[bm["domain"] == dom]
        rows = []
        for m, lab in MODELS:
            x41, x39 = d[d["model"] == m], d[(d["model"] == m) & (d["source"] == "IWEC2")]
            r = {"Modelo": lab}
            for comp in ("DNI", "DHI"):
                for met, f in (("RMSE", "{:.1f}"), ("MBE", "{:+.1f}")):
                    a, b = x39[f"{comp}_{met}"].mean(), x41[f"{comp}_{met}"].mean()
                    r[f"{comp} {met} 39 → 41"] = f"{f.format(a)} → {f.format(b)} ({b - a:+.2f})"
                    csv_rows.append(dict(domain=dom, model=lab, comp=comp, metric=met, mean_39=a, mean_41=b, diff=b - a))
            rows.append(r)
        rk = lambda sel: d[sel].groupby("model")["DNI_RMSE"].mean().loc[[m for m, _ in MODELS]].rank().astype(int).tolist()
        L += [f"## 2. Sensibilidad: medias entre estaciones con las 39 IWEC2 y con las 41 (dominio {dom}; W/m²)", "", md_table(pd.DataFrame(rows)), "",
              f"Orden de los 4 modelos por DNI RMSE medio ({', '.join(l for _, l in MODELS)}): con 39 {rk(d['source'] == 'IWEC2')}; con 41 {rk(d['source'] != '__none__')}.", ""]
    pd.DataFrame(csv_rows).to_csv(PROC / "r24_sensitivity.csv", index=False)
    # 3. resultados individuales
    ind = []
    for dom in ("c20", "all"):
        d = bm[(bm["domain"] == dom) & bm["city"].isin(other)]
        rows = []
        for c in other:
            x = d[d["city"] == c].set_index("model")
            order = x.loc[[m for m, _ in MODELS], "DNI_RMSE"].rank().astype(int)
            for m, lab in MODELS:
                rows.append({"Archivo": c, "Modelo": lab, "DNI RMSE": f"{x.loc[m, 'DNI_RMSE']:.1f}", "DNI MBE": f"{x.loc[m, 'DNI_MBE']:+.1f}", "DNI R²": f"{x.loc[m, 'DNI_R2']:.3f}",
                             "Puesto (DNI RMSE)": int(order[m]), "DHI RMSE": f"{x.loc[m, 'DHI_RMSE']:.1f}", "DHI MBE": f"{x.loc[m, 'DHI_MBE']:+.1f}"})
                ind.append(dict(domain=dom, file=c, model=lab, DNI_RMSE=x.loc[m, "DNI_RMSE"], DNI_MBE=x.loc[m, "DNI_MBE"], DNI_R2=x.loc[m, "DNI_R2"], rank=int(order[m])))
        L += [f"## 3. Resultados de cada archivo (dominio {dom}; W/m²)", "", md_table(pd.DataFrame(rows)), ""]
    pd.DataFrame(ind).to_csv(PROC / "r24_individual.csv", index=False)
    d20 = bm[bm["domain"] == "c20"]
    names = dict(MODELS)
    def order(sel):
        g = d20[sel].groupby("model")["DNI_RMSE"].mean().loc[[m for m, _ in MODELS]].sort_values()
        return " < ".join(names[m] for m in g.index)
    ords = "; ".join(f"{c}: {order(d20['city'] == c)}" for c in other)
    L += [f"Orden de los 4 modelos por DNI RMSE (c20): {ords}. Con las 39 IWEC2 (media entre estaciones): {order(d20['source'] == 'IWEC2')}. "
          "Es coherente con que, contra IWEC2, Reindl-2 y Erbs ganan por parecerse a la partición Gompertz que generó la referencia; con otra procedencia el orden cambia. Son 2 archivos y las dos procedencias difieren entre sí: no es una prueba.", "",
          "## 4. Qué no está verificado", "",
          "- Cómo se generó el DNI/DHI de Andalgalá (Meteonorm 7.x) y de General Pico (TMY2): solo se conoce lo que dice el encabezado del EPW.",
          "- Con n = 2 no hay inferencia estadística; solo se muestra el contraste.",
          "- Hipótesis sin verificar: el DIRINT de Perez queda a 15.7 W/m² de DNI en General Pico (R² 0.997), mucho más cerca que en cualquier otro archivo; podría indicar que ese DNI se generó con un modelo parecido al DIRINT. Sin verificación, solo una observación.", ""]
    (REV / "response" / "R2-4_tabla.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito response/R2-4_tabla.md")


if __name__ == "__main__":
    main()
