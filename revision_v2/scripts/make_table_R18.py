"""make_table_R18.py — tabla para R1-8 (RMSE y nRMSE de verano austral frente a invierno austral), decisión del autor 5/10 (opción 1): DJF y JJA para las 39 IWEC2 y separadas por latitud (|lat| ≤ 40° y > 40°,
el corte del manuscrito, p. 26), DNI y DHI; la versión del dominio «all» va al suplemento.
Entradas: data/processed/reanalysis_seasonal_{c20,all}.csv (por estación y estación del año; DJF = dic–feb, JJA = jun–ago por mes del TMY, hemisferio sur) y benchmark_v2_per_station.csv (latitud).
Salidas : response/R1-8_tabla.md y data/processed/r18_seasonal_{c20,all}.csv. Medias entre estaciones (no estadísticos agrupados). Sin texto para el manuscrito. Provisorio.
Uso (desde revision_v2, venv activo): python scripts/make_table_R18.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
MODELS = [("reindl2", "Reindl-2"), ("erbs", "Erbs"), ("dirint_std_noTd", "DIRINT Perez"), ("dirint_v1", "DIRINT v1.0.0")]
HILAT = 40.0


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in df.columns) + " |")
    return "\n".join(out)


def main() -> None:
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    lat = bm.drop_duplicates("city").set_index("city")["lat"]
    L = ["# R1-8: error de verano contra invierno (provisorio)", "",
         "Generado por `scripts/make_table_R18.py` desde `data/processed/reanalysis_seasonal_{c20,all}.csv`. **Decisión del autor (5/10): DJF y JJA para las 39 IWEC2 y separadas por latitud (|lat| ≤ 40° y > 40°); «all» al suplemento.** "
         "DJF = dic–feb y JJA = jun–ago (por mes del TMY, hemisferio sur). Medias entre estaciones de RMSE, nRMSE (RMSE como % del DNI o DHI medio de referencia de la estación en esa estación del año) y MBE; referencia IWEC2 (cercanía a la partición Gompertz, no exactitud). "
         "Sin texto para el manuscrito.", ""]
    for dom in ("c20", "all"):
        s = pd.read_csv(PROC / f"reanalysis_seasonal_{dom}.csv")
        s = s[s["model"].isin([m for m, _ in MODELS]) & s["season"].isin(["DJF", "JJA"])].copy()
        s["lat"] = s["city"].map(lat)
        if s["lat"].isna().any():
            raise SystemExit("hay estaciones sin latitud en benchmark_v2_per_station.csv")
        s["grupo"] = ["latitud absoluta > 40°" if abs(x) > HILAT else "latitud absoluta ≤ 40°" for x in s["lat"]]
        groups = [("Las 39 IWEC2", s), ("latitud absoluta ≤ 40°", s[s["grupo"] == "latitud absoluta ≤ 40°"]), ("latitud absoluta > 40°", s[s["grupo"] == "latitud absoluta > 40°"])]
        hi_cities = ", ".join(sorted(s.loc[s["grupo"] == "latitud absoluta > 40°", "city"].unique()))
        csv = []
        for comp in ("DNI", "DHI"):
            rows = []
            for gname, g in groups:
                ns = g["city"].nunique()
                for m, lab in MODELS:
                    x = g[(g["model"] == m) & (g["comp"] == comp)]
                    r = {"Grupo": f"{gname} (n = {ns})", "Modelo": lab}
                    for met, col, f in (("RMSE", "RMSE", "{:.1f}"), ("nRMSE (%)", "nRMSE_pct", "{:.1f}"), ("MBE", "MBE", "{:+.1f}")):
                        a, b = x[x["season"] == "DJF"][col].mean(), x[x["season"] == "JJA"][col].mean()
                        r[f"{met} DJF / JJA"] = f"{f.format(a)} / {f.format(b)}"
                        csv.append(dict(domain=dom, comp=comp, group=gname, n_stations=ns, model=lab, metric=met, DJF=a, JJA=b, JJA_minus_DJF=b - a))
                    r["Horas por estación DJF / JJA"] = f"{x[x['season'] == 'DJF']['n'].mean():.0f} / {x[x['season'] == 'JJA']['n'].mean():.0f}"
                    rows.append(r)
            L += [f"## {comp}, dominio {dom} (W/m² salvo nRMSE)", "", md_table(pd.DataFrame(rows)), ""]
        pd.DataFrame(csv).to_csv(PROC / f"r18_seasonal_{dom}.csv", index=False)
        L += [f"Estaciones con |latitud| > 40° ({dom}): {hi_cities}.", ""]
    # lectura
    d = pd.read_csv(PROC / "r18_seasonal_c20.csv")
    def v(group, model, metric, comp="DNI"):
        r = d[(d["group"].str.startswith(group)) & (d["model"] == model) & (d["metric"] == metric) & (d["comp"] == comp)].iloc[0]
        return r["DJF"], r["JJA"]
    parts = []
    for lab in ("Reindl-2", "Erbs", "DIRINT Perez"):
        a1, b1 = v("latitud absoluta ≤", lab, "nRMSE (%)")
        a2, b2 = v("latitud absoluta >", lab, "nRMSE (%)")
        parts.append(f"{lab}: ≤ 40° {a1:.1f} → {b1:.1f} %, > 40° {a2:.1f} → {b2:.1f} %")
    L += ["## Lectura (hechos, DNI, c20)", "",
          "nRMSE de DNI de verano (DJF) a invierno (JJA): " + "; ".join(parts) + ". "
          "En las latitudes altas el nRMSE sube en invierno en los tres modelos; en el resto del país casi no cambia (en Erbs baja). Las 7 estaciones de latitud alta tienen además menos horas de invierno por estación. "
          "Con 7 estaciones es descriptivo, sin inferencia estadística.", "",
          "DHI (las 39, c20), DJF → JJA: " + "; ".join(f"{lab}: RMSE {v('Las 39', lab, 'RMSE', 'DHI')[0]:.1f} → {v('Las 39', lab, 'RMSE', 'DHI')[1]:.1f}, nRMSE {v('Las 39', lab, 'nRMSE (%)', 'DHI')[0]:.1f} → {v('Las 39', lab, 'nRMSE (%)', 'DHI')[1]:.1f} %" for lab in ("Reindl-2", "Erbs", "DIRINT Perez")) + ". "
          "En DHI el RMSE baja o casi no cambia en invierno, pero el nRMSE sube porque el DHI medio de invierno es menor; conviene leer ambas métricas.", "",
          "## Qué no está verificado", "",
          "- El efecto del ángulo solar no se aísla de otros factores de invierno (más nubosidad, menos horas): la tabla muestra dónde cambia el error, no por qué.",
          "- El corte de 40° es el del manuscrito (p. 26), no uno calculado; la agrupación en dos grupos es gruesa.", ""]
    (REV / "response" / "R1-8_tabla.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito response/R1-8_tabla.md")


if __name__ == "__main__":
    main()
