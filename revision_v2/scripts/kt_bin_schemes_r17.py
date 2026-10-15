"""kt_bin_schemes_r17.py — compara esquemas de intervalos de Kt para R1-7 (errores por intervalo de índice de claridad), insumo de la decisión P4 del autor.
Esquemas: (A) los 6 intervalos propuestos en el handoff [0, .2, .35, .5, .65, .75, ∞) (los de RECALCULO §2); (B) 3 clases [0, .3), [.3, .7), [.7, ∞); (C) intervalos uniformes de 0.1 (0–0.1 … ≥ 0.8).
Para cada esquema, dominio y modelo: horas (y %), RMSE y MBE de DNI y de DHI contra IWEC2 (39 estaciones, horas agrupadas). No elige nada: muestra qué cambia con el esquema. Provisorio.
Entrada: data/processed/hourly_v2.pkl.gz. Salida: data/processed/r17_kt_schemes_{c20,all}.csv; response/R1-7_esquemas_Kt.md.
Uso (desde revision_v2, venv activo): python scripts/kt_bin_schemes_r17.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
DOMS = {"c20": 0.2, "all": 0.0174}
SCHEMES = {"A (handoff, 6)": [0, 0.2, 0.35, 0.5, 0.65, 0.75, np.inf], "B (3 clases)": [0, 0.3, 0.7, np.inf], "C (0.1, 9)": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, np.inf]}
MODELS = [("reindl2", "Reindl-2"), ("erbs", "Erbs"), ("dirint_std_noTd", "DIRINT Perez"), ("dirint_v1", "DIRINT v1.0.0")]


def lab(lo: float, hi: float) -> str:
    return f"<{hi:g}" if lo == 0 else (f"≥{lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}")


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join(out)


def main() -> None:
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    h = h[h["source"] == "IWEC2"]
    rows = []
    for dom, c0 in DOMS.items():
        d = h[h["cosz"] >= c0]
        for sname, edges in SCHEMES.items():
            kb = pd.cut(d["kt"], edges, right=False)
            for b, g in d.groupby(kb, observed=True):
                for m, mlab in MODELS:
                    r = dict(domain=dom, scheme=sname, bin=lab(b.left, b.right), n=len(g), share_pct=100 * len(g) / len(d), model=mlab)
                    for comp in ("dni", "dhi"):
                        e = (g[f"{comp}_{m}"] - g[f"{comp}_ref"]).to_numpy()
                        e = e[np.isfinite(e)]
                        r[f"{comp.upper()}_RMSE"] = float(np.sqrt(np.mean(e ** 2)))
                        r[f"{comp.upper()}_MBE"] = float(e.mean())
                    rows.append(r)
        pd.DataFrame([r for r in rows if r["domain"] == dom]).to_csv(PROC / f"r17_kt_schemes_{dom}.csv", index=False)
    df = pd.DataFrame(rows)
    L = ["# R1-7: esquemas de intervalos de Kt (provisorio, insumo de la decisión P4)", "",
         "Generado por `scripts/kt_bin_schemes_r17.py` desde `data/processed/hourly_v2.pkl.gz` (39 IWEC2, horas agrupadas, referencia IWEC2). Muestra qué cambia con el esquema; **no elige uno**. "
         "A = los 6 intervalos del handoff (los de `RECALCULO_IWEC2_dirint_corregido.md` §2); B = 3 clases (Kt < 0.3, 0.3–0.7, ≥ 0.7); C = intervalos de 0.1.", ""]
    for dom in DOMS:
        for sname in SCHEMES:
            g = df[(df.domain == dom) & (df.scheme == sname)]
            order = list(dict.fromkeys(g["bin"]))
            rows_ = []
            for b in order:
                x = g[g.bin == b]
                r = {"Kt": b, "% de horas": f"{x.share_pct.iloc[0]:.1f}"}
                for _, mlab in MODELS:
                    y = x[x.model == mlab].iloc[0]
                    r[mlab] = f"{y.DNI_RMSE:.0f} ({y.DNI_MBE:+.0f})"
                rows_.append(r)
            L += [f"## DNI RMSE (MBE) en W/m², dominio {dom}, esquema {sname}", "", md_table(pd.DataFrame(rows_)), ""]
    (REV / "response" / "R1-7_esquemas_Kt.md").write_text("\n".join(L), encoding="utf-8")
    print("escrito response/R1-7_esquemas_Kt.md")
    g = df[(df.domain == "c20") & (df.scheme == "B (3 clases)")]
    print(g[["bin", "share_pct", "model", "DNI_RMSE", "DNI_MBE", "DHI_RMSE", "DHI_MBE"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
