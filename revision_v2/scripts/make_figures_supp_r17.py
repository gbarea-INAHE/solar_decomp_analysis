"""make_figures_supp_r17.py — figuras y tablas de R1-7 (errores por intervalo de Kt), decisión del autor 5/10: opción 2 = TRES clases en el texto (Kt < 0.3, 0.3–0.7, ≥ 0.7) y
intervalos de 0.1 en el material suplementario (tabla y figura).
  Tabla de texto  : TableR17_Kt_3classes_{c20,all}.md/.csv (DNI y DHI, RMSE y MBE; 4 filas de modelo: Reindl-2, Erbs, DIRINT de Perez, DIRINT v1.0.0).
  Suplemento      : TableS_Kt_0p1_{c20,all}.md/.csv y FigS3_Kt_0p1_{c20,all}.png/pdf (2 × 2: MBE y RMSE de DNI y de DHI contra Kt; tres modelos, sin la v1.0.0, que aplastaría la escala).
Referencia = IWEC2 (cercanía a la partición Gompertz, no exactitud). 39 estaciones IWEC2, horas agrupadas. Provisorio (umbrales 0.3 y 0.7: convención sin cita verificada).
Entrada: data/processed/r17_kt_schemes_{dom}.csv (scripts/kt_bin_schemes_r17.py). Salida: figures/supplement_r17/ (+ FIGURAS_datos_para_leyendas.md, hechos y no leyendas).
Uso (desde revision_v2, venv activo): python scripts/make_figures_supp_r17.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures_v2 as mf  # noqa: E402  (también fija el estilo)

REV, PROC, WM2 = mf.REV, mf.PROC, mf.WM2
OUT = REV / "figures" / "supplement_r17"
ALL4 = ["Reindl-2", "Erbs", "DIRINT Perez", "DIRINT v1.0.0"]
FIG3 = [("DIRINT Perez", "DIRINT", "#CC79A7", "^"), ("Erbs", "Erbs", "#0072B2", "s"), ("Reindl-2", "Reindl-2", "#009E73", "o")]
S3, S9 = "B (3 clases)", "C (0.1, 9)"


def md_table(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join(out)


def wide(df: pd.DataFrame, comp: str) -> pd.DataFrame:
    rows = []
    for b in list(dict.fromkeys(df["bin"])):
        x = df[df["bin"] == b]
        r = {"Kt": b, "n (horas)": f"{int(x.n.iloc[0]):,}", "% de horas": f"{x.share_pct.iloc[0]:.1f}"}
        for m in ALL4:
            y = x[x.model == m].iloc[0]
            r[m] = f"{y[comp + '_RMSE']:.1f} ({y[comp + '_MBE']:+.1f})"
        rows.append(r)
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame, dom: str) -> None:
    order = list(dict.fromkeys(df["bin"]))
    x = np.arange(len(order))
    sh = df[df.model == "Erbs"].set_index("bin").loc[order, "share_pct"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for (r, c), (comp, metric) in zip(((0, 0), (0, 1), (1, 0), (1, 1)), (("DNI", "MBE"), ("DNI", "RMSE"), ("DHI", "MBE"), ("DHI", "RMSE"))):
        ax = axes[r, c]
        for m, lab, col, mk in FIG3:
            g = df[df.model == m].set_index("bin").loc[order, f"{comp}_{metric}"]
            ax.plot(x, g.to_numpy(), marker=mk, color=col, ms=4.5, mec="black", mew=0.4, zorder=3)
        if metric == "MBE":
            ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel(f"{comp} {metric} ({WM2})")
        ax.yaxis.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"({'abcd'[r * 2 + c]}) {comp} {metric}", loc="left", fontsize=8.5, fontweight="bold")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b.replace('–', '–' + chr(10))}{'' if '–' in b else chr(10)}\n{sh[b]:.0f} %" for b in order], fontsize=6.8)
        ax.set_xlabel("Clearness index $K_t$ (share of hours below)")
    handles = [Line2D([], [], marker=mk, color=col, mfc=col, mec="black", mew=0.4, ms=5, label=lab) for _, lab, col, mk in FIG3]
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.3)
    mf.save(fig, OUT, f"FigS3_Kt_0p1_{dom}")


def main() -> int:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Datos para las leyendas de las figuras y tablas de R1-7 — NO son leyendas", "",
         "Generado por `scripts/make_figures_supp_r17.py` desde `data/processed/r17_kt_schemes_{c20,all}.csv`. Son hechos; las leyendas las redacta el autor. **Provisorio.** Referencia IWEC2 (cercanía a la partición Gompertz, no exactitud); 39 estaciones IWEC2, horas agrupadas. "
         "Decisión del autor (5/10): tres clases en el texto (Kt < 0.3, 0.3–0.7, ≥ 0.7) y intervalos de 0.1 en el suplemento. Los umbrales 0.3 y 0.7 son una convención sin cita verificada. «DIRINT» = Perez 1992 sin punto de rocío; «DIRINT v1.0.0» = tabla propia enviada.", ""]
    for dom in ("c20", "all"):
        d = pd.read_csv(PROC / f"r17_kt_schemes_{dom}.csv")
        d3, d9 = d[d["scheme"] == S3], d[d["scheme"] == S9]
        for comp in ("DNI", "DHI"):
            t = wide(d3, comp)
            t.to_csv(OUT / f"TableR17_Kt_3classes_{comp}_{dom}.csv", index=False)
        with open(OUT / f"TableR17_Kt_3classes_{dom}.md", "w", encoding="utf-8") as f:
            f.write(f"# R1-7, tabla de texto: error por clase de Kt (dominio {dom}; RMSE (MBE) en W/m²; provisorio)\n\n")
            for comp in ("DNI", "DHI"):
                f.write(f"## {comp}\n\n" + md_table(wide(d3, comp)) + "\n\n")
        for comp in ("DNI", "DHI"):
            wide(d9, comp).to_csv(OUT / f"TableS_Kt_0p1_{comp}_{dom}.csv", index=False)
        with open(OUT / f"TableS_Kt_0p1_{dom}.md", "w", encoding="utf-8") as f:
            f.write(f"# R1-7, material suplementario: error por intervalo de Kt de 0.1 (dominio {dom}; RMSE (MBE) en W/m²; provisorio)\n\n")
            for comp in ("DNI", "DHI"):
                f.write(f"## {comp}\n\n" + md_table(wide(d9, comp)) + "\n\n")
        figure(d9, dom)
        # hechos
        for comp in ("DNI", "DHI"):
            best = []
            for b in list(dict.fromkeys(d3["bin"])):
                x = d3[d3["bin"] == b]
                x = x[x.model != "DIRINT v1.0.0"]
                m = x.loc[x[f"{comp}_RMSE"].idxmin()]
                best.append(f"{b}: {m.model} ({m[comp + '_RMSE']:.1f} W/m²)")
            L.append(f"- **Tres clases, {comp}, {dom}:** modelo de menor RMSE entre Reindl-2, Erbs y DIRINT de Perez en cada clase: " + "; ".join(best) + ". Participación de horas: " + ", ".join(f"{b} {d3[(d3.bin == b) & (d3.model == 'Erbs')].share_pct.iloc[0]:.1f} %" for b in dict.fromkeys(d3['bin'])) + ".")
        top = d9[(d9.model == "Reindl-2") & (d9.bin == list(dict.fromkeys(d9['bin']))[-1])].iloc[0]
        L.append(f"- **Intervalo Kt ≥ 0.8 ({dom}):** {top.share_pct:.1f} % de las horas ({int(top.n):,} h); Reindl-2: DNI RMSE {top.DNI_RMSE:.0f} W/m², MBE {top.DNI_MBE:+.0f}.")
    L += ["- **Figura S3:** 2 × 2 (DNI y DHI; MBE y RMSE) contra Kt en intervalos de 0.1, tres modelos (Reindl-2, Erbs, DIRINT de Perez); la v1.0.0 no se dibuja (su RMSE supera 100 W/m² en buena parte del rango y aplasta la escala) y está en las tablas. El porcentaje bajo cada intervalo es la parte de las horas que cae en él.", ""]
    (OUT / "FIGURAS_datos_para_leyendas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito", OUT / "FIGURAS_datos_para_leyendas.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
