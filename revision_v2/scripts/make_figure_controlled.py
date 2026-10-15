"""make_figure_controlled.py — Fig. 13 del manuscrito: experimento controlado del efecto de la referencia.

(a) Reindl-2 y (b) DIRINT: ΔRMSE = RMSE(modelo) − RMSE(Erbs) por estación y clase de Kt, contra el DNI medido (círculos) y contra Gompertz con la misma GHI (cuadrados).
Mendoza usa la difusa medida. (c) Dentro de IWEC2: ΔRMSE (Reindl-2 − Erbs) contra el % de horas con Kt ≥ 0.7 de cada estación; se superponen las estaciones medidas.
Lee data/processed/controlled_ref_effect*.csv (los genera controlled_reference_effect.py). Salida: figures/external_check/Fig13_controlled_reference_effect.png|pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures_v2 as mf  # noqa: E402  (fija el estilo)

PROC = mf.PROC
OUT = mf.REV / "figures" / "external_check"
WM2 = mf.WM2
NAMES = {"PTR": "Petrolina", "SMS": "São Martinho", "DRA": "Desert Rock", "GOLDEN": "Golden", "MENDOZA": "Mendoza (DHI)"}


def main() -> int:
    d = pd.concat([pd.read_csv(PROC / "controlled_ref_effect.csv"), pd.read_csv(PROC / "controlled_ref_effect_mendoza.csv")])
    cross = pd.read_csv(PROC / "controlled_ref_effect_cross_station.csv")
    rows = [(s, c) for s in NAMES for c in ("0.3-0.7", ">=0.7")]
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.08)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for ax, mod, title in ((axa, "reindl2", "(a) Reindl-2"), (axb, "dirint", "(b) DIRINT")):
        for i, (s, c) in enumerate(rows):
            y = len(rows) - 1 - i
            for ref, mk, mfc, dy in (("medido", "o", "black", 0.16), ("Gompertz", "s", "white", -0.16)):
                r = d[(d.estacion == s) & (d.clase == c) & (d.modelo == mod) & (d.ref == ref)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.d_vs_erbs, y + dy, xerr=[[r.d_vs_erbs - r.d_lo], [r.d_hi - r.d_vs_erbs]], fmt=mk, mfc=mfc, mec="black", color="black", ms=4.6, elinewidth=0.9, capsize=1.6, zorder=4)
        ax.axvline(0, color="black", lw=0.8)
        for k in range(1, len(NAMES)):
            ax.axhline(len(rows) - 1 - 2 * k + 0.5, color="0.8", lw=0.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([""] * len(rows))
        ax.xaxis.grid(True, color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.set_xlabel(f"RMSE(model) − RMSE(Erbs) ({WM2})")
    axa.set_yticks(range(len(rows)))
    axa.set_yticklabels([f"{NAMES[s]}, " + ("$K_t$ 0.3–0.7" if c == "0.3-0.7" else "$K_t$ ≥ 0.7") for s, c in rows][::-1], fontsize=6.6)
    axa.set_yticklabels([f"{NAMES[s]}, " + ("$K_t$ 0.3–0.7" if c == "0.3-0.7" else "$K_t$ ≥ 0.7") for s, c in rows][::-1], fontsize=6.6)
    axb.tick_params(labelleft=False)
    axa.set_xlim(-34, 82)
    axb.set_xlim(-34, 82)
    for ax in (axa, axb):
        ax.set_ylim(-0.6, len(rows) - 1 + 1.9)  # margen superior libre para la leyenda
    axb.legend(handles=[Line2D([], [], marker="o", ls="", mfc="black", mec="black", ms=4.6, label="against measurement"),
                        Line2D([], [], marker="s", ls="", mfc="white", mec="black", ms=4.6, label="against Gompertz reference")], loc="upper right", fontsize=6.4)
    # (c) experimento natural dentro de IWEC2 + estaciones medidas
    lr = stats.linregress(cross.share_kt07, cross.d_reindl_erbs)
    rho = stats.spearmanr(cross.share_kt07, cross.d_reindl_erbs)[0]
    axc.scatter(cross.share_kt07, cross.d_reindl_erbs, s=16, color="#009E73", edgecolors="black", linewidths=0.4, zorder=6, label="IWEC2 stations (39)")
    xs = np.array([cross.share_kt07.min(), cross.share_kt07.max()])  # solo el rango observado: no se extrapola
    axc.plot(xs, lr.intercept + lr.slope * xs, color="#009E73", lw=1.0, ls="--", zorder=2)
    for s in ("PTR", "SMS", "DRA", "GOLDEN"):
        for ref, mk, mfc in (("medido", "o", "black"), ("Gompertz", "s", "white")):
            r = d[(d.estacion == s) & (d.clase == "all") & (d.modelo == "reindl2") & (d.ref == ref)].iloc[0]
            sh = d[(d.estacion == s) & (d.clase == ">=0.7") & (d.modelo == "erbs") & (d.ref == ref)].pct.iloc[0]
            axc.scatter(sh, r.d_vs_erbs, marker=mk, s=26, facecolors=mfc, edgecolors="black", linewidths=0.8, zorder=4)
            if ref == "medido":
                axc.annotate(NAMES[s], (sh, r.d_vs_erbs), textcoords="offset points", xytext=(4, -9), fontsize=6.2)
    axc.axhline(0, color="black", lw=0.8)
    axc.set_xlim(0, 68)
    axc.set_xlabel("Hours with $K_t$ ≥ 0.7 (%)")
    axc.set_ylabel(f"RMSE(Reindl-2) − RMSE(Erbs) ({WM2})")
    axc.xaxis.grid(True, color="0.92", lw=0.5)
    axc.set_axisbelow(True)
    axc.set_title("(c)", loc="left", fontsize=8, fontweight="bold")
    axc.legend(handles=[Line2D([], [], marker="o", ls="", mfc="#009E73", mec="black", ms=4.6, label="IWEC2 stations (against IWEC2)"),
                        Line2D([], [], marker="o", ls="", mfc="black", mec="black", ms=4.6, label="measured stations (against measurement)"),
                        Line2D([], [], marker="s", ls="", mfc="white", mec="black", ms=4.6, label="measured stations (against Gompertz)")], loc="upper left", fontsize=6.4)
    mf.save(fig, OUT, "Fig13_controlled_reference_effect")
    print("ok", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
