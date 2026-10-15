"""make_figures_supp_r14.py — dos figuras de SUPLEMENTO para R1-4 (decisión del autor, 5/10: opción 2, figuras de suplemento):
  FigS1  sesgo de DNI (MBE) contra la masa de aire, por modelo, dominios c20 (AM ≲ 5) y all;
  FigS2  sesgo de DNI de la tabla v1.0.0 y del DIRINT de Perez (sin punto de rocío) en la grilla Kt′ × ΔKt′ (dominios c20 y all).
Mismo estilo que make_figures_v2 (sin títulos ni cajas, Okabe-Ito, 300 dpi PNG + PDF). Referencia = IWEC2 (cercanía a la partición Gompertz, no exactitud).
Entradas: data/processed/r14_am_bins_{dom}.csv y r14_ktp_dktp_{dom}.csv (scripts/dirint_diagnosis_r14.py). Salidas: figures/supplement_r14/, FIGURAS_datos_para_leyendas.md (hechos, no leyendas).
Uso (desde revision_v2, venv activo): python scripts/make_figures_supp_r14.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_figures_v2 as mf  # noqa: E402  (también fija el estilo)

REV, PROC, WM2 = mf.REV, mf.PROC, mf.WM2
OUT = REV / "figures" / "supplement_r14"
MODELS = [("dirint_v1", "SD 1.0.0 table", "#D55E00", "D"), ("dirint_std_noTd", "DIRINT", "#CC79A7", "^"), ("erbs", "Erbs", "#0072B2", "s"), ("reindl2", "Reindl-2", "#009E73", "o")]
MIN_N = 100          # celdas de la grilla con menos horas se atenúan


def edges(s: str) -> tuple[float, float]:
    a, b = re.findall(r"[-\d.e]+|inf", s.replace("−", "-"))[:2]
    return float(a), float(b)


def am_label(s: str) -> str:
    lo, hi = edges(s)
    if lo == 0:
        return f"<{hi:g}"
    return f">{lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}"


def dk_label(s: str) -> str:
    lo, hi = edges(s)
    if lo < 0 or lo == 0:
        return f"≤{hi:g}"
    return f">{lo:g}" if np.isinf(hi) else f"{lo:g}–{hi:g}"


def kp_label(s: str) -> str:
    lo, hi = edges(s)
    if lo == 0:
        return f"<{hi:g}"
    return f"≥{lo:g}" if hi > 1 else f"{lo:g}–{hi:g}"


def fig_s1() -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, gridspec_kw=dict(width_ratios=[4, 6]))
    facts = {}
    for ax, dom, letter in zip(axes, ("c20", "all"), "ab"):
        a = pd.read_csv(PROC / f"r14_am_bins_{dom}.csv")
        order = sorted(a["am_bin"].unique(), key=lambda s: edges(s)[0])
        x = np.arange(len(order))
        for m, lab, col, mk in MODELS:
            g = a[a["model"] == m].set_index("am_bin").loc[order]
            ax.plot(x, g["MBE"].to_numpy(), marker=mk, color=col, label=lab, ms=5, mec="black", mew=0.4, zorder=3)
        sh = a[a["model"] == "erbs"].set_index("am_bin").loc[order, "share_pct"]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{am_label(b)}\n{sh[b]:.0f} %" for b in order])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel("Relative air mass (share of hours below)")
        ax.set_title(f"({letter}) {'cos $z_\\theta$ ≥ 0.2' if dom == 'c20' else 'all hours'}", loc="left", fontsize=8.5, fontweight="bold")
        ax.yaxis.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
        facts[dom] = a
    axes[0].set_ylabel(f"DNI MBE against IWEC2 ({WM2})")
    handles = [Line2D([], [], marker=mk, color=col, mfc=col, mec="black", mew=0.4, ms=5, label=lab) for _, lab, col, mk in MODELS]
    fig.tight_layout(w_pad=0.8)
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), columnspacing=1.2, handletextpad=0.3)
    mf.save(fig, OUT, "FigS1_DNI_MBE_vs_air_mass")
    return facts


def fig_s2(dom: str) -> pd.DataFrame:
    j = pd.read_csv(PROC / f"r14_ktp_dktp_{dom}.csv")
    krows = sorted(j["ktp_bin"].unique(), key=lambda s: edges(s)[0])
    dcols = sorted(j["dktp_bin"].unique(), key=lambda s: edges(s)[1])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    norm = TwoSlopeNorm(vmin=-200, vcenter=0, vmax=200)
    im = None
    for ax, (m, lab) in zip(axes, (("dirint_v1", "SD 1.0.0 table"), ("dirint_std_noTd", "DIRINT"))):
        g = j[j["model"] == m].set_index(["ktp_bin", "dktp_bin"])
        mat = np.full((len(krows), len(dcols)), np.nan)
        nn = np.zeros_like(mat)
        for i, kb in enumerate(krows):
            for k, db in enumerate(dcols):
                if (kb, db) in g.index:
                    mat[i, k] = g.loc[(kb, db), "MBE"]
                    nn[i, k] = g.loc[(kb, db), "n"]
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, origin="lower", aspect="auto")
        for i in range(len(krows)):
            for k in range(len(dcols)):
                if np.isnan(mat[i, k]):
                    continue
                few = nn[i, k] < MIN_N
                ax.text(k, i, f"{mat[i, k]:+.0f}", ha="center", va="center", fontsize=7, color=("0.55" if few else ("white" if abs(mat[i, k]) > 130 else "black")), style="italic" if few else "normal")
        ax.set_xticks(range(len(dcols)))
        ax.set_xticklabels([dk_label(d) for d in dcols], fontsize=7)
        ax.set_xlabel("Hourly variability $\\Delta K_t'$")
        ax.set_yticks(range(len(krows)))
        ax.set_yticklabels([kp_label(k) for k in krows])
        ax.set_title(f"({'a' if m == 'dirint_v1' else 'b'}) {lab}", loc="left", fontsize=8.5, fontweight="bold")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Modified clearness index $K_t'$")
    fig.tight_layout(w_pad=0.8)
    cax = fig.add_axes([0.25, -0.06, 0.5, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"DNI MBE against IWEC2 ({WM2})")
    cb.outline.set_visible(False)
    mf.save(fig, OUT, f"FigS2_DNI_MBE_Ktp_x_dKtp_{dom}")
    return j


def main() -> int:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    fa = fig_s1()
    fb = {dom: fig_s2(dom) for dom in ("c20", "all")}
    L = ["# Datos para las leyendas de las figuras de suplemento de R1-4 — NO son leyendas", "",
         "Generado por `scripts/make_figures_supp_r14.py` desde `data/processed/r14_*.csv`. Son hechos de cada figura; las leyendas las redacta el autor. **Provisorio:** bins propuestos (P4), referencia IWEC2 (cercanía a la partición Gompertz, no exactitud). "
         "«DIRINT» = Perez 1992 sin punto de rocío (decisión del autor); «DIRINT v1.0.0» = tabla propia enviada. 39 estaciones IWEC2, horas agrupadas.", ""]
    for dom, a in fa.items():
        order = sorted(a["am_bin"].unique(), key=lambda s: edges(s)[0])
        rows = []
        for m, lab, _, _ in MODELS:
            g = a[a["model"] == m].set_index("am_bin").loc[order]
            rows.append(f"{lab}: " + ", ".join(f"{am_label(b)} {g.loc[b, 'MBE']:+.0f}" for b in order))
        L.append(f"- **Fig. S1 ({dom}):** DNI MBE (W/m²) por intervalo de masa de aire; porcentaje de horas por intervalo: " + ", ".join(f"{am_label(b)} {a[(a.model == 'erbs') & (a.am_bin == b)].share_pct.iloc[0]:.1f} %" for b in order) + ". " + "; ".join(rows) + ".")
    L.append("- **Fig. S1, nota:** con cos z ≥ 0.2 la masa de aire no supera ≈ 5; los intervalos de 5 en adelante solo existen en el dominio «all».")
    for dom, j in fb.items():
        for m, lab in (("dirint_v1", "SD 1.0.0 table"), ("dirint_std_noTd", "DIRINT")):
            g = j[j["model"] == m]
            top = g.loc[g["MBE"].idxmax()]
            low = g.loc[g["MBE"].idxmin()]
            few = int((g["n"] < MIN_N).sum())
            L.append(f"- **Fig. S2 ({dom}), {lab}:** MBE máximo {top.MBE:+.0f} W/m² en Kt′ {kp_label(top.ktp_bin)} × ΔKt′ {dk_label(top.dktp_bin)} ({int(top.n):,} h); mínimo {low.MBE:+.0f} W/m² en Kt′ {kp_label(low.ktp_bin)} × ΔKt′ {dk_label(low.dktp_bin)} ({int(low.n):,} h); {few} celdas con menos de {MIN_N} horas (en gris cursiva).")
    L += ["- **Fig. S2, nota:** ΔKt′ = |Kt′ᵢ − Kt′ᵢ₋₁| entre filas horarias diurnas consecutivas de la misma estación (diferencia hacia atrás; no es exactamente la de Perez, que promedia dos vecinos); la primera hora de cada día queda fuera. Escala de color común a los dos paneles (−200 a +200 W/m²); los valores fuera de ese rango quedan en el color extremo y el número anotado es el valor real.", ""]
    (OUT / "FIGURAS_datos_para_leyendas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito", OUT / "FIGURAS_datos_para_leyendas.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
