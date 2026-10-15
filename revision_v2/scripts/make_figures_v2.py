"""make_figures_v2.py — the 10 figures of the manuscript redone for the revision (Editor comment 5, 2026-10-05).

Rules applied (Editor + author defaults): no figure titles (captions are written by the author), no boxes around plots or legends, readable fonts and
marker/line sizes at reduced size, 300 dpi PNG + vector PDF, English labels, Okabe-Ito palette (colour-blind safe), one palette for the three models.

Content: same ten figures as the submitted paper, recomputed from the revised analysis (not from the v1.0.0 pipeline):
  * hourly series of benchmark_v2.py (geometry G2: cos z and ETR of the file itself), domain c20 (cos z >= 0.2; author decision 3) by default;
  * models: DIRINT = Perez 1992 WITHOUT dew point (author decision 5/10), Erbs, Reindl-2 (the v1.0.0 DIRINT table is not drawn);
  * reference = IWEC2 DNI/DHI, which is a Gompertz split of the file's own GHI: the figures show closeness to that reference, not accuracy;
  * per-site displays (Figs 1, 4, 5, 8) show all 41 files (the 2 non-IWEC2 files are flagged with a dagger); statistical summaries
    (Figs 2, 3, 6, 7, 9, 10) use the 39 IWEC2 files, as the revised tables do.
Station numbers = row order of the submitted Table S1 (north to south); data/… station_index.csv lists them.
Per-station metrics are recomputed from the hourly series and ASSERTED equal to benchmark_v2_per_station.csv.

Usage (from revision_v2, venv active):  python scripts/make_figures_v2.py [--domain c20|all]
Outputs: figures/revised_<domain>/FigN_*.png|pdf, station_index.csv, FIGURAS_datos_para_leyendas.md
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REV = HERE.parent
sys.path.insert(0, str(HERE))
from benchmark_v2 import DOMAINS, metr  # noqa: E402

PROC = REV / "data" / "processed"
S1 = REV / "v1_enviada_completa" / "03_tablas_y_datos" / "TableS1_sites.csv"
MODELS = ["dirint_std_noTd", "erbs", "reindl2"]
LAB = {"dirint_std_noTd": "DIRINT", "erbs": "Erbs", "reindl2": "Reindl-2"}
COL = {"dirint_std_noTd": "#CC79A7", "erbs": "#0072B2", "reindl2": "#009E73"}        # Okabe-Ito
MRK = {"dirint_std_noTd": "^", "erbs": "s", "reindl2": "o"}
# Clases de la lista de ASHRAE de IWEC2 (decisión D7 del autor, 5/10; opción --koppen ashrae): orden B, C, E; paleta Okabe-Ito
KOP_ASH = {"BSk": "#F0E442", "BWk": "#000000", "Cfa": "#009E73", "Cwa": "#D55E00", "Cwb": "#CC79A7", "Csb": "#56B4E9", "ET": "#0072B2"}
KOP = {"Aw": "#D55E00", "BSh": "#E69F00", "BSk": "#F0E442", "BWh": "#CC79A7", "BWk": "#000000", "Cfa": "#009E73", "Cfb": "#56B4E9", "ET": "#0072B2"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WM2 = r"W m$^{-2}$"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3, "lines.markersize": 4.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.grid": False,
})


def norm(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().upper())


def nice(city: str) -> str:
    return city.replace("-AERO", "").replace("-", " ").title().replace("Bs As", "Buenos Aires").replace("Airp.", "").strip()


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  ", name)


def clean_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------------------------------------------------------------------ data
def load(domain: str):
    c0 = DOMAINS[domain]
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    h = h[h["cosz"] >= c0].copy()
    s1 = pd.read_csv(S1, encoding="utf-8")
    # the submitted Table S1 CSV stores 'Andalgalá' as the mojibake 'Andalgalã¡': repair only names that show that pattern
    fixname = lambda s: s.replace("ã¡", "á").replace("Ã¡", "Á") if ("ã¡" in s or "Ã¡" in s) else s
    nb = {norm(c): c for c in h["city"].unique()}
    rows = []
    for i, r in s1.iterrows():
        hit = [c for k, c in nb.items() if k.startswith(norm(r.City)) or norm(r.City).startswith(k)]
        assert len(hit) == 1, f"no se pudo emparejar {r.City}: {hit}"
        rows.append(dict(no=i + 1, city=hit[0], name=fixname(r.City), wmo=r.WMO, source="IWEC2" if r.Source == "IWEC2" else "other", koppen=r.Koppen,
                         lat=-float(r.Lat_S), lon=-float(r.Lon_W)))
    sites = pd.DataFrame(rows)
    assert len(sites) == 41 and sites["city"].is_unique and (sites["source"] == "IWEC2").sum() == 39
    sites["label"] = sites["no"].astype(str) + np.where(sites["source"] == "other", "†", "")
    # per-station metrics recomputed from the hourly series (and checked against the benchmark table)
    recs = []
    for city, g in h.groupby("city"):
        for m in MODELS:
            r = metr(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())
            recs.append(dict(city=city, model=m, RMSE=r["RMSE"], MBE=r["MBE"], R2=r["R2"], nRMSE=r["nRMSE"], n=r["n"]))
    st = pd.DataFrame(recs).merge(sites, on="city")
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    bm = bm[bm["domain"] == domain].rename(columns={"DNI_RMSE": "b_RMSE", "DNI_MBE": "b_MBE", "DNI_R2": "b_R2", "DNI_nRMSE": "b_nRMSE", "n": "b_n"})
    chk = st.merge(bm[["city", "model", "b_RMSE", "b_MBE", "b_R2", "b_nRMSE", "b_n"]], on=["city", "model"], validate="one_to_one")
    assert len(chk) == len(st) == 41 * len(MODELS)
    for a, b in (("RMSE", "b_RMSE"), ("MBE", "b_MBE"), ("R2", "b_R2"), ("nRMSE", "b_nRMSE"), ("n", "b_n")):
        assert np.allclose(chk[a], chk[b], rtol=0, atol=1e-8), f"las métricas por estación difieren del benchmark ({a})"
    print(f"[OK] métricas por estación recalculadas == benchmark_v2_per_station.csv (dominio {domain}, 41 sitios x {len(MODELS)} modelos)")
    return h, sites, st


# ------------------------------------------------------------------------------------------------------------------ figures
def place_labels(fig, ax, best, fs=6.5):
    """Greedy label placement: for each site pick, among 8 offsets, the one that overlaps least with markers and with labels already placed."""
    fig.canvas.draw()
    k = fig.dpi / 72.0
    xy = ax.transData.transform(best[["lon", "lat"]].to_numpy())
    h_pt = fs * 1.15
    markers = [(x - 5 * k, y - 5 * k, x + 5 * k, y + 5 * k) for x, y in xy]
    placed = []

    def area(a, b):
        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])
        return dx * dy if dx > 0 and dy > 0 else 0.0

    # crowded sites first
    dist = np.array([[np.hypot(*(p - q)) for q in xy] for p in xy])
    order = np.argsort((dist < 25 * k).sum(axis=1))[::-1]
    for i in order:
        label = best.iloc[i]["label"]
        w = 0.62 * fs * len(label)
        cands = [(5, 3), (-5 - w, 3), (5, -h_pt - 3), (-5 - w, -h_pt - 3), (-w / 2, 7), (-w / 2, -h_pt - 7), (8, -h_pt / 2), (-8 - w, -h_pt / 2)]
        best_c, best_s = None, None
        for dx, dy in cands:
            rect = (xy[i, 0] + dx * k, xy[i, 1] + dy * k, xy[i, 0] + (dx + w) * k, xy[i, 1] + (dy + h_pt) * k)
            score = sum(area(rect, m) for m in markers) + 2 * sum(area(rect, p) for p in placed)
            if best_s is None or score < best_s - 1e-9:
                best_c, best_s, best_rect = (dx, dy), score, rect
        placed.append(best_rect)
        ax.annotate(label, (best.iloc[i]["lon"], best.iloc[i]["lat"]), xytext=best_c, textcoords="offset points", fontsize=fs, color="black", ha="left", va="bottom")


def draw_outline(ax):
    """Country outlines from Natural Earth 1:10m (public domain; Argentine point-of-view version, which includes the Malvinas), drawn below the data."""
    import geo_outline as go
    c = go.load_countries({"Argentina", "Chile", "Bolivia", "Paraguay", "Brazil", "Uruguay"})
    for name, rings in c.items():
        for r in rings:
            if r[:, 0].max() < -76 or r[:, 0].min() > -50 or r[:, 1].max() < -58 or r[:, 1].min() > -19:
                continue                                              # ring outside the map window (far islands, etc.)
            if name == "Argentina":
                ax.fill(r[:, 0], r[:, 1], fc="0.95", ec="0.3", lw=0.7, zorder=1)
            else:
                ax.plot(r[:, 0], r[:, 1], color="0.7", lw=0.5, zorder=1)


def fig1_map(sites, st, out):
    best = st.loc[st.groupby("city")["RMSE"].idxmin()].set_index("city")
    lo, hi = np.floor(best["R2"].min() * 100) / 100, np.ceil(best["R2"].max() * 100) / 100
    fig, ax = plt.subplots(figsize=(4.3, 6.4))
    cmap = plt.get_cmap("cividis")
    nrm = Normalize(lo, hi)
    for m in MODELS:
        sel = best[best["model"] == m]
        if sel.empty:
            continue
        ax.scatter(sel["lon"], sel["lat"], c=sel["R2"], cmap=cmap, norm=nrm, marker=MRK[m], s=42, edgecolors="black", linewidths=0.6, zorder=4)
    ax.set_xlim(-74.5, -52)
    ax.set_ylim(-56, -21)
    ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, color="0.9", lw=0.4)
    ax.set_axisbelow(True)
    draw_outline(ax)
    handles = [Line2D([], [], marker=MRK[m], ls="", mfc="0.75", mec="black", mew=0.6, ms=6, label=f"{LAB[m]} (n = {int((best['model'] == m).sum())})") for m in MODELS if (best["model"] == m).any()]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, borderaxespad=0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=nrm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, aspect=18)
    cb.set_label("$R^2$ (DNI), best model")
    cb.outline.set_visible(False)
    place_labels(fig, ax, best)
    save(fig, out, "Fig1_map_best_model")
    return best


def fig23_scatter(h, var, out, name):
    iw = h[h["source"] == "IWEC2"]
    ref = f"{var.lower()}_ref"
    lim = (0, 1050) if var == "DNI" else (0, 550)
    n = len(MODELS)
    nr, nc = (1, n) if n <= 3 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=(7.2, 2.75) if n <= 3 else (5.6, 5.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    stats = {}
    for ia, (ax, m, letter) in enumerate(zip(axes, MODELS, "abcd")):
        o, p = iw[ref].to_numpy(), iw[f"{var.lower()}_{m}"].to_numpy()
        ok = np.isfinite(o) & np.isfinite(p) & (o > 0)
        hb = ax.hexbin(o[ok], p[ok], gridsize=55, cmap="Blues", norm=LogNorm(vmin=1, vmax=None), mincnt=1, extent=(*lim, *lim), linewidths=0)
        ax.plot(lim, lim, color="black", lw=0.8, ls="--")
        r = metr(o[ok], p[ok])
        stats[m] = r
        ax.text(0.04, 0.96, f"({letter}) {LAB[m]}", transform=ax.transAxes, va="top", fontsize=8.5, fontweight="bold")
        ax.text(0.04, 0.855, f"$R^2$ = {r['R2']:.3f}\nRMSE = {r['RMSE']:.0f} {WM2}\nMBE = {r['MBE']:+.0f}".replace("= -", "= −") + f" {WM2}", transform=ax.transAxes, va="top", fontsize=7, linespacing=1.3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        if ia >= n - nc:
            ax.set_xlabel(f"{var} reference ({WM2})")
    for ax in axes[::nc]:
        ax.set_ylabel(f"{var} modelled ({WM2})")
    fig.tight_layout(w_pad=0.6)
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.015, aspect=22)
    cb.set_label("Hours (log scale)")
    cb.outline.set_visible(False)
    save(fig, out, name)
    return stats


def fig4_heatmap(sites, st, out):
    p = st.pivot(index="label", columns="model", values="R2")
    order = st[st["model"] == "reindl2"].sort_values("R2")["label"].tolist()
    p = p.loc[order, MODELS]
    names = sites.set_index("label")["name"].to_dict()
    vmin, vmax = np.floor(np.nanmin(p.values) * 100) / 100, np.ceil(np.nanmax(p.values) * 100) / 100
    n = len(MODELS)
    fig, ax = plt.subplots(figsize=(3.5 if n <= 3 else 4.5, 8.6))
    im = ax.imshow(p.values, aspect="auto", cmap="cividis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels([LAB[m] for m in MODELS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(p)))
    ax.set_yticklabels([f"{names[l]} ({l})" for l in p.index], fontsize=7)
    mid = (vmin + vmax) / 2
    for i in range(p.shape[0]):
        for j in range(n):
            v = p.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.8, color="white" if v < mid else "black", fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.7, aspect=25)
    cb.set_label("$R^2$ (DNI)")
    cb.outline.set_visible(False)
    save(fig, out, "Fig4_heatmap_R2_DNI")


def fig5_bars(sites, st, out):
    p = st.pivot(index="label", columns="model", values="nRMSE")
    order = st[st["model"] == "reindl2"].sort_values("nRMSE", ascending=False)["label"].tolist()
    p = p.loc[order, MODELS]
    names = sites.set_index("label")["name"].to_dict()
    n = len(MODELS)
    x, w = np.arange(len(p)), 0.81 / n
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - (n - 1) / 2) * w, p[m], w, color=COL[m], label=LAB[m], linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{names[l]} ({l})" for l in p.index], rotation=60, ha="right", fontsize=6.8)
    ax.set_ylabel("DNI nRMSE (%)")
    ax.set_xlim(-0.7, len(p) - 0.3)
    ax.yaxis.grid(True, color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=n, loc="upper right")
    save(fig, out, "Fig5_nRMSE_by_site")


def fig6_mbe(st, out):
    iw = st[st["source"] == "IWEC2"]
    n = len(MODELS)
    fig, ax = plt.subplots(figsize=(3.4 if n <= 3 else 4.3, 3.0))
    data = [iw[iw["model"] == m]["MBE"].to_numpy() for m in MODELS]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=0.9),
                    capprops=dict(lw=0.9), boxprops=dict(lw=0.9))
    rng = np.random.default_rng(20261005)
    for i, (patch, m) in enumerate(zip(bp["boxes"], MODELS)):
        patch.set_facecolor(COL[m])
        patch.set_alpha(0.55)
        ax.scatter(i + 1 + rng.uniform(-0.16, 0.16, len(data[i])), data[i], s=7, color=COL[m], edgecolors="black", linewidths=0.3, zorder=4)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([LAB[m] for m in MODELS], fontsize=7 if n > 3 else 7.5)
    ax.set_ylabel(f"DNI MBE ({WM2})")
    ax.yaxis.grid(True, color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, out, "Fig6_MBE_boxplot")


def fig7_taylor(h, out):
    """Taylor diagram in Cartesian form (x = sigma*R, y = sigma*sqrt(1-R^2)), zoomed on the region that holds the points."""
    iw = h[h["source"] == "IWEC2"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3) if len(MODELS) <= 3 else (5.0, 4.3))
    pts, xy_all = {}, {}
    for m in MODELS:
        xs, ys, Rs, ss = [], [], [], []
        for city, g in iw.groupby("city"):
            o, p = g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy()
            ok = np.isfinite(o) & np.isfinite(p)
            o, p = o[ok], p[ok]
            R = float(np.corrcoef(o, p)[0, 1])
            sg = float(p.std() / o.std())
            xs.append(sg * R)
            ys.append(sg * np.sqrt(max(0.0, 1 - R * R)))
            Rs.append(R)
            ss.append(sg)
        pts[m] = (np.array(Rs), np.array(ss))
        xy_all[m] = (xs, ys)
    if len(MODELS) <= 3:
        xlim, ylim = (0.72, 1.28), (0.0, 0.62)
    else:                                                                    # wider window when a model with lower correlation is drawn
        ax_x = np.concatenate([xy_all[m][0] for m in MODELS])
        ax_y = np.concatenate([xy_all[m][1] for m in MODELS])
        xlim = (np.floor((ax_x.min() - 0.05) * 20) / 20, np.ceil((ax_x.max() + 0.05) * 20) / 20)
        ylim = (0.0, np.ceil((ax_y.max() + 0.06) * 20) / 20)
    for m in MODELS:
        ax.scatter(*xy_all[m], c=COL[m], marker=MRK[m], s=24, edgecolors="black", linewidths=0.3, alpha=0.9, zorder=4)
    ax.scatter([1.0], [0.0], marker="*", s=90, c="black", zorder=5, clip_on=False)
    th = np.linspace(0, np.pi / 2, 400)
    wide = len(MODELS) > 3
    for sg in ((0.6, 0.8, 1.0, 1.2, 1.4) if wide else (0.8, 0.9, 1.0, 1.1, 1.2)):      # standard-deviation arcs
        ax.plot(sg * np.cos(th), sg * np.sin(th), color="0.8", lw=0.6, zorder=1)
        if xlim[0] < sg < xlim[1] - 0.01:                                      # label only arcs that cross the window
            ax.text(sg + 0.004, 0.012, f"{sg:g}", fontsize=6.8, color="0.35", ha="left", va="bottom")
    for R in ((0.6, 0.8, 0.9, 0.95, 0.99) if wide else (0.9, 0.95, 0.97, 0.98, 0.99)):  # correlation rays
        a = np.arccos(R)
        ax.plot([0, 1.8 * np.cos(a)], [0, 1.8 * np.sin(a)], color="0.8", lw=0.6, zorder=1)
        if wide:                                                              # label only rays that cross the window; place the label near where they leave it
            r_in, r_out = xlim[0] / np.cos(a), min(xlim[1] / np.cos(a), ylim[1] / np.sin(a))
            if r_in >= r_out * 0.97:
                continue
            rr = r_out * 0.94
        else:
            rr = 1.215
        ax.text(rr * np.cos(a), rr * np.sin(a), f"{R:g}", fontsize=6.8, color="0.25", ha="left", va="bottom", rotation=np.degrees(a), rotation_mode="anchor")
    t = np.linspace(0, np.pi, 400)
    for rms in ((0.2, 0.4, 0.6, 0.8) if wide else (0.1, 0.2, 0.3, 0.4)):    # centred normalised RMS difference around the reference
        x, y = 1 + rms * np.cos(t), rms * np.sin(t)
        keep = (y >= 0) & (x >= xlim[0]) & (x <= xlim[1]) & (y <= ylim[1])
        ax.plot(x[keep], y[keep], color="0.45", lw=0.7, ls="--", zorder=2)
        a25 = np.radians(25)
        if 1 + rms * np.cos(a25) < xlim[1] - 0.02 and rms * np.sin(a25) < ylim[1]:
            ax.text(1 + rms * np.cos(a25) + 0.006, rms * np.sin(a25), f"{rms:g}", fontsize=6.8, color="0.3", ha="left", va="center")
        elif rms + 0.012 < ylim[1] - 0.02 and xlim[0] < 1.0 < xlim[1]:          # arc too far right for its label: put it on top of the arc
            ax.text(1.0, rms + 0.012, f"{rms:g}", fontsize=6.8, color="0.3", ha="center", va="bottom")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("Normalised standard deviation")
    ax.set_ylabel("Normalised standard deviation")
    ax.text(0.55, 0.985, "Correlation", transform=ax.transAxes, fontsize=8, ha="center", va="bottom")
    handles = [Line2D([], [], marker=MRK[m], ls="", mfc=COL[m], mec="black", mew=0.3, ms=5.5, label=LAB[m]) for m in MODELS]
    handles.append(Line2D([], [], marker="*", ls="", mfc="black", mec="black", ms=9, label="Reference (IWEC2)"))
    handles.append(Line2D([], [], color="0.45", lw=0.9, ls="--", label="Centred RMS difference"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), fontsize=7.5, columnspacing=1.1, handletextpad=0.4)
    save(fig, out, "Fig7_Taylor_DNI")
    return pts, (xlim, ylim)


def fig8_monthly(h, sites, st, out):
    best = st.loc[st.groupby("city")["RMSE"].idxmin()].set_index("city")["model"]
    rows = {}
    for city, g in h.groupby("city"):
        m = best[city]
        r = []
        for mo in range(1, 13):
            gg = g[g["month"] == mo]
            e = (gg[f"dni_{m}"] - gg["dni_ref"]).dropna()
            r.append(float(e.mean()) if len(e) >= 30 else np.nan)
        rows[city] = r
    sites = sites.copy()
    mat = np.array([rows[c] for c in sites["city"]])
    vmax = np.nanmax(np.abs(mat))
    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTHS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([f"{r['name']} ({r['label']})" for _, r in sites.iterrows()], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(12):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, str(int(round(v))).replace("-", "−"), ha="center", va="center", fontsize=6.3, color="white" if abs(v) > 0.6 * vmax else "black")
    letters = {"dirint_std_noTd": "D", "erbs": "E", "reindl2": "R"}
    for i, c in enumerate(sites["city"]):
        ax.text(11.75, i, letters[best[c]], ha="left", va="center", fontsize=6.8, color="0.25")
    ax.text(11.75, -1.2, "Best", ha="left", va="center", fontsize=6.8, color="0.25")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02, shrink=0.55, aspect=30)
    cb.set_label(f"DNI MBE of the best model per site ({WM2})")
    cb.outline.set_visible(False)
    save(fig, out, "Fig8_MBE_monthly")
    return best, mat


def fig9_kt(h, st, out):
    iw = st[st["source"] == "IWEC2"]
    ktm = h[h["source"] == "IWEC2"].groupby("city")["kt"].mean().rename("kt_mean")
    n = len(MODELS)
    nr, nc = (1, n) if n <= 3 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=(7.2, 2.9) if n <= 3 else (6.0, 5.4), sharey=True, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    slopes = {}
    for ia, (ax, m, letter) in enumerate(zip(axes, MODELS, "abcd")):
        d = iw[iw["model"] == m].merge(ktm, left_on="city", right_index=True)
        for k, g in d.groupby("koppen"):
            ax.scatter(g["kt_mean"], g["R2"], c=KOP[k], s=24, edgecolors="black", linewidths=0.4, zorder=4)
        z = np.polyfit(d["kt_mean"], d["R2"], 1)
        xp = np.linspace(d["kt_mean"].min(), d["kt_mean"].max(), 50)
        ax.plot(xp, np.polyval(z, xp), color="black", lw=1.0, ls="--", zorder=3)
        rho = np.corrcoef(d["kt_mean"], d["R2"])[0, 1]
        slopes[m] = (float(z[0]), float(rho))
        ax.text(0.04, 0.05, f"slope = {z[0]:+.2f}", transform=ax.transAxes, fontsize=7.5)
        ax.set_title(f"({letter}) {LAB[m]}", loc="left", fontsize=8.5, fontweight="bold")
        low = d.nsmallest(3, "R2")
        for _, r in low.iterrows():
            ax.annotate(r["label"], (r["kt_mean"], r["R2"]), xytext=(3, -7), textcoords="offset points", fontsize=6.5)
        if ia >= n - nc:
            ax.set_xlabel("Mean clearness index $K_t$ (–)")
        ax.yaxis.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[::nc]:
        ax.set_ylabel("$R^2$ (DNI)")
    handles = [Line2D([], [], marker="o", ls="", mfc=KOP[k], mec="black", mew=0.4, ms=5.5, label=f"{k} ({int((iw[iw['model'] == 'erbs']['koppen'] == k).sum())})") for k in KOP]
    fig.tight_layout(w_pad=0.6)
    fig.legend(handles=handles, loc="lower center", ncol=8 if n <= 3 else 4, bbox_to_anchor=(0.5, -0.06 if n <= 3 else -0.1), fontsize=7.5, columnspacing=0.9, handletextpad=0.2)
    save(fig, out, "Fig9_R2_vs_Kt")
    return slopes


def fig10_koppen(st, out):
    iw = st[st["source"] == "IWEC2"]
    classes = [k for k in KOP if k in set(iw["koppen"])]
    n = len(MODELS)
    nr, nc = (1, n) if n <= 3 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=(7.2, 2.9) if n <= 3 else (6.0, 5.4), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    counts = {k: int((iw[iw["model"] == "erbs"]["koppen"] == k).sum()) for k in classes}
    for ax, m, letter in zip(axes, MODELS, "abcd"):
        d = iw[iw["model"] == m]
        rng = np.random.default_rng(20261005)
        for i, k in enumerate(classes, start=1):
            v = d[d["koppen"] == k]["nRMSE"].to_numpy()
            if len(v) >= 3:
                bp = ax.boxplot([v], positions=[i], widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.4),
                                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8), boxprops=dict(lw=0.8))
                bp["boxes"][0].set_facecolor(KOP[k])
                bp["boxes"][0].set_alpha(0.35)
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=11, color=KOP[k], edgecolors="black", linewidths=0.35, zorder=4)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([f"{k}\n({counts[k]})" for k in classes], fontsize=7)
        ax.set_title(f"({letter}) {LAB[m]}", loc="left", fontsize=8.5, fontweight="bold")
        ax.yaxis.grid(True, color="0.9", lw=0.5)
        ax.set_axisbelow(True)
    for ax in axes[::nc]:
        ax.set_ylabel("DNI nRMSE (%)")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "Fig10_nRMSE_by_Koppen")
    return counts


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="c20")
    ap.add_argument("--koppen", choices=["s1", "ashrae"], default="s1", help="clases Köppen de las Figs. 9 y 10: Tabla S1 enviada (defecto) o lista de ASHRAE de IWEC2 (data/processed/koppen_ashrae_vs_s1.csv; solo las 39 IWEC2)")
    ap.add_argument("--with-v1", action="store_true", help="also draw the DIRINT of v1.0.0 (own table) as a fourth model; Figs 1 and 8 are not redrawn (they do not change)")
    a = ap.parse_args()
    if a.with_v1:
        MODELS[:] = ["dirint_std_noTd", "dirint_v1", "erbs", "reindl2"]
        LAB["dirint_v1"], COL["dirint_v1"], MRK["dirint_v1"] = "SD 1.0.0 table", "#D55E00", "D"
    out = REV / "figures" / ((f"revised_{a.domain}_with_v1" if a.with_v1 else f"revised_{a.domain}") + ("_ashrae" if a.koppen == "ashrae" else ""))
    out.mkdir(parents=True, exist_ok=True)
    h, sites, st = load(a.domain)
    if a.koppen == "ashrae":
        ca = pd.read_csv(PROC / "koppen_ashrae_vs_s1.csv").set_index("city")["koppen_ashrae"]
        iw_cities = set(sites.loc[sites["source"] == "IWEC2", "city"])
        assert set(ca.index) == iw_cities, "las 39 estaciones IWEC2 de koppen_ashrae_vs_s1.csv no coinciden con las del benchmark"
        assert set(ca) <= set(KOP_ASH), f"clases de ASHRAE sin color definido: {set(ca) - set(KOP_ASH)}"
        sites["koppen_s1"] = sites["koppen"]
        sites["koppen"] = [ca.get(c, k) for c, k in zip(sites["city"], sites["koppen_s1"])]
        st["koppen"] = st["city"].map(sites.set_index("city")["koppen"])
        KOP.clear()
        KOP.update(KOP_ASH)                       # las Figs. 9 y 10 usan solo las 39 IWEC2; las 2 de otra procedencia (clase de S1) no se dibujan por clase
    sites[["no", "label", "name", "wmo", "source", "koppen", "lat", "lon"]].to_csv(out / "station_index.csv", index=False)
    # submitted Table S1 + a 'No.' column (the station numbers used in the figures); only the mojibake of 'Andalgalá' is repaired, nothing else changes
    t1 = pd.read_csv(S1, encoding="utf-8", dtype={"WMO": str})
    t1["City"] = [("Andalgalá" if "ã¡" in c or "Ã¡" in c else c) for c in t1["City"]]
    t1.insert(0, "No.", range(1, len(t1) + 1))
    if a.koppen == "ashrae":
        t1["Koppen_ASHRAE"] = [k if s_ == "IWEC2" else "n/a (not in the ASHRAE list)" for k, s_ in zip(sites["koppen"], sites["source"])]
    t1.to_csv(out / "TableS1_sites_with_number.csv", index=False, encoding="utf-8")
    print(f"Figures ({a.domain}) -> {out}")
    if a.with_v1:
        best, mat = st.loc[st.groupby("city")["RMSE"].idxmin()].set_index("city"), np.full((1, 1), np.nan)
    else:
        best = fig1_map(sites, st, out)
    s2 = fig23_scatter(h, "DNI", out, "Fig2_scatter_DNI")
    s3 = fig23_scatter(h, "DHI", out, "Fig3_scatter_DHI")
    fig4_heatmap(sites, st, out)
    fig5_bars(sites, st, out)
    fig6_mbe(st, out)
    pts, win = fig7_taylor(h, out)
    if not a.with_v1:
        bm, mat = fig8_monthly(h, sites, st, out)
    slopes = fig9_kt(h, st, out)
    counts = fig10_koppen(st, out)

    # facts per figure for the author's captions (no caption text is written here)
    iw = st[st["source"] == "IWEC2"]
    L = [f"# Datos para las leyendas de las figuras (dominio {a.domain}) — NO son leyendas", "",
         "Generado por `scripts/make_figures_v2.py`. Son hechos de cada figura (qué se grafica, muestra, cifras); las leyendas las redacta el autor. Provisorio.", "",
         "Regla común: DIRINT = Perez 1992 sin punto de rocío; referencia = IWEC2 (partición Gompertz de la GHI del archivo: cercanía a la referencia, no exactitud). "
         "Figuras por sitio (1, 4, 5, 8): 41 archivos, los 2 de otra procedencia marcados con †. Resúmenes estadísticos (2, 3, 6, 7, 9, 10): los 39 IWEC2. "
         "Numeración de estaciones = orden de la Tabla S1 enviada (norte a sur); ver `station_index.csv` (si se usa, agregar la columna «No.» a la Tabla S1).", "",
         f"- **Fig. 1 (mapa):** mejor modelo por sitio = menor RMSE de DNI en {a.domain}; forma = modelo, color = R² de ese modelo. Cuenta de mejores: "
         + ", ".join(f"{LAB[m]} {int((best['model'] == m).sum())}" for m in MODELS) + f". R² del mejor modelo entre {best['R2'].min():.3f} y {best['R2'].max():.3f}. "
         "Contorno de países de Natural Earth 1:10m (dominio público), versión con el punto de vista de Argentina (incluye las islas Malvinas); archivo y hash en `data/geo/`.",
         f"- **Fig. 2 (dispersión DNI):** {int(s2[MODELS[0]]['n']):,} horas IWEC2 con DNI de referencia > 0, hexbin en escala log. " + "; ".join(f"{LAB[m]}: R² {s2[m]['R2']:.3f}, RMSE {s2[m]['RMSE']:.0f}, MBE {s2[m]['MBE']:+.0f}" for m in MODELS) + " (W/m², estadísticos agrupados, no promedios por estación).",
         f"- **Fig. 3 (dispersión DHI):** igual con DHI; " + "; ".join(f"{LAB[m]}: R² {s3[m]['R2']:.3f}, RMSE {s3[m]['RMSE']:.0f}, MBE {s3[m]['MBE']:+.0f}" for m in MODELS) + ".",
         f"- **Fig. 4 (mapa de calor de R²):** 41 sitios × {len(MODELS)} modelos, ordenados por el R² de Reindl-2 (creciente); R² de DNI " + "; ".join(f"{LAB[m]}: {st[st['model'] == m]['R2'].min():.3f}–{st[st['model'] == m]['R2'].max():.3f}" for m in MODELS) + ".",
         "- **Fig. 5 (barras de nRMSE):** 41 sitios ordenados por el nRMSE de Reindl-2 (decreciente); nRMSE medio de DNI (39 IWEC2): " + "; ".join(f"{LAB[m]} {iw[iw['model'] == m]['nRMSE'].mean():.1f} %" for m in MODELS) + ".",
         "- **Fig. 6 (caja de MBE):** distribución entre los 39 IWEC2 del sesgo de DNI; mediana (W/m²): " + "; ".join(f"{LAB[m]} {iw[iw['model'] == m]['MBE'].median():+.1f}" for m in MODELS) + ". Se superponen los puntos individuales.",
         "- **Fig. 7 (Taylor):** un punto por estación y modelo (39 IWEC2); desviación estándar normalizada con la de **cada estación** (en la versión enviada se normalizaba con la desviación de todas las estaciones juntas, lo que no corresponde a un diagrama de Taylor por sitio). Vista **ampliada** (desviación normalizada de " + f"{win[0][0]:.2f} a {win[0][1]:.2f}" + ") porque todos los puntos caen ahí: correlación entre " + f"{min(pts[m][0].min() for m in MODELS):.3f} y {max(pts[m][0].max() for m in MODELS):.3f}" + " y desviación normalizada entre " + f"{min(pts[m][1].min() for m in MODELS):.2f} y {max(pts[m][1].max() for m in MODELS):.2f}" + "; curvas punteadas = diferencia RMS centrada normalizada.",
         f"- **Fig. 8 (MBE mensual):** MBE de DNI del mejor modelo de cada sitio, por mes del TMY (celdas con menos de 30 horas quedan en blanco); la letra de la derecha indica el modelo (D = DIRINT, E = Erbs, R = Reindl-2); rango {np.nanmin(mat):.0f} a {np.nanmax(mat):.0f} W/m².",
         "- **Fig. 9 (R² contra Kt medio):** una columna por modelo, color por clase Köppen " + ("de la lista de ASHRAE de IWEC2 (D7 decidida el 5/10)" if a.koppen == "ashrae" else "de la Tabla S1 (D7 abierta)") + ", 39 IWEC2; recta de mínimos cuadrados; pendientes: " + "; ".join(f"{LAB[m]} {slopes[m][0]:+.2f} (r = {slopes[m][1]:+.2f})" for m in MODELS) + ". Se rotulan los 3 sitios de menor R² de cada panel.",
         "- **Fig. 10 (nRMSE por clase Köppen):** cajas solo para clases con n ≥ 3, puntos para todas las estaciones; n por clase: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", "",
         "Verificación: las métricas por estación se recalculan desde la serie horaria y se comprueban iguales a `benchmark_v2_per_station.csv`."]
    if a.with_v1:
        L = [ln for ln in L if not (ln.startswith("- **Fig. 1") or ln.startswith("- **Fig. 8"))]
        L.insert(2, "**Variante con el DIRINT de la v1.0.0 (tabla propia) dibujado como cuarto modelo** (diamante rojo-anaranjado). Las Figs. 1 y 8 no cambian (el DIRINT v1.0.0 nunca es el mejor modelo de un sitio) y no se redibujan; usar las de la carpeta sin `_with_v1`.")
    (out / "FIGURAS_datos_para_leyendas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito", out / "FIGURAS_datos_para_leyendas.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
