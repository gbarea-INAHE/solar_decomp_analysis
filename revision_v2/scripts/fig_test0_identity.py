"""
fig_test0_identity.py — Figure for the circularity result (Test 0).

Panels (DNI in W/m2, hours with GHI >= 10 W/m2 and cos z >= 0.2, cos z and extraterrestrial radiation taken from the file):
  a, b  IWEC2 files (Mendoza, Ushuaia): DNI in the file against the Gompertz split of the file's own GHI (Zhang et al. 2004, port of the
        Fortran used in the IWEC2 chain). Points lie on the 1:1 line to rounding level.
  c     Histogram of Gompertz - file DNI for all IWEC2 files pooled (log counts).
  d     A non-IWEC2 file (General Pico, TMY2): same comparison; high R2 but not identity. Shows that R2 alone cannot tell "identical"
        from "similar".
Output: figures/fig_test0_identity.png (300 dpi, Okabe-Ito, no in-figure title). Usage: python fig_test0_identity.py --epw-dir <folder>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

from benchmark_v2 import prepare  # noqa: E402
from epw_io import epw_files, read_epw  # noqa: E402
from gompertz import gompertz_kn  # noqa: E402

FIG = ROOT / "revision_v2" / "figures"
COS_MIN, MIN_GHI = 0.2, 10.0
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"


def pairs(path):
    meta, df = read_epw(path)
    d, _ = prepare(meta, df)
    ghi, dni = d["GHI_h"].to_numpy(), df["dni"].to_numpy()
    etr_h, etr_n, cz = d["G0h"].to_numpy(), d["G0n"].to_numpy(), d["cos_Z"].to_numpy()
    m = (ghi >= MIN_GHI) & (cz >= COS_MIN) & np.isfinite(dni) & (etr_h > 0)
    pred = gompertz_kn(ghi[m] / etr_h[m], cz[m]) * etr_n[m]
    return meta, dni[m], pred


def r2(o, p):
    return 1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()


def main() -> None:
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(); ap.add_argument("--epw-dir", required=True); a = ap.parse_args()
    files = epw_files(a.epw_dir)
    pick = {k: next(f for f in files if k in f.name.upper()) for k in ("MENDOZA", "USHUAIA", "GENERAL")}
    allres, stats = [], {}
    for f in files:
        meta, o, p = pairs(f)
        stats[meta["city"]] = (meta["source"], len(o), r2(o, p), float(np.abs(p - o).max()), float((np.abs(p - o) <= 2).mean()))
        if meta["source"] == "IWEC2":
            allres.append(p - o)
    res = np.concatenate(allres)

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for ax, key, lab in ((axs[0, 0], "MENDOZA", "a"), (axs[0, 1], "USHUAIA", "b"), (axs[1, 1], "GENERAL", "d")):
        meta, o, p = pairs(pick[key])
        hb = ax.hexbin(o, p, gridsize=60, bins="log", mincnt=1, cmap="Blues" if key != "GENERAL" else "Oranges", linewidths=0.2)
        lim = [0, max(o.max(), p.max()) * 1.02]
        ax.plot(lim, lim, color="black", lw=0.7)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("DNI in the EPW file (W m$^{-2}$)"); ax.set_ylabel("Gompertz split of file GHI (W m$^{-2}$)")
        src = "IWEC2" if meta["source"] == "IWEC2" else "non-IWEC2"
        city = meta["city"].replace("-Aero", "").replace(" Airp.", "").title()
        ax.text(0.04, 0.96, f"({lab}) {city}, {src}", transform=ax.transAxes, fontsize=7.5, va="top")
        ax.text(0.97, 0.20, f"$R^2$ = {r2(o, p):.4f}\n|diff| $\\leq$ 2 W m$^{{-2}}$: {100*np.mean(np.abs(p - o) <= 2):.0f} % of hours",
                transform=ax.transAxes, fontsize=7, va="top", ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.05); cb.set_label("hours (log)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax = axs[1, 0]
    ax.hist(res, bins=np.arange(-12.5, 12.51, 1.0), color=BLUE, edgecolor="white", linewidth=0.3)
    ax.set_yscale("log"); ax.set_xlabel("Gompertz $-$ file DNI (W m$^{-2}$)"); ax.set_ylabel("hours (log)")
    ax.set_ylim(1, 1e8)
    ax.text(0.97, 0.96, "(c) 39 IWEC2 files pooled", transform=ax.transAxes, fontsize=7.5, va="top", ha="right")
    ax.text(0.97, 0.86, f"n = {len(res):,} h\nmax |diff| = {np.abs(res).max():.1f} W m$^{{-2}}$".replace(",", " "), transform=ax.transAxes, fontsize=7, va="top", ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "fig_test0_identity.png", dpi=300)
    print("saved", FIG / "fig_test0_identity.png")
    iw = [v for v in stats.values() if v[0] == "IWEC2"]
    print(f"IWEC2 files: {len(iw)}; pooled hours {len(res)}; max |diff| {np.abs(res).max():.2f}; share within 2 W/m2 {np.mean(np.abs(res) <= 2):.3f}; min R2 {min(v[2] for v in iw):.5f}")
    for c, v in stats.items():
        if v[0] != "IWEC2":
            print(f"non-IWEC2 {c}: R2 {v[2]:.4f}, max|diff| {v[3]:.1f}, within 2 W/m2 {v[4]:.3f}")


if __name__ == "__main__":
    main()
