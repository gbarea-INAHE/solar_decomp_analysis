"""
reanalysis_v2.py — Task 3 analyses on the hourly series produced by benchmark_v2.py (geometry G2).

Updated 2026-10-05 (decision 3 of the author): every analysis is reported for TWO domains, `c20` (cos z >= 0.2; the main figure) and `all`
(GHI >= 10 W/m2 and cos z > 0.0174, as the submitted paper); outputs carry the domain in the file name (`*_c20.csv`, `*_all.csv`). A third DIRINT
row was added: `dirint_std_noTd` = Perez 1992 without dew point (Perez's «unknown precipitable water» bin) = the default of Solar Decomp v1.1.0.
The previous (single-domain) outputs are kept in data/processed/_prev_20261005/.

PROVISIONAL parameters (not confirmed by the author; see revision_v2/DECISIONS_PROVISIONAL.md):
  KT_BINS            [0, .2, .35, .5, .65, .75, inf)   (handoff 3.1 proposal)
  KTP_BINS           [0, .2, .4, .56, .7, .8, 1.1)     (pvlib/Perez Kt' classes, for the DIRINT diagnosis)
  closure thresholds absolute 5/10/20/50 W/m2 and relative 2/5/10 % of GHI (several thresholds, no single choice)
  seasons            DJF (Dec-Feb) vs JJA (Jun-Aug) by TMY month, southern hemisphere
  bootstrap          2000 resamples of stations within class, seed 20261005, classes with n >= 3 only
  tests              Wilcoxon signed-rank on station-level paired values, Bonferroni within each family
Primary sample: the 39 IWEC2 files. The 2 other files are analysed separately (sensitivity).

Outputs: CSVs in data/processed/reanalysis_*.csv and figures in figures/ (300 dpi, Okabe-Ito, no in-figure titles).
Usage: python reanalysis_v2.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "revision_v2" / "data" / "processed"
FIG = ROOT / "revision_v2" / "figures"
MODELS = ["erbs", "reindl2", "dirint_v1", "dirint_std", "dirint_std_noTd", "disc", "orgill_hollands", "erbs_driesse", "boland"]
KEY = ["erbs", "reindl2", "dirint_v1", "dirint_std", "dirint_std_noTd", "disc"]
DIAG = ("dirint_v1", "dirint_std", "dirint_std_noTd", "erbs", "reindl2")
DOMS = {"c20": 0.2, "all": 0.0174}               # c20 = main figure; all = separate table
KT_BINS = [0.0, 0.2, 0.35, 0.5, 0.65, 0.75, np.inf]
KTP_BINS = [0.0, 0.2, 0.4, 0.56, 0.7, 0.8, 1.1]
SEED = 20261005
OKABE = {"erbs": "#0072B2", "reindl2": "#009E73", "dirint_v1": "#D55E00", "dirint_std": "#CC79A7", "dirint_std_noTd": "#CC79A7", "disc": "#E69F00",
         "orgill_hollands": "#56B4E9", "erbs_driesse": "#999999", "boland": "#000000"}
LSTYLE = {"dirint_std_noTd": "--"}
LABEL = {"erbs": "Erbs", "reindl2": "Reindl-2", "dirint_v1": "DIRINT (v1.0.0 table)", "dirint_std": "DIRINT (Perez 1992, pvlib, dew point)",
         "dirint_std_noTd": "DIRINT (Perez 1992, pvlib, no dew point)", "disc": "DISC", "orgill_hollands": "Orgill-Hollands",
         "erbs_driesse": "Erbs-Driesse", "boland": "Boland"}
PAIRS_ORIG = [("reindl2", "erbs"), ("reindl2", "dirint_std"), ("erbs", "dirint_std"), ("dirint_std", "dirint_v1"), ("reindl2", "disc"), ("erbs", "disc")]
PAIRS_NOTD = [("dirint_std_noTd", "dirint_std"), ("dirint_std_noTd", "erbs"), ("dirint_std_noTd", "dirint_v1"), ("reindl2", "dirint_std_noTd")]


def rmse_mbe(ref, pred):
    m = np.isfinite(ref) & np.isfinite(pred)
    e = (pred - ref)[m]
    return len(e), float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan, float(e.mean()) if len(e) else np.nan, float(ref[m].mean()) if m.any() else np.nan


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def kt_bins(iw, plt):
    """3.1: Kt bins on the pooled IWEC2 hours, both domains in one file."""
    rows = []
    for dom, c0 in DOMS.items():
        d = iw[iw["cosz"] >= c0]
        kb = pd.cut(d["kt"], KT_BINS, right=False)
        for comp, ref in (("DNI", "dni_ref"), ("DHI", "dhi_ref")):
            for m in MODELS:
                for b, g in d.groupby(kb, observed=True):
                    n, rmse, mbe, mref = rmse_mbe(g[ref].to_numpy(), g[f"{comp.lower()}_{m}"].to_numpy())
                    rows.append(dict(domain=dom, comp=comp, model=m, kt_bin=str(b), n=n, RMSE=rmse, MBE=mbe, ref_mean=mref,
                                     nRMSE_pct=100 * rmse / mref if mref and mref > 0 else np.nan))
    ktb = pd.DataFrame(rows)
    ktb.to_csv(OUT / "reanalysis_kt_bins.csv", index=False)
    for dom in DOMS:
        for metric in ("MBE", "RMSE"):
            sh = ktb[(ktb.domain == dom) & (ktb.comp == "DNI") & ktb.model.isin(KEY)].pivot(index="kt_bin", columns="model", values=metric).round(1)
            print(f"\n[3.1] DNI {metric} (W/m2) by Kt bin, IWEC2 pooled, domain {dom}:")
            print(sh[KEY].to_string())
        print(f"hours per Kt bin ({dom}):", ktb[(ktb.domain == dom) & (ktb.comp == "DNI") & (ktb.model == "erbs")].set_index("kt_bin")["n"].to_dict())
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
        for ax, metric in zip(axs, ("MBE", "RMSE")):
            for m in KEY:
                g = ktb[(ktb.domain == dom) & (ktb.comp == "DNI") & (ktb.model == m)]
                ax.plot(range(len(g)), g[metric].to_numpy(), marker="o", ms=3.5, lw=1.2, ls=LSTYLE.get(m, "-"), color=OKABE[m], label=LABEL[m])
            ax.set_xticks(range(len(KT_BINS) - 1)); ax.set_xticklabels(["<0.2", "0.2-0.35", "0.35-0.5", "0.5-0.65", "0.65-0.75", ">=0.75"], fontsize=7, rotation=30)
            ax.set_xlabel("Clearness index bin"); ax.set_ylabel(f"DNI {metric} (W m$^{{-2}}$)"); style(ax)
            if metric == "MBE":
                ax.axhline(0, color="black", lw=0.6)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.set_size_inches(7.2, 3.7)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5)      # fuera de los ejes: no tapa las curvas
        fig.savefig(FIG / f"fig_kt_bins_dni_{dom}.png", dpi=300); plt.close(fig)


def per_domain(h, bm_all, dom, c0, plt):
    iw = h[(h["source"] == "IWEC2") & (h["cosz"] >= c0)].copy()
    bm = bm_all[bm_all["domain"] == dom]
    print(f"\n{'=' * 100}\nDOMAIN {dom} (cos z >= {c0}): {len(iw):,} IWEC2 hours\n{'=' * 100}")

    # ---------------- 3.2 DIRINT diagnosis ----------------
    am = iw["am"].clip(upper=10.0)
    iw["ktp"] = iw["kt"] / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / am)) + 0.1)
    rows = []
    for b, g in iw.groupby(pd.cut(iw["ktp"], KTP_BINS, right=False), observed=True):
        for m in DIAG:
            n, rmse, mbe, mref = rmse_mbe(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())
            rows.append(dict(ktp_bin=str(b), model=m, n=n, RMSE=rmse, MBE=mbe, ref_mean=mref))
    ktp = pd.DataFrame(rows); ktp.to_csv(OUT / f"reanalysis_ktp_bins_{dom}.csv", index=False)
    print(f"\n[3.2] DNI MBE (W/m2) by Kt' bin (IWEC2 pooled, {dom}):")
    print(ktp.pivot(index="ktp_bin", columns="model", values="MBE").round(1)[list(DIAG)].to_string())
    print("hours per Kt' bin:", ktp[ktp.model == "erbs"].set_index("ktp_bin")["n"].to_dict())

    per = []
    for c, g in iw.groupby("city"):
        r = dict(city=c, koppen=g["koppen_s1"].iloc[0], f_kt_gt_065=float((g["kt"] > 0.65).mean()), f_ktp_ge_08=float((g["ktp"] >= 0.8).mean()),
                 am_median=float(g["am"].median()), kt_mean=float(g["kt"].mean()))
        for m in DIAG:
            r[f"mbe_{m}"] = rmse_mbe(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())[2]
        per.append(r)
    per = pd.DataFrame(per); per.to_csv(OUT / f"reanalysis_station_aridity_{dom}.csv", index=False)
    cor = []
    print(f"\n[3.2] station-level correlation of DNI MBE with aridity proxy F(Kt>0.65) (Spearman rho, p), {dom}:")
    for m in DIAG:
        rho, p = stats.spearmanr(per["f_kt_gt_065"], per[f"mbe_{m}"])
        cor.append(dict(domain=dom, model=m, spearman_rho=float(rho), p=float(p), n_stations=len(per)))
        print(f"   {m:<16} rho={rho:+.2f} p={p:.3g}")
    pd.DataFrame(cor).to_csv(OUT / f"reanalysis_aridity_corr_{dom}.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for m in DIAG:
        ax.scatter(per["f_kt_gt_065"], per[f"mbe_{m}"], s=12, color=OKABE[m], label=LABEL[m], linewidths=0, marker="^" if m == "dirint_std_noTd" else "o")
    ax.axhline(0, color="black", lw=0.6); ax.set_xlabel("Fraction of hours with Kt > 0.65"); ax.set_ylabel("DNI MBE (W m$^{-2}$)")
    handles, labels = ax.get_legend_handles_labels()
    style(ax)
    fig.set_size_inches(3.8, 4.3)
    fig.tight_layout(rect=(0, 0.28, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=5.5, markerscale=0.9)    # debajo del gráfico
    fig.savefig(FIG / f"fig_dirint_mbe_vs_clear_fraction_{dom}.png", dpi=300); plt.close(fig)

    # ---------------- 3.3 Buenos Aires (Aeroparque) vs Mendoza ----------------
    rows = []
    for c in ("AEROPARQUE-BS-AS", "MENDOZA-AERO"):
        g = iw[iw["city"] == c]
        kb = pd.cut(g["kt"], KT_BINS, right=False)
        for b, gg in g.groupby(kb, observed=True):
            r = dict(city=c, kt_bin=str(b), share=len(gg) / len(g), n=len(gg))
            for m in DIAG:
                r[f"rmse_{m}"] = rmse_mbe(gg["dni_ref"].to_numpy(), gg[f"dni_{m}"].to_numpy())[1]
            rows.append(r)
    ba = pd.DataFrame(rows); ba.to_csv(OUT / f"reanalysis_ba_vs_mendoza_{dom}.csv", index=False)
    print(f"\n[3.3] Aeroparque (Buenos Aires) vs Mendoza ({dom}): share of hours and DNI RMSE by Kt bin")
    print(ba.round(3).to_string(index=False))

    # ---------------- 3.4 seasonal ----------------
    rows = []
    for c, g in iw.groupby("city"):
        for sea, months in (("DJF", [12, 1, 2]), ("JJA", [6, 7, 8])):
            gg = g[g["month"].isin(months)]
            for m in MODELS:
                for comp in ("dni", "dhi"):
                    n, rmse, mbe, mref = rmse_mbe(gg[f"{comp}_ref"].to_numpy(), gg[f"{comp}_{m}"].to_numpy())
                    rows.append(dict(city=c, koppen=g["koppen_s1"].iloc[0], season=sea, model=m, comp=comp.upper(), n=n, RMSE=rmse, MBE=mbe, nRMSE_pct=100 * rmse / mref))
    sea = pd.DataFrame(rows); sea.to_csv(OUT / f"reanalysis_seasonal_{dom}.csv", index=False)
    s = sea[sea.comp == "DNI"].groupby(["model", "season"])[["RMSE", "nRMSE_pct", "MBE"]].mean().round(2).unstack("season")
    print(f"\n[3.4] DNI by season, mean over IWEC2 stations (RMSE, nRMSE %, MBE), {dom}:")
    print(s.loc[KEY].to_string())

    # ---------------- 3.5 closure of the mixed model (Reindl-2 DNI + Erbs DHI) ----------------
    d = iw.copy()
    d["res_mix"] = d["dni_reindl2"] * d["cosz"] + d["dhi_erbs"] - d["ghi"]
    d["res_reindl"] = d["dni_reindl2"] * d["cosz"] + d["dhi_reindl2"] - d["ghi"]
    d["res_erbs"] = d["dni_erbs"] * d["cosz"] + d["dhi_erbs"] - d["ghi"]
    rows = []
    for name in ("res_mix", "res_reindl", "res_erbs"):
        r_ = d[name].dropna()
        g_ = d.loc[r_.index, "ghi"]
        rows.append(dict(domain=dom, case=name, n=len(r_), MBE=r_.mean(), RMSE=float(np.sqrt((r_ ** 2).mean())), p50_abs=r_.abs().quantile(.5), p95_abs=r_.abs().quantile(.95),
                         p99_abs=r_.abs().quantile(.99), **{f"gt_{t}Wm2": float((r_.abs() > t).mean()) for t in (5, 10, 20, 50)},
                         **{f"gt_{t}pct": float((r_.abs() / g_ > t / 100).mean()) for t in (2, 5, 10)}))
    clo = pd.DataFrame(rows); clo.to_csv(OUT / f"reanalysis_closure_{dom}.csv", index=False)
    print(f"\n[3.5] closure residual GHI_rec - GHI (W/m2), IWEC2 pooled, {dom}:")
    print(clo.round(3).to_string(index=False))

    # ---------------- 3.6 sensitivity to the two non-IWEC2 files ----------------
    rows = []
    for m in MODELS:
        a39 = bm[(bm.model == m) & (bm.source == "IWEC2")]; a41 = bm[bm.model == m]
        rows.append(dict(domain=dom, model=m, R2_39=a39.DNI_R2.mean(), R2_41=a41.DNI_R2.mean(), RMSE_39=a39.DNI_RMSE.mean(), RMSE_41=a41.DNI_RMSE.mean(),
                         d_RMSE=a41.DNI_RMSE.mean() - a39.DNI_RMSE.mean(), SD_RMSE_39=a39.DNI_RMSE.std(), SD_RMSE_41=a41.DNI_RMSE.std()))
    sens = pd.DataFrame(rows); sens.to_csv(OUT / f"reanalysis_sensitivity_non_iwec2_{dom}.csv", index=False)
    print(f"\n[3.6] DNI summary with (41) and without (39) the two non-IWEC2 files, {dom}:")
    print(sens.round(3).to_string(index=False))
    rk39 = sens.set_index("model")["RMSE_39"].rank().astype(int); rk41 = sens.set_index("model")["RMSE_41"].rank().astype(int)
    print("rank by mean DNI RMSE (39 vs 41):", {m: (int(rk39[m]), int(rk41[m])) for m in MODELS})
    non = bm[bm.source != "IWEC2"]
    for c in non.city.unique():
        z = {}
        for m in KEY:
            ref = bm[(bm.model == m) & (bm.source == "IWEC2")]["DNI_RMSE"]
            z[m] = (float(non[(non.city == c) & (non.model == m)]["DNI_RMSE"].iloc[0]) - ref.mean()) / ref.std()
        print(f"   {c}: z-score of DNI RMSE vs IWEC2 distribution:", {k: round(v, 2) for k, v in z.items()})

    # ---------------- 3.7 class robustness: bootstrap CIs + paired tests ----------------
    rng = np.random.default_rng(SEED)
    iwb = bm[bm.source == "IWEC2"]
    rows = []
    for cl, g in iwb.groupby("koppen_s1"):
        ns = g["city"].nunique()
        for m in KEY:
            v = g[g.model == m].set_index("city")
            r = dict(domain=dom, koppen=cl, n_stations=ns, model=m, DNI_RMSE_mean=v.DNI_RMSE.mean(), DNI_R2_mean=v.DNI_R2.mean(), DNI_MBE_mean=v.DNI_MBE.mean())
            if ns >= 3:
                idx = rng.integers(0, ns, size=(2000, ns))
                for k in ("DNI_RMSE", "DNI_R2", "DNI_MBE"):
                    bs = v[k].to_numpy()[idx].mean(axis=1)
                    r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = np.quantile(bs, [0.025, 0.975])
            else:
                r["note"] = "indicative only (n<=2)"
            rows.append(r)
    cls = pd.DataFrame(rows); cls.to_csv(OUT / f"reanalysis_class_bootstrap_{dom}.csv", index=False)
    print(f"\n[3.7] DNI RMSE by Koppen class (manuscript Table S1 classes), IWEC2 only; bootstrap 95% CI where n>=3, {dom}:")
    print(cls[cls.model.isin(["erbs", "reindl2", "dirint_std", "dirint_std_noTd", "dirint_v1"])][["koppen", "n_stations", "model", "DNI_RMSE_mean", "DNI_RMSE_ci_lo", "DNI_RMSE_ci_hi", "note"]].round(1).to_string(index=False))

    rows = []
    for comp in ("DNI", "DHI"):
        for fam_name, pairs in (("original_6", PAIRS_ORIG), ("noTd_4", PAIRS_NOTD)):
            fam = []
            for a_, b_ in pairs:
                va = iwb[iwb.model == a_].set_index("city")[f"{comp}_RMSE"]; vb = iwb[iwb.model == b_].set_index("city")[f"{comp}_RMSE"]
                w = stats.wilcoxon(va, vb.loc[va.index])
                fam.append(dict(domain=dom, family=fam_name, comp=comp, a=a_, b=b_, mean_diff_RMSE=float((va - vb.loc[va.index]).mean()), W=float(w.statistic), p=float(w.pvalue)))
            for f in fam:
                f["p_bonferroni"] = min(1.0, f["p"] * len(fam))
            rows += fam
    tst = pd.DataFrame(rows); tst.to_csv(OUT / f"reanalysis_paired_tests_{dom}.csv", index=False)
    print(f"\n[3.7] paired Wilcoxon on station-level RMSE (a - b), Bonferroni within each family (component x pair set), {dom}:")
    print(tst.round(5).to_string(index=False))


def main() -> None:
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.read_pickle(OUT / "hourly_v2.pkl.gz")
    iw = h[h["source"] == "IWEC2"].copy()
    print(f"hours: {len(h)} total, {len(iw)} IWEC2 ({iw['city'].nunique()} stations)")
    pd.set_option("display.width", 220, "display.max_rows", 300)
    kt_bins(iw, plt)
    bm_all = pd.read_csv(OUT / "benchmark_v2_per_station.csv")
    for dom, c0 in DOMS.items():
        per_domain(h, bm_all, dom, c0, plt)


if __name__ == "__main__":
    main()
