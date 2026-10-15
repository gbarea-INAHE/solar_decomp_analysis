"""make_figures_external_check.py — figuras del «chequeo externo» (decisión del autor, 5/10: subsección corta en el artículo con datos y gráficos).
Compara con MEDICIONES lo que el benchmark contra IWEC2 no puede mostrar. Mismo estilo que make_figures_v2 (sin títulos ni cajas, Okabe-Ito, 300 dpi PNG + PDF).

  E1  Gompertz (el generador de la referencia IWEC2) contra lo medido: DNI en 4 estaciones medidas y DHI en Mendoza (hexbin, dominio c20).
  E2  ¿Se sostiene el ranking de IWEC2? Diferencia de RMSE respecto de Erbs (modelo − Erbs): contra IWEC2 (39 estaciones) y contra las mediciones.
  E3  Distancia de cada modelo a Gompertz frente a su distancia a lo medido (por debajo de la diagonal = más cerca de Gompertz que de la realidad).
  E4  Mendoza: RMSE de difusa por modelo, total (con IC 95 %) y por estación del año.
Provisorio: Mendoza con latitud, longitud, hora de reloj y marca de fin de minuto confirmadas por el técnico (2026-10-06) pero con altitud supuesta (750 m) y sin calibración informada; piloto = 4 estaciones de un año, ninguna argentina. El DNI de Mendoza es derivado; solo se usa la difusa medida.
Salidas: figures/external_check/*.png|pdf, external_check_summary.csv, FIGURAS_datos_para_leyendas.md.
Uso (desde revision_v2, venv activo): python scripts/make_figures_external_check.py     (tarda unos minutos: corre el piloto y Mendoza)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
import dirint_dewpoint_anchor as dd  # noqa: E402
import make_figures_v2 as mf  # noqa: E402  (también fija el estilo de matplotlib)
import run_mendoza_provisional as rmp  # noqa: E402
from benchmark_v2 import metr  # noqa: E402

REV = mf.REV
PROC = mf.PROC
OUT = REV / "figures" / "external_check"
WM2 = mf.WM2
STN = [("PTR", "Petrolina (BSRN, Brazil)"), ("SMS", "São Martinho da Serra (BSRN, Brazil)"), ("DRA", "Desert Rock (SURFRAD, USA)"), ("GOLDEN", "Golden (SRRL, USA; hourly)")]
MCOL = {"dirint_noTd": "#CC79A7", "erbs": "#0072B2", "reindl2": "#009E73", "dirint_v1": "#D55E00", "brl_hourly": "#E69F00", "engerer2_1min": "#56B4E9", "gompertz": "#000000"}
MLAB = {"dirint_noTd": "DIRINT", "erbs": "Erbs", "reindl2": "Reindl-2", "dirint_v1": "SD 1.0.0 table", "brl_hourly": "BRL", "engerer2_1min": "Engerer2 (1 min)", "gompertz": "Gompertz",
        "engerer2_60 (expl.)": "Engerer2 (60 min, exploratory)"}


def load_pilot():
    """Per station: hourly measured DNI, series of models (incl. DIRINT of Perez without dew point), c20 mask, day index."""
    out = {}
    for stn, _ in STN:
        if stn == "GOLDEN":
            h = asg.golden_base_frame()
            epw = asg.golden_epw(h)
            res, sim, series, mask = asg.evaluate(epw, h["dni"].to_numpy())
            lon = asg.G_LON
        else:
            minute, geo, meta = asg.minute_station(stn)
            res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
            epw = ba.to_epw_like(h, "avg")
            lon = meta["lon"]
        _, noTd, _ = dd.dirint_variants(epw)
        ser = {"gompertz": series["gompertz (T1)"], "erbs": series["erbs"], "reindl2": series["reindl2"], "dirint_v1": series["dirint_v1"], "dirint_noTd": noTd}
        day = (pd.DatetimeIndex(h.index) + pd.Timedelta(hours=lon / 15.0)).floor("D").to_numpy()
        out[stn] = dict(dni=h["dni"].to_numpy(float), ser=ser, mask=mask, day=day)
        print("  piloto cargado:", stn, int(mask.sum()), "horas c20", flush=True)
    return out


def load_mendoza(lat=-32.89, lon=-68.87, elev=750.0):                  # lat/lon: técnico, 2026-10-06; elev: SUPUESTO (no informada)
    P = rmp.process(rmp.mendoza_minute("end"), lat, lon, elev)         # marca de tiempo = final de minuto (técnico, 2026-10-06)
    day = (pd.DatetimeIndex(P["h"].index) + pd.Timedelta(hours=lon / 15.0)).floor("D").to_numpy()
    ser = {"gompertz": P["series"]["gompertz (T1)"], "erbs": P["series"]["erbs"], "reindl2": P["series"]["reindl2"], "dirint_v1": P["series"]["dirint_v1"],
           "dirint_noTd": P["series"]["dirint_perez"], "brl_hourly": P["series"]["brl_hourly"], "engerer2_1min": P["series"]["engerer2_1min"]}
    dhi = {k: rmp.dhi_from_dni(P["ghi"], v, P["cosz"]) for k, v in ser.items()}
    mk = rmp.common_mask(P)
    print("  Mendoza cargada:", int(mk.sum()), "horas c20", flush=True)
    return dict(P=P, dhi=dhi, mask=mk, day=day)


def e1_gompertz(pil, men):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    axes = axes.ravel()
    stats = {}
    last = None
    for ax, (stn, label), letter in zip(axes[:4], STN, "abcd"):
        d = pil[stn]
        mk = d["mask"] & np.isfinite(d["ser"]["gompertz"]) & np.isfinite(d["dni"])
        o, p = d["dni"][mk], d["ser"]["gompertz"][mk]
        last = ax.hexbin(o, p, gridsize=45, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, extent=(0, 1050, 0, 1050), linewidths=0)
        ax.plot([0, 1050], [0, 1050], color="black", lw=0.8, ls="--")
        r = metr(o, p)
        stats[stn] = r
        ax.set_title(f"({letter}) {label}".replace("São Martinho da Serra", "São Martinho"), loc="left", fontsize=7.2, fontweight="bold")
        ax.text(0.97, 0.04, f"$R^2$ = {r['R2']:.3f}\nRMSE = {r['RMSE']:.0f} {WM2}\nMBE = {r['MBE']:+.0f}".replace("= -", "= −") + f" {WM2}", transform=ax.transAxes, va="bottom", ha="right", fontsize=6.8, linespacing=1.25)
        ax.set_xlim(0, 1050)
        ax.set_ylim(0, 1050)
        ax.set_aspect("equal")
        ax.set_xlabel(f"DNI measured ({WM2})")
        if letter in "ad":
            ax.set_ylabel(f"DNI of the Gompertz split ({WM2})")
    # Mendoza (difusa medida)
    ax = axes[4]
    mk, P = men["mask"], men["P"]
    o, p = P["dhi_m"][mk], men["dhi"]["gompertz"][mk]
    last = ax.hexbin(o, p, gridsize=45, cmap="Blues", norm=LogNorm(vmin=1), mincnt=1, extent=(0, 550, 0, 550), linewidths=0)
    ax.plot([0, 550], [0, 550], color="black", lw=0.8, ls="--")
    r = metr(o, p)
    stats["MENDOZA"] = r
    ax.set_title("(e) Mendoza (INAHE, Argentina)", loc="left", fontsize=7.2, fontweight="bold")
    ax.text(0.97, 0.04, f"$R^2$ = {r['R2']:.3f}\nRMSE = {r['RMSE']:.0f} {WM2}\nMBE = {r['MBE']:+.0f}".replace("= -", "= −") + f" {WM2}", transform=ax.transAxes, va="bottom", ha="right", fontsize=6.8, linespacing=1.25)
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 550)
    ax.set_aspect("equal")
    ax.set_xlabel(f"DHI measured ({WM2})")
    ax.set_ylabel(f"DHI of the Gompertz split ({WM2})")
    axes[5].axis("off")
    cax = axes[5].inset_axes([0.15, 0.12, 0.07, 0.65])
    cb = fig.colorbar(last, cax=cax)
    cb.set_label("Hours (log scale)")
    cb.outline.set_visible(False)
    fig.tight_layout(w_pad=0.6, h_pad=1.0)
    mf.save(fig, OUT, "FigE1_Gompertz_vs_measured")
    return stats


def boot_diff(d, models, rng, B=2000):
    """Day-block bootstrap of RMSE(model) − RMSE(Erbs) on the common sample (c20)."""
    ref = d["dni"]
    ser = {k: d["ser"][k] for k in ["erbs"] + models}
    bt = asg.day_bootstrap(ref, ser, d["mask"], d["day"], rng).set_index("model")
    return {m: (bt.loc[m, "dRMSE_vs_erbs"], bt.loc[m, "dRMSE_lo"], bt.loc[m, "dRMSE_hi"]) for m in models}


def e2_ranking(pil, men):
    models = ["reindl2", "dirint_noTd", "dirint_v1"]
    rows = []
    # contra IWEC2 (media de 39 estaciones, c20), IC por bootstrap de estaciones
    b = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    b = b[(b["source"] == "IWEC2") & (b["domain"] == "c20")]
    key = {"reindl2": "reindl2", "dirint_noTd": "dirint_std_noTd", "dirint_v1": "dirint_v1"}
    rng = np.random.default_rng(asg.SEED)
    e = b[b["model"] == "erbs"].set_index("city")["DNI_RMSE"]
    idx = rng.integers(0, len(e), size=(2000, len(e)))
    for m in models:
        v = b[b["model"] == key[m]].set_index("city")["DNI_RMSE"].reindex(e.index)
        diff = (v - e).to_numpy()
        bs = diff[idx].mean(axis=1)
        rows.append(("Against IWEC2 (DNI, mean of 39 sites)", m, diff.mean(), *np.quantile(bs, [0.025, 0.975])))
    for stn, label in STN:
        r = boot_diff(pil[stn], models, np.random.default_rng(asg.SEED))
        for m in models:
            rows.append((f"Against measured DNI: {label.split(' (')[0]}", m, *r[m]))
    bt = pd.read_csv(PROC / "mendoza_prov_bootstrap.csv")
    bt = bt[bt["ref"] == "DHI"].set_index("model")
    mk = {"reindl2": "reindl2", "dirint_noTd": "dirint_perez", "dirint_v1": "dirint_v1"}
    for m in models:
        x = bt.loc[mk[m]]
        rows.append(("Against measured DHI: Mendoza", m, x.dRMSE_vs_erbs, x.dRMSE_lo, x.dRMSE_hi))
    df = pd.DataFrame(rows, columns=["dataset", "model", "d", "lo", "hi"])
    sets = list(dict.fromkeys(df["dataset"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for i, s in enumerate(sets):
        for j, m in enumerate(models):
            r = df[(df["dataset"] == s) & (df["model"] == m)].iloc[0]
            y = len(sets) - 1 - i + (1 - j) * 0.2
            ax.errorbar(r.d, y, xerr=[[r.d - r.lo], [r.hi - r.d]], fmt=mf.MRK.get({"dirint_noTd": "dirint_std_noTd"}.get(m, m), "D") if False else {"reindl2": "o", "dirint_noTd": "^", "dirint_v1": "D"}[m],
                        color=MCOL[m], mec="black", mew=0.4, ms=5.5, elinewidth=1.0, capsize=2, zorder=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(len(sets) - 1.5 + 0.0, color="0.75", lw=0.6)
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels(list(reversed(sets)), fontsize=7.5)
    ax.set_xlabel(f"RMSE(model) − RMSE(Erbs) ({WM2}); negative = better than Erbs")
    ax.xaxis.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker={"reindl2": "o", "dirint_noTd": "^", "dirint_v1": "D"}[m], ls="", mfc=MCOL[m], mec="black", mew=0.4, ms=6, label=MLAB[m]) for m in models]
    ax.legend(handles=handles, loc="lower right", ncol=1)
    fig.tight_layout()
    mf.save(fig, OUT, "FigE2_ranking_IWEC2_vs_measured")
    return df


def e3_distance(pil, men):
    sym = {"PTR": "o", "SMS": "s", "DRA": "^", "GOLDEN": "v"}
    models = ["erbs", "reindl2", "dirint_noTd", "dirint_v1"]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    ax = axes[0]
    for stn, label in STN:
        d = pil[stn]
        mk = d["mask"] & np.isfinite(d["dni"])
        for m in models + ["gompertz"]:
            mk &= np.isfinite(d["ser"][m])
        for m in models:
            dm = float(np.sqrt(np.mean((d["ser"][m][mk] - d["dni"][mk]) ** 2)))
            dg = float(np.sqrt(np.mean((d["ser"][m][mk] - d["ser"]["gompertz"][mk]) ** 2)))
            rows.append(("pilot DNI", stn, m, dm, dg, dg / dm))
            ax.scatter(dm, dg, marker=sym[stn], color=MCOL[m], edgecolors="black", linewidths=0.4, s=34, zorder=4)
    ax.plot([0, 260], [0, 260], color="black", lw=0.8, ls="--")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 260)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Distance to the measured DNI, RMSD ({WM2})")
    ax.set_ylabel(f"Distance to the Gompertz split, RMSD ({WM2})")
    ax.text(0.04, 0.96, "(a) Measured DNI, 4 stations", transform=ax.transAxes, va="top", fontsize=8, fontweight="bold")
    ax = axes[1]
    mk = men["mask"]
    P = men["P"]
    mm = ["erbs", "reindl2", "dirint_noTd", "dirint_v1"]            # texto principal; BRL y Engerer2 van al suplemento (Tabla S5, Fig. S4)
    for m in mm:
        dm = float(np.sqrt(np.mean((men["dhi"][m][mk] - P["dhi_m"][mk]) ** 2)))
        dg = float(np.sqrt(np.mean((men["dhi"][m][mk] - men["dhi"]["gompertz"][mk]) ** 2)))
        rows.append(("Mendoza DHI", "MENDOZA", m, dm, dg, dg / dm))
        ax.scatter(dm, dg, marker="*", color=MCOL[m], edgecolors="black", linewidths=0.4, s=95, zorder=4)
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls="--")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Distance to the measured DHI, RMSD ({WM2})")
    ax.text(0.04, 0.96, "(b) Measured DHI, Mendoza", transform=ax.transAxes, va="top", fontsize=8, fontweight="bold")
    h1 = [Line2D([], [], marker="o", ls="", mfc=MCOL[m], mec="black", mew=0.4, ms=6, label=MLAB[m]) for m in ["erbs", "reindl2", "dirint_noTd", "dirint_v1"]]
    h2 = [Line2D([], [], marker=sym[s], ls="", mfc="0.8", mec="black", mew=0.4, ms=6, label=l.split(" (")[0]) for s, l in STN]
    fig.tight_layout()
    fig.legend(handles=h1 + h2, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17), fontsize=7.2, columnspacing=1.0, handletextpad=0.3)
    mf.save(fig, OUT, "FigE3_distance_to_Gompertz_vs_to_measured")
    return pd.DataFrame(rows, columns=["set", "station", "model", "RMSD_to_measured", "RMSD_to_Gompertz", "ratio"])


MAIN_E4 = ["gompertz (T1)", "erbs", "reindl2", "dirint_perez", "dirint_v1"]
SUPP_E4 = MAIN_E4 + ["brl_hourly", "engerer2_1min"]                  # Engerer2 horario (exploratorio) no se informa


def e4_mendoza(men, models, name, figsize):
    bt = pd.read_csv(PROC / "mendoza_prov_bootstrap.csv")
    bt = bt[bt["ref"] == "DHI"].set_index("model")
    se = pd.read_csv(PROC / "mendoza_prov_by_season.csv")
    se = se[se["ref"] == "DHI"].copy()
    se["season"] = se["season"].replace({"DEF": "DJF"})          # misma rotulación que la Tabla 3 (DJF = diciembre–febrero)
    order = [m for m in bt["RMSE"].sort_values().index.tolist() if m in models]
    nice = {"gompertz (T1)": "Gompertz", "erbs": "Erbs", "reindl2": "Reindl-2", "dirint_v1": "SD 1.0.0 table", "dirint_perez": "DIRINT", "brl_hourly": "BRL",
            "engerer2_60 (expl.)": "Engerer2 (60 min)", "engerer2_1min": "Engerer2 (1 min)"}
    scol = {"DJF": "#D55E00", "MAM": "#009E73", "JJA": "#0072B2", "SON": "#E69F00"}
    fig, ax = plt.subplots(figsize=figsize)
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        x = bt.loc[m]
        ax.errorbar(x.RMSE, y, xerr=[[x.RMSE - x.RMSE_lo], [x.RMSE_hi - x.RMSE]], fmt="D", color="black", ms=5, elinewidth=1.1, capsize=2.5, zorder=5)
        for (s, c), dy in zip(scol.items(), (0.27, 0.09, -0.09, -0.27)):
            v = se[(se["model"] == m) & (se["season"] == s)]["RMSE"]
            if len(v):
                ax.scatter(float(v.iloc[0]), y + dy, color=c, s=18, edgecolors="black", linewidths=0.3, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice[m] for m in reversed(order)])
    ax.set_xlabel(f"DHI RMSE against the measured diffuse ({WM2})")
    ax.xaxis.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="D", ls="", mfc="black", mec="black", ms=5, label="All hours (95 % CI)")] + [
        Line2D([], [], marker="o", ls="", mfc=c, mec="black", mew=0.3, ms=5, label=s) for s, c in scol.items()]
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=7.2, columnspacing=0.9, handletextpad=0.3)
    mf.save(fig, OUT, name)
    return bt, se


def table_external_check(pil, men):
    """Datos de la subsección: RMSE y MBE (IC 95 % por bootstrap de días) por modelo en cada conjunto medido (c20)."""
    order = ["gompertz", "erbs", "reindl2", "dirint_noTd", "dirint_v1", "brl_hourly", "engerer2_1min", "engerer2_60 (expl.)"]
    rows = []
    for stn, label in STN:
        d = pil[stn]
        ser = {k: d["ser"][k] for k in ["gompertz", "erbs", "reindl2", "dirint_noTd", "dirint_v1"]}
        bt = asg.day_bootstrap(d["dni"], ser, d["mask"], d["day"], np.random.default_rng(asg.SEED))
        ok = d["mask"] & np.isfinite(d["dni"])
        for k in ser:
            ok &= np.isfinite(ser[k])
        mm = pd.read_csv(PROC / f"modern_models_bootstrap_{stn}.csv").set_index("model")
        for m in order:
            if m in ser:
                x = bt.set_index("model").loc[m]
                src = "day-block bootstrap, this script"
            else:
                if m not in mm.index:  # Golden es horario: no hay Engerer2 de 1 min
                    continue
                x = mm.loc[m]
                src = "modern_models_bootstrap_%s.csv" % stn
            rows.append(dict(dataset=label.split(" (")[0], variable="DNI", model=m, n_hours=int(x.n_hours), n_days=int(x.n_days), mean_measured=float(d["dni"][ok].mean()),
                             RMSE=x.RMSE, RMSE_lo=x.RMSE_lo, RMSE_hi=x.RMSE_hi, MBE=x.MBE, MBE_lo=x.MBE_lo, MBE_hi=x.MBE_hi,
                             dRMSE_vs_erbs=x.dRMSE_vs_erbs, dRMSE_lo=x.dRMSE_lo, dRMSE_hi=x.dRMSE_hi, source=src))
    bt = pd.read_csv(PROC / "mendoza_prov_bootstrap.csv")
    bt = bt[bt["ref"] == "DHI"].set_index("model")
    key = {"gompertz": "gompertz (T1)", "erbs": "erbs", "reindl2": "reindl2", "dirint_noTd": "dirint_perez", "dirint_v1": "dirint_v1", "brl_hourly": "brl_hourly", "engerer2_1min": "engerer2_1min", "engerer2_60 (expl.)": "engerer2_60 (expl.)"}
    mean_dhi = float(men["P"]["dhi_m"][men["mask"]].mean())
    for m in order:
        x = bt.loc[key[m]]
        rows.append(dict(dataset="Mendoza", variable="DHI", model=m, n_hours=int(x.n_hours), n_days=int(x.n_days), mean_measured=mean_dhi,
                         RMSE=x.RMSE, RMSE_lo=x.RMSE_lo, RMSE_hi=x.RMSE_hi, MBE=x.MBE, MBE_lo=x.MBE_lo, MBE_hi=x.MBE_hi,
                         dRMSE_vs_erbs=x.dRMSE_vs_erbs, dRMSE_lo=x.dRMSE_lo, dRMSE_hi=x.dRMSE_hi, source="mendoza_prov_bootstrap.csv"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "TableE1_external_check_long.csv", index=False)
    sets = list(dict.fromkeys(df["dataset"]))
    dif = []
    for s_ in sets:
        n0 = int(df[(df.dataset == s_) & (df.model == "erbs")].n_hours.iloc[0])
        for m_ in ("brl_hourly", "engerer2_1min", "engerer2_60 (expl.)"):
            r_ = df[(df.dataset == s_) & (df.model == m_)]
            if len(r_) and int(r_.n_hours.iloc[0]) != n0:
                dif.append(f"{s_}, {MLAB[m_]}: {int(r_.n_hours.iloc[0]):,} h frente a {n0:,} h")
    difn = ("la muestra de BRL/Engerer2 difiere en: " + "; ".join(dif) + " (comparación aproximada en esas columnas)." if dif else "las muestras coinciden.")
    hdr = ["Model"] + [f"{s} ({df[df.dataset == s].variable.iloc[0]})" for s in sets]

    def write_table(models, title, intro, stem, delta):
        md = [title, "", intro, "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        for m in models:
            cells = [MLAB[m]]
            for s in sets:
                rr = df[(df.dataset == s) & (df.model == m)]
                if rr.empty:
                    cells.append("n/a (hourly data)")
                    continue
                r = rr.iloc[0]
                c = f"{r.RMSE:.1f} [{r.RMSE_lo:.1f}–{r.RMSE_hi:.1f}] ({r.MBE:+.1f})"
                if delta and m != "erbs":
                    c += f"; Δ vs Erbs {r.dRMSE_vs_erbs:+.1f} [{r.dRMSE_lo:+.1f}, {r.dRMSE_hi:+.1f}]"
                cells.append(c)
            md.append("| " + " | ".join(cells) + " |")
        md.append("| n (hours) | " + " | ".join(f"{int(df[(df.dataset == s) & (df.model == 'erbs')].n_hours.iloc[0]):,}" for s in sets) + " |")
        md.append("| mean measured | " + " | ".join(f"{df[df.dataset == s].mean_measured.iloc[0]:.0f}" for s in sets) + " |")
        (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        try:
            wide = pd.DataFrame([[c.strip() for c in ln.strip("|").split("|")] for ln in md[md.index("| " + " | ".join(hdr) + " |") + 2:]], columns=hdr)
            with pd.ExcelWriter(OUT / f"{stem}.xlsx") as xw:
                wide.to_excel(xw, sheet_name="RMSE_CI_MBE", index=False)
                df[df.model.isin(models)].to_excel(xw, sheet_name="long", index=False)
        except Exception as e:  # noqa: BLE001  (xlsx es opcional; CSV/MD ya escritos)
            print("xlsx no escrito:", e)

    common = ("RMSE en W/m² contra la medición horaria, dominio c20 (cos z ≥ 0.2), con IC 95 % por bootstrap de días [2000 réplicas, semilla 20261005] y sesgo medio (MBE) entre paréntesis. "
              "DNI medido en las cuatro estaciones del piloto; difusa (DHI) medida en Mendoza (el DNI de Mendoza es derivado y no se usa para comparar modelos). "
              "Mendoza: latitud, longitud, hora de reloj y marca de fin de minuto confirmadas por el técnico; altitud supuesta (750 m); piranómetros Eppley 8-48 con banda de sombra propia (datos con la corrección de LeBaron et al. y los filtros de Nollas et al.). ")
    write_table(["gompertz", "erbs", "reindl2", "dirint_noTd", "dirint_v1"], "# Tabla E1: datos del chequeo externo (provisoria)",
                common + "Los cinco modelos comparten la muestra horaria en cada conjunto (remuestreo propio con muestra común de 5 series). BRL y Engerer2 están en la Tabla S5 (suplemento).", "TableE1_external_check", False)
    # --- Tabla S5: UNA muestra común por conjunto (la de todos los modelos corridos en él), de modo que el Erbs de referencia, el n y los días remuestreados
    # son los mismos para todas las filas. Se lee de modern_models_bootstrap_*.csv (estaciones con DNI) y mendoza_prov_bootstrap.csv (DHI de Mendoza).
    s5_models = ["erbs", "gompertz", "brl_hourly", "engerer2_1min", "engerer2_60 (expl.)"]
    key8 = {"erbs": "erbs", "gompertz": "gompertz (T1)", "brl_hourly": "brl_hourly", "engerer2_1min": "engerer2_1min", "engerer2_60 (expl.)": "engerer2_60 (expl.)"}
    rows8 = []
    for stn, label in STN:
        t = pd.read_csv(PROC / f"modern_models_bootstrap_{stn}.csv").set_index("model")
        for m in s5_models:
            if key8[m] not in t.index:
                continue
            x = t.loc[key8[m]]
            rows8.append(dict(dataset=label.split(" (")[0], variable="DNI", model=m, n_hours=int(x.n_hours), n_days=int(x.n_days), mean_measured=float(x.mean_obs),
                              RMSE=x.RMSE, RMSE_lo=x.RMSE_lo, RMSE_hi=x.RMSE_hi, MBE=x.MBE, MBE_lo=x.MBE_lo, MBE_hi=x.MBE_hi, RMSE_erbs_same_hours=float(x.RMSE_erbs_same_hours),
                              dRMSE_vs_erbs=x.dRMSE_vs_erbs, dRMSE_lo=x.dRMSE_lo, dRMSE_hi=x.dRMSE_hi, source=f"modern_models_bootstrap_{stn}.csv"))
    tm = pd.read_csv(PROC / "mendoza_prov_bootstrap.csv")
    tm = tm[tm["ref"] == "DHI"].set_index("model")
    erbs_m = float(tm.loc["erbs", "RMSE"])
    for m in s5_models:
        x = tm.loc[key8[m]]
        rows8.append(dict(dataset="Mendoza", variable="DHI", model=m, n_hours=int(x.n_hours), n_days=int(x.n_days), mean_measured=mean_dhi,
                          RMSE=x.RMSE, RMSE_lo=x.RMSE_lo, RMSE_hi=x.RMSE_hi, MBE=x.MBE, MBE_lo=x.MBE_lo, MBE_hi=x.MBE_hi, RMSE_erbs_same_hours=erbs_m,
                          dRMSE_vs_erbs=x.dRMSE_vs_erbs, dRMSE_lo=x.dRMSE_lo, dRMSE_hi=x.dRMSE_hi, source="mendoza_prov_bootstrap.csv"))
    df8 = pd.DataFrame(rows8)
    # comprobaciones de la estructura pareada: misma muestra en todas las filas de un conjunto; Δ = RMSE(modelo) − RMSE(Erbs de esas horas)
    for s_, g in df8.groupby("dataset", sort=False):
        assert g.n_hours.nunique() == 1 and g.n_days.nunique() == 1, f"S5: la muestra no es común en {s_}"
        assert np.allclose(g.RMSE - g.RMSE_erbs_same_hours, g.dRMSE_vs_erbs, atol=1e-9), f"S5: Δ no es RMSE(modelo) − RMSE(Erbs, mismas horas) en {s_}"
    assert int(men["mask"].sum()) == int(df8[df8.dataset == "Mendoza"].n_hours.iloc[0]), "S5: la muestra de Mendoza no coincide con la máscara común"
    df8.to_csv(OUT / "TableS5_long.csv", index=False)
    sets8 = list(dict.fromkeys(df8["dataset"]))
    hdr8 = ["Model"] + [f"{s} ({df8[df8.dataset == s].variable.iloc[0]})" for s in sets8]
    md = ["# Tabla S5 (suplemento): BRL y Engerer2 contra las mediciones (provisoria)", "",
          common + "**Muestra y contraste.** En cada columna todos los modelos se evalúan sobre las mismas horas (las válidas para todos los modelos de ese conjunto; n al pie), de modo que el Erbs de la tabla es el Erbs de esas horas. "
          "Δ vs Erbs = RMSE(modelo) − RMSE(Erbs) sobre esas horas (negativa = mejor que Erbs); su IC 95 % sale del mismo remuestreo por bloques de días, con los mismos días en cada réplica para el modelo y para Erbs. "
          "Las cifras pueden diferir en 0.1 W/m² de la resta de los RMSE redondeados mostrados, porque Δ se calcula con los valores sin redondear. Los RMSE de Erbs y Gompertz pueden diferir de los de la Tabla E1 y del texto principal, que usan la muestra común de 5 modelos (más horas en Desert Rock: 3,440 frente a 3,371 h; en Golden: 3,555 frente a 3,554 h). "
          "Los modelos comparados con Erbs son BRL, Engerer2 y Gompertz; BRL y Engerer2 con sus parámetros publicados, sin ajustar a estas estaciones (los sitios y períodos del ajuste global de Engerer2 se solapan con esta comparación según la Tabla I de Bright y Engerer, 2019, verificada contra el PDF del artículo el 2026-10-14; la coincidencia exacta de registros no se evaluó, y en Desert Rock se usan datos de SURFRAD). "
          "**Limitaciones:** un año por estación y ninguna estación argentina excepto Mendoza; Engerer2 (1 min) recibe datos de un minuto y los demás medias horarias; los datos de Mendoza fueron filtrados con un procedimiento (Nollas et al., 2023) cuyo filtro 8 usa Engerer3/Engerer2 [CONFIRMAR con el técnico que se aplicó], de modo que el resultado de Engerer2 en Mendoza no es independiente de los datos. "
          "Engerer2 (60 min, exploratory): juego horario tomado del código de los autores del modelo (coincide con la columna de 1 hora de la Tabla II del artículo, verificado contra el PDF el 2026-10-14); el artículo describe los promedios como medias de los t minutos siguientes y no indica otra convención de marca de tiempo del ajuste; "
          "se aplica a las medias horarias y se informa para que la comparación no dependa de un solo juego de parámetros.", "",
          "| " + " | ".join(hdr8) + " |", "|" + "---|" * len(hdr8)]
    for m in s5_models:
        cells = [MLAB[m] + (" (reference)" if m == "erbs" else "")]
        for s in sets8:
            rr = df8[(df8.dataset == s) & (df8.model == m)]
            if rr.empty:
                cells.append("n/a (hourly data)")
                continue
            r = rr.iloc[0]
            c = f"{r.RMSE:.1f} [{r.RMSE_lo:.1f}–{r.RMSE_hi:.1f}] ({r.MBE:+.1f})"
            if m != "erbs":
                c += f"; Δ vs Erbs {r.dRMSE_vs_erbs:+.1f} [{r.dRMSE_lo:+.1f}, {r.dRMSE_hi:+.1f}]"
            cells.append(c)
        md.append("| " + " | ".join(cells) + " |")
    md.append("| n (hours; days), same for all models | " + " | ".join(f"{int(df8[df8.dataset == s].n_hours.iloc[0]):,}; {int(df8[df8.dataset == s].n_days.iloc[0]):,}" for s in sets8) + " |")
    md.append("| mean measured (same hours) | " + " | ".join(f"{df8[df8.dataset == s].mean_measured.iloc[0]:.0f}" for s in sets8) + " |")
    (OUT / "TableS5_modern_models_supplement.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        with pd.ExcelWriter(OUT / "TableS5_modern_models_supplement.xlsx") as xw:
            df8.to_excel(xw, sheet_name="long", index=False)
    except Exception as e:  # noqa: BLE001
        print("xlsx no escrito:", e)
    return df


def main() -> int:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    print("Cargando piloto y Mendoza …", flush=True)
    pil = load_pilot()
    men = load_mendoza()
    print("Figuras →", OUT)
    s1 = e1_gompertz(pil, men)
    d2 = e2_ranking(pil, men)
    d3 = e3_distance(pil, men)
    bt, se = e4_mendoza(men, MAIN_E4, "FigE4_Mendoza_DHI_by_model_and_season", (5.2, 3.2))
    bt_s, _ = e4_mendoza(men, SUPP_E4, "FigS4_Mendoza_DHI_all_models_supplement", (5.2, 3.8))
    tab = table_external_check(pil, men)
    print(tab.groupby("dataset").n_hours.first().to_string())
    d2.to_csv(OUT / "external_check_E2_ranking.csv", index=False)
    d3.to_csv(OUT / "external_check_E3_distances.csv", index=False)
    L = ["# Datos para las leyendas de las figuras del chequeo externo — NO son leyendas", "",
         "Generado por `scripts/make_figures_external_check.py`. Son hechos de cada figura; las leyendas las redacta el autor. **Provisorio:** Mendoza con latitud, longitud, hora de reloj y marca de fin de minuto confirmadas por el técnico (2026-10-06), altitud supuesta (750 m, no informada), instrumentos Eppley 8-48 con banda de sombra de construcción propia, y datos filtrados con un procedimiento (Nollas et al., 2023) cuyo filtro 8 usa Engerer2/Engerer3, así que Engerer2 no es independiente de estos datos; "
         "el piloto son 4 estaciones de un año, ninguna argentina. Dominio c20 (cos z ≥ 0.2).", "",
         "- **Fig. E1:** DNI de la partición Gompertz (el generador de la referencia IWEC2) contra el DNI **medido**, medias horarias (hexbin, escala log) en cuatro estaciones medidas, y DHI de la partición Gompertz contra la difusa **medida** en Mendoza. "
         + "; ".join(f"{k}: R² {v['R2']:.3f}, RMSE {v['RMSE']:.0f}, MBE {v['MBE']:+.0f} W/m² (n = {v['n']:,})" for k, v in s1.items()) + ".",
         "- **Fig. E2:** diferencia de RMSE respecto de Erbs (modelo − Erbs; negativa = mejor que Erbs) para Reindl-2, DIRINT (Perez, sin punto de rocío) y DIRINT v1.0.0. Fila superior: contra IWEC2, media de 39 estaciones, IC 95 % por bootstrap de estaciones; filas siguientes: contra las mediciones (DNI en las 4 estaciones, DHI en Mendoza), IC 95 % por bootstrap de días (2000 réplicas, semilla 20261005). "
         + "Contra IWEC2: " + "; ".join(f"{MLAB[m]} {d2[(d2.dataset.str.startswith('Against IWEC2')) & (d2.model == m)].d.iloc[0]:+.1f}" for m in ["reindl2", "dirint_noTd", "dirint_v1"]) + " W/m². Contra la medición, Reindl-2 − Erbs: "
         + "; ".join(f"{s.split(': ')[1]} {d2[(d2.dataset == s) & (d2.model == 'reindl2')].d.iloc[0]:+.1f}" for s in d2.dataset.unique() if 'measured' in s) + " W/m².",
         "- **Fig. E3:** cada punto es un modelo en una estación; eje x = RMSD entre el modelo y la medición; eje y = RMSD entre el modelo y la partición Gompertz; por debajo de la diagonal el modelo se parece más a Gompertz que a lo medido. "
         f"Panel (a): DNI medido, 4 estaciones; panel (b): difusa medida de Mendoza (Erbs, Reindl-2, DIRINT y DIRINT v1.0.0; BRL y Engerer2 están en el suplemento). Fracción de puntos por debajo de la diagonal: (a) {(d3[d3.set == 'pilot DNI'].ratio < 1).mean():.0%}; (b) {(d3[d3.set == 'Mendoza DHI'].ratio < 1).mean():.0%}.",
         "- **Fig. E4:** Mendoza (2022–2025), RMSE de la difusa por modelo (rombo negro: todas las horas, IC 95 % por bootstrap de días) y por estación del año (DJF = dic–feb, MAM, JJA, SON); modelos ordenados por RMSE total (Gompertz, Erbs, Reindl-2, DIRINT y DIRINT v1.0.0).",
         "- **Fig. S4 (suplemento):** igual que la Fig. E4 con BRL y Engerer2 (1 min) agregados; RMSE de la difusa en Mendoza: "
         + "; ".join(f"{k} {bt_s.loc[m, 'RMSE']:.1f}" for k, m in (("BRL", "brl_hourly"), ("Engerer2 (1 min)", "engerer2_1min"), ("Erbs", "erbs"), ("Gompertz", "gompertz (T1)"))) + " W/m². El resultado de Engerer2 no es independiente de los datos (filtro 8 de Nollas et al.); Engerer2 horario (exploratorio) no se muestra.",
         "- **Tabla S5 (suplemento):** `TableS5_modern_models_supplement.md/.xlsx`: BRL y Engerer2 (1 min) contra las mediciones, con la diferencia de RMSE respecto de Erbs.", ""]
    (OUT / "FIGURAS_datos_para_leyendas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("escrito", OUT / "FIGURAS_datos_para_leyendas.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
