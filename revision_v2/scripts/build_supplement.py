"""build_supplement.py — arma el material suplementario de la revisión (Tablas S1–S12 y Figs. S1–S4) en inglés, a partir de los CSV del análisis.

Nada se tipea a mano: todas las cifras salen de data/processed/, figures/ y TableS5_long.csv. El documento reemplaza al suplemento enviado (que tenía S1 y S2):
la antigua Tabla S2 (estadísticos por estación) es ahora la Tabla S12; el resto es nuevo. La numeración sigue las citas del manuscrito revisado.
Salidas (carpeta supplement/): Supplementary_Material_v2.docx, Supplementary_Tables_v2.xlsx, SUPLEMENTO_notas.md (qué contiene cada tabla y de dónde sale).
Las leyendas y notas de este documento son texto redactado con asistencia de IA a partir de los datos y de los scripts (véase la declaración de uso de IA del manuscrito).
Uso (desde revision_v2, venv activo): python scripts/build_supplement.py
"""
from __future__ import annotations

import re
import struct
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
OUT = ROOT / "supplement"
MIN = "−"
MODELS4 = [("reindl2", "Reindl-2"), ("erbs", "Erbs"), ("dirint_std_noTd", "DIRINT"), ("dirint_v1", "SD 1.0.0 table")]
BLOCKS: list = []          # (tipo, contenido)
SHEETS: dict = {}          # hoja de Excel -> DataFrame


# ---------------------------------------------------------------------------------------------------------------------------- utilidades
def f(x, nd=1):
    return f"{x:.{nd}f}".replace("-", MIN)


def sg(x, nd=1):
    return ("+" if x >= 0 else MIN) + f"{abs(x):.{nd}f}"


def ci(lo, hi, nd=1):
    return f"[{f(lo, nd)}, {f(hi, nd)}]"


SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


def norm(x: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().upper())


def fixmoji(s: str) -> str:
    s = str(s)
    return s.replace("Ã¡", "á").replace("ã¡", "á").replace("�", "á") if ("ã" in s or "Ã" in s or "�" in s) else s


ES = {"las 39": "All 39", "baja altitud (≤ 500 m)": "Low elevation (≤ 500 m)", "más de 500 m": "Above 500 m", "clase Cfa (todas son de baja altitud)": "Class Cfa (all low elevation)",
      "las 39 reponderadas por altitud a la composición de las 54 (44 de ≤ 500 m, 10 de más)": "The 39 reweighted by elevation to the composition of the 54 (44 ≤ 500 m, 10 above)",
      "las 39 reponderadas por altitud a la composición de las 54": "The 39 reweighted by elevation to the composition of the 54",
      "54 de la lista de IWEC2": "54 stations of the IWEC2 list", "39 analizadas": "39 analyzed", "15 no analizadas": "15 not analyzed",
      "Reindl-2 menos Erbs": "Reindl-2 minus Erbs", "Erbs menos DIRINT (Perez et al., 1992)": "Erbs minus DIRINT", "DIRINT (Perez et al., 1992) menos DIRINT v1.0.0": "DIRINT minus SD 1.0.0 table",
      "Reindl-2 menos DIRINT v1.0.0": "Reindl-2 minus SD 1.0.0 table", "sesgo medio de DIRINT v1.0.0 (MBE)": "Mean bias (MBE) of SD 1.0.0 table", "sí": "yes", "no": "no"}


def en(s):
    s = str(s)
    if s in ES:
        return ES[s]
    s = re.sub(r"(\d+) de (\d+)", r"\1 of \2", s)
    s = s.replace("DIRINT (Perez et al., 1992)", "DIRINT").replace("DIRINT v1.0.0", "SD 1.0.0 table").replace("DIRINT Perez", "DIRINT")
    return s.replace("-", MIN) if re.fullmatch(r"[-+\d.,\s×\[\]]+", s) else s


def station_map(cities, names):
    """names (Tabla S1 enviada) -> clave de ciudad del benchmark, con el mismo emparejamiento por prefijo que make_figures_v2."""
    nb = {norm(c): c for c in cities}
    out = {}
    for n in names:
        hit = [c for k, c in nb.items() if k.startswith(norm(n)) or norm(n).startswith(k)]
        assert len(hit) == 1, f"no se pudo emparejar {n}: {hit}"
        out[n] = hit[0]
    return out


def H(t):
    BLOCKS.append(("h1", t))


def H2(t):
    BLOCKS.append(("h2", t))


def P(t):
    BLOCKS.append(("p", t))


def TAB(name, cap, head, rows, note=None, widths=None, sheet=None):
    BLOCKS.append(("table", (cap, head, rows, note, widths)))
    SHEETS[sheet or name] = pd.DataFrame(rows, columns=head)


def FIGB(png, cap):
    BLOCKS.append(("fig", (Path(png), cap)))


# ---------------------------------------------------------------------------------------------------------------------------- datos base
bench = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
CIT = list(bench.city.unique())
diag = pd.read_csv(PROC / "r14_station_diag_c20.csv").set_index("city")
ash = pd.read_csv(PROC / "koppen_ashrae_vs_s1.csv").set_index("city")


# ---------------------------------------------------------------------------------------------------------------------------- S1
def table_s1():
    t = pd.read_csv(FIG / "revised_c20_ashrae" / "TableS1_sites_with_number.csv", dtype=str, encoding="utf-8")
    t["City"] = t["City"].map(fixmoji)
    mp = station_map(CIT, t["City"])
    c20 = bench[(bench.domain == "c20") & (bench.model == "erbs")].set_index("city")["n"]
    rows = []
    for _, r in t.iterrows():
        key = mp[r.City]
        nonie = r.Source != "IWEC2"
        rows.append([r["No."] + ("†" if nonie else ""), ("not available" if nonie else str(r.WMO).replace(".0", "")), r.City, r.Lat_S, r.Lon_W, r.Elev_m, r.TZ.replace("-", MIN), r.Source,
                     "n/a" if nonie else r.Koppen_ASHRAE, r.Koppen, r.IRAM, f"{int(c20[key]):,}"])
    TAB("S1", "Table S1. Characteristics of the 41 study sites. No.: station number used in the figures († = the two files that are not in the IWEC2 list, analyzed separately). "
        "WMO: station code; Lat and Lon in decimal degrees (S and W positive); Elev: m a.s.l.; UTC offset in hours; Source: origin of the file; K–G (IWEC2 list): Köppen–Geiger class of the IWEC2 station list, used in the paper "
        "(n/a for files not in the list); K–G (first version): class used in the first version of this work; IRAM: thermal zone according to IRAM 11603; N hours: hours with cos θz ≥ 0.2 used in the main analysis. "
        "No WMO code was verified for the two files that are not in the IWEC2 list, so none is reported (“not available”): the header of the Andalgalá file gives the source “MN7” (read here as Meteonorm 7.x) and the placeholder station number 999, and that of the General Pico Airp. file (TMY2 format) gives the source “TMY2-87532” and no station number.",
        ["No.", "WMO", "City", "Lat (°S)", "Lon (°W)", "Elev (m)", "UTC offset", "Source", "K–G (IWEC2 list)", "K–G (first version)", "IRAM", "N hours"], rows,
        note="Source: data/processed/koppen_ashrae_vs_s1.csv, figures/revised_c20_ashrae/TableS1_sites_with_number.csv, benchmark_v2_per_station.csv.", sheet="S1_sites")
    return mp


# ---------------------------------------------------------------------------------------------------------------------------- S2
def table_s2():
    for comp in ("DNI", "DHI"):
        d = pd.read_csv(FIG / "supplement_r17" / f"TableS_Kt_0p1_{comp}_c20.csv", encoding="utf-8")
        d.columns = ["Kt", "n", "pct", "reindl2", "erbs", "dirint", "sd100"]
        rows = [[str(r.Kt).replace("<", "< ").replace("≥", "≥ "), f"{r.n}", f"{r.pct:.1f}", *[str(r[c]).replace("-0.0)", "0.0)").replace("-", MIN) for c in ("reindl2", "erbs", "dirint", "sd100")]] for _, r in d.iterrows()]
        TAB(f"S2{comp}", f"Table S2{'a' if comp == 'DNI' else 'b'}. {comp} RMSE (MBE) in W/m² against the IWEC2 reference by interval of the hourly clearness index Kₜ of 0.1 "
            f"(39 IWEC2 stations; hours with cos θz ≥ 0.2, n = 141,410). Values are RMSE with the MBE in parentheses. Intervals with fewer than 1% of the hours are shown for completeness; the conclusions of the main text do not depend on the thresholds "
            f"(Table 6 gives the three classes used there).", ["Kₜ interval", "Hours (n)", "% of hours", "Reindl-2", "Erbs", "DIRINT", "SD 1.0.0 table"], rows, sheet=f"S2{comp}_Kt_0p1")


# ---------------------------------------------------------------------------------------------------------------------------- S3
def table_s3():
    rows = []
    lab = {"res_mix": "Mixed: Reindl-2 DNI + Erbs DHI", "res_reindl": "Reindl-2 alone", "res_erbs": "Erbs alone"}
    for dom, dn in (("c20", "cos θz ≥ 0.2"), ("all", "all daytime hours")):
        d = pd.read_csv(PROC / f"reanalysis_closure_{dom}.csv")
        for _, r in d.iterrows():
            if r.case == "res_mix":
                cells = [f(r.MBE), f(r.RMSE), f(r.p50_abs), f(r.p95_abs), f(r.p99_abs)] + [f(100 * r[c]) for c in ("gt_5Wm2", "gt_10Wm2", "gt_20Wm2", "gt_50Wm2", "gt_2pct", "gt_5pct", "gt_10pct")]
            else:
                cells = ["≈ 0"] * 5 + ["0.0"] * 7
                assert abs(r.RMSE) < 1e-12 and r.gt_5Wm2 == 0, "closure of a single model should be exact"
            rows.append([dn, lab[r.case], f"{int(r.n):,}"] + cells)
    TAB("S3", "Table S3. Radiative-closure residual, GHI reconstructed as DNI·cos θz + DHI minus the GHI of the file, for the mixed model (DNI of Reindl-2 with the DHI of Erbs) and for each model alone, "
        "39 IWEC2 stations. MBE, RMSE and percentiles of the absolute residual in W/m²; the last seven columns give the percentage of hours in which the absolute residual exceeds the threshold (W/m² or % of the GHI). "
        "Reindl-2 alone and Erbs alone close to rounding error (of the order of 10⁻¹⁴ W/m²) because each obtains DNI from its DHI by closure.",
        ["Domain", "Case", "n", "MBE", "RMSE", "P50 |r|", "P95 |r|", "P99 |r|", "> 5 W/m²", "> 10 W/m²", "> 20 W/m²", "> 50 W/m²", "> 2% GHI", "> 5% GHI", "> 10% GHI"], rows, sheet="S3_closure")


# ---------------------------------------------------------------------------------------------------------------------------- S4
def table_s4():
    d = pd.read_csv(PROC / "sample_rep_composition.csv")
    rows = [[en(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2]).replace("�", "–").replace("-", "–") if "–" not in str(r.iloc[2]) else str(r.iloc[2]), f"{float(r.iloc[3]):.0f}", str(r.iloc[4]), str(r.iloc[5]), str(r.iloc[6])] for _, r in d.iterrows()]
    TAB("S4a", "Table S4a. Composition of the 39 IWEC2 files analyzed in relation to the 54 Argentine stations of the IWEC2 station list. Latitude range in °S; elevation in m; data-quality grades A/B/C are those of the IWEC2 station list.",
        ["Group", "n", "Latitude range (°S)", "Mean elevation (m)", "Maximum elevation (m)", "Stations above 500 m", "Quality A / B / C"], rows, sheet="S4a_composition")
    d = pd.read_csv(PROC / "sample_rep_classes.csv")
    rows = [[str(r.iloc[0]), str(r.iloc[1]), str(r.iloc[2]), str(r.iloc[3])] for _, r in d.iterrows()]
    TAB("S4b", "Table S4b. Köppen–Geiger classes (IWEC2 station list) of the 54 stations: stations not analyzed, analyzed and total.", ["Class", "Not analyzed", "Analyzed", "Total"], rows, sheet="S4b_classes")
    d = pd.read_csv(PROC / "sample_rep_subsets_rmse.csv")
    rows = [[en(r.iloc[0]), str(r.iloc[1]), *[f(float(r.iloc[k])) for k in (2, 3, 4, 5)], en(r.iloc[6])] for _, r in d.iterrows()]
    TAB("S4c", "Table S4c. Mean DNI RMSE (W/m²) against the IWEC2 reference by subset of the 39 stations (hours with cos θz ≥ 0.2) and order of the models from lowest to highest RMSE.",
        ["Subset", "n", "Reindl-2", "Erbs", "DIRINT", "SD 1.0.0 table", "Order (lowest to highest RMSE)"], rows, sheet="S4c_subsets")
    d = pd.read_csv(PROC / "sample_rep_subsets_diff.csv")
    rows = [[en(r.iloc[0]), str(r.iloc[1]), f(float(r.iloc[2])), en(r.iloc[3]), en(r.iloc[4]), f(float(r.iloc[5])), en(r.iloc[6])] for _, r in d.iterrows()]
    TAB("S4d", "Table S4d. Difference of the DNI RMSE between Reindl-2 and Erbs (negative: Reindl-2 closer to the reference) and mean bias of the SD 1.0.0 table by subset; 95% interval by bootstrap of stations.",
        ["Subset", "n", "Reindl-2 minus Erbs, mean (W/m²)", "95% interval", "Stations with Reindl-2 < Erbs", "Mean bias of SD 1.0.0 table (W/m²)", "Stations with positive bias"], rows, sheet="S4d_differences")
    d = pd.read_csv(PROC / "sample_rep_tipping.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([en(r.iloc[0]), f(float(r.iloc[1])), f(float(r.iloc[2])), f(float(r.iloc[3])), en(r.iloc[4]) if str(r.iloc[4]) != "nan" else "", en(r.iloc[5]) if str(r.iloc[5]) != "nan" else ""])
    TAB("S4e", "Table S4e. Sensitivity of the differences to the 15 stations not analyzed: mean (W/m²) that the 15 stations would need to have for the mean over the 54 to cancel the mean over the 39, compared with the most extreme value "
        "observed at one of the 39. This is a sensitivity analysis and not a test, because the 15 stations were not analyzed.",
        ["Comparison", "Mean over the 39", "Mean the 15 would need (to cancel it)", "Most extreme value observed at one of the 39 (opposite direction)", "Is the tipping value outside the observed range?", "Stations with the sign opposite to the mean"], rows, sheet="S4e_tipping")


# ---------------------------------------------------------------------------------------------------------------------------- S5
def table_s5():
    d = pd.read_csv(FIG / "external_check" / "TableS5_long.csv")
    nm = {"erbs": "Erbs (reference)", "gompertz": "Gompertz split", "brl_hourly": "BRL", "engerer2_1min": "Engerer2 (1 min)", "engerer2_60 (expl.)": "Engerer2 (60 min, exploratory)"}
    rows = []
    for ds in dict.fromkeys(d.dataset):
        g = d[d.dataset == ds]
        for _, r in g.iterrows():
            rows.append([f"{ds} ({r.variable})", nm[r.model], f"{int(r.n_hours):,} ({int(r.n_days)})", f"{f(r.RMSE)} {ci(r.RMSE_lo, r.RMSE_hi)}", sg(r.MBE), f(r.RMSE_erbs_same_hours),
                         "—" if r.model == "erbs" else f"{sg(r.dRMSE_vs_erbs)} {ci(r.dRMSE_lo, r.dRMSE_hi)}"])
    TAB("S5", "Table S5. BRL and Engerer2 against the measurements, with the Erbs model of the same hours. For each station all models are evaluated on the same hours (those valid for all the models of that set; n and number of days in the third column), "
        "so the Erbs RMSE in the sixth column is the Erbs of the sample of that row. The difference with Erbs is RMSE(model) − RMSE(Erbs) on those hours (negative: closer to the measurements than Erbs); its 95% interval comes from a day-block bootstrap (2000 replicates) "
        "in which the same days are resampled for the model and for Erbs; the pairing keeps the comparison within the retained hours and does not make the results independent of the coverage and availability criteria that retain them. The difference is computed from unrounded values and can differ by 0.1 W/m² from the difference of the rounded RMSE shown. The RMSE of Erbs and of the Gompertz split can differ slightly from those of Table 7, which uses the common sample "
        "of five models (3,440 h at Desert Rock and 3,555 h at Golden). Measured DNI at the four stations; measured DHI at Mendoza. Hours with cos θz ≥ 0.2. BRL and Engerer2 were run with their published parameters, without adjustment to these stations; "
        "the sites and periods of the global parameterization of Engerer2 (Bright and Engerer, 2019) overlap with this comparison (its Table I lists Petrolina, São Martinho da Serra and Desert Rock, with BSRN data from January 2013 to January 2018); exact record overlap was not assessed, and the Desert Rock data used here are from SURFRAD. "
        "The hourly set of Engerer2 comes from the code of the authors of the model and agrees with the 1-hour column of their Table II; Bright and Engerer describe their averages as means of the following t minutes of the one-minute data, and the time-stamp convention of the fit is not stated otherwise. "
        "The Mendoza data were filtered with a procedure whose test 8 uses the Engerer models, so the Engerer2 result at Mendoza is not independent of the data. One year per station; no station with measured DNI in Argentina.",
        ["Set (variable)", "Model", "n hours (days)", "RMSE [95% CI] (W/m²)", "MBE (W/m²)", "RMSE of Erbs, same hours", "Difference with Erbs [95% CI] (W/m²)"], rows, sheet="S5_modern_models")


# ---------------------------------------------------------------------------------------------------------------------------- S6
def table_s6():
    d = pd.read_csv(PROC / "r14_proxy_corr_c20.csv")
    prox = [("kt_mean", "Mean clearness index K̄ₜ"), ("f_kt065", "Fraction of hours with Kₜ > 0.65"), ("f_ktp08", "Fraction of hours with Kₜ′ ≥ 0.8"), ("am_median", "Median relative air mass")]
    rows = []
    for k, lab in prox:
        r = [lab]
        for mk, _ in MODELS4:
            x = d[(d.proxy == k) & (d.model == mk)].iloc[0]
            r.append(f"{sg(x.rho, 2)} (p = {sci(x.p)})")
        rows.append(r)
    TAB("S6", "Table S6. Spearman rank correlation (ρ, with its p-value) between the station-level DNI mean bias error against IWEC2 and four descriptors of clearness and geometry, across the 39 IWEC2 stations "
        "(hours with cos θz ≥ 0.2). The p-values are not adjusted for multiple comparisons; the table is exploratory.", ["Descriptor"] + [n for _, n in MODELS4], rows, sheet="S6_spearman")


# ---------------------------------------------------------------------------------------------------------------------------- S7
def table_s7():
    c = bench[(bench.domain == "c20") & (bench.source != "IWEC2")]
    names = {"MN7": "Andalgalá (Meteonorm 7.x)", "TMY2-87532": "General Pico Airp. (TMY2 format)"}
    rows = []
    for src, nm in names.items():
        for mk, mn in MODELS4:
            r = c[(c.source == src) & (c.model == mk)].iloc[0]
            rows.append([nm, mn, f"{int(r.n):,}", f(r.DNI_RMSE), sg(r.DNI_MBE), f(r.DNI_nRMSE), f(r.DNI_R2, 3), f(r.DHI_RMSE), sg(r.DHI_MBE), f(r.DHI_R2, 3)])
    TAB("S7a", "Table S7a. Results at the two files that are not in the IWEC2 list (hours with cos θz ≥ 0.2), shown separately as two descriptive cases. They do not follow the Gompertz split of IWEC2 and their provenance cannot be inferred from two files. "
        "RMSE and MBE in W/m²; nRMSE in %.", ["File", "Model", "n", "DNI RMSE", "DNI MBE", "DNI nRMSE", "DNI R²", "DHI RMSE", "DHI MBE", "DHI R²"], rows, sheet="S7a_non_IWEC2")
    s = pd.read_csv(PROC / "reanalysis_sensitivity_non_iwec2_c20.csv").set_index("model")
    rows = []
    for mk, mn in MODELS4:
        r = s.loc[mk]
        rows.append([mn, f(r.R2_39, 3), f(r.R2_41, 3), f(r.RMSE_39), f(r.RMSE_41), sg(r.d_RMSE, 2)])
    TAB("S7b", "Table S7b. Sensitivity of the station means to the two non-IWEC2 files: DNI R² and RMSE (W/m²) averaged over the 39 IWEC2 stations and over the 41 files, and the change of the mean RMSE. The order of the four models does not change.",
        ["Model", "R², 39 stations", "R², 41 files", "RMSE, 39 stations", "RMSE, 41 files", "Change of the mean RMSE"], rows, sheet="S7b_sensitivity")


# ---------------------------------------------------------------------------------------------------------------------------- S8
def table_s8():
    c = bench[(bench.domain == "all") & (bench.source == "IWEC2")]
    assert c.city.nunique() == 39
    rows = []
    for comp in ("DNI", "DHI"):
        for mk, mn in MODELS4:
            s = c[c.model == mk]
            assert len(s) == 39
            rows.append([comp, mn, f"{s[comp + '_RMSE'].mean():.1f} ± {s[comp + '_RMSE'].std():.1f}", f"{sg(s[comp + '_MBE'].mean())} ± {s[comp + '_MBE'].std():.1f}",
                         f"{s[comp + '_nRMSE'].mean():.1f} ± {s[comp + '_nRMSE'].std():.1f}", f"{s[comp + '_R2'].mean():.3f} ± {s[comp + '_R2'].std():.3f}"])
    n_all = int(c[c.model == "erbs"].n.sum())
    TAB("S8", f"Table S8. Global statistics of the consistency of the models with the IWEC2 reference in the sensitivity domain \"all\" (hours with GHI ≥ 10 W/m² and cos θz > 0.0174; n = {n_all:,}); mean ± SD across the 39 IWEC2 stations. "
        "The main text uses the domain cos θz ≥ 0.2 (Table 2). RMSE and MBE in W/m²; nRMSE in %.", ["Component", "Model", "RMSE", "MBE", "nRMSE", "R²"], rows, sheet="S8_domain_all")


# ---------------------------------------------------------------------------------------------------------------------------- S9
def table_s9():
    a = pd.concat([pd.read_csv(PROC / "controlled_ref_effect.csv"), pd.read_csv(PROC / "controlled_ref_effect_mendoza.csv")])
    cl = {"all": "All hours", "<0.3": "Kₜ < 0.3", "0.3-0.7": "0.3 ≤ Kₜ < 0.7", ">=0.7": "Kₜ ≥ 0.7", "0.7-0.9": "0.7 ≤ Kₜ < 0.9"}
    sn = {"PTR": "Petrolina", "SMS": "São Martinho da Serra", "DRA": "Desert Rock", "GOLDEN": "Golden", "MENDOZA": "Mendoza (DHI)"}
    rows = []
    for stn in ("PTR", "SMS", "DRA", "GOLDEN", "MENDOZA"):
        for ref in ("medido", "Gompertz"):
            for k in cl:
                g = a[(a.estacion == stn) & (a.ref == ref) & (a.clase == k)]
                if g.empty:
                    continue
                g = g.set_index("modelo")
                e = g.loc["erbs"]
                r = [sn[stn], "Measurement" if ref == "medido" else "Gompertz split", cl[k], f"{int(e.n):,} ({e.pct:.1f})", f(e.RMSE)]
                for mk in ("reindl2", "dirint", "sd100"):
                    x = g.loc[mk]
                    r.append(f"{f(x.RMSE)}; {sg(x.d_vs_erbs)} {ci(x.d_lo, x.d_hi)}")
                rows.append(r)
    TAB("S9a", "Table S9a. Controlled comparison of the effect of the reference, by station, reference and class of the hourly clearness index. In each station the same hours (cos θz ≥ 0.2, common sample), the same GHI, the same geometry and the same models are evaluated "
        "against the measured value and against the Gompertz split computed from that same GHI and geometry. Columns: n of hours (% of the sample), RMSE of Erbs (W/m²), and RMSE of the other models with the difference with Erbs [95% day-block bootstrap interval] "
        "(negative: closer than Erbs). At Mendoza the variable is DHI.", ["Station", "Reference", "Class", "n (%)", "RMSE Erbs", "Reindl-2: RMSE; difference with Erbs", "DIRINT: RMSE; difference with Erbs", "SD 1.0.0 table: RMSE; difference with Erbs"], rows, sheet="S9a_by_class")
    p = pd.read_csv(PROC / "controlled_ref_effect_paired.csv")
    mn = {"reindl2": "Reindl-2", "dirint": "DIRINT", "sd100": "SD 1.0.0 table"}
    rows = []
    for stn in ("PTR", "SMS", "DRA", "GOLDEN", "MENDOZA"):
        for k in ("all", "<0.3", "0.3-0.7", ">=0.7"):
            g = p[(p.estacion == stn) & (p.clase == k)].set_index("modelo")
            rows.append([sn[stn], cl[k]] + [f"{sg(g.loc[m].d_ref)} {ci(g.loc[m].d_ref_lo, g.loc[m].d_ref_hi)}" for m in mn])
    TAB("S9b", "Table S9b. Effect of replacing the measurement by the Gompertz split as reference on the difference of each model with Erbs: Δref = [RMSE(M,G) − RMSE(E,G)] − [RMSE(M,O) − RMSE(E,O)], with M the model, E = Erbs, G the Gompertz split and O the observation, in W/m²; "
        "95% interval with the same resampled days for the four RMSE in each replicate. A positive value means that the reference moves the difference against the model.",
        ["Station", "Class", "Reindl-2", "DIRINT", "SD 1.0.0 table"], rows, sheet="S9b_delta_ref")
    w = pd.read_csv(PROC / "controlled_ref_effect_reweight.csv").set_index(["estacion", "ref"])
    wc = pd.read_csv(PROC / "controlled_ref_effect_reweight_ci.csv")
    rows = []
    for _, r in wc.iterrows():
        sh = w.loc[(r.estacion, r.ref), "pct_Kt>=0.7_propio"] if (r.estacion, r.ref) in w.index else np.nan
        own = w.loc[(r.estacion, r.ref), "d_reindl_erbs_mezcla_propia"] if (r.estacion, r.ref) in w.index else np.nan
        rows.append([sn[r.estacion], "Measurement" if r.ref == "medido" else "Gompertz split", "" if np.isnan(own) else sg(own), sg(r.d_reindl_erbs_mezcla_IWEC2) + " " + ci(r.lo, r.hi), "" if np.isnan(sh) else f(sh)])
    TAB("S9c", "Table S9c. Difference of the RMSE between Reindl-2 and Erbs with the sky mix of the station and with the sky mix of IWEC2 (frequencies of the three classes 0.160, 0.733 and 0.107, treated as fixed), W/m². "
        "For the IWEC2 mix the days were resampled and the class errors recomputed in each replicate (2000 replicates); the intervals do not include the uncertainty of the weights. "
        "The last column is the percentage of hours with Kₜ ≥ 0.7 in the sample of the station (blank at Mendoza, where the table was not computed for the own sky mix).",
        ["Station", "Reference", "With the own sky mix", "With the IWEC2 sky mix [95% interval]", "% of hours with Kₜ ≥ 0.7"], rows, sheet="S9c_reweighting")


# ---------------------------------------------------------------------------------------------------------------------------- S10 / S11
def table_s10():
    d = pd.read_csv(PROC / "dirint_dewpoint_anchor.csv")
    nm = {"PTR": "Petrolina", "SMS": "São Martinho da Serra", "DRA": "Desert Rock", "GOLDEN": "Golden"}
    rows = [[nm[r.station], f"{int(r.n_hours):,} ({int(r.n_days)})", f"{f(r.RMSE_sin_td)} {ci(r.RMSE_sin_td_lo, r.RMSE_sin_td_hi)}", f"{f(r.RMSE_con_td)} {ci(r.RMSE_con_td_lo, r.RMSE_con_td_hi)}",
             f"{sg(r.dRMSE_con_menos_sin)} {ci(r.dRMSE_lo, r.dRMSE_hi)}", sg(r.MBE_sin_td), sg(r.MBE_con_td)] for _, r in d.iterrows()]
    TAB("S10", "Table S10. DIRINT (Perez et al., 1992, as implemented in pvlib) with and without dew point against the measured DNI. Same hours for both variants (cos θz ≥ 0.2); n hours (days) in the second column. "
        "The dew point was derived from the air temperature and the relative humidity (Magnus formula with the coefficients of the WMO) at Petrolina, São Martinho da Serra and Desert Rock, and measured at Golden. "
        "The difference is RMSE(with) − RMSE(without) with its 95% interval from a day-block bootstrap (2000 replicates) on paired samples; a positive value means a larger error with the dew point. "
        "The variant without dew point was adopted after seeing this result, so the intervals reported for DIRINT in the main text do not incorporate the uncertainty of that choice. RMSE and MBE in W/m².",
        ["Station", "n hours (days)", "RMSE without dew point [95% CI]", "RMSE with dew point [95% CI]", "Difference (with − without) [95% CI]", "MBE without", "MBE with"], rows, sheet="S10_dirint_dewpoint")


def table_s11():
    d = pd.read_csv(PROC / "dirint_segments_check.csv")
    d["city"] = d["city"].map(fixmoji)
    rows = [[c, f"{int(r.n_c20):,}", str(int(r.tramos_por_paso)), str(int(r.tramos_por_mes)), str(int(r.n_dif_paso)), f(r.max_abs_paso, 2), str(int(r.n_dif_mes)), f(r.max_abs_mes, 2)] for c, (_, r) in zip(d.city, d.iterrows())]
    TAB("S11a", "Table S11a. Effect of splitting the hourly series into continuous segments on the DNI of DIRINT in the 41 files (hours with cos θz ≥ 0.2). The benchmark applies pvlib to the 8760 rows of each file, and pvlib 0.16.1 computes ΔKₜ′ from the neighboring rows without looking at the clock; "
        "the software v1.1.0 splits the series where the time step is not exactly 1 h. Columns: number of segments when splitting at every step different from 1 h and at every change of month; number of hours in which the DNI differs from the continuous call "
        "(difference > 10⁻⁹ W/m²) and maximum absolute difference (W/m²) for each way of splitting.",
        ["File", "Hours (n)", "Segments (time step)", "Segments (month)", "Hours that differ (time step)", "Max |diff| (time step)", "Hours that differ (month)", "Max |diff| (month)"], rows, sheet="S11a_dirint_segments_IWEC")
    m = pd.read_csv(PROC / "dirint_measured_check.csv")
    nm = {"PTR": "Petrolina", "SMS": "São Martinho da Serra", "DRA": "Desert Rock", "GOLDEN": "Golden"}
    rows = [[nm[r.estacion], f"{int(r.filas):,}", "yes" if r.grilla_regular else "no", f"{int(r.horas_ghi_nan):,}", str(int(r.tramos)), f"{int(r.horas_c20):,}", str(int(r.horas_distintas_c20)), f(r.max_dif_c20, 2)] for _, r in m.iterrows()]
    TAB("S11b", "Table S11b. The same check for the four measured series. The hourly series are kept on a complete and regular hourly grid, with NaN in the hours without sufficient coverage (not discarded rows), and the evaluation mask is applied after running the models. "
        "DIRINT on continuous segments of valid hours (a new segment opens after every hour without GHI) gives the same DNI as the call used in the study in all hours.",
        ["Station", "Rows", "Regular hourly grid", "Hours without GHI", "Segments of valid hours", "Hours of the evaluation domain", "Hours that differ", "Max |diff| (W/m²)"], rows, sheet="S11b_dirint_segments_measured")


# ---------------------------------------------------------------------------------------------------------------------------- S12
def table_s12(mp):
    c = bench[bench.domain == "c20"]
    t1 = pd.read_csv(FIG / "revised_c20_ashrae" / "TableS1_sites_with_number.csv", dtype=str)
    t1["City"] = t1["City"].map(fixmoji)
    recs = []
    for _, r in t1.iterrows():
        key = mp[r.City]
        nonie = r.Source != "IWEC2"
        cls = r.Koppen if nonie else r.Koppen_ASHRAE
        kt = diag.loc[key, "kt_mean"] if key in diag.index else np.nan
        recs.append((r["No."] + ("†" if nonie else ""), r.City, cls, kt, key, nonie))
    order = sorted(recs, key=lambda x: (x[5], x[2], -np.nan_to_num(x[3], nan=-1)))
    for comp in ("DNI", "DHI"):
        rows = []
        for no, name, cls, kt, key, nonie in order:
            r = [no, name, cls, "—" if np.isnan(kt) else f"{kt:.3f}"]
            for mk, _ in MODELS4[:3]:
                x = c[(c.city == key) & (c.model == mk)].iloc[0]
                r += [f(x[f"{comp}_R2"], 3), f(x[f"{comp}_RMSE"]), sg(x[f"{comp}_MBE"]), f(x[f"{comp}_nRMSE"])]
            rows.append(r)
        head = ["No.", "Station", "K–G", "K̄ₜ"] + [f"{m} {k}" for m, _ in zip(("Reindl-2", "Erbs", "DIRINT"), range(3)) for k in ("R²", "RMSE", "MBE", "nRMSE")]
        head = ["No.", "Station", "K–G", "K̄ₜ"] + [f"{m}: {k}" for m in ("Reindl-2", "Erbs", "DIRINT") for k in ("R²", "RMSE", "MBE", "nRMSE")]
        TAB(f"S12{comp}", f"Table S12{'a' if comp == 'DNI' else 'b'}. Station-level {comp} statistics of the three models against the reference of each file (hours with cos θz ≥ 0.2). Stations are ordered by Köppen–Geiger class "
            f"(IWEC2 station list; the two files that are not in the list, marked †, carry the class of the first version and are placed last) and, within each class, by decreasing mean clearness index K̄ₜ (computed from the GHI of the file; — for the non-IWEC2 files). "
            "R²: coefficient of determination; RMSE and MBE in W/m²; nRMSE in %. This table replaces Table S2 of the first version.", head, rows, sheet=f"S12{comp}_by_station")


# ---------------------------------------------------------------------------------------------------------------------------- S13
def table_s13():
    rows = [["Abal et al. [43]; Uruguay; ten hourly diffuse-fraction models, five sites", "Best models were multi-predictor and locally fitted; uncertainty of the hourly diffuse fraction about 18% of the mean",
             "Compatible: locally fitted models are expected to outperform generic ones; not tested here"],
            ["Lemos et al. [29], Starke et al. [42]; Brazil", "BRL fitted to nine SONDA stations reduced the mean relative error of the diffuse fraction from 16–51% to 9–26%; adapted to one-minute data",
             "Not comparable: those studies compare fitted and original BRL, whereas here the original BRL was compared with Erbs and no locally fitted version was evaluated (Section 3.15)"],
            ["García Rojas et al. [35]; Chile, Costa Rica, Australia", "BRL appears little sensitive to regional climate", "Not contradicted: the dependence on class is weak in our data and limited by the number of stations per class"],
            ["Gueymard and Ruiz-Arias [4]; worldwide, one-minute data", "DIRINT mean bias of +3.5% (arid) and +12.7% (temperate); positive bias at high Kₜ attributed to hourly-developed models without cloud enhancement; Engerer2 recommended",
             "At the measured stations the bias of DIRINT was of variable sign; the large positive bias was specific to the look-up table of version 1.0.0"],
            ["Every et al. [48]; Starke et al. [49]; Ruiz-Arias and Gueymard [50]", "Köppen–Geiger or climate-feature fitting of BRL and one-minute models; no significant benefit of conditioning by class in GISPLIT",
             "Weak or inconsistent class dependence in our data; the class also enters the generator of IWEC2"]]
    TAB("S13", "Table S13. Coincidences and differences between the present results and regional or related studies (qualitative: most of those studies evaluate diffuse-fraction models against local measurements, whereas the main benchmark here is against IWEC2). "
        "Reference numbers are those of the reference list of the main text. This table was Table 8 of the revised main text and was moved here to shorten it.", ["Study", "Reported", "Relation to the present results"], rows,
        widths=[4400, 5400, 5600], sheet="S13_regional_studies")


# ---------------------------------------------------------------------------------------------------------------------------- figuras
def figures():
    FIGS = [(FIG / "supplement_r14" / "FigS1_DNI_MBE_vs_air_mass.png",
             "Fig. S1. DNI mean bias error (MBE) of the models against the IWEC2 reference by interval of relative air mass: (a) hours with cos θz ≥ 0.2 (main analysis) and (b) all daytime hours. The percentage under each interval is the share of the hours that fall in it. "
             "SD 1.0.0 table: look-up table of the first software version (not the DIRINT model of Perez et al.); DIRINT: Perez et al. without dew point. 39 IWEC2 stations."),
            (FIG / "supplement_r14" / "FigS2_DNI_MBE_Ktp_x_dKtp_c20.png",
             "Fig. S2. DNI MBE (W/m²) against IWEC2 by cell of the modified clearness index Kₜ′ and of the hourly variability ΔKₜ′ for (a) the SD 1.0.0 table and (b) DIRINT (Perez et al., without dew point); 39 IWEC2 stations, hours with cos θz ≥ 0.2. "
             "Here ΔKₜ′ = |Kₜ′(i) − Kₜ′(i−1)| between consecutive daytime hourly rows of the same station (backward difference; it is not exactly that of Perez et al., which averages the two neighbors), "
             "and the first hour of each day is left out. The colour scale is common to both panels and values outside ±200 W/m² take the extreme colour; the number in each cell is the actual value."),
            (FIG / "supplement_r17" / "FigS3_Kt_0p1_c20.png",
             "Fig. S3. Mean bias error (MBE) and RMSE of DNI (a, b) and of DHI (c, d) against the IWEC2 reference by interval of the hourly clearness index Kₜ of 0.1, for Reindl-2, Erbs and DIRINT (Perez et al., without dew point); 39 IWEC2 stations, "
             "hours with cos θz ≥ 0.2. The percentage under each interval is the share of the hours that fall in it. The SD 1.0.0 table is not drawn because its RMSE exceeds 100 W/m² over much of the range (see Table S2)."),
            (FIG / "external_check" / "FigS4_Mendoza_DHI_all_models_supplement.png",
             "Fig. S4. DHI RMSE at Mendoza (INAHE, 2022–2025) against the measured diffuse irradiance by model for all hours (black diamond; 95% interval by day bootstrap) and by season (DJF = December–February, MAM, JJA, SON); models are ordered by total RMSE. "
             "BRL and Engerer2 (one-minute data) are added to the models of Fig. 14. The result of Engerer2 is not independent of the data, because the quality control of the Mendoza series includes a test based on the Engerer models.")]
    for p, c in FIGS:
        assert p.exists(), p
        FIGB(p, c)


# ---------------------------------------------------------------------------------------------------------------------------- escritura
def run(t, b=False, i=False, sz=None, sup=False):
    pr = ("<w:b/>" if b else "") + ("<w:i/>" if i else "") + (f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else "") + ('<w:vertAlign w:val="superscript"/>' if sup else "")
    return f'<w:r>{"<w:rPr>" + pr + "</w:rPr>" if pr else ""}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def para(runs, style=None, jc=None, spacing=None, keep=False):
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:keepNext/>" if keep else "") + (f'<w:spacing w:before="{spacing[0]}" w:after="{spacing[1]}"/>' if spacing else "") + (f'<w:jc w:val="{jc}"/>' if jc else "")
    return f'<w:p>{"<w:pPr>" + ppr + "</w:pPr>" if ppr else ""}{runs}</w:p>'


def cap_runs(cap, sz=19):
    m = re.match(r"^((?:Table|Fig\.) S\d+[a-z]?\.)(.*)$", cap, flags=re.S)
    return (run(m.group(1), b=True, sz=sz) + run(m.group(2), sz=sz)) if m else run(cap, sz=sz)


def table_xml(cap, head, rows, note, widths):
    n = len(head)
    sz = 15 if n >= 12 else (16 if n >= 8 else 17)
    bd = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    ws = widths or [int(15400 / n)] * n

    def cell(t, w, b=False, shade=False):
        sh = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if shade else ""
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{sh}</w:tcPr>{para(run(str(t), b=b, sz=sz), jc="left", spacing=(20, 20))}</w:tc>'

    x = f'<w:tbl><w:tblPr><w:tblW w:w="{sum(ws)}" w:type="dxa"/><w:tblBorders>{bd}</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr>'
    x += "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in ws) + "</w:tblGrid>"
    x += "<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" + "".join(cell(h, w, True, True) for h, w in zip(head, ws)) + "</w:tr>"
    for r in rows:
        x += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cell(c, w) for c, w in zip(r, ws)) + "</w:tr>"
    x += "</w:tbl>"
    out = para(cap_runs(cap), jc="both", spacing=(200, 80), keep=True) + x
    if note:
        out += para(run(note, i=True, sz=15), spacing=(40, 120))
    return out + para("", spacing=(0, 160))


class Pkg:
    def __init__(self):
        self.media, self.rels, self.n = [], [], 0

    def image(self, png: Path, max_w_emu: int):
        data = png.read_bytes()
        w, h = struct.unpack(">II", data[16:24])
        cx = min(int(w / 300 * 914400), max_w_emu)
        cy = int(cx * h / w)
        self.n += 1
        rid, name = f"rId{100 + self.n}", f"media/supp_{self.n}.png"
        self.media.append(("word/" + name, data))
        self.rels.append((rid, name))
        return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{self.n}" name="{png.stem}"/>'
                f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{self.n}" name="{png.stem}"/><pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
                f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


def write_docx(path: Path):
    pk = Pkg()
    body = []
    for kind, v in BLOCKS:
        if kind == "h1":
            body.append(para(run(v, b=True, sz=28), style="Heading1", keep=True))
        elif kind == "h2":
            body.append(para(run(v, b=True, sz=23), style="Heading2", keep=True))
        elif kind == "p":
            body.append(para(run(v, sz=21), jc="both", spacing=(0, 120)))
        elif kind == "table":
            body.append(table_xml(*v))
        elif kind == "fig":
            png, cap = v
            body.append(para(pk.image(png, int(8.2 * 914400)), jc="center", spacing=(240, 60), keep=True))
            body.append(para(cap_runs(cap), jc="both", spacing=(0, 200)))
        elif kind == "pb":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rId2"/><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
            '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    ns = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
          'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"')
    doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {ns}><w:body>' + "".join(body) + sect + "</w:body></w:document>"
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="21"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>'
              '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="259" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>'
              '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>'
              '</w:styles>')
    footer = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
              '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
             + "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{nm}"/>' for rid, nm in pk.rels) + "</Relationships>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", drels)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/document.xml", doc)
        z.writestr("word/footer1.xml", footer)
        for nm, data in pk.media:
            z.writestr(nm, data)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    H("Supplementary Material")
    P("Consistency of Three Global Irradiance Decomposition Models with IWEC2 Reference Data in Argentina. This supplement accompanies the revised manuscript. It was reorganized in the revision: Table S1 of the first version "
      "(characteristics of the 41 sites) is Table S1 here, and Table S2 of the first version (station-level statistics) is now Table S12; the other tables and the figures are new. "
      "Unless stated otherwise, the domain is the one of the main text (hours with cos θz ≥ 0.2), “DIRINT” is the model of Perez et al. (1992) as implemented in pvlib without dew point, and “SD 1.0.0 table” is the look-up table of the first version of the software, which is not the DIRINT model. "
      "All the values were computed with the scripts of the analysis from the files described in the main text; the file names of each table are listed in the accompanying notes.")
    mp = table_s1()
    table_s2(); table_s3(); table_s4(); table_s5(); table_s6(); table_s7(); table_s8(); table_s9(); table_s10(); table_s11(); table_s12(mp); table_s13()
    BLOCKS.append(("pb", None))
    H("Supplementary figures")
    figures()
    write_docx(OUT / "Supplementary_Material_v2.docx")
    with pd.ExcelWriter(OUT / "Supplementary_Tables_v2.xlsx") as xw:
        for k, df in SHEETS.items():
            df.to_excel(xw, sheet_name=k[:31], index=False)
    notes = ["# Material suplementario: qué contiene y de dónde sale cada tabla", "",
             "Generado por `scripts/build_supplement.py`. Las leyendas y notas son texto redactado con asistencia de IA a partir de los datos (véase la declaración de uso de IA del manuscrito). Cifras: todas desde `data/processed/` y `figures/`.", "",
             "| Tabla / figura | Contenido | Fuente |", "|---|---|---|"]
    src = {"S1": "Sitios (41), con las dos clasificaciones de Köppen–Geiger y N de horas c20", "S2": "Errores por intervalos de Kt de 0.1 (DNI y DHI) y Fig. S3", "S3": "Residuo de cierre radiativo (c20 y all)",
           "S4": "Composición de la muestra y sensibilidad a las 15 estaciones no analizadas", "S5": "BRL y Engerer2 contra las mediciones, con el Erbs de las mismas horas (TableS5_long.csv)",
           "S6": "Correlaciones de Spearman del sesgo con cuatro descriptores", "S7": "Resultados de los dos archivos no IWEC2 y sensibilidad", "S8": "Tabla 2 en el dominio «all»",
           "S9": "Experimento controlado: por clase de Kt, Δref pareado y reponderación", "S10": "DIRINT con y sin punto de rocío en las cuatro estaciones medidas (muestra común)",
           "S11": "Comprobación de huecos/tramos de DIRINT en los 41 archivos y en las cuatro series medidas", "S12": "Estadísticos por estación (antigua S2)", "S13": "Comparación cualitativa con estudios regionales (antigua Tabla 8 del texto principal)"}
    for k, v in src.items():
        notes.append(f"| {k} | {v} | ver docstring de `build_supplement.py` |")
    notes += ["| Figs. S1–S2 | Sesgo de DNI por masa de aire y por celdas Kt′ × ΔKt′ | `scripts/make_figures_supp_r14.py` |", "| Fig. S3 | MBE y RMSE por Kt de 0.1 | `scripts/make_figures_supp_r17.py` |",
              "| Fig. S4 | RMSE de DHI en Mendoza, todos los modelos | `scripts/make_figures_external_check.py` |", ""]
    (OUT / "SUPLEMENTO_notas.md").write_text("\n".join(notes), encoding="utf-8")
    n_t = sum(1 for k, _ in BLOCKS if k == "table")
    print(f"supplement/Supplementary_Material_v2.docx: {n_t} tablas, 4 figuras, {len(SHEETS)} hojas en el xlsx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
