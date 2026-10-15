"""ctrl_stats.py — cifras del experimento controlado que se citan en la carta de respuesta.

Lee data/processed/controlled_ref_effect*.csv (los genera controlled_reference_effect.py) y calcula, sin tipear nada a mano, los rangos por clase de Kt,
las proporciones de horas con Kt >= 0.7 y las estadísticas entre estaciones (Spearman, parcial, terciles, cruce por cero). Falla en voz alta si falta algo.
El manuscrito se verifica por separado en verify_manuscript_numbers.py (implementación independiente).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
S4 = ("PTR", "SMS", "DRA", "GOLDEN")

_ce = pd.read_csv(PROC / "controlled_ref_effect.csv")
_cm = pd.read_csv(PROC / "controlled_ref_effect_mendoza.csv")
_all = pd.concat([_ce, _cm], ignore_index=True)
_cs = pd.read_csv(PROC / "controlled_ref_effect_cross_station.csv")
_cls = pd.read_csv(PROC / "koppen_ashrae_vs_s1.csv").set_index("city")["koppen_ashrae"]


def _sel(mod, ref, cl, sts=S4):
    x = _all[(_all.modelo == mod) & (_all.ref == ref) & (_all.clase == cl) & _all.estacion.isin(sts)].set_index("estacion")
    if len(x) != len(sts):
        raise RuntimeError(f"faltan filas del experimento controlado: {mod} {ref} {cl} {sts}")
    return x


def rng(mod, ref, cl, sts=S4):
    """(mínimo, máximo) de ΔRMSE respecto de Erbs entre las estaciones."""
    d = _sel(mod, ref, cl, sts).d_vs_erbs
    return float(d.min()), float(d.max())


def val(mod, ref, cl, st):
    return float(_sel(mod, ref, cl, (st,)).d_vs_erbs.iloc[0])


def ci_excl0(mod, ref, cl, sts=S4):
    x = _sel(mod, ref, cl, sts)
    return int(((x.d_lo > 0) | (x.d_hi < 0)).sum())


def share(st, ref="medido"):
    return float(_sel("erbs", ref, ">=0.7", (st,)).pct.iloc[0])


def amplification(cl=">=0.7", sts=("SMS", "DRA", "GOLDEN")):
    m = _sel("reindl2", "medido", cl, sts).d_vs_erbs
    g = _sel("reindl2", "Gompertz", cl, sts).d_vs_erbs
    f = g / m
    return float(f.min()), float(f.max())


# ---- entre estaciones
share_rng = (float(_cs.share_kt07.min()), float(_cs.share_kt07.max()))
_rho = stats.spearmanr(_cs.share_kt07, _cs.d_reindl_erbs)
RHO, RHO_P = float(_rho.statistic), float(_rho.pvalue)
_rk = _cs[["share_kt07", "d_reindl_erbs", "abs_lat", "elev_m"]].rank()
_z = np.column_stack([np.ones(len(_rk)), _rk[["abs_lat", "elev_m"]].values])
_rx = _rk.share_kt07.values - _z @ np.linalg.lstsq(_z, _rk.share_kt07.values, rcond=None)[0]
_ry = _rk.d_reindl_erbs.values - _z @ np.linalg.lstsq(_z, _rk.d_reindl_erbs.values, rcond=None)[0]
RHO_PARTIAL = float(np.corrcoef(_rx, _ry)[0, 1])
RHO_LAT = float(stats.spearmanr(_cs.share_kt07, _cs.abs_lat).statistic)
RHO_ELEV = float(stats.spearmanr(_cs.share_kt07, _cs.elev_m).statistic)
_lr = stats.linregress(_cs.share_kt07, _cs.d_reindl_erbs)
SLOPE, CROSS = float(_lr.slope), float(-_lr.intercept / _lr.slope)
_s = _cs.sort_values("share_kt07").reset_index(drop=True)
TERC = [(float(_s.iloc[a:b].share_kt07.mean()), float(_s.iloc[a:b].d_reindl_erbs.mean())) for a, b in ((0, 13), (13, 26), (26, 39))]
_top = _s[_s.share_kt07 > CROSS]
N_TOP = len(_top)
N_TOP_ERBS = int((_top.d_reindl_erbs > 0).sum())
N_TOP_ARID = int(_top.city.map(_cls).isin(["BSk", "BWk"]).sum())
_s["cls"] = _s.city.map(_cls)
SHARE_CLASS = _s.groupby("cls").share_kt07.mean().to_dict()


# ---- segunda ronda: contraste pareado del efecto de la referencia e intervalos de la reponderación
_pr = pd.read_csv(PROC / "controlled_ref_effect_paired.csv")
_rc = pd.read_csv(PROC / "controlled_ref_effect_reweight_ci.csv")


def paired(st, cl, m, col="d_ref"):
    x = _pr[(_pr.estacion == st) & (_pr.clase == cl) & (_pr.modelo == m)]
    if len(x) != 1:
        raise RuntimeError(f"falta el contraste pareado de {st} {cl} {m}")
    return float(x[col].iloc[0])


def paired_rng(cl, m, sts=S4):
    v = [paired(s, cl, m) for s in sts]
    return min(v), max(v)


def reweighted(st, ref):
    x = _rc[(_rc.estacion == st) & (_rc.ref == ref)]
    if len(x) != 1:
        raise RuntimeError(f"falta la reponderación de {st} {ref}")
    x = x.iloc[0]
    return float(x.d_reindl_erbs_mezcla_IWEC2), float(x.lo), float(x.hi)


def kt_low_max():
    x = _all[(_all.clase == "<0.3") & (_all.modelo == "reindl2")]
    return float(x.d_vs_erbs.abs().max())
