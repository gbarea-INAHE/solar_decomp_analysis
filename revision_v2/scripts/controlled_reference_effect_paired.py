"""controlled_reference_effect_paired.py — contraste PAREADO del efecto de la referencia e intervalos de la reponderación.

Dos comprobaciones que faltaban en controlled_reference_effect.py (segunda revisión externa):
(1) Efecto de la referencia sobre la diferencia entre modelos, con los MISMOS días remuestreados en cada réplica para las cuatro cantidades:
    Δref(M) = [RMSE(M,G) − RMSE(E,G)] − [RMSE(M,O) − RMSE(E,O)],  M = Reindl-2, DIRINT, SD 1.0.0 table; E = Erbs; G = Gompertz de la misma GHI; O = observación.
    (Los intervalos de las dos referencias por separado no se pueden comparar a ojo: este contraste sí.)
(2) Intervalo bootstrap de días de la diferencia Reindl-2 − Erbs reponderada con la mezcla de cielos de IWEC2 (16.0 / 73.3 / 10.7 %), contra cada referencia.
Mismas muestras, clases y semilla que controlled_reference_effect.py. Mendoza (difusa) incluido.
Salidas: data/processed/controlled_ref_effect_paired.csv y controlled_ref_effect_reweight_ci.csv. Uso (desde revision_v2, venv): python scripts/controlled_reference_effect_paired.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controlled_reference_effect as cre  # noqa: E402
import anchor_sensitivity_golden as asg  # noqa: E402

PROC = cre.PROC
B = cre.B
W = {"<0.3": 0.160, "0.3-0.7": 0.733, ">=0.7": 0.107}
MODS = ("reindl2", "dirint", "sd100")
CL = [("all", None, None), ("<0.3", -np.inf, 0.3), ("0.3-0.7", 0.3, 0.7), (">=0.7", 0.7, np.inf)]


def run(d: dict, stn: str, rng):
    ok = d["mask"] & np.isfinite(d["ref"]) & np.isfinite(d["kt"])
    for k in d["ser"]:
        ok &= np.isfinite(d["ser"][k])
    days = np.unique(d["day"][ok])
    draws = rng.integers(0, len(days), size=(B, len(days)))
    refs = {"O": d["ref"], "G": d["ser"]["gompertz"]}
    names = ["erbs", *MODS]
    st = {}
    for cname, lo, hi in CL:
        sel = ok if lo is None else ok & (d["kt"] >= lo) & (d["kt"] < hi)
        if sel.sum() < 30:
            continue
        di = {x: np.flatnonzero(sel & (d["day"] == x)) for x in days}
        nd = np.array([len(di[x]) for x in days], float)
        sse = {(r, m): np.array([np.sum((d["ser"][m][di[x]] - refs[r][di[x]]) ** 2) for x in days]) for r in refs for m in names}
        st[cname] = (nd, sse)
    prow, rrow = [], []
    for cname, (nd, sse) in st.items():
        den = nd[draws].sum(axis=1)
        pt = {(r, m): np.sqrt(sse[(r, m)].sum() / nd.sum()) for r in refs for m in names}
        bs = {(r, m): np.sqrt(sse[(r, m)][draws].sum(axis=1) / den) for r in refs for m in names}
        for m in MODS:
            est = (pt[("G", m)] - pt[("G", "erbs")]) - (pt[("O", m)] - pt[("O", "erbs")])
            b = (bs[("G", m)] - bs[("G", "erbs")]) - (bs[("O", m)] - bs[("O", "erbs")])
            lo_, hi_ = np.percentile(b, [2.5, 97.5])
            prow.append(dict(estacion=stn, clase=cname, modelo=m, d_ref=est, d_ref_lo=lo_, d_ref_hi=hi_,
                             d_obs=pt[("O", m)] - pt[("O", "erbs")], d_gomp=pt[("G", m)] - pt[("G", "erbs")]))
    if all(c in st for c in W):
        for r, lab in (("O", "medido"), ("G", "Gompertz")):
            def rmse_w(m, s=None):
                if s is None:
                    return np.sqrt(sum(W[c] * st[c][1][(r, m)].sum() / st[c][0].sum() for c in W))
                return np.sqrt(sum(W[c] * st[c][1][(r, m)][s].sum(axis=1) / st[c][0][s].sum(axis=1) for c in W))
            est = rmse_w("reindl2") - rmse_w("erbs")
            b = rmse_w("reindl2", draws) - rmse_w("erbs", draws)
            lo_, hi_ = np.percentile(b, [2.5, 97.5])
            rrow.append(dict(estacion=stn, ref=lab, d_reindl_erbs_mezcla_IWEC2=est, lo=lo_, hi=hi_))
    return prow, rrow


def main() -> int:
    warnings.filterwarnings("ignore")
    P, R = [], []
    for stn in cre.STN:
        print("estación", stn, flush=True)
        p, r = run(cre.load_station(stn), stn, np.random.default_rng(asg.SEED))
        P += p
        R += r
    print("Mendoza (difusa)", flush=True)
    p, r = run(cre.load_mendoza_dhi(), "MENDOZA", np.random.default_rng(asg.SEED))
    P += p
    R += r
    pd.DataFrame(P).to_csv(PROC / "controlled_ref_effect_paired.csv", index=False)
    pd.DataFrame(R).to_csv(PROC / "controlled_ref_effect_reweight_ci.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(P).round(1).to_string(index=False))
    print(pd.DataFrame(R).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
