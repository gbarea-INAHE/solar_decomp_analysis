"""check_dirint_measured.py — DIRINT en las cuatro series con DNI medido: ¿la cadena usada conecta filas separadas por horas faltantes?

La cadena del estudio (bsrn_anchor.to_hourly -> to_epw_like -> dirint_dewpoint_anchor.dirint_variants -> máscara de evaluación) trabaja sobre una grilla horaria COMPLETA y regular;
las horas sin cobertura suficiente quedan como NaN de GHI (no se descartan filas). pvlib 0.16.1 calcula ΔKt' con las filas vecinas y omite los NaN (promedia las vecinas válidas),
que es lo mismo que parte la serie en el hueco. Este script lo comprueba: (a) la grilla es regular (todos los pasos de 1 h); (b) cuántas horas NaN hay; (c) DIRINT por tramos
continuos de horas con GHI válida frente a la cadena usada: horas distintas y diferencia máxima (todas las horas y dominio c20 con muestra común). La máscara se aplica DESPUÉS de DIRINT.
Salida: data/processed/dirint_measured_check.csv. Uso (desde revision_v2, venv activo): python scripts/check_dirint_measured.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
import controlled_reference_effect as cre  # noqa: E402
from benchmark_v2 import MIN_COSZ, prepare  # noqa: E402


def frame(stn: str):
    if stn == "GOLDEN":
        h = asg.golden_base_frame()
        epw = asg.golden_epw(h)
    else:
        minute, geo, meta = asg.minute_station(stn)
        res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
        epw = ba.to_epw_like(h, "avg")
    return h, epw


def main() -> int:
    warnings.filterwarnings("ignore")
    rows = []
    for stn in cre.STN:
        h, epw = frame(stn)
        times = pd.DatetimeIndex(epw["t_start"])
        steps = times.to_series().diff().dropna()
        regular = bool((steps == pd.Timedelta(hours=1)).all())
        d, _ = prepare({}, epw)
        ghi = d["GHI_h"].to_numpy()
        g = pd.Series(ghi, index=times)
        zen = pd.Series(d["zenith_deg"].to_numpy(), index=times)
        p = epw["p_pa"].to_numpy()
        kw = dict(use_delta_kt_prime=True, temp_dew=None, min_cos_zenith=MIN_COSZ, max_zenith=89.0)
        used = np.asarray(pvlib.irradiance.dirint(ghi=g, solar_zenith=zen, times=times, pressure=p, **kw), dtype=float)
        # tramos continuos: se parte en cada hora con GHI NaN (o con presión NaN)
        valid = np.isfinite(ghi)
        seg = np.cumsum(~valid)  # cada NaN abre un tramo nuevo; las filas NaN quedan solas
        alt = np.full(len(g), np.nan)
        for s in np.unique(seg):
            sl = (seg == s) & valid
            if sl.sum() == 0:
                continue
            alt[sl] = np.asarray(pvlib.irradiance.dirint(ghi=g[sl], solar_zenith=zen[sl], times=times[sl], pressure=p[sl], **kw), dtype=float)
        cosz = d["cos_Z"].to_numpy()
        both = np.isfinite(used) & np.isfinite(alt)
        diff = np.abs(used - alt)
        c20 = (cosz >= 0.2) & (ghi >= 10) & both
        rows.append(dict(estacion=stn, filas=len(g), grilla_regular=regular, horas_ghi_nan=int((~valid).sum()), tramos=int(len(np.unique(seg[valid]))),
                         horas_dirint_distintas_todas=int((diff[both] > 1e-9).sum()), max_dif_todas=float(np.nanmax(diff[both])) if both.any() else np.nan,
                         horas_c20=int(c20.sum()), horas_distintas_c20=int((diff[c20] > 1e-9).sum()), max_dif_c20=float(np.nanmax(diff[c20])) if c20.any() else np.nan))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(cre.PROC / "dirint_measured_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
