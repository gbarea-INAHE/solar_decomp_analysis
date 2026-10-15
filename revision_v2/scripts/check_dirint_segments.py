"""check_dirint_segments.py — ¿cambia el DIRINT del benchmark si se parte la serie en tramos continuos?

El benchmark (benchmark_v2.py) llama a pvlib.irradiance.dirint sobre las 8760 filas de cada archivo; la herramienta v1.1.0 parte la serie en tramos de
pasos de exactamente 1 h y llama a pvlib por tramo. En un TMY los meses vienen de años distintos, y pvlib 0.16.1 calcula ΔKt' con las filas vecinas
(shift ±1) sin mirar el reloj. Este script mide, en los 41 archivos y en el dominio c20 (GHI >= 10 W/m2, cos z >= 0.2), cuánto difiere el DNI de la
llamada continua respecto de la llamada por tramos (partiendo donde el paso de t_start no es 1 h, que es la regla de la herramienta), y además
partiendo en cada cambio de mes. Salida: data/processed/dirint_segments_check.csv. Falla en voz alta si falta algún archivo.
Uso (desde revision_v2, venv activo): python scripts/check_dirint_segments.py --epw-dir data/raw
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import benchmark_v2 as B  # noqa: E402
from epw_io import epw_files, read_epw  # noqa: E402


def dirint_by_segments(g, zen, times, p, seg):
    out = np.full(len(g), np.nan)
    for s in np.unique(seg):
        sl = seg == s
        out[sl] = np.asarray(pvlib.irradiance.dirint(
            ghi=g[sl], solar_zenith=zen[sl], times=times[sl], pressure=p[sl], use_delta_kt_prime=True, temp_dew=None,
            min_cos_zenith=B.MIN_COSZ, max_zenith=89.0), dtype=float)
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    a = ap.parse_args()
    files = list(epw_files(a.epw_dir))
    if len(files) != 41:
        raise RuntimeError(f"se esperaban 41 archivos EPW y hay {len(files)}")
    rows = []
    for f in files:
        meta, df = read_epw(f)
        d, _ = B.prepare(meta, df)
        times = pd.DatetimeIndex(df["t_start"])
        g = pd.Series(d["GHI_h"].to_numpy(), index=times)
        zen = pd.Series(d["zenith_deg"].to_numpy(), index=times)
        p = pd.Series(df["p_pa"].to_numpy(), index=times)
        cont = np.asarray(pvlib.irradiance.dirint(ghi=g, solar_zenith=zen, times=times, pressure=p, use_delta_kt_prime=True, temp_dew=None,
                                                  min_cos_zenith=B.MIN_COSZ, max_zenith=89.0), dtype=float)
        brk_gap = (times.to_series().diff() != pd.Timedelta(hours=1)).to_numpy().copy()
        brk_gap[0] = True
        seg_gap = np.cumsum(brk_gap)
        seg_mon = np.cumsum(np.r_[True, np.diff(df["month"].to_numpy()) != 0])
        mask = (d["GHI_h"].to_numpy() >= B.MIN_GHI) & (d["cos_Z"].to_numpy() >= 0.20)
        rec = {"city": meta["city"], "n_c20": int(mask.sum()), "tramos_por_paso": int(seg_gap.max()), "tramos_por_mes": int(seg_mon.max())}
        for lab, seg in (("paso", seg_gap), ("mes", seg_mon)):
            alt = dirint_by_segments(g, zen, times, p, seg)
            diff = np.abs(cont - alt)[mask]
            both_nan = np.isnan(cont[mask]) & np.isnan(alt[mask])
            diff = np.where(both_nan, 0.0, diff)
            if np.isnan(diff).any():
                rec[f"n_nan_distinto_{lab}"] = int(np.isnan(diff).sum())
                diff = np.nan_to_num(diff, nan=np.inf)
            rec[f"max_abs_{lab}"] = float(diff.max())
            rec[f"n_dif_{lab}"] = int((diff > 1e-9).sum())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(B.OUT / "dirint_segments_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.describe().T[["min", "max"]])
    print("archivos con diferencia por paso:", int((out.n_dif_paso > 0).sum()), "| por mes:", int((out.n_dif_mes > 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
