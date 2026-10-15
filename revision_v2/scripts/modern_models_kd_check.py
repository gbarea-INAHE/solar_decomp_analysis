"""modern_models_kd_check.py — control independiente de la implementación de Engerer2 y BRL: la FRACCIÓN DIFUSA predicha se compara con la medida
(Kd = DHI/GHI), que no es la métrica de DNI usada en el resto del análisis. Un error de implementación (signos, unidades, tiempo solar) se vería aquí
como sesgo o dispersión grandes. (No se compara con valores de la literatura: es solo un control de coherencia.)
Uso: python scripts/modern_models_kd_check.py   (desde revision_v2, venv activo). Escribe data/processed/modern_models_kd_check.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import modern_models as mm  # noqa: E402
import run_modern_models_anchor as rm  # noqa: E402


def stats(kd_meas, kd_mod, label, rows, stn, n_min=200):
    m = np.isfinite(kd_meas) & np.isfinite(kd_mod)
    d = kd_mod[m] - kd_meas[m]
    r = float(np.corrcoef(kd_meas[m], kd_mod[m])[0, 1])
    rows.append(dict(station=stn, model=label, n=int(m.sum()), RMSE_Kd=float(np.sqrt(np.mean(d ** 2))), MBE_Kd=float(d.mean()), r=r,
                     kd_meas_mean=float(kd_meas[m].mean()), kd_mod_mean=float(kd_mod[m].mean())))
    print(f"  {stn:6s} {label:22s} n={int(m.sum()):6d}  RMSE(Kd)={rows[-1]['RMSE_Kd']:.3f}  MBE(Kd)={rows[-1]['MBE_Kd']:+.3f}  r={r:.3f}  "
          f"Kd medio medido/modelo = {rows[-1]['kd_meas_mean']:.3f}/{rows[-1]['kd_mod_mean']:.3f}")


def main() -> int:
    warnings.filterwarnings("ignore")
    rows = []
    for stn in ("PTR", "SMS", "DRA", "GOLDEN"):
        if stn == "GOLDEN":
            h = asg.golden_base_frame()
            lat, lon = asg.G_LAT, asg.G_LON
        else:
            minute, geo, meta = asg.minute_station(stn)
            lat, lon = meta["lat"], meta["lon"]
            res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
            qc = asg.qc_variant(minute, geo)
            # Engerer2 1 min: Kd modelado vs medido en minutos válidos con GHI > 50 W/m2 y Z < 80
            t_mid = pd.DatetimeIndex(minute.index + pd.Timedelta(seconds=30)).tz_convert("UTC").tz_localize(None)
            sun = (geo["cosz"] > 0).to_numpy()
            valid = ((~qc["invalid"].to_numpy()) | ~sun) & ~qc["missing"].to_numpy()
            ghi = np.where(valid & sun, minute["ghi"].to_numpy(float), np.nan)
            _, kd = mm.engerer2_dni(ghi, t_mid, lat, lon, 1)
            kd_meas = np.where((ghi > 50) & (geo["zenith"].to_numpy() < 80) & valid, minute["dhi"].to_numpy(float) / ghi, np.nan)
            stats(kd_meas, kd, "engerer2_1min (minuto)", rows, stn)
        # horario: Kd medido = DHI_h/GHI_h (Golden trae DHI horario; en 1 min se usa la media horaria de DHI)
        ghi_h = h["ghi"].to_numpy(float)
        mid, ast, e_n, e_h, cosz, day_id = rm.hourly_geometry(h, lat, lon)
        z = np.degrees(np.arccos(np.clip(cosz, 0, 1)))
        t_naive = mid.tz_convert("UTC").tz_localize(None)
        _, kd_e2 = mm.engerer2_dni(ghi_h, t_naive, lat, lon, 60, 0.015, e_h_override=e_h, z_override=z)
        _, kd_brl, kt, psi, ktd = mm.brl_dni_hourly(ghi_h, e_n, e_h, cosz, ast, day_id)
        kd_meas_h = np.where((ghi_h > 50) & (z < 80), h["dhi"].to_numpy(float) / ghi_h, np.nan)
        stats(kd_meas_h, kd_e2, "engerer2_60 (hora)", rows, stn)
        stats(kd_meas_h, kd_brl, "brl_hourly (hora)", rows, stn)
    out = pd.DataFrame(rows)
    out.to_csv(asg.OUT / "modern_models_kd_check.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
