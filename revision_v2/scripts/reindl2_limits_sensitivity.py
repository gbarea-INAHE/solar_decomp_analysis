"""reindl2_limits_sensitivity.py — efecto de los límites por tramo de Reindl-2 sobre los 39 archivos IWEC2 (dominio c20).

El código del estudio (decomposition.reindl2_single) solo recorta Kd a [0, 1] y Kt a [0, 1]. La formulación que se atribuye al artículo original (Reindl, Beckman y
Duffie, 1990) agrega límites por tramo: Kd <= 1.0 (Kt <= 0.3), 0.1 <= Kd <= 0.97 (0.3 < Kt < 0.78), Kd >= 0.1 (Kt >= 0.78). Estos límites NO pudieron
verificarse contra el texto original (la página de Sandia PVPMC, fuente secundaria, da las ecuaciones sin límites); este script mide cuánto importarían si se
aplicaran. Reconstruye Reindl-2 con la serie horaria de hourly_v2.pkl.gz (sin α = cos z), comprueba que reproduce el DNI/DHI guardados y compara con la variante
con límites. Salida: data/processed/reindl2_limits_sensitivity.csv y resumen por pantalla.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REV = Path(__file__).resolve().parents[1]
PROC = REV / "data" / "processed"
DNI_MAX = 1100.0  # se verifica contra decomposition.DNI_MAX_W_M2 abajo


def kd_global(kt, sa):
    k = np.clip(kt, 0, 1)
    kd = np.where(k <= 0.30, 1.020 - 0.254 * k + 0.0123 * sa, np.where(k < 0.78, 1.400 - 1.749 * k + 0.177 * sa, 0.486 * k - 0.182 * sa))
    return np.clip(kd, 0, 1)


def kd_limits(kt, sa):
    k = np.clip(kt, 0, 1)
    kd = np.where(k <= 0.30, np.minimum(1.020 - 0.254 * k + 0.0123 * sa, 1.0),
                  np.where(k < 0.78, np.clip(1.400 - 1.749 * k + 0.177 * sa, 0.1, 0.97), np.maximum(0.486 * k - 0.182 * sa, 0.1)))
    return np.clip(kd, 0, 1)


def dni_from_kd(kd, ghi, cosz):
    dhi = np.minimum(np.maximum(0.0, kd * ghi), ghi)
    return np.clip((ghi - dhi) / cosz, 0, DNI_MAX), dhi


def main() -> int:
    import sys
    sys.path.insert(0, str(REV.parent))
    import decomposition as dec
    global DNI_MAX
    DNI_MAX = float(dec.DNI_MAX_W_M2)
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    h = h[(h.source == "IWEC2") & (h.cosz >= 0.2)].copy()
    ghi, cosz, kt = h.ghi.to_numpy(float), h.cosz.to_numpy(float), h.kt.to_numpy(float)
    sa = cosz  # sin(altura solar) = cos(cenit)
    kd0, kd1 = kd_global(kt, sa), kd_limits(kt, sa)
    dni0, dhi0 = dni_from_kd(kd0, ghi, cosz)
    dni1, dhi1 = dni_from_kd(kd1, ghi, cosz)
    # el modelo guardado usa Kt = GHI/G0h con G0h entero; se acepta una tolerancia chica por el redondeo de cosz
    err = np.nanmax(np.abs(dni0 - h.dni_reindl2.to_numpy(float)))
    print(f"reconstrucción vs DNI guardado de Reindl-2: error máximo {err:.2f} W/m2 (el redondeo de cos z entra en sin α)")
    h["dni_r1"], h["dhi_r1"] = dni1, dhi1
    changed = np.abs(kd1 - kd0) > 1e-12
    print(f"horas c20: {len(h)}; con Kd distinto al aplicar los límites: {int(changed.sum())} ({100 * changed.mean():.3f} %)")
    print(f"  del tramo Kt<=0.3 (límite Kd<=1): {int((changed & (kt <= 0.3)).sum())}; tramo intermedio: {int((changed & (kt > 0.3) & (kt < 0.78)).sum())}; tramo Kt>=0.78: {int((changed & (kt >= 0.78)).sum())}")
    rows = []
    for city, g in h.groupby("city"):
        ref, dref = g.dni_ref.to_numpy(), g.dhi_ref.to_numpy()
        r = {"city": city, "n": len(g), "n_cambia": int(changed[h.city.to_numpy() == city].sum())}
        for lab, col in (("erbs", "dni_erbs"), ("r_global", "dni_reindl2"), ("r_limites", "dni_r1")):
            r[f"rmse_{lab}"] = float(np.sqrt(np.mean((g[col].to_numpy() - ref) ** 2)))
        r["rmse_dhi_r_global"] = float(np.sqrt(np.mean((g.dhi_reindl2.to_numpy() - dref) ** 2)))
        r["rmse_dhi_r_limites"] = float(np.sqrt(np.mean((g.dhi_r1.to_numpy() - dref) ** 2)))
        rows.append(r)
    out = pd.DataFrame(rows)
    out["d_global"] = out.rmse_r_global - out.rmse_erbs
    out["d_limites"] = out.rmse_r_limites - out.rmse_erbs
    out.to_csv(PROC / "reindl2_limits_sensitivity.csv", index=False)
    print(f"RMSE DNI medio entre estaciones: Reindl-2 global {out.rmse_r_global.mean():.2f}; con límites {out.rmse_r_limites.mean():.2f}; Erbs {out.rmse_erbs.mean():.2f}")
    print(f"diferencia media Reindl-2 − Erbs: global {out.d_global.mean():+.2f}; con límites {out.d_limites.mean():+.2f}")
    print(f"cambio máximo del RMSE de Reindl-2 por estación: {np.abs(out.rmse_r_limites - out.rmse_r_global).max():.3f} W/m2")
    print(f"estaciones con Reindl-2 más cerca que Erbs: global {(out.d_global < 0).sum()}; con límites {(out.d_limites < 0).sum()}")
    print(f"RMSE DHI medio: global {out.rmse_dhi_r_global.mean():.2f}; con límites {out.rmse_dhi_r_limites.mean():.2f}")
    print(f"cambio máximo del DNI en una hora: {np.abs(dni1 - dni0).max():.1f} W/m2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
