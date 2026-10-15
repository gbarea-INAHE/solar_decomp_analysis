"""check_dhi_dni_identity.py — el error de DHI de cada modelo es, salvo redondeo, -cos(z) por su error de DNI.

La referencia IWEC2 cierra GHI = DNI*cos(z) + DHI, y cada modelo del estudio cierra la misma ecuación (DHI por cierre desde DNI, o al revés).
Por lo tanto e_DHI = DHI_modelo - DHI_ref = -cos(z) * (DNI_modelo - DNI_ref) = -cos(z) * e_DNI, salvo el redondeo de los campos enteros del archivo.
Este script mide ese redondeo (máximo y p99 de |e_DHI + cos(z) e_DNI|) en las 39 estaciones IWEC2, dominio c20, y la mediana del ΔR² de DHI entre Reindl-2 y Erbs.
Salida: data/processed/identity_dhi_dni_c20.csv y un resumen por pantalla (falla en voz alta si algo no cierra).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
PROC = REV / "data" / "processed"
MODELS = ["erbs", "reindl2", "dirint_std_noTd", "dirint_v1"]


def main() -> int:
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    b = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    iwec2 = set(b.loc[b["source"].astype(str).str.upper().str.contains("IWEC"), "city"])
    if len(iwec2) != 39:
        # la columna 'source' del benchmark puede tener otro rotulado: se cae a la lista del experimento controlado
        iwec2 = set(pd.read_csv(PROC / "controlled_ref_effect_cross_station.csv")["city"])
    if len(iwec2) != 39:
        raise RuntimeError(f"se esperaban 39 estaciones IWEC2 y hay {len(iwec2)}")
    d = h[h["city"].isin(iwec2) & (h["cosz"] >= 0.2)].copy()
    if len(d) != 141410:
        raise RuntimeError(f"se esperaban 141410 horas c20 y hay {len(d)}")
    rows = []
    ref_close = np.abs(d["dni_ref"] * d["cosz"] + d["dhi_ref"] - d["ghi"])
    rows.append({"modelo": "referencia (cierre GHI)", "max_abs": ref_close.max(), "p99": ref_close.quantile(0.99), "mediana": ref_close.median()})
    for m in MODELS:
        e_dni = d[f"dni_{m}"] - d["dni_ref"]
        e_dhi = d[f"dhi_{m}"] - d["dhi_ref"]
        res = np.abs(e_dhi + d["cosz"] * e_dni)
        rows.append({"modelo": m, "max_abs": res.max(), "p99": res.quantile(0.99), "mediana": res.median()})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "identity_dhi_dni_c20.csv", index=False)
    print(out.to_string(index=False))
    # mediana de ΔR² (Reindl-2 - Erbs) por estación, DNI y DHI
    c = b[(b["domain"] == "c20") & b["city"].isin(iwec2)]
    piv_dhi = c.pivot(index="city", columns="model", values="DHI_R2")
    piv_dni = c.pivot(index="city", columns="model", values="DNI_R2")
    print("n estaciones:", len(piv_dhi))
    print("mediana dR2 DHI (Reindl-2 - Erbs): %.4f" % (piv_dhi["reindl2"] - piv_dhi["erbs"]).median())
    print("mediana dR2 DNI (Reindl-2 - Erbs): %.4f" % (piv_dni["reindl2"] - piv_dni["erbs"]).median())
    return 0


if __name__ == "__main__":
    sys.exit(main())
