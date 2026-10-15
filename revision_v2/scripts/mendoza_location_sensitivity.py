"""mendoza_location_sensitivity.py — ¿cambia el resultado de la corrida de Mendoza si la ubicación real difiere de la usada?
Latitud y longitud las confirmó el técnico (2026-10-06: -32.89, -68.87); la altitud NO fue informada (750 m es un supuesto). Marca de tiempo = final de minuto (confirmada).
Prueba variantes de latitud, longitud y altitud alrededor de (-32.89, -68.87, 750 m) y reporta el RMSE de DHI (c20, muestra común de cada variante) de los
modelos principales y su diferencia con la base. Solo lectura de los datos; escribe data/processed/mendoza_location_sensitivity.csv.
Uso (desde revision_v2, venv activo): python scripts/mendoza_location_sensitivity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_mendoza_provisional as rmp  # noqa: E402

VARIANTS = {
    "base (-32.89, -68.87, 750 m)": (-32.89, -68.87, 750.0),
    "lon -68.90 (3 km al oeste)": (-32.89, -68.90, 750.0),
    "lat -32.87 (2 km al norte)": (-32.87, -68.87, 750.0),
    "altitud 650 m": (-32.89, -68.87, 650.0),
    "altitud 850 m": (-32.89, -68.87, 850.0),
    "altitud 1000 m": (-32.89, -68.87, 1000.0),
    "lat -32.87, lon -68.90, 850 m": (-32.87, -68.90, 850.0),
}


def main() -> int:
    warnings.filterwarnings("ignore")
    minute = rmp.mendoza_minute("end")                 # marca de tiempo = final de minuto (confirmada por el técnico)
    rows = []
    for name, (lat, lon, elev) in VARIANTS.items():
        P = rmp.process(minute, lat, lon, elev)
        mk = rmp.common_mask(P)
        for k in rmp.MAIN:
            pred = rmp.dhi_from_dni(P["ghi"], P["series"][k], P["cosz"])
            e = pred[mk] - P["dhi_m"][mk]
            rows.append(dict(variant=name, model=k, n=int(mk.sum()), RMSE_DHI=float(np.sqrt(np.mean(e ** 2))), MBE_DHI=float(e.mean())))
        print("  hecho:", name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(rmp.OUT / "mendoza_location_sensitivity.csv", index=False)
    piv = df.pivot(index="model", columns="variant", values="RMSE_DHI").reindex(rmp.MAIN)[list(VARIANTS)]
    print(piv.round(2).to_string())
    print("máxima diferencia de RMSE de DHI respecto de la base: %.3f W/m2" % float(piv.sub(piv.iloc[:, 0], axis=0).abs().max().max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
