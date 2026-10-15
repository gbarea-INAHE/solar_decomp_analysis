"""kt_exceedance_measured.py — horas de la muestra común de las 4 estaciones medidas con Kt por encima de umbrales (c20, mismas horas del experimento controlado)."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import controlled_reference_effect as cre  # noqa: E402

warnings.filterwarnings("ignore")
rows = []
for stn in cre.STN:
    d = cre.load_station(stn)
    ok = d["mask"] & np.isfinite(d["ref"]) & np.isfinite(d["kt"])
    for k in d["ser"]:
        ok &= np.isfinite(d["ser"][k])
    kt = d["kt"]
    rows.append(dict(estacion=stn, n_muestra_comun=int(ok.sum()), kt_gt_0p9=int((ok & (kt > 0.9)).sum()), kt_gt_1p0=int((ok & (kt > 1.0)).sum()), kt_gt_1p05=int((ok & (kt > 1.05)).sum()),
                     kt_max=float(np.nanmax(kt[ok]))))
out = pd.DataFrame(rows)
out.to_csv(cre.PROC / "kt_exceedance_measured.csv", index=False)
print(out.to_string(index=False))
