"""dni_clip_check.py — ¿cuántas horas c20 tocan el recorte de DNI (DNI_MAX_W_M2) en los modelos, y cuántas veces la referencia lo supera?"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
import decomposition as dec  # noqa: E402
import controlled_reference_effect as cre  # noqa: E402

warnings.filterwarnings("ignore")
MAXV = float(dec.DNI_MAX_W_M2)
h = pd.read_pickle(cre.PROC / "hourly_v2.pkl.gz")
h = h[(h.source == "IWEC2") & (h.cosz >= 0.2)]
rows = [dict(serie="IWEC2 (39, c20)", n=len(h), ref_gt_max=int((h.dni_ref > MAXV).sum()), ref_max=float(h.dni_ref.max()),
             **{f"{m}_en_max": int((h[f"dni_{m}"] >= MAXV - 1e-9).sum()) for m in ("erbs", "reindl2", "dirint_std_noTd")},
             dirint_gt_max=int((h.dni_dirint_std_noTd > MAXV).sum()), dirint_max=float(h.dni_dirint_std_noTd.max()))]
for stn in cre.STN:
    d = cre.load_station(stn)
    ok = d["mask"] & np.isfinite(d["ref"])
    for k in d["ser"]:
        ok &= np.isfinite(d["ser"][k])
    s = d["ser"]
    rows.append(dict(serie=stn, n=int(ok.sum()), ref_gt_max=int((d["ref"][ok] > MAXV).sum()), ref_max=float(d["ref"][ok].max()),
                     erbs_en_max=int((s["erbs"][ok] >= MAXV - 1e-9).sum()), reindl2_en_max=int((s["reindl2"][ok] >= MAXV - 1e-9).sum()),
                     dirint_std_noTd_en_max=int((s["dirint"][ok] >= MAXV - 1e-9).sum()), dirint_gt_max=int((s["dirint"][ok] > MAXV).sum()), dirint_max=float(s["dirint"][ok].max())))
out = pd.DataFrame(rows)
out.to_csv(cre.PROC / "dni_clip_check.csv", index=False)
pd.set_option("display.width", 220)
print("DNI_MAX_W_M2 =", MAXV)
print(out.to_string(index=False))
