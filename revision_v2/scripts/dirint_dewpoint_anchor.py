"""dirint_dewpoint_anchor.py — ¿Cuánto mejora el DIRINT de Perez 1992 al usar el punto de rocío (desde humedad relativa) frente al bin de «agua
precipitable desconocida»? Contra el DNI medido, 4 estaciones, dominio c20, misma muestra de horas para las dos variantes, bootstrap por días
emparejado (B=2000, semilla fija del piloto).

Variante CON punto de rocío = la `dirint_std` del piloto (Magnus-OMM desde T y HR en PTR/SMS/DRA; Tdew medido en Golden).
Variante SIN punto de rocío = misma llamada a pvlib con temp_dew=None.
Verificación: la llamada replicada aquí con punto de rocío reproduce exactamente `dirint_std` del piloto (aserción).

Uso (desde revision_v2, venv activo): python scripts/dirint_dewpoint_anchor.py
Escribe data/processed/dirint_dewpoint_anchor.csv y logs/dirint_dewpoint_<ts>.txt
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import anchor_sensitivity_golden as asg  # noqa: E402
import bsrn_anchor as ba  # noqa: E402
from benchmark_v2 import MIN_COSZ, arr, call, metr, prepare  # noqa: E402

OUT, LOGS = asg.OUT, asg.LOGS


def dirint_variants(epw: pd.DataFrame):
    d, _ = prepare({}, epw)
    ghi, cosz = d["GHI_h"].to_numpy(), d["cos_Z"].to_numpy()
    times = pd.DatetimeIndex(epw["t_start"])
    g = pd.Series(ghi, index=times)
    zen = pd.Series(d["zenith_deg"].to_numpy(), index=times)
    kw = dict(ghi=g, solar_zenith=zen, times=times, pressure=epw["p_pa"].to_numpy(), use_delta_kt_prime=True,
              min_cos_zenith=MIN_COSZ, max_zenith=89.0)
    with_td = arr(call(pvlib.irradiance.dirint, temp_dew=pd.Series(epw["t_dew"].to_numpy(), index=times), **kw))
    without = arr(call(pvlib.irradiance.dirint, temp_dew=None, **kw))
    return with_td, without, d


def main() -> int:
    warnings.filterwarnings("ignore")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"DIRINT con y sin punto de rocío vs DNI medido (c20) {ts}")
    rows, by_bin = [], []
    for stn in ("PTR", "SMS", "DRA", "GOLDEN"):
        if stn == "GOLDEN":
            h = asg.golden_base_frame()
            epw = asg.golden_epw(h)
            res, sim, series, mask = asg.evaluate(epw, h["dni"].to_numpy())
            lon = asg.G_LON
        else:
            minute, geo, meta = asg.minute_station(stn)
            res, sim, series, mask, h, cov = asg.run_minute_variant(minute, geo, meta, asg.MIN_VARIANTS["base"])
            epw = ba.to_epw_like(h, "avg")
            lon = meta["lon"]
        with_td, without, d = dirint_variants(epw)
        assert np.array_equal(with_td, series["dirint_std"], equal_nan=True), f"{stn}: la llamada replicada no reproduce dirint_std del piloto"
        log(f"  [{stn}] la llamada replicada con punto de rocío reproduce dirint_std del piloto (igualdad exacta)")
        n_td_nan = int((np.isfinite(epw["ghi"]) & ~np.isfinite(epw["t_dew"])).sum())
        dni = h["dni"].to_numpy(float)
        day = (pd.DatetimeIndex(h.index) + pd.Timedelta(hours=lon / 15.0)).floor("D").to_numpy()
        # el clave "erbs" es solo la «referencia» interna de day_bootstrap: aquí es DIRINT sin punto de rocío
        bt = asg.day_bootstrap(dni, {"erbs": without, "con_td": with_td}, mask, day, np.random.default_rng(asg.SEED))
        a = bt.set_index("model")
        # verificación independiente de RMSE en la muestra común (pandas)
        ok = mask & np.isfinite(with_td) & np.isfinite(without)
        for k, v in (("erbs", without), ("con_td", with_td)):
            e = pd.Series(v[ok] - dni[ok])
            assert abs(float(np.sqrt((e ** 2).mean())) - a.loc[k, "RMSE"]) < 1e-9
        r = dict(station=stn, n_hours=int(a.loc["con_td", "n_hours"]), n_days=int(a.loc["con_td", "n_days"]),
                 RMSE_sin_td=a.loc["erbs", "RMSE"], RMSE_sin_td_lo=a.loc["erbs", "RMSE_lo"], RMSE_sin_td_hi=a.loc["erbs", "RMSE_hi"],
                 RMSE_con_td=a.loc["con_td", "RMSE"], RMSE_con_td_lo=a.loc["con_td", "RMSE_lo"], RMSE_con_td_hi=a.loc["con_td", "RMSE_hi"],
                 dRMSE_con_menos_sin=a.loc["con_td", "dRMSE_vs_erbs"], dRMSE_lo=a.loc["con_td", "dRMSE_lo"], dRMSE_hi=a.loc["con_td", "dRMSE_hi"],
                 MBE_sin_td=a.loc["erbs", "MBE"], MBE_con_td=a.loc["con_td", "MBE"], horas_con_ghi_sin_td=n_td_nan)
        rows.append(r)
        # diagnóstico: dónde se produce la diferencia, por bin de agua precipitable de Perez (W = exp(0.07·Td − 0.075) [cm])
        w = np.exp(0.07 * epw["t_dew"].to_numpy(float) - 0.075)
        wbin = np.digitize(w, [1.0, 2.0, 3.0])          # 0: W<1, 1: 1–2, 2: 2–3, 3: >3 cm
        for b_i, lab in enumerate(("W<1", "1<=W<2", "2<=W<3", "W>=3")):
            m = ok & (wbin == b_i)
            if m.sum() < 30:
                continue
            e0, e1 = without[m] - dni[m], with_td[m] - dni[m]
            by_bin.append(dict(station=stn, w_bin_cm=lab, n=int(m.sum()), RMSE_sin_td=float(np.sqrt(np.mean(e0 ** 2))),
                               RMSE_con_td=float(np.sqrt(np.mean(e1 ** 2))), MBE_sin_td=float(e0.mean()), MBE_con_td=float(e1.mean())))
        log(f"  [{stn}] n={r['n_hours']} h / {r['n_days']} días | RMSE sin Td {r['RMSE_sin_td']:.2f} [{r['RMSE_sin_td_lo']:.2f}, {r['RMSE_sin_td_hi']:.2f}]"
            f" | con Td {r['RMSE_con_td']:.2f} [{r['RMSE_con_td_lo']:.2f}, {r['RMSE_con_td_hi']:.2f}]"
            f" | dif {r['dRMSE_con_menos_sin']:+.2f} [{r['dRMSE_lo']:+.2f}, {r['dRMSE_hi']:+.2f}] | MBE {r['MBE_sin_td']:+.1f} → {r['MBE_con_td']:+.1f}"
            f" | horas con GHI y sin Td: {n_td_nan}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "dirint_dewpoint_anchor.csv", index=False)
    bb = pd.DataFrame(by_bin)
    bb.to_csv(OUT / "dirint_dewpoint_by_wbin.csv", index=False)
    log("\nPor bin de agua precipitable (misma muestra c20):\n" + bb.round(2).to_string(index=False))
    (LOGS / f"dirint_dewpoint_{ts}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\nescrito {OUT / 'dirint_dewpoint_anchor.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
