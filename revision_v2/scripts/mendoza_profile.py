"""mendoza_profile.py — Perfil de los datos medidos de Mendoza (estación del INAHE), SOLO LECTURA: huecos, rangos, cierre D<=G, hora de la medición.

Entradas: data/Mendoza_Datos_Medidos/{2022..2025}_minutal.csv  (columnas: IRGLO = global horizontal, IRDIF_C = difusa horizontal corregida; 1 min).
Salidas : data/processed/mendoza_profile_by_year.csv, mendoza_profile_by_month.csv, mendoza_time_check.csv, data/MANIFEST_Mendoza.csv,
          logs/mendoza_profile_<ts>.txt.  No modifica los datos.

ADVERTENCIA: las coordenadas se pasan por línea de comandos (--lat --lon --elev) y NO hay valores por defecto: la estación debe ser confirmada por
quien midió. Este perfil solo usa la geometría para (a) separar día/noche y (b) probar si la hora del archivo es UTC o local; no depende de
diferencias de 0.01° en las coordenadas.

Uso (desde revision_v2, venv activo):  python scripts/mendoza_profile.py --lat <grados N, negativo=sur> --lon <grados E, negativo=oeste> --elev <m>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

REV = Path(__file__).resolve().parents[1]
RAW = REV / "data" / "Mendoza_Datos_Medidos"
OUT = REV / "data" / "processed"
LOGS = REV / "logs"
YEARS = (2022, 2023, 2024, 2025)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_year(y: int) -> pd.DataFrame:
    p = RAW / f"{y}_minutal.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if list(df.columns) != ["IRGLO", "IRDIF_C"]:
        raise ValueError(f"{p.name}: columnas inesperadas {list(df.columns)}")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{p.name}: marcas de tiempo duplicadas o desordenadas")
    steps = df.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(minutes=1):
        raise ValueError(f"{p.name}: la grilla no es regular de 1 min: pasos {list(steps)[:5]}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--elev", type=float, required=True)
    a = ap.parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"Perfil de datos de Mendoza {ts}  (lat={a.lat}, lon={a.lon}, elev={a.elev} — a confirmar por quien midió)")
    # manifiesto de hashes (los datos no se modifican)
    man = [dict(file=f"{y}_minutal.csv", bytes=(RAW / f"{y}_minutal.csv").stat().st_size, sha256=sha256(RAW / f"{y}_minutal.csv")) for y in YEARS]
    pd.DataFrame(man).to_csv(REV / "data" / "MANIFEST_Mendoza.csv", index=False)

    rows_y, rows_m, tcheck = [], [], []
    for y in YEARS:
        df = read_year(y)
        # hipótesis de hora: (L) hora local ART = UTC-3 ; (U) UTC. La marca se toma como inicio de minuto en ambas.
        res = {}
        for hyp, off in (("local_UTC-3", 3), ("UTC", 0)):
            t_utc = (df.index + pd.Timedelta(hours=off)).tz_localize("UTC")
            sp = pvlib.solarposition.get_solarposition(t_utc + pd.Timedelta(seconds=30), a.lat, a.lon, altitude=a.elev)
            res[hyp] = (t_utc, sp["apparent_zenith"].to_numpy())
        t_utc, zen = res["local_UTC-3"]                      # para el perfil se usa la hipótesis local; se contrasta abajo
        cosz = np.cos(np.radians(zen))
        g, d = df["IRGLO"].to_numpy(float), df["IRDIF_C"].to_numpy(float)
        sun = zen < 85.0
        e0 = pvlib.irradiance.get_extra_radiation(t_utc).to_numpy()
        gv, dv = np.isfinite(g), np.isfinite(d)
        both = gv & dv
        n_sun = int(sun.sum())
        row = dict(year=y, rows=len(df), expected_rows=525600 + (1440 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 0),
                   sun_minutes=n_sun, ghi_nan_pct_all=100 * float((~gv).mean()), dhi_nan_pct_all=100 * float((~dv).mean()),
                   sun_ghi_valid_pct=100 * float(gv[sun].mean()), sun_dhi_valid_pct=100 * float(dv[sun].mean()), sun_both_valid_pct=100 * float(both[sun].mean()),
                   ghi_min=float(np.nanmin(g)), ghi_max=float(np.nanmax(g)), dhi_min=float(np.nanmin(d)), dhi_max=float(np.nanmax(d)),
                   sun_ghi_negative=int((g[sun & gv] < 0).sum()), sun_dhi_negative=int((d[sun & dv] < 0).sum()),
                   sun_dhi_gt_ghi=int((d[sun & both] > g[sun & both]).sum()), sun_dhi_gt_ghi_by_more_than_20=int((d[sun & both] > g[sun & both] + 20).sum()),
                   ghi_gt_1p2_etr_cosz=int((g[sun & gv] > 1.2 * (e0 * cosz)[sun & gv]).sum()),
                   night_ghi_gt_20=int((g[(~sun) & gv] > 20).sum()), night_dhi_gt_20=int((d[(~sun) & dv] > 20).sum()))
        rows_y.append(row)
        m = pd.DataFrame({"g": g, "d": d, "sun": sun, "gv": gv, "dv": dv}, index=df.index)
        for mo, grp in m.groupby(m.index.month):
            s = grp[grp["sun"]]
            rows_m.append(dict(year=y, month=mo, sun_minutes=len(s), both_valid_pct=100 * float((s["gv"] & s["dv"]).mean())))
        # --- hora del archivo: centroide diario de la GHI de días despejados (Kt diario alto) vs mediodía solar de cada hipótesis
        daily = pd.DataFrame({"g": np.where(gv, g, np.nan)}, index=df.index)
        day_key = daily.index.floor("D")
        full_day = daily.groupby(day_key)["g"].apply(lambda s: s.notna().mean() > 0.97)
        kt_day = daily.groupby(day_key)["g"].sum() * 60 / 1e6 / pd.Series(
            pvlib.irradiance.get_extra_radiation(pd.DatetimeIndex(full_day.index).tz_localize("UTC")).to_numpy(), index=full_day.index) / 24.0
        clear_days = full_day[full_day & (kt_day > kt_day.quantile(0.85))].index
        for hyp in res:
            tu, zz = res[hyp]
            dfh = pd.DataFrame({"g": np.where(gv, np.clip(g, 0, None), np.nan), "z": zz, "t": tu}, index=df.index)
            offs = []
            for dd in clear_days:
                seg = dfh.loc[dd:dd + pd.Timedelta(hours=23, minutes=59)]
                seg = seg[(seg["g"] > 50) & seg["g"].notna()]
                if len(seg) < 300:
                    continue
                tt = (seg.index - dd).total_seconds().to_numpy() / 3600.0         # hora del reloj del archivo [h]
                centroid = float(np.average(tt, weights=seg["g"].to_numpy()))
                # mediodía solar del día en hora de reloj del archivo = instante del cenit mínimo
                noon = float(tt[np.argmin(seg["z"].to_numpy())])
                offs.append(centroid - noon)
            tcheck.append(dict(year=y, hypothesis=hyp, clear_days_used=len(offs), median_centroid_minus_noon_min=60 * float(np.median(offs)) if offs else np.nan,
                               p05_min=60 * float(np.percentile(offs, 5)) if offs else np.nan, p95_min=60 * float(np.percentile(offs, 95)) if offs else np.nan))
    ty, tm, tc = pd.DataFrame(rows_y), pd.DataFrame(rows_m), pd.DataFrame(tcheck)
    ty.to_csv(OUT / "mendoza_profile_by_year.csv", index=False)
    tm.to_csv(OUT / "mendoza_profile_by_month.csv", index=False)
    tc.to_csv(OUT / "mendoza_time_check.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    log("\nPOR AÑO (día = cenit < 85°, hora local UTC−3):\n" + ty.round(2).T.to_string())
    log("\nCOBERTURA MENSUAL (% de minutos de día con global y difusa válidas):\n" + tm.pivot(index="month", columns="year", values="both_valid_pct").round(1).to_string())
    log("\nHORA DEL ARCHIVO: centroide de la GHI de días despejados menos el mediodía solar (min). Cerca de 0 = hipótesis correcta:\n" + tc.round(2).to_string(index=False))
    (LOGS / f"mendoza_profile_{ts}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
