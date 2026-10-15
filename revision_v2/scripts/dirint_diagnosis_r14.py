"""dirint_diagnosis_r14.py — diagnóstico para R1-4 / R2-3 sobre la serie horaria de las 39 estaciones IWEC2 (benchmark_v2).
Pone a prueba, con los datos, las explicaciones que dio el manuscrito enviado y la que sugiere el revisor 1 (masa de aire alta y rango estrecho de Kt′ bajo cielo claro persistente):
  A. Sesgo de DNI y RMSE por intervalo de masa de aire (AM), con la parte de horas de cada intervalo. En c20 (cos z ≥ 0.2) AM ≤ ~5: la masa de aire > 5 solo existe en el dominio «all».
  B. Sesgo por intervalo de ΔKt′ (variabilidad hora a hora de Kt′; diferencia hacia atrás entre filas consecutivas de la misma estación) y por Kt′ × ΔKt′ para la tabla v1.0.0.
  C. R² de DNI por estación para cada DIRINT: cuántas estaciones bajo 0.5 y bajo 0 (el manuscrito enviado reportó dos con R² negativo: Esquel y Ushuaia).
  D. Estaciones de latitud alta (|lat| > 40°) frente al resto: Kt′ y AM medianos, sesgo por modelo.
  E. Correlación de Spearman, entre estaciones, del sesgo con tres proxies de claridad/aridez (F(Kt>0.65), F(Kt′≥0.8), Kt medio) y con la masa de aire mediana.
Sin hipótesis previas: imprime y guarda lo que sale. Provisorio (bins P4; ver DECISIONS_PROVISIONAL.md). No modifica nada existente.
Salidas: data/processed/r14_*.csv.   Uso (desde revision_v2, venv activo): python scripts/dirint_diagnosis_r14.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REV = HERE.parent
PROC = REV / "data" / "processed"
DOMS = {"c20": 0.2, "all": 0.0174}
DIAG = ["dirint_v1", "dirint_std", "dirint_std_noTd", "erbs", "reindl2"]
AM_BINS = [0.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
DKT_BINS = [-1e-9, 0.015, 0.035, 0.07, 0.15, np.inf]          # los bins de ΔKt′ de Perez (y de la tabla v1.0.0)
KTP_COARSE = [0.0, 0.4, 0.56, 0.7, 0.8, 1.1]
HILAT = 40.0


def rm(ref, pred):
    m = np.isfinite(ref) & np.isfinite(pred)
    e = (pred - ref)[m]
    return len(e), float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan, float(e.mean()) if len(e) else np.nan


def prep() -> pd.DataFrame:
    h = pd.read_pickle(PROC / "hourly_v2.pkl.gz")
    h = h[h["source"] == "IWEC2"].copy()
    h["ktp"] = h["kt"] / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / h["am"].clip(upper=10.0))) + 0.1)
    h = h.sort_values(["city", "month", "doy", "hour"]).reset_index(drop=True)
    # ΔKt′ hacia atrás: solo si la fila anterior es la hora inmediata anterior de la misma estación y mismo día (o 23 h → 0 h del día siguiente, que no existe en filas diurnas)
    prev_ktp = h.groupby("city")["ktp"].shift(1)
    prev_hour = h.groupby("city")["hour"].shift(1)
    prev_doy = h.groupby("city")["doy"].shift(1)
    ok = (prev_doy == h["doy"]) & (prev_hour == h["hour"] - 1)
    h["dktp"] = np.where(ok, (h["ktp"] - prev_ktp).abs(), np.nan)
    return h


def station_table(h: pd.DataFrame, bm: pd.DataFrame) -> pd.DataFrame:
    lat = bm.drop_duplicates("city").set_index("city")["lat"]
    return lat


def main() -> int:
    h = prep()
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    bm = bm[bm["source"] == "IWEC2"]
    lat = bm.drop_duplicates("city").set_index("city")["lat"]
    h["lat"] = h["city"].map(lat)
    out = {}
    for dom, c0 in DOMS.items():
        d = h[h["cosz"] >= c0]
        # ---------------- A. masa de aire
        rows = []
        for b, g in d.groupby(pd.cut(d["am"], AM_BINS, right=False), observed=True):
            for m in DIAG:
                n, r, mb = rm(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())
                rows.append(dict(domain=dom, am_bin=str(b), share_pct=100 * len(g) / len(d), model=m, n=n, RMSE=r, MBE=mb, ref_mean=float(g["dni_ref"].mean())))
        pd.DataFrame(rows).to_csv(PROC / f"r14_am_bins_{dom}.csv", index=False)
        # ---------------- B. ΔKt′
        rows = []
        dd = d[d["dktp"].notna()]
        for b, g in dd.groupby(pd.cut(dd["dktp"], DKT_BINS), observed=True):
            for m in DIAG:
                n, r, mb = rm(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())
                rows.append(dict(domain=dom, dktp_bin=str(b), share_pct=100 * len(g) / len(dd), model=m, n=n, RMSE=r, MBE=mb))
        pd.DataFrame(rows).to_csv(PROC / f"r14_dktp_bins_{dom}.csv", index=False)
        rows = []
        for (kb, db), g in dd.groupby([pd.cut(dd["ktp"], KTP_COARSE, right=False), pd.cut(dd["dktp"], DKT_BINS)], observed=True):
            for m in ("dirint_v1", "dirint_std_noTd"):
                n, r, mb = rm(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())
                rows.append(dict(domain=dom, ktp_bin=str(kb), dktp_bin=str(db), model=m, n=n, share_pct=100 * len(g) / len(dd), MBE=mb, RMSE=r))
        pd.DataFrame(rows).to_csv(PROC / f"r14_ktp_dktp_{dom}.csv", index=False)
        # ---------------- C. R² por estación
        b = bm[bm["domain"] == dom]
        rows = []
        for m in DIAG:
            x = b[b["model"] == m]
            rows.append(dict(domain=dom, model=m, n_stations=len(x), R2_mean=x["DNI_R2"].mean(), R2_median=x["DNI_R2"].median(), R2_min=x["DNI_R2"].min(),
                             city_min=x.loc[x["DNI_R2"].idxmin(), "city"], n_lt0=int((x["DNI_R2"] < 0).sum()), n_lt05=int((x["DNI_R2"] < 0.5).sum()),
                             cities_lt0="; ".join(x.loc[x["DNI_R2"] < 0, "city"]), MBE_mean=x["DNI_MBE"].mean(), MBE_min=x["DNI_MBE"].min(), MBE_max=x["DNI_MBE"].max(),
                             n_MBE_pos=int((x["DNI_MBE"] > 0).sum())))
        pd.DataFrame(rows).to_csv(PROC / f"r14_station_r2_{dom}.csv", index=False)
        # ---------------- D. latitud alta y E. correlaciones
        per = []
        for c, g in d.groupby("city"):
            r = dict(city=c, lat=float(lat[c]), hilat=bool(abs(lat[c]) > HILAT), ktp_median=float(g["ktp"].median()), ktp_p90=float(g["ktp"].quantile(0.9)), ktp_sd=float(g["ktp"].std()),
                     am_median=float(g["am"].median()), am_p90=float(g["am"].quantile(0.9)), kt_mean=float(g["kt"].mean()), f_kt065=float((g["kt"] > 0.65).mean()),
                     f_ktp08=float((g["ktp"] >= 0.8).mean()), dktp_median=float(g["dktp"].median()))
            for m in DIAG:
                r[f"mbe_{m}"] = rm(g["dni_ref"].to_numpy(), g[f"dni_{m}"].to_numpy())[2]
            per.append(r)
        per = pd.DataFrame(per)
        per.to_csv(PROC / f"r14_station_diag_{dom}.csv", index=False)
        rows = []
        for hi, g in per.groupby("hilat"):
            r = dict(domain=dom, hilat=bool(hi), n_stations=len(g), ktp_median=g["ktp_median"].mean(), ktp_sd=g["ktp_sd"].mean(), am_median=g["am_median"].mean(), kt_mean=g["kt_mean"].mean())
            for m in DIAG:
                r[f"mbe_{m}"] = g[f"mbe_{m}"].mean()
            rows.append(r)
        pd.DataFrame(rows).to_csv(PROC / f"r14_hilat_{dom}.csv", index=False)
        rows = []
        for proxy in ("f_kt065", "f_ktp08", "kt_mean", "am_median", "ktp_sd"):
            for m in DIAG:
                rho, p = stats.spearmanr(per[proxy], per[f"mbe_{m}"])
                rows.append(dict(domain=dom, proxy=proxy, model=m, rho=float(rho), p=float(p), n=len(per)))
        pd.DataFrame(rows).to_csv(PROC / f"r14_proxy_corr_{dom}.csv", index=False)
        out[dom] = dict(am=pd.read_csv(PROC / f"r14_am_bins_{dom}.csv"), r2=pd.read_csv(PROC / f"r14_station_r2_{dom}.csv"))
        print(f"\n=== {dom}: n={len(d):,} h")
        print(out[dom]["am"].pivot(index="am_bin", columns="model", values="MBE").round(1)[DIAG].to_string())
        print(out[dom]["r2"][["model", "R2_mean", "R2_min", "city_min", "n_lt0", "n_lt05", "cities_lt0", "MBE_mean", "n_MBE_pos"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
