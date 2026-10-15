"""modern_models.py — Engerer2 (Bright & Engerer 2019) y BRL (Ridley, Boland & Lauret 2010) para el análisis contra mediciones.

NO es parte de la herramienta Solar Decomp: es código de análisis de la revisión, escrito a partir de fórmulas y coeficientes confirmados en
DOS fuentes independientes (2026-10-05):
  Engerer2  : repositorio del autor (JamieMBright/Engerer2-separation-model, Engerer2Separation.py) y jararias/splitting_models (models.py).
              Fórmula y parámetros de 1 min: coinciden en ambas. Parámetros de 60 min: SOLO el repositorio del autor (README y código
              coinciden entre sí; no cotejados con el artículo ni con una segunda implementación) -> resultados horarios EXPLORATORIOS.
  BRL       : jararias/splitting_models (models.py) y la Tabla 5 de arXiv:2007.01639 (A1..A6 = -5.38, 6.63, 0.006, -0.007, 1.75, 1.31, ec. 8):
              coinciden. La definición exacta de la persistencia psi y del Kt diario sigue la implementación de splitting_models
              (promedio centrado de las horas vecinas); el artículo original no se pudo leer (acceso por VPN).
Starke3 NO se implementa: coeficientes de una sola fuente (splitting_models, "vía Yang 2022"), predictores (Kt horario, GHI de cielo claro)
con definiciones no confirmadas.
Cada modelo está escrito dos veces (vectorizado y escalar con math) y selftest() exige igualdad.

Convenciones: GHI en W/m2; tiempos UTC (marca = centro del intervalo); Z, alpha en grados; AST en horas.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------- parámetros
# Kd = C + (1-C)/(1+exp(b0 + b1*Kt + b2*AST + b3*Z + b4*dKtc)) + b5*Kde      (Engerer 2015; Bright & Engerer 2019)
ENGERER2 = {
    1:  dict(C=0.10562,    b0=-4.1332, b1=8.2578, b2=0.010087, b3=0.0008880, b4=-4.9302, b5=0.44378),   # 2 fuentes concordantes
    60: dict(C=-0.0097539, b0=-5.3169, b1=8.5084, b2=0.013241, b3=0.0074356, b4=-3.0329, b5=0.56403),   # fuente única (autor)
}
BRL_A = (-5.38, 6.63, 0.006, -0.007, 1.75, 1.31)       # A1..A6; 2 fuentes concordantes


# ---------------------------------------------------------------------------------------------------------------- geometría (Bright)
def geometry_bright(t_utc: pd.DatetimeIndex, lat: float, lon: float):
    """Zenith [deg], AST [h], E_extn y E_exth [W/m2], DOY, con las fórmulas del código de referencia de Engerer2."""
    t = pd.DatetimeIndex(t_utc)
    doy = t.dayofyear.to_numpy(float)
    dec_t = t.hour.to_numpy(float) + t.minute.to_numpy(float) / 60.0 + t.second.to_numpy(float) / 3600.0
    beta_e = np.radians(360.0 / 365.242 * (doy - 1))
    eot = 0.258 * np.cos(beta_e) - 7.416 * np.sin(beta_e) - 3.648 * np.cos(2 * beta_e) - 9.228 * np.sin(2 * beta_e)
    lsn = 12.0 - lon / 15.0 - eot / 60.0
    w = (dec_t - lsn) * 15.0
    w = (w + 180.0) % 360.0 - 180.0
    ph = 2 * np.pi * (doy + dec_t / 24.0 - 1) / 365.0
    dec = (0.006918 - 0.399912 * np.cos(ph) + 0.070257 * np.sin(ph) - 0.006758 * np.cos(2 * ph) + 0.000907 * np.sin(2 * ph)
           - 0.002697 * np.cos(3 * ph) + 0.001480 * np.sin(3 * ph))
    cz = np.sin(np.radians(lat)) * np.sin(dec) + np.cos(np.radians(lat)) * np.cos(dec) * np.cos(np.radians(w))
    z = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    b = 2 * np.pi * doy / 365.0
    e_n = 1366.1 * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b) + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))
    e_h = np.where(z <= 90, e_n * np.cos(np.radians(z)), 0.0)
    return z, w / 15.0 + 12.0, e_n, e_h, doy


def tuned_jiang(z_deg, doy):
    """Cielo claro 'Tuned Jiang' del código de referencia de Engerer2: devuelve GHI_cs, DNI_cs."""
    a = 1160 + 75 * np.sin(np.radians(360 * (doy - 275) / 365))
    k = 0.174 + 0.035 * np.sin(np.radians(360 * (doy - 100) / 365))
    c = 0.095 + 0.04 * np.sin(np.radians(360 * (doy - 100) / 365))
    cz = np.cos(np.radians(z_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        dni = np.where(z_deg <= 90, a * np.exp(-k / np.where(cz > 0, cz, np.nan)), 0.0)
    dni = np.nan_to_num(dni, nan=0.0)
    return dni * cz * (z_deg <= 90) + c * dni, dni


# ---------------------------------------------------------------------------------------------------------------- Engerer2
def engerer2_kd(ghi, z, ast, e_h, ghi_cs, par, kde_offset=0.015):
    """Vectorizado. Devuelve Kd en [0,1] (NaN donde Z>=85, Kt no definido o GHI<=0)."""
    ghi = np.asarray(ghi, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = ghi / e_h
        ktc = ghi_cs / e_h
        dktc = ktc - kt
        kde = np.maximum(0.0, ghi - ghi_cs - kde_offset) / ghi
        arg = par["b0"] + par["b1"] * kt + par["b2"] * ast + par["b3"] * z + par["b4"] * dktc
        kd = par["C"] + (1 - par["C"]) / (1 + np.exp(arg)) + par["b5"] * kde
    kd = np.clip(kd, 0.0, 1.0)
    return np.where((z < 85.0) & np.isfinite(kt) & (e_h > 0) & (ghi > 0), kd, np.nan)


def engerer2_kd_scalar(ghi, z, ast, e_h, ghi_cs, par, kde_offset=0.015):
    """Referencia escalar independiente (math)."""
    if not (z < 85.0 and e_h > 0 and ghi > 0 and math.isfinite(ghi)):
        return float("nan")
    kt, ktc = ghi / e_h, ghi_cs / e_h
    dktc = ktc - kt
    kde = max(0.0, ghi - ghi_cs - kde_offset) / ghi
    arg = par["b0"] + par["b1"] * kt + par["b2"] * ast + par["b3"] * z + par["b4"] * dktc
    kd = par["C"] + (1 - par["C"]) / (1 + math.exp(arg)) + par["b5"] * kde
    return min(1.0, max(0.0, kd))


def engerer2_dni(ghi, t_utc, lat, lon, resolution: int, kde_offset=0.015, e_h_override=None, z_override=None):
    """DNI de Engerer2 con el cielo claro y la geometría del código de referencia. *_override permite pasar Z y E_exth de un intervalo
    (uso horario: coseno promedio del intervalo)."""
    z, ast, e_n, e_h, doy = geometry_bright(t_utc, lat, lon)
    if z_override is not None:
        z = z_override
        e_h = e_n * np.cos(np.radians(z))
    if e_h_override is not None:
        e_h = e_h_override
    ghi_cs, _ = tuned_jiang(z, doy)
    kd = engerer2_kd(ghi, z, ast, e_h, ghi_cs, ENGERER2[resolution], kde_offset)
    cz = np.cos(np.radians(z))
    with np.errstate(divide="ignore", invalid="ignore"):
        dni = np.asarray(ghi, float) * (1 - kd) / cz
    return np.where(np.isfinite(kd), np.clip(dni, 0.0, e_n), np.nan), kd


# ---------------------------------------------------------------------------------------------------------------- BRL (horario)
def brl_kd(kt, ast, alpha_deg, kt_daily, psi, a=BRL_A):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(a[0] + a[1] * kt + a[2] * ast + a[3] * alpha_deg + a[4] * kt_daily + a[5] * psi))


def brl_kd_scalar(kt, ast, alpha_deg, kt_daily, psi, a=BRL_A):
    x = a[0] + a[1] * kt + a[2] * ast + a[3] * alpha_deg + a[4] * kt_daily + a[5] * psi
    return 1.0 / (1.0 + math.exp(min(x, 700.0)))


def persistence(kt: np.ndarray, daytime: np.ndarray, night_as_zero: bool = False) -> np.ndarray:
    """psi = media de Kt de las horas vecinas (anterior y siguiente). Por defecto solo cuentan los vecinos con sol y valor válido (en la
    salida/puesta de sol se usa el único vecino); con night_as_zero=True un vecino sin sol aporta 0."""
    k = np.where(daytime, kt, np.nan)
    if night_as_zero:
        k = np.where(~daytime, 0.0, k)
    prev = np.full(len(k), np.nan)
    nxt = np.full(len(k), np.nan)
    prev[1:] = k[:-1]
    nxt[:-1] = k[1:]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.vstack([prev, nxt]), axis=0)


def daily_kt(ghi: np.ndarray, e_h: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Kt diario = sum(GHI) / sum(E_exth) sobre las horas con GHI válido y sol del día local; se repite en cada hora del día."""
    ok = np.isfinite(ghi) & (e_h > 0)
    df = pd.DataFrame({"g": np.where(ok, ghi, 0.0), "e": np.where(ok, e_h, 0.0), "d": day_id})
    s = df.groupby("d").sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (s["g"] / s["e"]).where(s["e"] > 0)
    return np.clip(df["d"].map(k).to_numpy(float), 0.0, 1.0)


def brl_dni_hourly(ghi, e_n, e_h, cosz_avg, ast, day_id, night_as_zero_psi=False):
    """BRL horario. Kt horario = GHI/E_exth del intervalo; alpha = elevación a partir del coseno cenital promedio del intervalo."""
    ghi = np.asarray(ghi, float)
    day = (e_h > 0) & np.isfinite(ghi)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(e_h > 0, ghi / e_h, np.nan)
    psi = persistence(kt, e_h > 0, night_as_zero_psi)
    ktd = daily_kt(ghi, e_h, day_id)
    alpha = np.degrees(np.arcsin(np.clip(cosz_avg, 0, 1)))
    kd = np.where(day & np.isfinite(kt) & np.isfinite(psi), brl_kd(kt, ast, alpha, ktd, psi), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dni = ghi * (1 - kd) / np.where(cosz_avg > 0, cosz_avg, np.nan)
    return np.where(np.isfinite(kd), np.clip(dni, 0.0, e_n), np.nan), kd, kt, psi, ktd


# ---------------------------------------------------------------------------------------------------------------- autoprueba
def selftest(verbose=True) -> None:
    rng = np.random.default_rng(7)
    # 1) Engerer2: vectorizado == escalar sobre 4000 puntos aleatorios, ambas resoluciones y ambas definiciones de Kde
    t = pd.to_datetime("2016-01-01") + pd.to_timedelta(rng.integers(0, 366 * 1440, 4000), unit="min") + pd.Timedelta(seconds=30)
    lat, lon = -29.44, -53.82
    z, ast, e_n, e_h, doy = geometry_bright(t, lat, lon)
    ghi_cs, _ = tuned_jiang(z, doy)
    ghi = np.where(z < 90, ghi_cs * rng.uniform(0.05, 1.15, len(t)), 0.0)
    for res in (1, 60):
        for off in (0.015, 0.0):
            v = engerer2_kd(ghi, z, ast, e_h, ghi_cs, ENGERER2[res], off)
            s = np.array([engerer2_kd_scalar(ghi[i], z[i], ast[i], e_h[i], ghi_cs[i], ENGERER2[res], off) for i in range(len(t))])
            assert np.allclose(v, s, equal_nan=True, rtol=0, atol=1e-12), (res, off)
    # 2) BRL: vectorizado == escalar
    kt = rng.uniform(0, 1.1, 4000); a_ = rng.uniform(0, 24, 4000); al = rng.uniform(1, 90, 4000); kd_ = rng.uniform(0, 1, 4000); ps = rng.uniform(0, 1.1, 4000)
    assert np.allclose(brl_kd(kt, a_, al, kd_, ps), [brl_kd_scalar(*x) for x in zip(kt, a_, al, kd_, ps)], rtol=0, atol=1e-12)
    # 3) propiedades: Kd en [0,1]; Engerer2 sube con ΔKtc negativo (cielo despejado) -> menos difusa; BRL baja con Kt
    k = engerer2_kd(ghi, z, ast, e_h, ghi_cs, ENGERER2[1])
    assert np.nanmin(k) >= 0 and np.nanmax(k) <= 1
    k1 = brl_kd(np.array([0.2, 0.5, 0.8]), 12, 60, 0.5, 0.5)
    assert k1[0] > k1[1] > k1[2], "BRL debe disminuir con Kt"
    # 4) punto calculado a mano (Engerer2, 60 min): Kt=0.5, ktc=0.8 -> dKtc=0.3, AST=12, Z=40, Kde=0
    p = ENGERER2[60]
    arg = p["b0"] + p["b1"] * 0.5 + p["b2"] * 12 + p["b3"] * 40 + p["b4"] * 0.3
    esperado = p["C"] + (1 - p["C"]) / (1 + math.exp(arg))
    got = engerer2_kd_scalar(500.0, 40.0, 12.0, 1000.0, 800.0, p)
    assert abs(got - min(1, max(0, esperado))) < 1e-15
    # 5) geometría de Bright contra pvlib (SPA): diferencias pequeñas
    import pvlib
    zz = pvlib.solarposition.get_solarposition(t.tz_localize("UTC"), lat, lon, method="nrel_numpy")["zenith"].to_numpy()
    m = zz < 85
    dz = np.abs(z[m] - zz[m]).max()
    e0 = pvlib.irradiance.get_extra_radiation(t.tz_localize("UTC"), method="spencer").to_numpy()
    de = np.abs(e_n / e0 - 1).max()
    assert dz < 0.5 and de < 0.003, (dz, de)
    if verbose:
        print(f"selftest OK: vectorizado == escalar (Engerer2 x2 resoluciones x2 Kde, BRL); límites y propiedades; punto a mano; "
              f"geometría de Bright vs pvlib: max|dZ|={dz:.3f} deg, max|dE0/E0|={de:.4f}")


if __name__ == "__main__":
    selftest()
