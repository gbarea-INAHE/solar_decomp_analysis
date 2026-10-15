"""
gompertz.py — Gompertz direct-normal split (Zhang et al. 2004), as used by the
IWEC2 weather-file generation chain (ZHM GHI -> Gompertz DNI -> ASHRAE clear sky).

Port of the Fortran subroutine GOMPERTZ posted by J. Huang (Bldg-sim list,
31 May 2015, coded by D. Seo). Source text kept verbatim in
revision_v2/scripts/gompertz_ref.f.txt so the port can be re-validated.

Validation status: see validate_gompertz_port.py (agrees with a literal
Fortran->C transliteration in REAL*4 and REAL*8).
"""
from __future__ import annotations

import numpy as np

# Coefficients exactly as in the Fortran source.
_A1 = (-0.1556, 0.1028, 1.3748)   # quadratic in cos(z): c2, c1, c0
_A2 = (0.7973, 0.1509, 3.035)
_A3 = (5.4307, 7.2182)            # linear in cos(z): c1, c0
_A4 = 2.99


def gompertz_kn(kt: np.ndarray, cos_zenith: np.ndarray) -> np.ndarray:
    """Direct-normal transmittance KN = DNI / I0 for clearness index kt = GHI/(I0 cos z)."""
    a1 = _A1[0] * cos_zenith**2 + _A1[1] * cos_zenith + _A1[2]
    a2 = _A2[0] * cos_zenith**2 + _A2[1] * cos_zenith + _A2[2]
    a3 = _A3[0] * cos_zenith + _A3[1]
    kn1 = 1.0 / a2 ** (_A4 * kt)
    return a1 * a2 ** (-a3 * kn1)


def gompertz_dni(ghi_wm2: np.ndarray, cos_zenith: np.ndarray, i0_wm2: float) -> np.ndarray:
    """Gompertz DNI [W/m2] from GHI [W/m2] and cos(zenith) [-].

    i0_wm2 is the solar constant used by the generator (1355 in the original
    code; the comment in the source recommends 1367). It is a *parameter to be
    tested*, never defaulted here.
    """
    ghi_wm2 = np.asarray(ghi_wm2, dtype=float)
    cos_zenith = np.asarray(cos_zenith, dtype=float)
    if ghi_wm2.shape != cos_zenith.shape:
        raise ValueError(f"shape mismatch: ghi {ghi_wm2.shape} vs cos_zenith {cos_zenith.shape}")
    sun_up = cos_zenith > 0.0
    safe_cos = np.where(sun_up, cos_zenith, 1.0)
    kt = np.where(sun_up, ghi_wm2 / (i0_wm2 * safe_cos), 0.0)
    kn = gompertz_kn(kt, np.where(sun_up, cos_zenith, 0.0))
    return np.where(sun_up, kn * i0_wm2, 0.0)


def gompertz_dhi(ghi_wm2: np.ndarray, cos_zenith: np.ndarray, i0_wm2: float) -> np.ndarray:
    """DHI by closure, DHI = GHI - DNI cos z (assumed, to be confirmed against EPW data)."""
    dni = gompertz_dni(ghi_wm2, cos_zenith, i0_wm2)
    return np.asarray(ghi_wm2, dtype=float) - dni * np.asarray(cos_zenith, dtype=float)
