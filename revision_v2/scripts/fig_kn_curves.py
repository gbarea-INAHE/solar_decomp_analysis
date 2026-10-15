"""
fig_kn_curves.py — Direct-normal transmittance Kn = DNI / G0n as a function of the clearness index, for each model.

Because the IWEC2 DNI is the Gompertz split of its own GHI (Test 0), the benchmark measures how close each model's
curve is to the Gompertz curve. This figure shows those curves at fixed cos z, using the same functions the
benchmark uses (solar_decomp v1.0.0 for Erbs / Reindl-2 / DIRINT table; pvlib for DIRINT-standard).

 * Erbs, Reindl-2 : closed form through the diffuse fraction, Kn = Kt (1 - Kd) with DNI cosz = GHI - DHI.
 * Gompertz       : gompertz.gompertz_kn(Kt, cosz).
 * DIRINT v1.0.0  : the table in decomposition.py, evaluated at the Kt' implied by (Kt, AM), W bin and dKt' bin fixed
                    (W = 1.5 cm; dKt' = 0.0, i.e. persistent conditions) — a slice of a 3-D table, not the whole model.
 * DIRINT standard: pvlib.irradiance.dirint on a synthetic constant-Kt day (dKt' off), W from a 5 C dew point.

Output: figures/fig_kn_curves.png (300 dpi, Okabe-Ito, no title).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import decomposition as dec  # noqa: E402
from gompertz import gompertz_kn  # noqa: E402
from solar_geometry import air_mass_kasten  # noqa: E402

FIG = ROOT / "revision_v2" / "figures"
OKABE = {"gomp": "#000000", "erbs": "#0072B2", "reindl2": "#009E73", "v1": "#D55E00", "std": "#CC79A7"}


def kn_erbs(kt):
    kd = np.where(kt <= 0.22, 1 - 0.09 * kt, np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))
    return kt * (1 - np.clip(kd, 0, 1))


def kn_reindl2(kt, cosz):
    s = cosz
    kd = np.where(kt <= 0.3, 1.020 - 0.254 * kt + 0.0123 * s, np.where(kt < 0.78, 1.400 - 1.749 * kt + 0.177 * s, 0.486 * kt - 0.182 * s))
    return kt * (1 - np.clip(kd, 0, 1))


def kn_dirint_v1(kt, cosz, w_cm=1.5, dkt=0.0):
    am = air_mass_kasten(float(np.degrees(np.arccos(cosz))))
    out = []
    for k in kt:
        ktp = dec._kt_prime(float(k), am)
        wb, db = dec._bin_index(w_cm, dec._W_EDGES), dec._bin_index(dkt, dec._DKT_EDGES)
        out.append(dec._kn_interpolated(wb, db, min(max(ktp, 0.0), 1.0)))
    return np.array(out)


def kn_dirint_std(kt_grid, cosz):
    """pvlib DIRINT on a constant-condition synthetic day; returns Kn for each Kt in the grid at (almost) fixed cos z."""
    lat, lon = -32.83, -68.78
    times = pd.date_range("2001-12-21 08:00", periods=1, freq="h", tz="Etc/GMT+3")
    kn = []
    for k in kt_grid:
        z = float(np.degrees(np.arccos(cosz)))
        e0 = pvlib.irradiance.get_extra_radiation(times, method="spencer").to_numpy()[0]
        ghi = k * e0 * cosz
        t = pd.date_range("2001-12-21 10:00", periods=3, freq="h", tz="Etc/GMT+3")
        r = pvlib.irradiance.dirint(pd.Series([ghi] * 3, index=t), pd.Series([z] * 3, index=t), t, pressure=93000.0,
                                    use_delta_kt_prime=True, temp_dew=pd.Series([5.0] * 3, index=t), min_cos_zenith=0.0174, max_zenith=89.0)
        kn.append(float(r.iloc[1]) / e0)
    return np.array(kn)


def main() -> None:
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kt = np.linspace(0.05, 0.9, 60)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, cz in zip(axs, (0.35, 0.8)):
        ax.plot(kt, gompertz_kn(kt, np.full_like(kt, cz)), color=OKABE["gomp"], lw=1.8, label="Gompertz (IWEC2 reference)")
        ax.plot(kt, kn_erbs(kt), color=OKABE["erbs"], lw=1.2, label="Erbs")
        ax.plot(kt, kn_reindl2(kt, cz), color=OKABE["reindl2"], lw=1.2, label="Reindl-2")
        ax.plot(kt, kn_dirint_v1(kt, cz), color=OKABE["v1"], lw=1.2, ls="--", label="DIRINT, v1.0.0 table (slice)")
        ax.plot(kt, kn_dirint_std(kt, cz), color=OKABE["std"], lw=1.2, ls="-.", label="DIRINT, Perez 1992 (pvlib)")
        ax.set_xlabel("Clearness index $K_t$"); ax.text(0.04, 0.9, f"cos z = {cz}", transform=ax.transAxes, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("$K_n$ = DNI / $G_{0n}$")
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_kn_curves.png", dpi=300)
    print("saved", FIG / "fig_kn_curves.png")

    print("\nKn at selected Kt for cos z = 0.8:")
    ks = np.array([0.3, 0.45, 0.6, 0.7, 0.8])
    cz = 0.8
    tbl = pd.DataFrame({"Kt": ks, "Gompertz": gompertz_kn(ks, np.full_like(ks, cz)), "Erbs": kn_erbs(ks), "Reindl2": kn_reindl2(ks, cz),
                        "DIRINT_v1": kn_dirint_v1(ks, cz), "DIRINT_std": kn_dirint_std(ks, cz)})
    print(tbl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
