"""
reproduce_paper_pvlib_claim.py — Try to reproduce the cross-validation statement in the
submitted manuscript (Sect. 2, software description):

  "The DIRINT and Erbs implementations were cross-validated against pvlib-python 0.10.4
   on the included synthetic test dataset (test_data_15min.csv); residual deviations were
   below 0.5 W/m2 across the full annual record, attributable to floating-point rounding
   in the Kt' interpolation step."

Same dataset, same repo pipeline as test_pipeline.py (Buenos Aires, tz -3, altitude 0),
then pvlib on the identical hourly GHI. The test CSV has no dew point, so pvlib DIRINT
is run without temp_dew (its documented 'missing W' bin). Two zenith conventions are
tried for pvlib: pvlib's own solar position at the interval centre, and solar_decomp's.

Pass criterion stated by the manuscript: max |difference| < 0.5 W/m2.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from decomposition import run_decomposition  # noqa: E402
from io_handler import load_file  # noqa: E402
from preprocessor import aggregate_to_hourly  # noqa: E402

LAT, LON, TZ = -34.60, -58.38, -3.0   # as in test_pipeline.py


def diff_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    m = mask & np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    return dict(n=int(m.sum()), max_abs=float(np.abs(d).max()), mean_diff=float(d.mean()),
                rmse=float(np.sqrt((d ** 2).mean())), frac_gt_0p5=float((np.abs(d) > 0.5).mean()))


def main() -> None:
    warnings.filterwarnings("ignore")
    print(f"pvlib {pvlib.__version__}  (manuscript states 0.10.4)")
    ld = load_file(ROOT / "test_data_15min.csv")
    df_h = aggregate_to_hourly(ld, lat_deg=LAT, lon_deg=LON, tz_hr=TZ)
    rep = run_decomposition(df_h, primary="DIRINT")

    centre = pd.DatetimeIndex(pd.to_datetime(rep["timestamp_center"])).tz_localize("Etc/GMT+3")
    loc = pvlib.location.Location(LAT, LON, altitude=0.0, tz="Etc/GMT+3")
    sp = loc.get_solarposition(centre)
    ghi = pd.Series(rep["GHI_h"].to_numpy(), index=centre)
    p_pa = rep["pressure_kPa"].to_numpy() * 1000.0

    rows = []
    for zlabel, zen in (("pvlib zenith", sp["zenith"]), ("solar_decomp zenith", pd.Series(rep["zenith_deg"].to_numpy(), index=centre))):
        pv_d = pvlib.irradiance.dirint(ghi, zen, centre, pressure=p_pa, use_delta_kt_prime=True,
                                       temp_dew=None, min_cos_zenith=0.065, max_zenith=87)
        pv_e = pvlib.irradiance.erbs(ghi, zen, centre)
        day = (rep["G0h"].to_numpy() > 0) & (rep["GHI_h"].to_numpy() >= 10.0)
        hi = day & (rep["cos_Z"].to_numpy() > 0.3)
        for mlabel, mask in (("all daytime h (GHI>=10)", day), ("cos z > 0.3", hi)):
            rows.append(dict(zenith=zlabel, hours=mlabel, comparison="DIRINT DNI",
                             **diff_stats(rep["DNI_dirint"].to_numpy(), pv_d.to_numpy(), mask)))
            rows.append(dict(zenith=zlabel, hours=mlabel, comparison="Erbs DNI",
                             **diff_stats(rep["DNI_erbs"].to_numpy(), pv_e["dni"].to_numpy(), mask)))
            rows.append(dict(zenith=zlabel, hours=mlabel, comparison="Erbs DHI",
                             **diff_stats(rep["DHI_erbs"].to_numpy(), pv_e["dhi"].to_numpy(), mask)))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nsolar_decomp minus pvlib, W/m2 (manuscript claims max |diff| < 0.5):")
    print(out.round(3).to_string(index=False))
    claim_ok = bool((out["max_abs"] < 0.5).all())
    print("\nManuscript claim (all comparisons max|diff| < 0.5 W/m2) reproduced:", claim_ok)
    out.to_csv(ROOT / "revision_v2" / "data" / "processed" / "claim_check_pvlib_test_data.csv", index=False)


if __name__ == "__main__":
    main()
