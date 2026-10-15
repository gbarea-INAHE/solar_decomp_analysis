"""
audit_dirint_vs_pvlib.py — Implementation-fidelity audit of solar_decomp DIRINT / Erbs
against pvlib, on identical synthetic hourly inputs.

WHY SYNTHETIC: the licensed IWEC2 EPWs are not reachable from this container yet
(network policy). This audit asks a narrower question that does not need them:
"given the same GHI/time/pressure/dew point, does solar_decomp's DIRINT return the
same DNI as an independent reference implementation of Perez et al. (1992)?"

Synthetic inputs: Ineichen clear-sky GHI (pvlib) x a seeded persistent cloud process,
for one arid and one humid site. Nothing here is a claim about real irradiance.

Variants of the solar_decomp run, to localise the source of any divergence:
  A  batch-faithful   : rows with GHI < 10 W/m2 dropped before dirint_series
                        (exactly what Paper/validate_epw_batch.py did), W from T,P (RH=70 %)
  B  continuous       : full 8760 h series incl. night rows, W from T,P (RH=70 %)
  C  continuous + W   : as B but W from dew point, w = exp(0.07 Td - 0.075) [cm]
                        (the DIRINT standard; same formula pvlib uses)
Reference: pvlib.irradiance.dirint on the continuous series with dew point.
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

import decomposition as dec  # noqa: E402
from decomposition import run_decomposition  # noqa: E402
from solar_geometry import compute_geometry_df  # noqa: E402

OUT = ROOT / "revision_v2" / "data" / "processed"
FIG = ROOT / "revision_v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MIN_GHI = 10.0      # as in validate_epw_batch.py
MIN_COSZ = 0.0174   # as in validate_epw_batch.py
SEED = 20261005

SITES = {
    "Mendoza (arid)":  dict(lat=-32.833, lon=-68.783, alt=704.0, tz=-3.0, linke=2.0, p_clear=0.70, tdew=0.0),
    "Posadas (humid)": dict(lat=-27.367, lon=-55.967, alt=125.0, tz=-3.0, linke=3.5, p_clear=0.40, tdew=14.0),
}


def synth_site(name: str, s: dict) -> dict:
    rng = np.random.default_rng(SEED + abs(hash(name)) % 1000 if False else SEED + len(name))
    start = pd.date_range("2001-01-01 00:00", periods=8760, freq="h")
    center = start + pd.Timedelta(minutes=30)
    loc = pvlib.location.Location(s["lat"], s["lon"], altitude=s["alt"], tz="Etc/GMT+3")
    c_tz = center.tz_localize("Etc/GMT+3")
    cs = loc.get_clearsky(c_tz, model="ineichen", linke_turbidity=s["linke"])
    sp = loc.get_solarposition(c_tz)

    # persistent cloud process: daily regime + hourly AR(1)
    n_days = 365
    clear_day = rng.random(n_days) < s["p_clear"]
    mu_day = rng.normal(0.0, 1.0, n_days)
    x = np.zeros(8760)
    eps = rng.normal(0.0, 0.5, 8760)
    for i in range(1, 8760):
        x[i] = 0.85 * x[i - 1] + eps[i]
    day_idx = np.arange(8760) // 24
    sig = 1.0 / (1.0 + np.exp(-(x + mu_day[day_idx])))
    s_cloud = np.where(clear_day[day_idx], 1.0 - np.abs(rng.normal(0.0, 0.01, 8760)), 0.15 + 0.80 * sig)
    ghi = np.clip(cs["ghi"].to_numpy() * s_cloud, 0.0, None)

    doy = center.dayofyear.to_numpy()
    seas = np.sin(2 * np.pi * (doy - 80) / 365.0)
    hod = center.hour.to_numpy() + 0.5
    temp = 16.0 - 8.0 * seas * np.sign(s["lat"]) * -1 + 6.0 * np.sin(2 * np.pi * (hod - 9) / 24.0)
    tdew = s["tdew"] + 4.0 * seas * np.sign(-s["lat"]) * -1 + rng.normal(0, 1.5, 8760)
    tdew = np.minimum(tdew, temp - 1.0)
    p_pa = 101325.0 * (1.0 - 2.25577e-5 * s["alt"]) ** 5.25588

    return dict(start=start, center=center, c_tz=c_tz, ghi=ghi, ghi_clear=cs["ghi"].to_numpy(),
                dni_clear=cs["dni"].to_numpy(), zen=sp["zenith"].to_numpy(), temp=temp, tdew=tdew, p_pa=p_pa)


def run_repo(d: dict, s: dict, ghi: np.ndarray, filt: bool, w_from_dew: bool) -> pd.DataFrame:
    base = pd.DataFrame({"timestamp_start": d["start"], "GHI": ghi, "temp_C": d["temp"],
                         "pressure_kPa": d["p_pa"] / 1000.0, "tdew": d["tdew"]})
    if filt:
        base = base[base["GHI"] >= MIN_GHI].reset_index(drop=True)
    hour = pd.DataFrame({
        "timestamp_start": base["timestamp_start"],
        "timestamp_center": base["timestamp_start"] + pd.Timedelta(minutes=30),
        "GHI_h": base["GHI"], "coverage": 1.0,
        "temp_C": base["temp_C"], "pressure_kPa": base["pressure_kPa"],
    })
    geo = compute_geometry_df(hour, s["lat"], s["lon"], s["tz"])
    for k_out, k_geo in (("cos_Z", "cos_Z"), ("G0n", "G0n"), ("G0h", "G0h"), ("zenith_deg", "zenith_deg"), ("AM", "AM")):
        hour[k_out] = geo[k_geo]
    g0h = geo["G0h"]
    hour["Kt"] = np.where(g0h > 0, base["GHI"].to_numpy() / g0h, np.nan)
    if w_from_dew:
        w = pd.Series(np.exp(0.07 * base["tdew"].to_numpy() - 0.075), index=hour.index)
        hour = dec.dirint_series(hour, w_cm_series=w, min_cosz=MIN_COSZ)
        er = dec.erbs_series(hour, min_cosz=MIN_COSZ)
        hour["DNI_erbs"], hour["DHI_erbs"] = er["DNI_erbs"], er["DHI_erbs"]
        return hour
    return run_decomposition(hour, primary="DIRINT", min_cosz=MIN_COSZ)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    return dict(n=int(m.sum()), mean_a=float(a.mean()), mean_b=float(b.mean()), MBE_a_minus_b=float(d.mean()),
                RMSE=float(np.sqrt((d ** 2).mean())), corr=float(np.corrcoef(a, b)[0, 1]))


def main() -> None:
    warnings.filterwarnings("ignore")
    print(f"pvlib {pvlib.__version__}")
    print("\n[structure]")
    pv_coeffs = pvlib.irradiance._get_dirint_coeffs()
    print(f"  pvlib DIRINT coefficient array shape : {pv_coeffs.shape}  ([kt_prime, zenith, delta_kt_prime, w] bins; dni = DISC dni * coeff)")
    print(f"  solar_decomp _KN_TABLE shape         : {dec._KN_TABLE.shape}  ([w, dKt', Kt'] bins; returns Kn directly, no DISC)")
    print(f"  solar_decomp table max Kn            : {dec._KN_TABLE.max():.3f}  (DNI_MAX cap {dec.DNI_MAX_W_M2:.0f} W/m2)")

    rows = []
    fig_data = {}
    for name, s in SITES.items():
        d = synth_site(name, s)
        idx = d["c_tz"]
        ghi_s = pd.Series(d["ghi"], index=idx)
        zen_s = pd.Series(d["zen"], index=idx)
        td_s = pd.Series(d["tdew"], index=idx)

        def pv_dirint(g: np.ndarray) -> np.ndarray:
            return pvlib.irradiance.dirint(pd.Series(g, index=idx), zen_s, idx, pressure=d["p_pa"],
                                           use_delta_kt_prime=True, temp_dew=td_s,
                                           min_cos_zenith=0.065, max_zenith=87).to_numpy()

        pv = pv_dirint(d["ghi"])
        pv_erbs = pvlib.irradiance.erbs(ghi_s, zen_s, idx)
        res = {
            "A batch-faithful": run_repo(d, s, d["ghi"], filt=True, w_from_dew=False),
            "B continuous": run_repo(d, s, d["ghi"], filt=False, w_from_dew=False),
            "C continuous + W(dew)": run_repo(d, s, d["ghi"], filt=False, w_from_dew=True),
        }
        ref = pd.DataFrame({"timestamp_start": d["start"], "dni_pv": pv, "dni_pv_erbs": pv_erbs["dni"].to_numpy(),
                            "dhi_pv_erbs": pv_erbs["dhi"].to_numpy(), "ghi": d["ghi"], "zen": d["zen"],
                            "dni_clear": d["dni_clear"]})
        for label, r in res.items():
            j = r.merge(ref, on="timestamp_start", how="inner")
            mask = (j["GHI_h"] >= MIN_GHI) & (j["zen"] < 80.0)
            st = stats(j.loc[mask, "DNI_dirint"].to_numpy(), j.loc[mask, "dni_pv"].to_numpy())
            rows.append(dict(site=name, comparison=f"DIRINT solar_decomp[{label}] vs pvlib", **st))
            if label == "A batch-faithful":
                fig_data[name] = (j.loc[mask, "dni_pv"].to_numpy(), j.loc[mask, "DNI_dirint"].to_numpy())
        # Erbs fidelity (variant B)
        jb = res["B continuous"].merge(ref, on="timestamp_start", how="inner")
        mask = (jb["GHI_h"] >= MIN_GHI) & (jb["zen"] < 80.0)
        rows.append(dict(site=name, comparison="Erbs DNI solar_decomp vs pvlib.erbs",
                         **stats(jb.loc[mask, "DNI_erbs"].to_numpy(), jb.loc[mask, "dni_pv_erbs"].to_numpy())))
        rows.append(dict(site=name, comparison="Erbs DHI solar_decomp vs pvlib.erbs",
                         **stats(jb.loc[mask, "DHI_erbs"].to_numpy(), jb.loc[mask, "dhi_pv_erbs"].to_numpy())))

        # Pure clear-sky physical check: feed clear-sky GHI; truth = the clear-sky generator's DNI
        pv_c = pv_dirint(d["ghi_clear"])
        rc = run_repo(d, s, d["ghi_clear"], filt=False, w_from_dew=False)
        refc = pd.DataFrame({"timestamp_start": d["start"], "dni_pv_c": pv_c, "dni_truth": d["dni_clear"], "zen": d["zen"]})
        jc = rc.merge(refc, on="timestamp_start", how="inner")
        m = (jc["zen"] < 60.0) & (jc["GHI_h"] >= 100.0)
        rows.append(dict(site=name, comparison="CLEAR-SKY input: solar_decomp DIRINT vs Ineichen DNI",
                         **stats(jc.loc[m, "DNI_dirint"].to_numpy(), jc.loc[m, "dni_truth"].to_numpy())))
        rows.append(dict(site=name, comparison="CLEAR-SKY input: pvlib DIRINT vs Ineichen DNI",
                         **stats(jc.loc[m, "dni_pv_c"].to_numpy(), jc.loc[m, "dni_truth"].to_numpy())))
        rows.append(dict(site=name, comparison="CLEAR-SKY input: solar_decomp Erbs vs Ineichen DNI",
                         **stats(jc.loc[m, "DNI_erbs"].to_numpy(), jc.loc[m, "dni_truth"].to_numpy())))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print("\n[results]  (a = first-named, b = second-named; W/m2; mid-day-ish hours: GHI>=10, zenith<80 deg; clear-sky rows: zenith<60 deg)")
    print(df.round(2).to_string(index=False))
    df.to_csv(OUT / "audit_dirint_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    okabe = {"Mendoza (arid)": "#0072B2", "Posadas (humid)": "#D55E00"}
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for name, (x, y) in fig_data.items():
        ax.scatter(x, y, s=3, alpha=0.25, color=okabe[name], label=name, linewidths=0)
    lim = [0, 1150]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("pvlib DIRINT DNI (W m$^{-2}$)")
    ax.set_ylabel("solar_decomp DIRINT DNI (W m$^{-2}$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, markerscale=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "audit_dirint_scatter.png", dpi=300)
    print(f"\nsaved {OUT / 'audit_dirint_summary.csv'} and {FIG / 'audit_dirint_scatter.png'}")


if __name__ == "__main__":
    main()
