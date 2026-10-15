"""
test0_gompertz.py — Task 1 / Test 0: is the DNI in an IWEC2 EPW the Gompertz split of its own GHI?

Hypothesis (HANDOFF section 6): IWEC2 GHI is modelled (ZHM) and DNI = Gompertz(GHI, cos z), DHI by closure.
If true, applying gompertz.gompertz_dni to the EPW GHI must reproduce the EPW DNI to rounding level,
and the EPW DHI must equal GHI - DNI cos z.

What identity would look like (stated before looking at any real file): EPW irradiance fields are integer
Wh/m2, so rounding of GHI, DNI and DHI alone leaves |DNI error| of order 1-2 W/m2. Identity therefore means
that for one geometry variant almost all hours fall within a few W/m2. High R2 with a broad error cloud
means "structurally similar", NOT identical; this script reports both so the two cases cannot be confused.

Variants tested (nothing is assumed; all are reported):
  cosz kind : inst_mid  (cos z at hour-0.5)  | inst_end (cos z at hour) | avg_interval (mean of max(cos z,0)
              over the EPW interval [hour-1, hour]; the Fortran variable is named COSZAV)
  I0        : 1355 and 1367 W/m2 (the Fortran comment says 1355, recommends 1367)
Independent diagnostics that do not use Gompertz at all:
  closure   : residual GHI - DNI cos z - DHI per cosz kind (is DHI closed by construction, and with which cos z?)
  etr       : implied solar constant from the EPW extraterrestrial horizontal field, per cosz kind

Hour selection (replicates the submitted batch so the numbers are comparable; flagged as a QC rule to confirm):
  GHI >= MIN_GHI and cos z(inst_mid) > MIN_COSZ.

Solar position: pvlib NREL SPA (nrel_numpy), true zenith, geometric (no refraction), fixed UTC offset from
the EPW header.

Usage:  python test0_gompertz.py --epw-dir <folder> [--tag NAME]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epw_io import epw_files, read_epw  # noqa: E402
from gompertz import gompertz_dni  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision_v2" / "data" / "processed"
FIG = ROOT / "revision_v2" / "figures"

MIN_GHI = 10.0      # W/m2, as in the submitted batch
MIN_COSZ = 0.0174   # as in the submitted batch
I0_VALUES = (1355.0, 1367.0)
COSZ_KINDS = ("inst_mid", "inst_end", "avg_interval")
N_SUB = 12          # 5-min sub-steps for the interval average


def cos_z_variants(meta: dict, t_start: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    tz = timezone(timedelta(hours=meta["tz"]))
    kw = dict(latitude=meta["lat"], longitude=meta["lon"], altitude=meta["elev"], method="nrel_numpy")

    def cosz(times: pd.DatetimeIndex) -> np.ndarray:
        z = pvlib.solarposition.get_solarposition(times.tz_localize(tz), **kw)["zenith"].to_numpy()
        return np.clip(np.cos(np.radians(z)), 0.0, None)

    out = {"inst_mid": cosz(t_start + pd.Timedelta(minutes=30)),
           "inst_end": cosz(t_start + pd.Timedelta(minutes=60))}
    sub = np.tile((np.arange(N_SUB) + 0.5) * 60.0 / N_SUB, len(t_start))
    t_sub = pd.DatetimeIndex(np.repeat(t_start.to_numpy(), N_SUB)) + pd.to_timedelta(sub, unit="min")
    out["avg_interval"] = cosz(t_sub).reshape(len(t_start), N_SUB).mean(axis=1)
    return out


def spencer_e0(doy: np.ndarray) -> np.ndarray:
    g = 2.0 * np.pi * (doy - 1) / 365.0
    return (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
            + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))


def metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    d = pred - obs
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return dict(n=int(len(obs)), r2=float(1.0 - (d ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
                mbe=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())), max_abs=float(np.abs(d).max()),
                frac_within_2=float((np.abs(d) <= 2.0).mean()), frac_within_10=float((np.abs(d) <= 10.0).mean()))


def run_station(path: Path) -> tuple[list[dict], dict]:
    meta, df = read_epw(path)
    cz = cos_z_variants(meta, pd.DatetimeIndex(df["t_start"]))
    ghi, dni, dhi, etr = (df[c].to_numpy() for c in ("ghi", "dni", "dhi", "etr_h"))
    mask = (ghi >= MIN_GHI) & (cz["inst_mid"] > MIN_COSZ) & np.isfinite(dni) & np.isfinite(dhi)
    if mask.sum() < 100:
        raise ValueError(f"{path.name}: only {int(mask.sum())} usable hours")
    base = {"file": meta["file"], "city": meta["city"], "source_field": meta["source"], "n_hours": int(mask.sum())}

    rows = []
    for kind in COSZ_KINDS:
        c = cz[kind]
        resid = ghi - dni * c - dhi
        m2 = mask & (c > 0)
        clos = dict(closure_mean_abs=float(np.abs(resid[m2]).mean()), closure_p99_abs=float(np.quantile(np.abs(resid[m2]), 0.99)),
                    closure_frac_within_2=float((np.abs(resid[m2]) <= 2.0).mean()))
        me = mask & np.isfinite(etr) & (cz["avg_interval"] > 0.1)
        implied = etr[me] / (spencer_e0(df["doy"].to_numpy()[me]) * c[me]) if me.any() else np.array([np.nan])
        etr_d = dict(etr_implied_i0_median=float(np.nanmedian(implied)), etr_implied_i0_iqr=float(np.nanquantile(implied, .75) - np.nanquantile(implied, .25)))
        for i0 in I0_VALUES:
            pred = gompertz_dni(ghi[mask], c[mask], i0)
            rows.append({**base, "cosz_kind": kind, "i0": i0, **{f"dni_{k}": v for k, v in metrics(dni[mask], pred).items()}, **clos, **etr_d})
    return rows, {"meta": meta, "df": df, "mask": mask, "cz": cz}


def scatter(ctx: dict, kind: str, i0: float, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ghi, dni = ctx["df"]["ghi"].to_numpy(), ctx["df"]["dni"].to_numpy()
    m = ctx["mask"]
    pred = gompertz_dni(ghi[m], ctx["cz"][kind][m], i0)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(dni[m], pred, s=3, alpha=0.3, color="#0072B2", linewidths=0)
    lim = [0, max(1100.0, float(np.nanmax(dni[m])))]
    ax.plot(lim, lim, color="black", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("EPW DNI (W m$^{-2}$)"); ax.set_ylabel("Gompertz DNI from EPW GHI (W m$^{-2}$)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    ap.add_argument("--tag", default="test0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    all_rows, best_rows = [], []
    for p in epw_files(a.epw_dir):
        rows, ctx = run_station(p)
        all_rows += rows
        best = max(rows, key=lambda r: (r["dni_frac_within_2"], r["dni_r2"]))
        best_rows.append(best)
        scatter(ctx, best["cosz_kind"], best["i0"], FIG / f"{a.tag}_scatter_{Path(p).stem}.png")
        print(f"{best['city'][:24]:<24} best: {best['cosz_kind']:<12} I0={best['i0']:.0f}  R2={best['dni_r2']:.5f}  "
              f"max|err|={best['dni_max_abs']:.1f}  within±2={best['dni_frac_within_2']:.3f}  "
              f"closure(within±2)={best['closure_frac_within_2']:.3f}")
    full = pd.DataFrame(all_rows)
    full.to_csv(OUT / f"{a.tag}_all_variants.csv", index=False)
    pd.DataFrame(best_rows).to_csv(OUT / f"{a.tag}_best_variant.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nMedian over stations, per variant (dni_*: Gompertz(EPW GHI) vs EPW DNI; closure: EPW GHI - DNI cos z - DHI):")
    agg = full.groupby(["cosz_kind", "i0"])[["dni_r2", "dni_max_abs", "dni_frac_within_2", "closure_frac_within_2",
                                               "etr_implied_i0_median"]].median().round(4)
    print(agg.to_string())
    print(f"\nsaved {OUT / (a.tag + '_all_variants.csv')} and {OUT / (a.tag + '_best_variant.csv')}")


if __name__ == "__main__":
    main()
