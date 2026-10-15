"""run_anchor_pilot.py — first pass of the external anchor (T1-T3) on REAL measured data: BSRN PTR and SMS (PANGAEA monthly tables)
and NOAA SURFRAD Desert Rock (daily files), year 2016. Files are in data/anchor_raw/ (see response/ANCLA_piloto_descargas.md).

This script does NOT modify bsrn_anchor.py (tested on synthetic data); it adds the two real-file readers that module lacks and calls its
`anchor`, `geometry`, `bsrn_qc`, `to_hourly` and `rank_agreement` unchanged.

Time conventions (verified from the sources, not assumed):
  * PANGAEA/BSRN tables: stamps run from 00:00 to 23:59 UTC in every day and the month ends at 23:59 -> stamp = START of the 1-min
    interval (inferred from that structure; the BSRN station-to-archive manual was not read). A sensitivity run with the opposite
    convention ("end") is reported.
  * SURFRAD README: "Reported times are the end times of the 1-min averaging periods" -> converted here to start-of-minute (t - 1 min).

Quality control: the physically-possible / extremely-rare limits and the closure test of bsrn_anchor.bsrn_qc were checked on 2026-10-05
against "BSRN Global Network recommended QC tests, V2.0" (C. N. Long and E. G. Dutton). They match; two differences are documented in
response/ANCLA_piloto_resultados.md (closure minimum is on GHI instead of Sum SW; the diffuse-ratio test is not implemented).
bsrn_anchor.LIMITS_VERIFIED is left as in the module (False) on purpose; the verification is recorded in the log of this script.

Usage (from revision_v2, venv active):  python scripts/run_anchor_pilot.py
Writes data/processed/anchor_pilot_<STN>_{metrics,t3,t2rank}.csv, anchor_pilot_coverage.csv and logs/anchor_pilot_<timestamp>.txt
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
ROOT = HERE.parents[1]
REV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import bsrn_anchor as ba  # noqa: E402

RAW = REV / "data" / "anchor_raw"
OUT = REV / "data" / "processed"
LOGS = REV / "logs"


# ----------------------------------------------------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------------------------------------------------
PANG_COLS = {"time_utc": "Date/Time", "ghi": "SWD [W/m**2]", "dni": "DIR [W/m**2]", "dhi": "DIF [W/m**2]",
             "t_c": "T2 [°C]", "rh": "RH [%]", "p_hpa": "PoPoPoPo [hPa]"}


def read_pangaea_month(path: Path) -> tuple[pd.DataFrame, dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.strip() == "*/"), None)
    if end is None:
        raise ValueError(f"{path.name}: metadata block end '*/' not found")
    meta_txt = "\n".join(lines[:end])
    import re
    m_ev = re.search(r"Event\(s\):\s*(\w+) \(([^)]*)\) \* LATITUDE: (-?[\d.]+) \* LONGITUDE: (-?[\d.]+) .*?ELEVATION: ([\d.]+) m", meta_txt)
    if not m_ev:
        raise ValueError(f"{path.name}: could not read station/lat/lon/elevation from the metadata block")
    meta = dict(code=m_ev.group(1), name=m_ev.group(2), lat=float(m_ev.group(3)), lon=float(m_ev.group(4)), elev=float(m_ev.group(5)))
    header = lines[end + 1].split("\t")
    idx, absent = {}, []
    for k, name in PANG_COLS.items():
        n = header.count(name)
        if n > 1 or (n == 0 and k == "time_utc"):
            raise ValueError(f"{path.name}: expected exactly one column {name!r}; header = {header}")
        if n == 0:
            idx[k] = None          # a month without that instrument/column (e.g. PTR 2016-09 has no DIF): NaN, and it is REPORTED
            absent.append(name)
        else:
            idx[k] = header.index(name)
    rows = [ln.split("\t") for ln in lines[end + 2:] if ln.strip()]
    df = pd.DataFrame({k: ([r[i] if i < len(r) else "" for r in rows] if i is not None else [""] * len(rows)) for k, i in idx.items()})
    for k in df.columns[1:]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    if df["time_utc"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate timestamps")
    meta["absent_columns"] = absent
    return df.set_index("time_utc").sort_index(), meta


def read_pangaea_station(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    frames, metas = [], []
    for p in files:
        d, m = read_pangaea_month(p)
        frames.append(d)
        metas.append(m)
    core = [{k: v for k, v in m.items() if k != "absent_columns"} for m in metas]
    if any(c != core[0] for c in core[1:]):
        raise ValueError("files disagree on station metadata; do not mix stations or description changes silently")
    df = pd.concat(frames).sort_index()
    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps across months")
    meta = dict(core[0])
    meta["absent_columns_by_file"] = {p.name: m["absent_columns"] for p, m in zip(files, metas) if m["absent_columns"]}
    return df, meta


SURF_VARS = ["dw_solar", "uw_solar", "direct_n", "diffuse", "dw_ir", "dw_casetemp", "dw_dometemp", "uw_ir", "uw_casetemp",
             "uw_dometemp", "uvb", "par", "netsolar", "netir", "totalnet", "temp", "rh", "windspd", "winddir", "pressure"]


def read_surfrad_year(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """SURFRAD daily files -> minute frame with START-of-minute UTC stamps. Values with a QC flag != 0 or -9999.9 become NaN."""
    recs, meta = [], None
    for p in files:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 3:
            raise ValueError(f"{p.name}: too short")
        name = lines[0].strip()
        t = lines[1].split()
        m = dict(name=name, lat=float(t[0]), lon=float(t[1]), elev=float(t[2]))
        if meta is None:
            meta = dict(m)
        else:
            # the header precision changes inside the year (36.62/-116.02 in the first 50 files, 36.624/-116.019 afterwards): same station
            same = (m["name"] == meta["name"] and m["elev"] == meta["elev"]
                    and abs(m["lat"] - meta["lat"]) < 0.01 and abs(m["lon"] - meta["lon"]) < 0.01)
            if not same:
                raise ValueError(f"{p.name}: station header differs from the first file beyond rounding: {m} vs {meta}")
            if len(str(m["lat"])) > len(str(meta["lat"])):
                meta.update(lat=m["lat"], lon=m["lon"])     # keep the more precise coordinates
        for ln in lines[2:]:
            f = ln.split()
            if not f:
                continue
            if len(f) != 8 + 2 * len(SURF_VARS):
                raise ValueError(f"{p.name}: expected {8 + 2 * len(SURF_VARS)} fields, found {len(f)}")
            recs.append(f)
    a = np.array(recs, dtype=float)
    t_end = pd.to_datetime(dict(year=a[:, 0].astype(int), month=a[:, 2].astype(int), day=a[:, 3].astype(int),
                                hour=a[:, 4].astype(int), minute=a[:, 5].astype(int)), utc=True)
    out = {}
    for i, v in enumerate(SURF_VARS):
        val, qc = a[:, 8 + 2 * i], a[:, 9 + 2 * i]
        val = np.where((qc != 0) | (val <= -9999.0), np.nan, val)
        out[v] = val
    df = pd.DataFrame({"ghi": out["dw_solar"], "dni": out["direct_n"], "dhi": out["diffuse"], "t_c": out["temp"],
                       "rh": out["rh"], "p_hpa": out["pressure"]}, index=pd.DatetimeIndex(t_end) - pd.Timedelta(minutes=1))
    df.index.name = "time_utc"
    if df.index.duplicated().any():
        raise ValueError("duplicate timestamps in SURFRAD files")
    return df.sort_index(), meta


# ----------------------------------------------------------------------------------------------------------------------
def regularise(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Complete 1-min grid for the study year (missing minutes become NaN rows) and trim to [start, end]."""
    idx = pd.date_range(start, end, freq="min", tz="UTC")
    return df.reindex(idx).rename_axis("time_utc")


def coverage_report(minute: pd.DataFrame, meta: dict, stamp: str) -> dict:
    geo = ba.geometry(minute.index, meta["lat"], meta["lon"], meta["elev"], stamp)
    qc = ba.bsrn_qc(minute, geo)
    sun = geo["cosz"] > 0
    h = ba.to_hourly(minute, geo, qc, stamp)
    day_h = h["cosz_avg"] > 0
    rep = {
        "minutes_total": len(minute), "minutes_missing_pct": 100 * float(qc["missing"].mean()),
        "sun_minutes_missing_pct": 100 * float(qc.loc[sun, "missing"].mean()),
        "sun_minutes_phys_fail_pct": 100 * float(qc.loc[sun, "phys"].mean()),
        "sun_minutes_rare_fail_pct": 100 * float(qc.loc[sun, "rare"].mean()),
        "sun_minutes_closure_fail_pct": 100 * float(qc.loc[sun, "closure"].mean()),
        "sun_minutes_invalid_pct": 100 * float(qc.loc[sun, "invalid"].mean()),
        "daytime_hours": int(day_h.sum()), "daytime_hours_kept_pct": 100 * float(h.loc[day_h, "ghi"].notna().mean()),
    }
    return rep


def main() -> int:
    warnings.filterwarnings("ignore")
    stamp_log = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LOGS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log(f"anchor pilot run {stamp_log} | python {sys.version.split()[0]} | pandas {pd.__version__} numpy {np.__version__} pvlib {pvlib.__version__}")
    log(f"bsrn_anchor.PARAMS = {ba.PARAMS} | bsrn_anchor.LIMITS_VERIFIED (module flag) = {ba.LIMITS_VERIFIED}")
    log("QC limits compared 2026-10-05 with Long & Dutton, BSRN Global Network recommended QC tests V2.0: physically possible and "
        "extremely rare limits and the 8 %/15 % closure tolerances MATCH; differences: closure minimum applied on GHI (document: Sum SW > 50), "
        "diffuse-ratio test not implemented, 'extremely rare' failures treated as invalid.")
    stations = {
        "PTR": ("pangaea", sorted((RAW / "PTR").glob("PTR_radiation_2016-*.tab"))),
        "SMS": ("pangaea", sorted((RAW / "SMS").glob("SMS_radiation_2016-*.tab"))),
        "DRA": ("surfrad", sorted((RAW / "DRA").glob("dra16???.dat"))),
    }
    bench = REV / "data" / "processed" / "benchmark_v2_per_station.csv"
    cov_rows = []
    pd.set_option("display.width", 220)
    for stn, (kind, files) in stations.items():
        log("\n" + "=" * 100)
        log(f"{stn}: {len(files)} files ({kind})")
        raw, meta = read_pangaea_station(files) if kind == "pangaea" else read_surfrad_year(files)
        log(f"  meta: {meta}")
        minute = regularise(raw, "2016-01-01 00:00", "2016-12-31 23:59")
        log(f"  minute records read: {len(raw)} | grid {len(minute)}")
        variants = [("start", "base")] if kind == "surfrad" else [("start", "base"), ("end", "stamp_end")]
        for stamp, tag in variants:
            rep = coverage_report(minute, meta, stamp)
            log(f"  [{tag}] coverage/QC: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
            cov_rows.append(dict(station=stn, variant=tag, **rep))
            res, sim = ba.anchor(minute, meta["lat"], meta["lon"], meta["elev"], stamp)
            res.to_csv(OUT / f"anchor_pilot_{stn}_{tag}_metrics.csv", index=False)
            sim.to_csv(OUT / f"anchor_pilot_{stn}_{tag}_t3.csv", index=False)
            rk = ba.rank_agreement(res, bench if bench.exists() else None, "c20")
            rk.to_csv(OUT / f"anchor_pilot_{stn}_{tag}_t2rank.csv", index=False)
            log(f"  [{tag}] T1/T2 (DNI vs measured), domain c20:")
            log(res[res["domain"] == "c20"].round(2).to_string(index=False))
            log(f"  [{tag}] T2 ranking (c20) and agreement with the IWEC2 ranking: Spearman = {rk.attrs.get('spearman')}")
            log(rk.round(2).to_string(index=False))
            log(f"  [{tag}] T3 (c20):")
            log(sim[sim["domain"] == "c20"].round(2).to_string(index=False))
    pd.DataFrame(cov_rows).to_csv(OUT / "anchor_pilot_coverage.csv", index=False)
    (LOGS / f"anchor_pilot_{stamp_log}.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log(f"\nlog: {LOGS / ('anchor_pilot_' + stamp_log + '.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
