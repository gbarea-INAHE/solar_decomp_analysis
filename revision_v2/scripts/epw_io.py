"""
epw_io.py — Strict EPW reader for the SEJ-D-26-02525 revision.

Differences from the Paper/epw_parser.py used for the submitted results:
  * fails loudly on malformed files (missing LOCATION line, != 8760 rows, bad dates);
  * keeps dew point, RH, extraterrestrial radiation and precipitable water;
  * hour h (1..24) is the interval ENDING at h in local standard time of the header
    time zone: t_start = date + (h-1) h. (The old parser mapped hour 24 to 00:00.)
  * EPW missing-value codes (9999, 999999, 999) become NaN for the fields used here.

Raw files are opened read-only and never modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 0-based column indices of the standard 35-column EPW data record
COLS = {
    0: "year", 1: "month", 2: "day", 3: "hour", 4: "minute", 5: "flags",
    6: "t_db", 7: "t_dew", 8: "rh", 9: "p_pa", 10: "etr_h", 11: "etr_n", 12: "hir",
    13: "ghi", 14: "dni", 15: "dhi", 28: "pw_mm",
}
MISSING = {"ghi": 9999.0, "dni": 9999.0, "dhi": 9999.0, "etr_h": 9999.0, "etr_n": 9999.0,
           "t_db": 99.9, "t_dew": 99.9, "rh": 999.0, "p_pa": 999999.0, "pw_mm": 999.0}
N_EXPECTED = 8760


def read_epw(path: str | Path) -> tuple[dict, pd.DataFrame]:
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")   # files written by ingest_epw_from_fetch.py
    except UnicodeDecodeError:
        text = raw.decode("latin-1")  # original EPW encoding
    lines = text.splitlines()
    if len(lines) < 9:
        raise ValueError(f"{path.name}: only {len(lines)} lines, not an EPW")
    header = lines[:8]
    loc = header[0].split(",")
    if loc[0].strip().upper() != "LOCATION" or len(loc) < 10:
        raise ValueError(f"{path.name}: first line is not a valid LOCATION record: {header[0][:80]!r}")
    meta = {
        "file": path.name,
        "city": loc[1].strip(), "state": loc[2].strip(), "country": loc[3].strip(),
        "source": loc[4].strip(), "wmo": loc[5].strip(),
        "lat": float(loc[6]), "lon": float(loc[7]), "tz": float(loc[8]), "elev": float(loc[9]),
        "header_lines": header,
    }
    rows = [ln.split(",") for ln in lines[8:] if ln.strip()]
    if len(rows) != N_EXPECTED:
        raise ValueError(f"{path.name}: {len(rows)} data rows, expected {N_EXPECTED}")
    short = [i for i, r in enumerate(rows) if len(r) < 29]
    if short:
        raise ValueError(f"{path.name}: {len(short)} data rows with <29 fields (first at row {short[0] + 9})")

    df = pd.DataFrame({name: [r[i] for r in rows] for i, name in COLS.items()})
    num = [c for c in df.columns if c != "flags"]
    df[num] = df[num].apply(pd.to_numeric, errors="raise")
    for col, code in MISSING.items():
        df[col] = df[col].where(df[col] < code, np.nan)

    if not df["hour"].between(1, 24).all():
        raise ValueError(f"{path.name}: hour field outside 1..24")
    date = pd.to_datetime(df[["year", "month", "day"]].astype(int), errors="raise")
    df["t_start"] = date + pd.to_timedelta(df["hour"].astype(int) - 1, unit="h")
    # TMY months come from different source years, so the full timestamp may step back at month joins.
    # What must hold is calendar order of (month, day, hour) across the file.
    cal_key = (df["month"] * 10000 + df["day"] * 100 + df["hour"]).to_numpy()
    if not (np.diff(cal_key) > 0).all():
        raise ValueError(f"{path.name}: (month, day, hour) not strictly increasing")
    df["doy"] = df["t_start"].dt.dayofyear
    return meta, df.reset_index(drop=True)


def epw_files(directory: str | Path) -> list[Path]:
    d = Path(directory)
    files = sorted(set(d.glob("*.epw")) | set(d.glob("*.EPW")))
    if not files:
        raise FileNotFoundError(f"no .epw/.EPW files in {d}")
    return files
