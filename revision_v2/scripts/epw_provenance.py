"""
epw_provenance.py — Task 0: provenance table for the EPW files.

Prints and saves the raw header fields that tell where each file comes from
(LOCATION source field, COMMENTS 1, COMMENTS 2, DATA PERIODS, first-row data-source flags).
It does NOT decide which files are IWEC2: it reports whether the string "IWEC" appears in the
header so the author can check the table by eye. Nothing is inferred from file names.

Usage:  python epw_provenance.py --epw-dir <folder with the 41 EPW files> [--out <csv>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epw_io import epw_files, read_epw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epw-dir", required=True)
    ap.add_argument("--out", default=str(ROOT / "revision_v2" / "data" / "processed" / "epw_provenance.csv"))
    a = ap.parse_args()

    rows = []
    for p in epw_files(a.epw_dir):
        meta, df = read_epw(p)
        h = meta["header_lines"]
        header_text = " ".join(h)
        rows.append({
            "file": meta["file"], "city": meta["city"], "country": meta["country"], "wmo": meta["wmo"],
            "lat": meta["lat"], "lon": meta["lon"], "tz": meta["tz"], "elev_m": meta["elev"],
            "location_source": meta["source"],
            "comments_1": h[5].split(",", 1)[-1].strip(),
            "comments_2": h[6].split(",", 1)[-1].strip(),
            "data_periods": h[7].strip(),
            "years_in_data": ",".join(str(y) for y in sorted(df["year"].unique())),
            "first_row_flags": df["flags"].iloc[0],
            "iwec_in_header": "IWEC" in header_text.upper(),
            "n_ghi_pos": int((df["ghi"] > 0).sum()),
            "n_dni_nan": int(df["dni"].isna().sum()), "n_dhi_nan": int(df["dhi"].isna().sum()),
            "etr_h_present": bool(df["etr_h"].notna().any()),
            "pw_mm_present": bool(df["pw_mm"].notna().any()),
        })
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 100)
    print(out[["file", "city", "wmo", "lat", "lon", "tz", "elev_m", "location_source", "iwec_in_header",
               "etr_h_present", "pw_mm_present"]].to_string(index=False))
    print(f"\n{len(out)} files; iwec_in_header = {int(out['iwec_in_header'].sum())}; saved {a.out}")
    print("\nCOMMENTS 1 / COMMENTS 2 (first 3 distinct values):")
    for c in ("comments_1", "comments_2"):
        print(f"  {c}:", out[c].drop_duplicates().head(3).tolist())


if __name__ == "__main__":
    main()
