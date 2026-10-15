"""download_anchor_pilot.py — pilot download of MEASURED radiation for the external anchor (Task 2).

Authorised by the author on 2026-10-05 (see response/ANCLA_piloto_descargas.md). Sources:
  * BSRN via PANGAEA, stations PTR (Petrolina) and SMS (Sao Martinho da Serra), year 2016, 24 monthly tab-delimited files.
    Licence: "Baseline Surface Radiation Network License 1.0" (https://bsrn.awi.de/data/conditions-of-data-release/):
    free for research, cite each dataset, no selling, do not pass the raw data to third parties without WRMC agreement.
  * NOAA GML SURFRAD, station Desert Rock NV (DRA), year 2016, 366 daily files + the format README. Licence: CC0 1.0.

Files are written to data/anchor_raw/<STATION>/ (raw data: never edited, set read-only). A manifest with source URL, size, SHA256 and
download time is appended to data/anchor_raw/MANIFEST_anchor_raw.csv. Existing files are never overwritten. Sequential requests with a
short pause. The exit code is the number of files that failed.

Usage (from revision_v2):  python scripts/download_anchor_pilot.py
"""
from __future__ import annotations

import csv
import hashlib
import os
import stat
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "anchor_raw"
MANIFEST = RAW / "MANIFEST_anchor_raw.csv"
UA = "Mozilla/5.0 (compatible; research-download; solar-decomp-revision)"
PAUSE_S = 0.4

# (month, PANGAEA dataset number, expected bytes from the HTTP HEAD of 2026-10-05)
BSRN = {
    "PTR": [(1, 889520, 4004395), (2, 889521, 3842636), (3, 889522, 4114949), (4, 889524, 3968752), (5, 889525, 3511876),
            (6, 889526, 3929371), (7, 889527, 4082775), (8, 889528, 3541841), (9, 889529, 3002406), (10, 889530, 2930190),
            (11, 889531, 3520912), (12, 889532, 4120464)],
    "SMS": [(1, 889301, 4104998), (2, 889302, 3834773), (3, 889361, 4038361), (4, 889303, 3083534), (5, 889304, 3703470),
            (6, 889306, 3877286), (7, 889307, 3994600), (8, 889308, 3999329), (9, 889311, 3683253), (10, 889309, 3704103),
            (11, 889426, 3616002), (12, 889310, 3988417)],
}
SURFRAD_BASE = "https://gml.noaa.gov/aftp/data/radiation/surfrad/Desert_Rock_NV/"


def jobs():
    for st, lst in BSRN.items():
        for mo, num, nbytes in lst:
            yield st, f"https://doi.pangaea.de/10.1594/PANGAEA.{num}?format=textfile", f"{st}_radiation_2016-{mo:02d}.tab", nbytes
    for doy in range(1, 367):
        yield "DRA", f"{SURFRAD_BASE}2016/dra16{doy:03d}.dat", f"dra16{doy:03d}.dat", None
    yield "DRA", f"{SURFRAD_BASE}README_SURFRAD.txt", "README_SURFRAD.txt", None


def fetch(url: str) -> tuple[bytes, int, int | None]:
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
                cl = r.headers.get("Content-Length")
                return data, r.status, int(cl) if cl else None
        except Exception as e:  # noqa: BLE001 - retried, then reported
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    new_manifest = not MANIFEST.exists()
    failed = []
    done = skipped = 0
    with MANIFEST.open("a", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        if new_manifest:
            w.writerow(["station", "file", "source_url", "bytes", "sha256", "downloaded_utc", "expected_bytes_head"])
        for n, (st, url, name, expected) in enumerate(jobs(), start=1):
            dest = RAW / st / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                skipped += 1
                continue
            try:
                data, status, cl = fetch(url)
                if status != 200:
                    raise RuntimeError(f"HTTP {status}")
                if cl is not None and cl != len(data):
                    raise RuntimeError(f"size mismatch: Content-Length {cl} != received {len(data)}")
                if expected is not None and expected != len(data):
                    print(f"  AVISO {name}: {len(data)} bytes, la consulta HEAD previa dio {expected}")
                if len(data) == 0:
                    raise RuntimeError("empty file")
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                os.chmod(dest, stat.S_IREAD)
                w.writerow([st, name, url, len(data), hashlib.sha256(data).hexdigest(),
                            datetime.now(timezone.utc).isoformat(timespec="seconds"), expected or ""])
                mf.flush()
                done += 1
            except Exception as e:  # noqa: BLE001
                failed.append((name, str(e)))
                print(f"  FALLO {name}: {e}")
            if n % 25 == 0:
                print(f"  {n} procesados | bajados {done} | ya existian {skipped} | fallidos {len(failed)}", flush=True)
            time.sleep(PAUSE_S)
    print(f"\nTERMINADO: bajados {done}, ya existian {skipped}, fallidos {len(failed)}")
    for name, err in failed:
        print("  FALLO", name, err)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
