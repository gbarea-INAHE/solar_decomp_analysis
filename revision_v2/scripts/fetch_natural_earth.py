"""fetch_natural_earth.py — descarga UNA vez el contorno de países de Natural Earth para dibujar el mapa de la Fig. 1 (autorizado por el autor el 2026-10-05).

Archivo : ne_10m_admin_0_countries_arg.zip  (Natural Earth 1:10m, «Admin 0 – Countries», versión con el punto de vista de Argentina)
Origen  : https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries_arg.zip   (sitio oficial; Content-Length 4 910 248 bytes)
Licencia: dominio público (https://www.naturalearthdata.com/about/terms-of-use/; no exige atribución).
Destino : data/geo/ (solo lectura después de bajarlo) + data/geo/MANIFEST_geo.csv (tamaño y SHA256). Falla si el tamaño no coincide con el anunciado.
Uso (desde revision_v2, venv activo): python scripts/fetch_natural_earth.py
"""
from __future__ import annotations

import csv
import hashlib
import os
import stat
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries_arg.zip"
EXPECTED = 4910248
DEST = Path(__file__).resolve().parents[1] / "data" / "geo"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / URL.split("/")[-1]
    if out.exists():
        print("ya existe:", out)
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "solar_decomp-revision-geo-fetch"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != EXPECTED:
            sys.exit(f"tamaño inesperado: {len(data)} bytes (esperado {EXPECTED}); no se guarda")
        out.write_bytes(data)
        os.chmod(out, stat.S_IREAD)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    man = DEST / "MANIFEST_geo.csv"
    new = not man.exists()
    with man.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "bytes", "sha256", "source_url", "downloaded_utc", "license"])
        w.writerow([out.name, out.stat().st_size, h, URL, datetime.now(timezone.utc).isoformat(timespec="seconds"), "public domain (Natural Earth terms of use)"])
    print(out.name, out.stat().st_size, "bytes  sha256", h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
