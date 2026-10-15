"""geo_outline.py — lector mínimo (sin dependencias) de polígonos de un shapefile de Natural Earth guardado en un .zip. Solo lectura.
Soporta el tipo de forma 5 (Polygon) y el .dbf con campos de texto; devuelve {nombre: [anillos Nx2]}.
"""
from __future__ import annotations

import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np

GEO = Path(__file__).resolve().parents[1] / "data" / "geo"
ZIP = GEO / "ne_10m_admin_0_countries_arg.zip"


def _check_manifest() -> None:
    import csv
    rows = list(csv.DictReader((GEO / "MANIFEST_geo.csv").open(encoding="utf-8")))
    rec = [r for r in rows if r["file"] == ZIP.name][-1]
    h = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    if h != rec["sha256"]:
        raise ValueError(f"{ZIP.name}: el SHA256 no coincide con el manifiesto (archivo modificado)")


def read_dbf(raw: bytes) -> list[dict]:
    n, hlen, rlen = struct.unpack("<IHH", raw[4:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\x00")[0].decode("ascii")
        ftype, flen = chr(raw[pos + 11]), raw[pos + 16]
        fields.append((name, ftype, flen))
        pos += 32
    recs = []
    for i in range(n):
        off = hlen + i * rlen
        if raw[off:off + 1] == b"*":
            recs.append(None)
            continue
        p, d = off + 1, {}
        for name, ftype, flen in fields:
            d[name] = raw[p:p + flen].decode("utf-8", errors="replace").replace("\x00", "").strip()
            p += flen
        recs.append(d)
    return recs


def read_polygons(raw: bytes) -> list[list[np.ndarray]]:
    shapes, pos = [], 100
    while pos + 8 <= len(raw):
        _, clen = struct.unpack(">ii", raw[pos:pos + 8])
        body = raw[pos + 8:pos + 8 + clen * 2]
        pos += 8 + clen * 2
        stype = struct.unpack("<i", body[:4])[0]
        if stype == 0:
            shapes.append([])
            continue
        if stype != 5:
            raise ValueError(f"tipo de forma no soportado: {stype}")
        npart, npts = struct.unpack("<ii", body[36:44])
        parts = list(struct.unpack(f"<{npart}i", body[44:44 + 4 * npart])) + [npts]
        pts = np.frombuffer(body[44 + 4 * npart:44 + 4 * npart + 16 * npts], dtype="<f8").reshape(-1, 2)
        shapes.append([pts[parts[k]:parts[k + 1]] for k in range(npart)])
    return shapes


def load_countries(names: set[str] | None = None, field: str = "ADMIN") -> dict[str, list[np.ndarray]]:
    _check_manifest()
    with zipfile.ZipFile(ZIP) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
        dbf = next(n for n in z.namelist() if n.endswith(".dbf"))
        attrs, shapes = read_dbf(z.read(dbf)), read_polygons(z.read(shp))
    assert len(attrs) == len(shapes)
    out = {}
    for a, s in zip(attrs, shapes):
        if a is None:
            continue
        if names is None or a.get(field) in names:
            out[a[field]] = s
    return out


if __name__ == "__main__":
    c = load_countries()
    print(len(c), "países")
    for k in ("Argentina", "Chile", "Uruguay", "Paraguay", "Bolivia", "Brazil", "Falkland Islands"):
        if k in c:
            rings = c[k]
            allp = np.vstack(rings)
            print(f"{k:<18} anillos={len(rings):3d} puntos={len(allp):6d} lon[{allp[:,0].min():.1f},{allp[:,0].max():.1f}] lat[{allp[:,1].min():.1f},{allp[:,1].max():.1f}]")
        else:
            print(k, "no está")
    print([k for k in c if "alk" in k or "Malv" in k or "Geor" in k])
