"""extract_ashrae_koppen.py — extrae del PDF «IWEC2 Station List» (ASHRAE RP-1477, Apéndice A; descargado con autorización del autor el 2026-10-05) las estaciones de Argentina
(estación, número WMO, lat, lon, elevación, huso, clase Köppen, grado de calidad DQ) y las compara con las clases de la Tabla S1 enviada, emparejando por número WMO del nombre del archivo EPW.
Entradas : data/ashrae_list/IWECStationList_1477-RP_Final_Report_20111019-2.pdf (solo lectura; hash en MANIFEST_ashrae_list.csv), data/raw/ARG_*_IW2.EPW (solo los nombres de archivo),
           data/processed/benchmark_v2_per_station.csv (clase de S1 por estación).
Salidas  : data/ashrae_list/ashrae_arg_stations.csv (todas las de Argentina del PDF), data/processed/koppen_ashrae_vs_s1.csv (comparación) y data/ashrae_list/IWECStationList_pages.json (texto derivado por página).
Requiere pypdf (no está en el venv): `PYPDF_PATH` = carpeta que lo contiene. Falla con error claro si el hash no coincide con el manifiesto, si no hay 39 estaciones IWEC2 o si alguna no se empareja.
Uso (desde revision_v2): $env:PYPDF_PATH='<carpeta con pypdf>'; python scripts/extract_ashrae_koppen.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
AL = REV / "data" / "ashrae_list"
PDF = AL / "IWECStationList_1477-RP_Final_Report_20111019-2.pdf"
RAW = REV / "data" / "raw"
PROC = REV / "data" / "processed"
# El texto extraído trae espacios sueltos dentro de nombres y números («ARG_T UCUMAN-AERO», «-26. 850»): se normalizan antes de aplicar el patrón.
ROW = re.compile(r"ARG_([A-Z0-9_.\- ]+?)\s+(\d{6})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+)\s+(-?\d+\.\d+)\s+([A-Z][A-Za-z]{1,2})\s+([A-D])\b")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    if os.environ.get("PYPDF_PATH"):
        sys.path.insert(0, os.environ["PYPDF_PATH"])
    from pypdf import PdfReader  # noqa: PLC0415  (dependencia opcional, fuera del venv)

    man = AL / "MANIFEST_ashrae_list.csv"
    if not man.exists():
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "bytes", "sha256", "source_url", "downloaded_utc", "authorized_by"])
            w.writerow([PDF.name, PDF.stat().st_size, sha256(PDF),
                        "https://www.ashrae.org/File%20Library/Technical%20Resources/Bookstore/IWECStationList_1477-RP_Final_Report_20111019-2.pdf", "2026-10-05",
                        "autor (chat, 2026-10-05): «Sí, autorizado no importa el tope, si realmente hace falta»"])
    rec = pd.read_csv(man)
    if rec.loc[0, "sha256"] != sha256(PDF) or int(rec.loc[0, "bytes"]) != PDF.stat().st_size:
        raise SystemExit("El PDF no coincide con el manifiesto (hash o tamaño): no se usa.")
    pages = [p.extract_text() or "" for p in PdfReader(str(PDF)).pages]
    (AL / "IWECStationList_pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    txt = re.sub(r"(\d\.) (\d)", r"\1\2", "\n".join(pages))
    n_arg = len(re.findall(r"ARG_", txt))
    rows = []
    for m in ROW.finditer(txt):
        name, wmo, lat, lon, elev, tz, kop, dq = m.groups()
        rows.append(dict(ashrae_station=name.replace(" ", ""), wmo=int(wmo), lat=float(lat), lon=float(lon), elev_m=int(elev), tz=float(tz), koppen_ashrae=kop, dq=dq))
    arg = pd.DataFrame(rows).drop_duplicates("wmo").sort_values("wmo")
    print(f"Estaciones de Argentina en el PDF: {len(arg)} (apariciones de «ARG_» en el texto: {n_arg}; filas que cumplen el patrón: {len(rows)})")
    if len(rows) != n_arg:
        raise SystemExit("Hay filas de Argentina que el patrón no reconoce: revisar antes de usar la comparación.")
    arg.to_csv(AL / "ashrae_arg_stations.csv", index=False)

    # emparejar con los EPW por número WMO del nombre de archivo ARG_<NOMBRE>_<WMO>_IW2.EPW
    files = {}
    for f in sorted(RAW.rglob("*.EPW")) + sorted(RAW.rglob("*.epw")):
        m = re.match(r"ARG_(.+)_(\d{6})_IW2\.EPW$", f.name, re.I)
        if m:
            files[int(m.group(2))] = m.group(1)
    if len(files) != 39:
        raise SystemExit(f"Se esperaban 39 EPW IWEC2 en data/raw, hay {len(files)}")
    bm = pd.read_csv(PROC / "benchmark_v2_per_station.csv")
    s1 = bm[bm["source"] == "IWEC2"].drop_duplicates("city").set_index("city")["koppen_s1"]
    out = []
    for wmo, city in files.items():
        a = arg[arg["wmo"] == wmo]
        if a.empty:
            raise SystemExit(f"WMO {wmo} ({city}) no está en la lista de ASHRAE")
        if city not in s1.index:
            raise SystemExit(f"{city} no está en benchmark_v2_per_station.csv")
        a = a.iloc[0]
        out.append(dict(city=city, wmo=wmo, ashrae_station=a["ashrae_station"], koppen_s1=s1[city], koppen_ashrae=a["koppen_ashrae"], dq=a["dq"], same=s1[city] == a["koppen_ashrae"]))
    cmp_ = pd.DataFrame(out).sort_values("city")
    cmp_.to_csv(PROC / "koppen_ashrae_vs_s1.csv", index=False)
    print(f"Emparejadas {len(cmp_)} de 39; misma clase: {int(cmp_['same'].sum())}; distinta: {int((~cmp_['same']).sum())}")
    print(cmp_[~cmp_["same"]].to_string(index=False))
    print("\nClases S1:", cmp_["koppen_s1"].value_counts().to_dict())
    print("Clases ASHRAE:", cmp_["koppen_ashrae"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
