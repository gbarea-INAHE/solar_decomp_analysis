# Solar Decomp consistency study — analysis scripts and derived data

Scripts and derived data behind the manuscript that evaluates how consistent three GHI decomposition models (Erbs, Reindl-2 and DIRINT) are with the DNI and DHI fields of IWEC2 weather files for Argentina, with a limited comparison against ground measurements (BSRN, SURFRAD, SRRL and the INAHE station in Mendoza).

**Authors:** Gustavo Barea Paci · Carolina Ganem — [INAHE-CONICET](https://www.mendoza-conicet.gob.ar/inahe/), Argentina.

This repository holds the **analysis**. The **tool** is a separate repository, [Solar Decomp](https://github.com/gbarea-INAHE/solar_decomp) (v1.1.0, DOI [10.5281/zenodo.23355088](https://doi.org/10.5281/zenodo.23355088); all versions: [10.5281/zenodo.20262706](https://doi.org/10.5281/zenodo.20262706)).

## What is here

```
decomposition.py, solar_geometry.py, io_handler.py,
preprocessor.py, config.py, validator.py   Solar Decomp v1.0.2 modules, byte-identical (commit 7db2498); the scripts import them
test_data_15min.csv                        synthetic test series shipped with Solar Decomp v1.0.2
requirements.txt                           exact package versions of the results
revision_v2/
  run_all.py                               runs the chain in order, with a log
  scripts/                                 the analysis scripts (+ scripts/legacy/legacy_epw_parser.py)
  data/processed/                          derived tables (station-level statistics, bootstrap results, sensitivity checks)
  data/Mendoza_Datos_Medidos/              one-minute series of the INAHE station in Mendoza, 2022–2025 (see DATA.md)
  data/*_manifest*.csv, data/*/MANIFEST_*  file names, sizes and SHA-256 of every input that is not redistributed here
```

Folder names follow the original working layout, so that the scripts run unchanged.

**Which software version the analysis uses.** The analysis imports the Erbs, Reindl-2 and legacy-DIRINT-table implementations of Solar Decomp **v1.0.2** (the modules in the repository root) and computes the DIRINT model of Perez et al. (1992) with `pvlib.irradiance.dirint` (pvlib 0.16.1). Solar Decomp v1.1.0, which replaces the DIRINT table by pvlib, is the version described in the manuscript; `check_dirint_segments.py` checks that its splitting of the series into continuous segments gives the same DNI as running pvlib on the whole series.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cd revision_v2
python run_all.py --list
python run_all.py --group selftests          # needs no external data
```

The `selftests` group (synthetic control files, Test 0, BSRN-anchor self-test) and the audits `fig_kn_curves`, `audit_dirint` and `reproduce_pvlib_claim` need nothing else. The steps that analyse the 41 IWEC2 files (`core`, `figures`, `reproduce_v1_batch`) need the files in `revision_v2/data/raw/`:

```bash
python run_all.py --group core --epw-dir data/raw
```

The IWEC2 files are ASHRAE material and are **not** redistributed here; `revision_v2/data/raw_manifest.csv` lists the exact files (name, size, SHA-256) used. The measured series of BSRN, SURFRAD and SRRL are also not redistributed; `scripts/download_anchor_pilot.py` downloads them from the providers and `revision_v2/data/anchor_raw/MANIFEST_anchor_raw.csv` lists the source URLs and hashes.

Checked on 2026-10-15 with Python 3.14.5 and the pinned requirements: the `selftests` group and `fig_kn_curves`, `audit_dirint` and `reproduce_pvlib_claim` pass, and the regenerated control tables (`controls_all_variants.csv`, `controls_best_variant.csv`) and `claim_check_pvlib_test_data.csv` are byte-identical to those in `data/processed/`. The steps that need the IWEC2 or measured raw files were not re-run for this release.

## Data and licences

- **Code:** MIT License (see `LICENSE`).
- **Derived tables and the Mendoza series:** see `DATA.md` for sources, columns, terms and how to cite each provider.

## Cite

See `CITATION.cff`. If you use the tool, cite Solar Decomp (DOI above).
