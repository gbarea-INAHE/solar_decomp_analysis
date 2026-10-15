"""run_all.py — run the revision_v2 analysis chain in order, with one log.

Usage (from <repo>/revision_v2, Python >= 3.11 with ../requirements.txt installed):
    python run_all.py --list
    python run_all.py --group selftests
    python run_all.py --group core  --epw-dir data/raw
    python run_all.py --group figures --epw-dir data/raw
    python run_all.py --group all   --epw-dir data/raw
    python run_all.py --only benchmark_v2 reanalysis_v2 --epw-dir data/raw

Groups
    selftests : checks that need no Argentine EPW files (Test 0 on the synthetic controls, BSRN-anchor self-test)
    core      : provenance, Test 0, geometry-consistent benchmark, re-analysis (needs the 41 EPW files)
    figures   : figures that need the EPW files
    audit     : DIRINT/Erbs audit against pvlib and reproduction of the submitted batch (needs EPW files)
    optional  : steps that need a C compiler (validate_gompertz_port)
    all       : selftests + core + figures + audit  (optional is never run unless asked)

Nothing here edits the repository root or anything outside revision_v2/{data/processed,figures,logs}.
A step that fails is reported and the run continues; the exit code is the number of failed steps.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "scripts"

# name -> (group, command relative to revision_v2, needs --epw-dir)
STEPS: dict[str, tuple[str, list[str], bool]] = {
    "make_synthetic_epw":   ("selftests", ["scripts/make_synthetic_epw.py"], False),
    "test0_controls":       ("selftests", ["scripts/test0_gompertz.py", "--epw-dir", "data/synthetic", "--tag", "controls"], False),
    "bsrn_selftest":        ("selftests", ["scripts/bsrn_anchor.py", "--selftest"], False),
    "epw_provenance":       ("core", ["scripts/epw_provenance.py"], True),
    "test0_final":          ("core", ["scripts/test0_final.py"], True),
    "test0_followup":       ("core", ["scripts/test0_followup.py"], True),
    "benchmark_v2":         ("core", ["scripts/benchmark_v2.py"], True),
    "reanalysis_v2":        ("core", ["scripts/reanalysis_v2.py"], False),   # reads data/processed/hourly_v2.pkl.gz
    "fig_test0_identity":   ("figures", ["scripts/fig_test0_identity.py"], True),
    "fig_kn_curves":        ("figures", ["scripts/fig_kn_curves.py"], False),
    "audit_dirint":         ("audit", ["scripts/audit_dirint_vs_pvlib.py"], False),
    "reproduce_pvlib_claim": ("audit", ["scripts/reproduce_paper_pvlib_claim.py"], False),
    "reproduce_v1_batch":   ("audit", ["scripts/reproduce_v1_batch.py"], True),
    "validate_gompertz_c":  ("optional", ["scripts/validate_gompertz_port.py"], False),   # needs gcc
}
ORDER_GROUPS = ["selftests", "core", "figures", "audit"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--group", choices=ORDER_GROUPS + ["optional", "all"])
    ap.add_argument("--only", nargs="+", choices=list(STEPS))
    ap.add_argument("--epw-dir", default=str(HERE / "data" / "raw"))
    a = ap.parse_args()

    if a.list or not (a.group or a.only):
        for name, (g, cmd, needs) in STEPS.items():
            print(f"{g:10s} {name:24s} {' '.join(cmd)}{'  [--epw-dir]' if needs else ''}")
        return 0

    if a.only:
        chosen = list(a.only)
    elif a.group == "all":
        chosen = [n for g in ORDER_GROUPS for n, (gg, _, _) in STEPS.items() if gg == g]
    else:
        chosen = [n for n, (gg, _, _) in STEPS.items() if gg == a.group]

    needs_epw = any(STEPS[n][2] for n in chosen)
    epw_dir = Path(a.epw_dir)
    if needs_epw:
        n_epw = len(list(epw_dir.glob("*.[eE][pP][wW]"))) if epw_dir.is_dir() else 0
        if n_epw == 0:
            print(f"ERROR: no .epw files in {epw_dir}. Put the 41 EPW files there (see README_LOCAL.md) or pass --epw-dir.", file=sys.stderr)
            return 2
        print(f"EPW directory: {epw_dir} ({n_epw} files)")

    (HERE / "logs").mkdir(exist_ok=True)
    log_path = HERE / "logs" / f"run_all_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.txt"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    results = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"python {sys.version}\n")
        for name in chosen:
            _, cmd, needs = STEPS[name]
            full = [sys.executable, *cmd] + (["--epw-dir", str(epw_dir)] if needs else [])
            print(f"\n=== {name}: {' '.join(full[1:])}", flush=True)
            t0 = time.time()
            p = subprocess.run(full, cwd=HERE, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            dt = time.time() - t0
            log.write(f"\n=== {name} rc={p.returncode} {dt:.1f}s\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}\n")
            tail = (p.stdout.strip().splitlines() or [""])[-3:]
            print("\n".join("   " + t for t in tail))
            if p.returncode:
                print("   STDERR tail: " + " | ".join((p.stderr.strip().splitlines() or [""])[-3:]))
            results.append((name, p.returncode, dt))
    print("\nSummary")
    for name, rc, dt in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name:24s} {dt:7.1f}s")
    print(f"log: {log_path}")
    return sum(1 for _, rc, _ in results if rc)


if __name__ == "__main__":
    sys.exit(main())
