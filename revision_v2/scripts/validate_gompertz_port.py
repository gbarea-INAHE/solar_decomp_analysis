"""
validate_gompertz_port.py — Validate gompertz.gompertz_dni against an independent
literal transliteration of the Fortran subroutine (gompertz_ref.f.txt).

Three implementations are compared on a dense grid plus random points:
  1. numpy port            (gompertz.py)
  2. scalar Python port    (line-by-line copy of the Fortran control flow)
  3. C transliteration     (compiled with gcc, REAL*8 and REAL*4 builds)

Pass criteria (stated before running):
  numpy vs C double : max relative difference < 1e-12
  numpy vs C float  : max relative difference < 1e-4   (REAL*4 rounding)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gompertz import gompertz_dni  # noqa: E402

C_SRC = r"""
#include <stdio.h>
#include <math.h>
#ifdef USE_FLOAT
typedef float real;
#define POW powf
#else
typedef double real;
#define POW pow
#endif
int main(void) {
    double ksol_d, cosz_d, i0_d;
    while (scanf("%lf %lf %lf", &ksol_d, &cosz_d, &i0_d) == 3) {
        real KSOL = (real)ksol_d, COSZAV = (real)cosz_d, I0 = (real)i0_d;
        real A1, A2, A3, A4, KT, KN0, KN1, KN2, KN, KDN = 0;
        A1 = (real)-0.1556*COSZAV*COSZAV + (real)0.1028*COSZAV + (real)1.3748;
        A2 = (real)0.7973*COSZAV*COSZAV + (real)0.1509*COSZAV + (real)3.035;
        A3 = (real)5.4307*COSZAV + (real)7.2182;
        A4 = (real)2.99;
        if (COSZAV == 0) KT = 0; else KT = KSOL / (I0 * COSZAV);
        KN0 = A4 * KT;
        KN1 = 1 / POW(A2, KN0);
        KN2 = -A3 * KN1;
        KN = A1 * POW(A2, KN2);
        if (COSZAV > 0) KDN = KN * I0;
        if (COSZAV <= 0) KDN = 0;
        printf("%.17g\n", (double)KDN);
    }
    return 0;
}
"""


def scalar_port(ksol: float, cosz: float, i0: float) -> float:
    """Line-by-line copy of the Fortran control flow, scalar float64."""
    a1 = -0.1556 * cosz * cosz + 0.1028 * cosz + 1.3748
    a2 = 0.7973 * cosz * cosz + 0.1509 * cosz + 3.035
    a3 = 5.4307 * cosz + 7.2182
    a4 = 2.99
    kt = 0.0 if cosz == 0 else ksol / (i0 * cosz)
    kn0 = a4 * kt
    kn1 = 1 / (a2 ** kn0)
    kn2 = -a3 * kn1
    kn = a1 * a2 ** kn2
    return kn * i0 if cosz > 0 else 0.0


def run_c(binary: Path, inputs: np.ndarray) -> np.ndarray:
    # float() first: numpy 2 repr() would emit "np.float64(...)", which scanf cannot parse
    text = "\n".join(f"{float(a)!r} {float(b)!r} {float(c)!r}" for a, b, c in inputs) + "\n"
    out = subprocess.run([str(binary)], input=text, capture_output=True, text=True, check=True)
    return np.array([float(x) for x in out.stdout.split()])


def main() -> int:
    rng = np.random.default_rng(20261005)
    cos_grid = np.concatenate([[0.0, 1e-4, 0.005, 0.01, 0.05], np.linspace(0.1, 1.0, 19)])
    kt_grid = np.linspace(0.0, 1.2, 25)
    rows = []
    for i0 in (1355.0, 1367.0):
        for c in cos_grid:
            for kt in kt_grid:
                rows.append((kt * i0 * c if c > 0 else 0.0, c, i0))
    # random points, including negative cos(z) (sun below horizon)
    n_rand = 5000
    cr = rng.uniform(-0.2, 1.0, n_rand)
    ktr = rng.uniform(0.0, 1.2, n_rand)
    i0r = rng.choice([1355.0, 1367.0], n_rand)
    rows += [(k * i * c if c > 0 else 0.0, c, i) for c, k, i in zip(cr, ktr, i0r)]
    inputs = np.array(rows)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "g.c").write_text(C_SRC)
        for name, flag in (("g64", []), ("g32", ["-DUSE_FLOAT"])):
            subprocess.run(["gcc", "-O0", "-o", str(td / name), str(td / "g.c"), "-lm", *flag], check=True)
        c64 = run_c(td / "g64", inputs)
        c32 = run_c(td / "g32", inputs)

    ghi, cosz, i0 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    # numpy port is called per i0 value (i0 is a scalar parameter by design)
    npy = np.empty(len(inputs))
    for v in np.unique(i0):
        m = i0 == v
        npy[m] = gompertz_dni(ghi[m], cosz[m], float(v))
    sc = np.array([scalar_port(a, b, c) for a, b, c in inputs])

    def maxrel(a, b):
        denom = np.maximum(np.abs(b), 1e-9)
        return float(np.max(np.abs(a - b) / denom))

    r_np_c64 = maxrel(npy, c64)
    r_sc_c64 = maxrel(sc, c64)
    r_np_c32 = maxrel(npy, c32)
    print(f"points tested                : {len(inputs)}")
    print(f"max rel diff numpy vs C f64  : {r_np_c64:.3e}   (criterion < 1e-12)")
    print(f"max rel diff scalar vs C f64 : {r_sc_c64:.3e}   (criterion < 1e-12)")
    print(f"max rel diff numpy vs C f32  : {r_np_c32:.3e}   (criterion < 1e-4)")
    ok = r_np_c64 < 1e-12 and r_sc_c64 < 1e-12 and r_np_c32 < 1e-4
    print("RESULT:", "PASS" if ok else "FAIL")

    print("\nGompertz curve, I0=1367, DNI [W/m2] (rows: cos z; cols: kt)")
    kts = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    print("cos z   " + "".join(f"kt={k:<6}" for k in kts))
    for c in (1.0, 0.8, 0.5, 0.3, 0.1):
        vals = [gompertz_dni(np.array([k * 1367.0 * c]), np.array([c]), 1367.0)[0] for k in kts]
        print(f"{c:<7}" + "".join(f"{v:<9.1f}" for v in vals))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
