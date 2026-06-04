"""Benchmark matrix across coverage methods — writes the table straight into
docs/coverage_methods.md between the BENCH_MATRIX markers (and prints it).

    python bench_matrix.py

Self-contained: builds the polygons from the bundled Antarctic drainage data and
times the hierarchical coverage methods on the current build. Timings are
machine/run dependent; cell counts are deterministic. The `old (flat)`
flood-fill baseline is not reproduced here (that code was removed) — it lives as
static prose in the doc.
"""
import time
from pathlib import Path
import numpy as np
import mortie

DATA = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
DOC = Path("docs/coverage_methods.md")
START, END = "<!-- BENCH_MATRIX:START -->", "<!-- BENCH_MATRIX:END -->"


def basin(n_verts=None):
    """Largest drainage basin (~81,595 verts); densified to ~n_verts if given."""
    d = np.loadtxt(DATA)
    b = d[d[:, 2] == 1]
    lat, lon = b[:, 0], b[:, 1]
    if n_verts is None or n_verts == len(lat):
        return lat.copy(), lon.copy()
    t = np.arange(len(lat))
    tt = np.linspace(0, len(lat) - 1, n_verts)
    return np.interp(tt, t, lat), np.interp(tt, t, lon)


# (column label, callable) — the methods shown across the matrix.
METHODS = [
    ("flat", lambda la, lo, o: mortie.morton_coverage(la, lo, order=o)),
    ("MOC", lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o)),
    ("MOC tol 0.5°", lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, tolerance=0.5)),
    ("MOC tol 0.05°", lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, tolerance=0.05)),
    ("MOC budget 2k", lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, max_cells=2000)),
    ("MOC budget 500", lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, max_cells=500)),
]

# (vertex target | None=full basin, order)
CASES = [(None, 8), (None, 10), (None, 12), (1_000_000, 10)]


def timed(fn, rep):
    ts, out = [], None
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), len(out)


def build_table():
    cols = [m[0] for m in METHODS]
    rows = ["| verts | order | " + " | ".join(cols) + " |",
            "|--:|--:|" + "|".join(["--"] * len(cols)) + "|"]
    for nverts, order in CASES:
        la, lo = basin(nverts)
        rep = 1 if len(la) >= 500_000 else 3
        cells = []
        for _, fn in METHODS:
            fn(la, lo, order)  # warmup
            t, n = timed(lambda: fn(la, lo, order), rep)
            cells.append(f"{n:,}c / {t * 1e3:.0f}ms")
        rows.append(f"| {len(la):,} | {order} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    table = build_table()
    print(table)
    doc = DOC.read_text()
    if START not in doc or END not in doc:
        raise SystemExit(f"markers {START} / {END} not found in {DOC}")
    pre = doc[: doc.index(START) + len(START)]
    post = doc[doc.index(END):]
    note = ("`c` = cell count, `ms` = milliseconds (machine/run dependent; cell "
            "counts are deterministic).")
    DOC.write_text(f"{pre}\n\n{table}\n\n{note}\n\n{post}")
    print(f"\nwrote table into {DOC}")


if __name__ == "__main__":
    main()
