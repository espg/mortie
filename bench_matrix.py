"""Benchmark matrix across coverage methods (reproduces docs/coverage_methods.md).

Self-contained: builds the polygons from the bundled Antarctic drainage data and
times the hierarchical coverage methods on the current build.

    python bench_matrix.py

(The `old (flat)` column in the docs comes from the previous flood-fill
implementation and is not reproduced here.)
"""
import time
from pathlib import Path
import numpy as np
import mortie

DATA = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")


def basin(n_verts=None):
    """Largest drainage basin; optionally sub/super-sampled to ~n_verts."""
    d = np.loadtxt(DATA)
    b = d[d[:, 2] == 1]            # basin 1, ~82k vertices
    lat, lon = b[:, 0], b[:, 1]
    if n_verts is None or n_verts == len(lat):
        return lat.copy(), lon.copy()
    t = np.arange(len(lat))
    tt = np.linspace(0, len(lat) - 1, n_verts)
    return np.interp(tt, t, lat), np.interp(tt, t, lon)


methods = {
    "flat":       lambda la, lo, o: mortie.morton_coverage(la, lo, order=o),
    "moc":        lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o),
    "moc+tol0.5": lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, tolerance=0.5),
    "moc+bud2k":  lambda la, lo, o: mortie.morton_coverage_moc(la, lo, order=o, max_cells=2000),
}

cases = [(82_000, 8), (82_000, 10), (82_000, 12), (1_000_000, 10)]


def timed(fn, rep):
    ts = []
    out = None
    for _ in range(rep):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), len(out)


print("| verts | order | " + " | ".join(methods) + " |")
print("|--:|--:|" + "|".join(["--"] * len(methods)) + "|")
for nverts, order in cases:
    la, lo = basin(nverts)
    rep = 1 if len(la) >= 500_000 else 3
    cells = []
    for name, fn in methods.items():
        fn(la, lo, order)  # warmup
        t, n = timed(lambda: fn(la, lo, order), rep)
        cells.append(f"{n}c / {t*1e3:.0f}ms")
    print(f"| {len(la):,} | {order} | " + " | ".join(cells) + " |")
