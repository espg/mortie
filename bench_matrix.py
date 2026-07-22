"""Benchmark matrix across coverage methods — writes the table straight into
docs/coverage_methods.md between the BENCH_MATRIX markers (and prints it).

    python bench_matrix.py

Self-contained: builds the polygons from the bundled Antarctic drainage data and
times the hierarchical coverage methods on the current build. Timings are
machine/run dependent; cell counts are deterministic. The `old (flat)`
flood-fill baseline is not reproduced here (that code was removed) — it lives as
static prose in the doc.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import mortie

DATA = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
DOC = Path("docs/coverage_methods.md")
START, END = "<!-- BENCH_MATRIX:START -->", "<!-- BENCH_MATRIX:END -->"
WSTART, WEND = "<!-- BENCH_WARMUP:START -->", "<!-- BENCH_WARMUP:END -->"


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


# A genuine "cold" (first-call) measurement can only be the *first* parallel call
# in a process — the rayon threadpool and caches warm globally after that. So each
# warm-up row is measured in its own fresh subprocess. Contrasts a tiny cover
# (fixed startup dominates) with the full basin (startup amortized away).
WARMUP_CASES = [("~1 km box", "box", 11), ("81.6k-vert basin", "basin", 10)]

_PROBE = """
import time, json, sys, numpy as np, mortie
kind, order = sys.argv[1], int(sys.argv[2])
if kind == "box":
    la = np.array([10.0, 10.0, 10.01, 10.01]); lo = np.array([20.0, 20.01, 20.01, 20.0])
else:
    d = np.loadtxt("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
    b = d[d[:, 2] == 1]; la, lo = b[:, 0].copy(), b[:, 1].copy()
t = time.perf_counter(); mortie.morton_coverage_moc(la, lo, order=order)
cold = (time.perf_counter() - t) * 1e3
ts = []
for _ in range(7):
    t = time.perf_counter(); mortie.morton_coverage_moc(la, lo, order=order)
    ts.append((time.perf_counter() - t) * 1e3)
print(json.dumps({"cold": cold, "warm": float(np.median(ts))}))
"""


def warmup_table():
    rows = ["| MOC cover | cold (first call) | warm (steady state) | ratio |",
            "|--|--:|--:|--:|"]
    for label, kind, order in WARMUP_CASES:
        out = subprocess.run([sys.executable, "-c", _PROBE, kind, str(order)],
                             capture_output=True, text=True, check=True)
        d = json.loads(out.stdout)
        rows.append(f"| {label}, order {order} | {d['cold']:.1f} ms | "
                    f"{d['warm']:.1f} ms | {d['cold'] / d['warm']:.1f}x |")
    return "\n".join(rows)


def replace_block(doc, start, end, body):
    if start not in doc or end not in doc:
        raise SystemExit(f"markers {start} / {end} not found in {DOC}")
    return f"{doc[: doc.index(start) + len(start)]}\n\n{body}\n\n{doc[doc.index(end):]}"


def main():
    table = build_table()
    print(table)
    warmup = warmup_table()
    print("\n" + warmup)
    note = ("`c` = cell count, `ms` = milliseconds. Matrix timings are the warm "
            "median (each method is called once to warm up, then timed); see the "
            "first-call warm-up table for the one-time cold cost. Timings are "
            "machine/run dependent; cell counts are deterministic.")
    doc = DOC.read_text()
    doc = replace_block(doc, START, END, f"{table}\n\n{note}")
    doc = replace_block(doc, WSTART, WEND, warmup)
    DOC.write_text(doc)
    print(f"\nwrote tables into {DOC}")


if __name__ == "__main__":
    main()
