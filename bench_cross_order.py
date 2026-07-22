"""Cross-order throughput benchmark — writes the table straight into
docs/benchmarks.md between the BENCH_CROSS_ORDER markers (and prints it).

    python bench_cross_order.py

Self-contained: fixed-seed coordinate arrays, no external data. Reports raw
throughput (millions of morton indices per second) for encode (`geo2mort`) and
decode (`mort2geo`), plus mixed-order coverage (`morton_coverage_moc`) timing,
at representative orders. Throughput/timings are machine/run dependent; cell
counts are deterministic (fixed seed / fixed polygon).

Note on the coverage column: coverage cost scales with the polygon's *boundary
length measured in cells* (~2**order per boundary edge for the MOC), so a fixed
polygon covered at order 29 explodes. The coverage input is therefore a small
fixed ~0.01 degree (~1 km) box, chosen so order 29 stays tractable; see
docs/coverage_methods.md for the flat-vs-MOC / precision / budget trade-offs on
real-world polygons. Flat `morton_coverage` is not benchmarked cross-order
because it scales as ~4**order along the boundary and exhausts memory well
before order 29.
"""
import time
from pathlib import Path
import numpy as np
import mortie

DOC = Path("docs/benchmarks.md")
START, END = "<!-- BENCH_CROSS_ORDER:START -->", "<!-- BENCH_CROSS_ORDER:END -->"

SEED = 20260722
N = 1_000_000          # coordinates encoded/decoded per order
ORDERS = [4, 12, 18, 29]

# Small fixed box (~0.01 degree ~ 1 km) for the coverage column; deliberately
# small so order-29 MOC coverage stays tractable (see module docstring).
BOX_LAT = np.array([10.0, 10.0, 10.01, 10.01])
BOX_LON = np.array([20.0, 20.01, 20.01, 20.0])


def coords(n=N, seed=SEED):
    """Fixed-seed lat/lon arrays covering the whole sphere."""
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-89.0, 89.0, n)
    lon = rng.uniform(-180.0, 180.0, n)
    return lat, lon


def timed(fn, rep):
    ts, out = [], None
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def build_table():
    lat, lon = coords()
    rows = ["| order | encode (M idx/s) | decode (M idx/s) | coverage (cells / ms) |",
            "|--:|--:|--:|--:|"]
    for order in ORDERS:
        # encode: geo2mort(lat, lon, order) -> uint64 morton array
        morton = mortie.geo2mort(lat, lon, order=order)  # warmup + decode input
        t_enc, _ = timed(lambda: mortie.geo2mort(lat, lon, order=order), 5)
        enc_mps = N / t_enc / 1e6

        # decode: mort2geo(morton) -> (lat, lon)
        mortie.mort2geo(morton)  # warmup
        t_dec, _ = timed(lambda: mortie.mort2geo(morton), 5)
        dec_mps = N / t_dec / 1e6

        # coverage: mixed-order cover of the small fixed box (rep=1, no warmup —
        # order 29 is seconds-scale).
        t_cov, cov = timed(
            lambda: mortie.morton_coverage_moc(BOX_LAT, BOX_LON, order=order), 1)

        rows.append(f"| {order} | {enc_mps:,.1f} | {dec_mps:,.1f} | "
                    f"{len(cov):,}c / {t_cov * 1e3:.0f}ms |")
    return "\n".join(rows)


def main():
    table = build_table()
    print(table)
    doc = DOC.read_text()
    if START not in doc or END not in doc:
        raise SystemExit(f"markers {START} / {END} not found in {DOC}")
    pre = doc[: doc.index(START) + len(START)]
    post = doc[doc.index(END):]
    note = (f"Encode/decode throughput measured over {N:,} fixed-seed coordinates; "
            "coverage over a fixed ~0.01 degree box. `M idx/s` = millions of morton "
            "indices per second, `c` = cell count, `ms` = milliseconds. Throughput "
            "and timings are machine/run dependent; cell counts are deterministic.")
    DOC.write_text(f"{pre}\n\n{table}\n\n{note}\n\n{post}")
    print(f"\nwrote table into {DOC}")


if __name__ == "__main__":
    main()
