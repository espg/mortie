"""
Issue #90 phase-2 measurement: how much of the `node_straddles` boundary
refinement is provable over-refinement, by cause?

Dev script — not wired to CI ("run and reported", the issue #78 benchmark
precedent).  Protocol (two builds):

    # 1. wall-time baseline on the normal release build (no instrumentation)
    maturin develop --release
    python benchmarks/measure_straddle_overrefinement.py timing -o timing.json

    # 2. cause-tagged stats on the instrumented build
    maturin develop --release --features descent-stats
    python benchmarks/measure_straddle_overrefinement.py stats -o stats.json

    # 3. merge into the report table
    python benchmarks/measure_straddle_overrefinement.py report timing.json \
        stats.json

For each straddle-stopped leaf the descent recorded, the polygon boundary is
re-tested independently of the descent's own predicates: every polygon edge
is densely sampled along its great circle (spacing 0.2 cells) and each sample
is classified with the exact HEALPix point-in-cell assignment (`geo2mort`).

  * a sample lands in the leaf          -> the boundary genuinely enters it;
  * min angular distance from the leaf centre to every sample exceeds the
    leaf's densified circumradius + half the sample spacing (+1e-6 rad
    safety) -> the boundary **provably misses** the cell = over-refinement;
  * anything between                    -> ambiguous, counted but not claimed.

`quad_touch` leaves (the #103 closed-set exact-incidence branch) are contract,
not waste: they are excluded from the over-refinement count and from the
hypothetical reduction regardless of the geometric verdict.

The achievable reduction assumes provably-missed leaves whose centre fill is
False (fully outside) are dropped; provably-missed fill=True leaves are fully
inside and already merge optimally in the MOC, so they change descent work
but not output size.  Hypothetical covers are computed exactly with
`moc_minus` + renormalization.
"""

import argparse
import json
import sys
import time

import numpy as np

import mortie
from mortie import _rustie, moc_minus, morton_coverage_moc

CAUSES = ["vertex_leaf", "quad_cross", "quad_touch", "corner_parity",
          "near_pole_bulge"]
QUAD_TOUCH = CAUSES.index("quad_touch")
ORDERS = [6, 9, 11]
SAMPLE_FRACTION = 0.2  # sample spacing as a fraction of the cell resolution
SAFETY_RAD = 1e-6


# ---------------------------------------------------------------------------
# Polygon suite
# ---------------------------------------------------------------------------

def _unit(lats, lons):
    la, lo = np.radians(lats), np.radians(lons)
    return np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1
    )


def _latlon(v):
    return (
        np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))),
        np.degrees(np.arctan2(v[:, 1], v[:, 0])),
    )


def box(lat_lo, lat_hi, lon_w, lon_e):
    return (
        np.array([lat_lo, lat_lo, lat_hi, lat_hi], float),
        np.array([lon_w, lon_e, lon_e, lon_w], float),
    )


def circle(n, lat0=-75.0, lon0=0.0, radius=5.0):
    """The bench circle: lat/lon-offset approximation (matches
    benchmarks/test_bench_coverage.py and the Rust coverage bench)."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return lat0 + radius * np.cos(ang), lon0 + radius * np.sin(ang)


def icesat2_swath(width_deg=0.06, n_track=260):
    """Synthetic ICESat-2-style swath: a ~130 deg near-polar ground-track
    segment buffered to ~6.6 km total width (the beam-pair span)."""
    a = _unit(np.array([-65.0]), np.array([-40.0]))[0]
    b = _unit(np.array([65.0]), np.array([20.0]))[0]
    omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
    t = np.linspace(0.0, 1.0, n_track)
    p = (
        np.sin((1 - t) * omega)[:, None] * a + np.sin(t * omega)[:, None] * b
    ) / np.sin(omega)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    # local track tangent and cross-track normal
    tan = np.gradient(p, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.cross(p, tan)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    h = np.radians(width_deg / 2.0)
    left = np.cos(h) * p + np.sin(h) * nrm
    right = np.cos(h) * p - np.sin(h) * nrm
    ring = np.concatenate([left, right[::-1]])
    return _latlon(ring)


def hemisphere_ring():
    """Hemisphere-plus ring: the pinned #22 world ring (lat -80..80,
    lon -90..90, vertex sum balanced so ingest trusts the winding), whose
    interior is the lon-0-facing hemisphere+ region including both poles —
    the shape `test_coverage_hemisphere.test_complement_world_minus_cap`
    validates.  Verified here to cover > half the sphere."""
    lats = np.array([-80.0, -80.0, 80.0, 80.0])
    lons = np.array([-90.0, 90.0, 90.0, -90.0])
    cells = morton_coverage_moc(lats, lons, order=3)
    frac = _rustie.rust_moc_to_order_count(
        np.asarray(cells, np.uint64), 3
    ) / (12 * 4 ** 3)
    if frac <= 0.5:
        raise RuntimeError(f"hemisphere+ ring covers only {frac:.3f}")
    return lats, lons


def shapes():
    tri_lat, tri_lon = np.array([20.0, 30.0, 25.0]), np.array(
        [-120.0, -120.0, -110.0]
    )
    sq_lat, sq_lon = np.array([20.0, 20.0, 30.0, 30.0]), np.array(
        [-125.0, -115.0, -115.0, -125.0]
    )
    trip_lat, trip_lon = np.array([-80.0, -88.0, -84.0]), np.array(
        [-120.0, -120.0, -100.0]
    )
    sqp_lat, sqp_lon = np.array([-80.0, -80.0, -87.0, -87.0]), np.array(
        [-130.0, -100.0, -100.0, -130.0]
    )
    return [
        # (class, name, lats, lons)
        ("ongrid_box", "box_belt_lon0", *box(20.0, 25.0, 0.0, 5.0)),
        ("ongrid_box", "box_cap_lon45", *box(60.0, 65.0, 44.0, 45.0)),
        ("ongrid_box", "box_equator", *box(0.0, 5.0, 100.0, 105.0)),
        ("midlat", "triangle_midlat", tri_lat, tri_lon),
        ("midlat", "square_midlat", sq_lat, sq_lon),
        ("polar", "triangle_polar", trip_lat, trip_lon),
        ("polar", "square_polar", sqp_lat, sqp_lon),
        ("circle", "circle32", *circle(32)),
        ("circle", "circle100", *circle(100)),
        ("circle", "circle500", *circle(500)),
        ("swath", "icesat2_swath", *icesat2_swath()),
        ("hemisphere", "hemisphere_ring", *hemisphere_ring()),
    ]


# ---------------------------------------------------------------------------
# Boundary sampling + independent leaf classification
# ---------------------------------------------------------------------------

def cell_resolution_rad(order):
    return np.sqrt(np.pi / 3.0) / (1 << order)


def sample_boundary(lats, lons, delta):
    """Sample every polygon edge along its great circle at spacing <= delta
    (vertices included).  Returns (unit vectors, actual max spacing)."""
    v = _unit(np.asarray(lats, float), np.asarray(lons, float))
    out, max_step = [], 0.0
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
        if omega < 1e-12:
            continue
        n = max(1, int(np.ceil(omega / delta)))
        max_step = max(max_step, omega / n)
        t = np.arange(n) / n  # endpoint b belongs to the next edge
        pts = (
            np.sin((1 - t) * omega)[:, None] * a + np.sin(t * omega)[:, None] * b
        ) / np.sin(omega)
        out.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    return np.concatenate(out), max_step


def min_angle_to_samples(centers, samples, chunk=2048):
    """Min angular distance from each centre to the sample set (chunked)."""
    best = np.full(len(centers), -1.0)
    for i in range(0, len(samples), chunk):
        np.maximum(best, (centers @ samples[i:i + chunk].T).max(axis=1),
                   out=best)
    return np.arccos(np.clip(best, -1.0, 1.0))


def measure_stats(lats, lons, order):
    take = _rustie.rust_descent_stats_take
    take()  # clear
    t0 = time.perf_counter()
    moc = np.asarray(morton_coverage_moc(lats, lons, order=order), np.uint64)
    wall = time.perf_counter() - t0
    st = take()

    delta = SAMPLE_FRACTION * cell_resolution_rad(order)
    samples, spacing = sample_boundary(lats, lons, delta)
    slat, slon = _latlon(samples)
    smort = np.asarray(mortie.geo2mort(slat, slon, order=order), np.uint64)

    morton = st["morton"]
    cause = st["cause"]
    fill = st["fill"]
    centers = np.stack([st["cx"], st["cy"], st["cz"]], axis=1)
    circ = st["circ"]

    hit = np.isin(morton, smort)  # a boundary sample lands in the leaf
    dist = np.full(len(morton), np.inf)
    cand = ~hit
    if cand.any():
        dist[cand] = min_angle_to_samples(centers[cand], samples)
    missed = cand & (dist > circ + spacing / 2.0 + SAFETY_RAD)
    ambiguous = cand & ~missed

    contract = cause == QUAD_TOUCH
    over = missed & ~contract
    removed = morton[over & ~fill]

    moc_hyp = moc_minus(moc, removed) if len(removed) else moc
    flat_cur = _rustie.rust_moc_to_order_count(moc, order)
    flat_hyp = _rustie.rust_moc_to_order_count(
        np.asarray(moc_hyp, np.uint64), order
    )

    # Unclaimable ceiling: what if every *ambiguous*, non-contract, fill=false
    # leaf were also dropped?  Not a safe relaxation (ambiguous cells may truly
    # intersect; dropping them can under-cover) — reported only to bound the
    # best case any relaxation could reach.
    ceil_removed = morton[(cand & ~contract & ~fill)]
    moc_ceil = moc_minus(moc, ceil_removed) if len(ceil_removed) else moc

    per_cause = {}
    for ci, name in enumerate(CAUSES):
        m = cause == ci
        per_cause[name] = {
            "total": int(m.sum()),
            "intersecting": int((m & hit).sum()),
            "provably_missed": int((m & missed).sum()),
            "ambiguous": int((m & ambiguous).sum()),
            "internal": int(st["internal_counts"][ci]),
        }
    return {
        "order": order,
        "moc_cells": int(len(moc)),
        "flat_cells": int(flat_cur),
        "straddle_leaves": int(len(morton)),
        "per_cause": per_cause,
        "over_refined": int(over.sum()),
        "over_refined_outside": int(len(removed)),
        "over_refined_inside": int((over & fill).sum()),
        "ambiguous": int(ambiguous.sum()),
        "moc_hyp": int(len(moc_hyp)),
        "flat_hyp": int(flat_hyp),
        "moc_reduction_pct": 100.0 * (1.0 - len(moc_hyp) / len(moc)),
        "flat_reduction_pct": 100.0 * (1.0 - flat_hyp / max(flat_cur, 1)),
        "ceiling_removed": int(len(ceil_removed)),
        "moc_ceiling": int(len(moc_ceil)),
        "moc_ceiling_reduction_pct": 100.0 * (1.0 - len(moc_ceil) / len(moc)),
        "samples": int(len(samples)),
        "sample_spacing_rad": float(spacing),
        "stats_build_wall_s": wall,
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_timing(out_path, reps=7):
    res = {}
    for cls, name, la, lo in shapes():
        for order in ORDERS:
            morton_coverage_moc(la, lo, order=order)  # warm
            best = min(
                _timed(lambda: morton_coverage_moc(la, lo, order=order))
                for _ in range(reps)
            )
            res[f"{name}@o{order}"] = {"class": cls, "wall_s": best}
            print(f"{name}@o{order}: {best * 1e3:.3f} ms", flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


def _timed(f):
    t0 = time.perf_counter()
    f()
    return time.perf_counter() - t0


def run_stats(out_path):
    if not hasattr(_rustie, "rust_descent_stats_take"):
        sys.exit(
            "extension lacks rust_descent_stats_take; rebuild with "
            "`maturin develop --release --features descent-stats`"
        )
    res = {}
    for cls, name, la, lo in shapes():
        for order in ORDERS:
            r = measure_stats(la, lo, order)
            r["class"] = cls
            res[f"{name}@o{order}"] = r
            print(
                f"{name}@o{order}: {r['straddle_leaves']} straddle leaves, "
                f"{r['over_refined']} provably over-refined "
                f"({r['over_refined_outside']} outside), "
                f"{r['ambiguous']} ambiguous, "
                f"MOC {r['moc_cells']} -> {r['moc_hyp']} "
                f"(-{r['moc_reduction_pct']:.2f}%)",
                flush=True,
            )
    json.dump(res, open(out_path, "w"), indent=1)


def run_report(timing_path, stats_path):
    timing = json.load(open(timing_path))
    stats = json.load(open(stats_path))
    cols = (
        "| shape | order | wall ms | straddle leaves | VL / QC / QT / CP / NP "
        "| over-refined (out+in) | ambig | MOC cur->hyp | MOC dv% | flat dv% "
        "| MOC ceiling dv% |"
    )
    print(cols)
    print("|" + "---|" * 11)
    for key, r in stats.items():
        c = r["per_cause"]
        vl, qc, qt, cp, np_ = (
            c["vertex_leaf"]["total"],
            c["quad_cross"]["total"],
            c["quad_touch"]["total"],
            c["corner_parity"]["total"],
            c["near_pole_bulge"]["total"],
        )
        wall = timing.get(key, {}).get("wall_s")
        wall_ms = f"{wall * 1e3:.2f}" if wall is not None else "-"
        print(
            f"| {key.split('@')[0]} | {r['order']} | {wall_ms} "
            f"| {r['straddle_leaves']} | {vl}/{qc}/{qt}/{cp}/{np_} "
            f"| {r['over_refined']} ({r['over_refined_outside']}+"
            f"{r['over_refined_inside']}) | {r['ambiguous']} "
            f"| {r['moc_cells']}->{r['moc_hyp']} "
            f"| {r['moc_reduction_pct']:.2f} | {r['flat_reduction_pct']:.2f} "
            f"| {r['moc_ceiling_reduction_pct']:.2f} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("timing", help="wall-time baseline (normal build)")
    t.add_argument("-o", "--out", default="straddle_timing.json")
    s = sub.add_parser("stats", help="cause-tagged stats (descent-stats build)")
    s.add_argument("-o", "--out", default="straddle_stats.json")
    r = sub.add_parser("report", help="merge the two JSONs into a table")
    r.add_argument("timing_json")
    r.add_argument("stats_json")
    args = ap.parse_args()
    if args.mode == "timing":
        run_timing(args.out)
    elif args.mode == "stats":
        run_stats(args.out)
    else:
        run_report(args.timing_json, args.stats_json)


if __name__ == "__main__":
    main()
