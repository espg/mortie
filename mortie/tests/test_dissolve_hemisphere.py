"""Hemisphere+ dissolve: the winding-free classifier (issue #147).

Phase 1 pins the orientation contract the classifier rests on, for the Python
reference oracle (:func:`mortie.dissolve._dissolved_rings_py`) side by side
with the Rust tests in ``src_rust/src/dissolve.rs``:

* every HEALPix cell boundary loop is emitted with a fixed handedness —
  interior on the LEFT at ``step == 1`` (CCW, positive spherical signed area)
  and on the RIGHT at ``step > 1`` (CW, negative), uniformly over every base
  cell and order;
* therefore, after one per-call calibration (reverse every chained ring iff
  the emission handedness is CW), every ring carries the covered region on
  its LEFT.  The property is local — each surviving directed edge bounds
  exactly one covered cell, on the emission side, and chaining preserves edge
  direction — so it holds at any cover scale, including exact-hemisphere and
  over-hemisphere covers the classifier used to reject.

Later phases add the classifier behaviour tests (hemisphere+ covers dissolve
to point-sampled-correct outlines) on both engines.
"""

import numpy as np
import pytest

import mortie
from mortie import _healpix as hp
from mortie import dissolve
from mortie.orders import _rust_mort2nested, _rust_nested2mort


def _cells(bases, order):
    """All nested cells of the given base cells at *order*."""
    n = 4 ** order
    return [b * n + i for b in bases for i in range(n)]


def _to_morton(nested, order):
    """Morton words for nested ids at a single order."""
    arr = np.ascontiguousarray(np.asarray(nested, dtype=np.uint64))
    return _rust_nested2mort(arr, np.full(len(nested), order, dtype=np.uint8))


def _cell_loop(order, nest, step):
    """One cell's boundary loop as an (M, 3) unit-vector array."""
    bnd = hp.boundaries(order, np.asarray([nest], dtype=np.int64), step=step)
    if bnd.ndim == 2:
        bnd = bnd[np.newaxis, ...]
    return np.transpose(bnd, (0, 2, 1))[0]


def _world_minus_one():
    nested = list(range(192))
    del nested[100]
    return nested


_AUDIT_COVERS = [
    ("base 0-3", _cells([0, 1, 2, 3], 1), 1),
    ("hemisphere 0-5", _cells([0, 1, 2, 3, 4, 5], 1), 1),
    ("over-hemisphere 0-6", _cells([0, 1, 2, 3, 4, 5, 6], 1), 1),
    ("exemplar 4/6/7/10", _cells([4, 6, 7, 10], 1), 1),
    ("exemplar 1/2/7", _cells([1, 2, 7], 1), 1),
    ("scattered", [16, 24, 25, 28, 40, 43], 1),
    ("world minus one order-2 cell", _world_minus_one(), 2),
]


# ── issue #181: stitcher failure states are curated ValueErrors ────────────


def test_stitch_missing_partner_raises_curated():
    # Issue #181: the stitching path's defensive failure states surface as
    # the curated ValueError convention (PR #111), never as a raw panic
    # crossing the FFI (Rust) or a RuntimeError (oracle).  A lone segment
    # starting and ending on +180° with no start above its end wraps the
    # north pole and finds no partner on -180°.  (The cover that used to
    # reach the old unbalanced-no-pole assert deterministically — base cells
    # {1, 2, 7} at order 1 — now dissolves; see the hemisphere+ family
    # below.)
    segs = [[(180.0, 10.0), (0.0, 15.0), (180.0, 20.0)]]
    with pytest.raises(ValueError, match="no partner segment"):
        dissolve._stitch_segments(segs)


# ── phase 1: the orientation contract ──────────────────────────────────────


def test_cell_boundary_emission_orientation_is_uniform():
    # Mirror of the Rust `cell_boundary_emission_orientation_is_uniform`.
    for order in range(6):
        n = 4 ** order
        exact = np.pi / (3.0 * 4.0 ** order)
        for step in (1, 2, 3, 4, 8):
            for base in range(12):
                for nest in (base * n, base * n + n // 2, base * n + n - 1):
                    a = dissolve._spherical_signed_area(
                        _cell_loop(order, nest, step))
                    assert (a > 0) == (step == 1), (order, step, nest, a)
                    assert abs(abs(a) - exact) < 0.35 * exact, \
                        (order, step, nest, a)


# ── phases 2+3: hemisphere+ covers dissolve, identically on both engines ───


def _both_caps_cover():
    lats, lons = np.meshgrid(np.arange(-88.0, 88.5, 2.0),
                             np.arange(-180.0, 180.0, 2.0))
    keep = np.abs(lats.ravel()) > 62.0
    return sorted(set(
        int(w) for w in mortie.geo2mort(
            lats.ravel()[keep], lons.ravel()[keep], 3)))


def _assert_centre_sampled(mp, cover, sample_order):
    """Point-sample *mp* against exact cover membership at cell centres.

    Every cell centre of ``sample_order`` (whole sphere) must be inside the
    emitted planar geometry iff its ancestor cell is in the cover — the
    Python mirror of the Rust ``assert_point_sampled`` oracle.  Centres are
    strictly interior to their covering cell, hence bounded away from the
    emitted outline (centres exactly on the ±180 seam are nudged west, with
    membership re-evaluated at the nudged point).
    """
    import shapely

    nested, depths = _rust_mort2nested(
        np.ascontiguousarray(np.asarray(cover, dtype=np.uint64)))
    order = int(depths.max())
    assert int(depths.min()) == order  # single-order fixtures
    assert sample_order >= order
    covered = set(int(c) for c in nested)
    npix = 12 * 4 ** sample_order
    pix = np.arange(npix, dtype=np.int64)
    lon, lat = hp.pix2ang(sample_order, pix)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    seam = np.abs(np.abs(lon) - 180.0) < 1e-9
    lon = np.where(seam, 180.0 - 1e-4, lon)
    fine = np.asarray(
        mortie.geo2mort(lat, lon, sample_order), dtype=np.uint64)
    fine_nested, _ = _rust_mort2nested(np.ascontiguousarray(fine))
    anc = fine_nested >> np.uint64(2 * (sample_order - order))
    want = np.array([int(a) in covered for a in anc])
    got = shapely.contains_xy(mp, lon, lat)
    mismatch = np.nonzero(want != got)[0]
    # The emit is a straight-chord discretization of the true cell-edge
    # arcs, so pointwise agreement is only required for samples farther
    # from the emitted boundary than the chord-vs-arc gap (band shrinks
    # with cell size); mirrors the Rust `assert_point_sampled` oracle.
    band = 4.0 / 2 ** order
    boundary = mp.boundary
    for i in mismatch:
        d = boundary.distance(shapely.Point(lon[i], lat[i]))
        assert d < band, (float(lon[i]), float(lat[i]), float(d))


@pytest.mark.parametrize("name,nested,order", _AUDIT_COVERS,
                         ids=[c[0] for c in _AUDIT_COVERS])
def test_hemisphere_plus_family_both_engines(name, nested, order):
    # Issues #147 phases 2/3: every audit cover — exact-hemisphere,
    # over-hemisphere, the PR #179 sweep exemplars, pole-touching,
    # world-minus-cell — dissolves on the Rust engine, the Python oracle
    # mirrors it to machine precision (same fixtures, so mirror drift is
    # caught here), the emit is valid, and every cell centre two orders
    # finer point-samples correctly against exact cover membership.
    import shapely

    from mortie import geometry

    cover = _to_morton(nested, order)
    mp = geometry.to_geometry(cover)
    assert mp.is_valid, name
    ext_py, holes_py = dissolve._dissolved_rings_py(cover, 1)
    mp_py = shapely.MultiPolygon(dissolve._nest_and_build(shapely, ext_py, holes_py))
    assert mp_py.is_valid, name
    assert mp.symmetric_difference(mp_py).area < 1e-9, name
    _assert_centre_sampled(mp, cover, order + 2)


def test_multipart_both_polar_caps_both_engines():
    # A multipart hemisphere-straddling cover: both polar caps (|lat| > 62°),
    # whose two boundary circles have cancelling net longitude windings — the
    # case the old global pole selection could not stitch (each cap must wrap
    # its own pole).  Two clean parts, both engines identical.
    import shapely

    from mortie import geometry

    cover = np.asarray(_both_caps_cover(), dtype=np.uint64)
    covered = set(int(w) for w in cover)
    mp = geometry.to_geometry(cover)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 2
    ext_py, holes_py = dissolve._dissolved_rings_py(cover, 1)
    mp_py = shapely.MultiPolygon(dissolve._nest_and_build(shapely, ext_py, holes_py))
    assert mp.symmetric_difference(mp_py).area < 1e-9
    for la in (-88.0, -70.0, -40.0, 0.0, 40.0, 70.0, 88.0):
        for lo in (-170.0, -60.0, 0.0, 60.0, 170.0):
            w = int(mortie.geo2mort(np.asarray([la]), np.asarray([lo]), 3)[0])
            assert mp.covers(shapely.Point(lo, la)) == (w in covered), (lo, la)


def test_sweep_corpus_slice_oracle_matches_rust():
    # A deterministic slice of the regenerated order-1 sweep corpus (all 1-
    # and 2-base-cell masks, plus every 25th of the larger masks): the
    # Python oracle must mirror the Rust emit to machine precision — the
    # corpus-scale mirror-drift detector (the full 1585-mask corpus is
    # point-sampled Rust-side in `sweep_corpus_order1_combos...`).
    import shapely

    from mortie import geometry

    idx = 0
    checked = 0
    for mask in range(1, 1 << 12):
        bits = bin(mask).count("1")
        if bits > 5:
            continue
        idx += 1
        if bits > 2 and idx % 25:
            continue
        bases = [b for b in range(12) if mask >> b & 1]
        cover = _to_morton(_cells(bases, 1), 1)
        mp = geometry.to_geometry(cover)
        ext_py, holes_py = dissolve._dissolved_rings_py(cover, 1)
        mp_py = shapely.MultiPolygon(
            dissolve._nest_and_build(shapely, ext_py, holes_py))
        assert mp.symmetric_difference(mp_py).area < 1e-9, bases
        checked += 1
    assert checked > 100


# ── phase 4: spherely goldens and the round-trip invariant ─────────────────

# spherely (s2geometry) areas of the emitted outlines, pinned 2026-08-09:
# Σ shell areas − Σ hole areas, with the map-frame shell counting 4π.  An
# inverted exterior/hole classification would land near 4π − A, far outside
# the chord-discretization band the assertion allows (step-1 order-1 chords
# carry a ~2-3% area deficit; order 2 ~0.006%).
_SPHERELY_GOLDENS = {
    "base 0-3": 4.089856949,
    "hemisphere 0-5": 6.283185307,
    "over-hemisphere 0-6": 7.379849486,
    "exemplar 4/6/7/10": 4.312456774,
    "exemplar 1/2/7": 3.141592654,
    "scattered": 1.621696737,
    "world minus one order-2 cell": 12.500225105,
}


def _spherely_ring(ring):
    """Planar closed ring -> spherely-ready open ring (spherical dedup).

    Seam pairs like (-180, φ)/(180, φ) and pole corners are one spherical
    point; s2 rejects the degenerate zero-length edges they form.
    """
    def sph(p):
        rlat, rlon = np.radians(p[1]), np.radians(p[0])
        return (np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon),
                np.sin(rlat))

    out, last = [], None
    for p in ring:
        x = sph(p)
        if last is not None and max(abs(a - b) for a, b in zip(x, last)) < 1e-12:
            continue
        out.append((float(p[0]), float(p[1])))
        last = x
    while len(out) > 1 and max(
            abs(a - b) for a, b in zip(sph(out[0]), sph(out[-1]))) < 1e-12:
        out.pop()
    return out


@pytest.mark.parametrize("name,nested,order", _AUDIT_COVERS,
                         ids=[c[0] for c in _AUDIT_COVERS])
def test_dissolve_outline_areas_match_spherely(name, nested, order):
    # External sphere-truth oracle on the *classification* (issue #147
    # phase 4, the #144/#145 golden route): s2 measures the emitted
    # outline's area — shells positive, holes negative, frame 4π — which
    # must match the cover's exact area within chord discretization, and
    # the pinned golden to machine precision.
    spherely = pytest.importorskip("spherely")
    from mortie import _rustie

    cover = _to_morton(nested, order)
    shells, holes = _rustie.rust_dissolve(np.ascontiguousarray(cover), 1)
    total = 0.0
    for s in shells:
        ring = [tuple(map(float, v)) for v in s]
        if ring == dissolve._frame_ring():
            total += 4.0 * np.pi
            continue
        p = spherely.create_polygon(shell=_spherely_ring(ring), oriented=True)
        total += spherely.area(p) / spherely.EARTH_RADIUS_METERS ** 2
    for h in holes:
        ring = _spherely_ring([tuple(map(float, v)) for v in h])
        p = spherely.create_polygon(shell=ring[::-1], oriented=True)
        total -= spherely.area(p) / spherely.EARTH_RADIUS_METERS ** 2
    exact = len(nested) * np.pi / (3.0 * 4.0 ** order)
    assert abs(total / exact - 1.0) < 0.05, (name, total, exact)
    assert abs(total - _SPHERELY_GOLDENS[name]) < 1e-6, (name, total)


def test_roundtrip_coverage_contains_cover_normalize_false():
    # Open question (2) of issue #147: cover → dissolve →
    # from_geometry(..., normalize=False) → cover?  Exact set equality is
    # structurally impossible — coverage is an *overlap* cover, so
    # boundary-touching neighbour cells always join — but the containment
    # half (every source cell survives the round trip) holds for every
    # hole-free emit, hemisphere+ included, and is pinned here.  Emits with
    # holes are excluded: the parity fill under normalize=False reads a
    # CW-wound (RFC 7946) hole ring as selecting its complement, which
    # re-fills the hole — see the PR discussion for the full report.
    from mortie import geometry

    for name, nested, order in _AUDIT_COVERS:
        if name == "world minus one order-2 cell":
            continue  # holed emit (see above)
        cover = _to_morton(nested, order)
        mp = geometry.to_geometry(cover)
        back = geometry.from_geometry(mp, order=order, normalize=False)
        missing = set(int(w) for w in cover) - set(int(w) for w in back)
        assert not missing, (name, len(missing))


@pytest.mark.parametrize("name,nested,order", _AUDIT_COVERS,
                         ids=[c[0] for c in _AUDIT_COVERS])
@pytest.mark.parametrize("step", [1, 3])
def test_oracle_chained_rings_carry_cover_on_left(name, nested, order, step):
    # Mirror of the Rust `chained_rings_carry_cover_on_left`: drives the
    # oracle's `_boundary_rings_xyz` directly (below any guard), calibrates
    # once from one cell's own loop, and probes each ring edge's midpoint a
    # quarter edge-length to the left (must be covered) and right (must not).
    cover = _to_morton(nested, order)
    covered = set(int(w) for w in cover)
    rings = dissolve._boundary_rings_xyz(cover, step)
    assert rings, name
    ccw = dissolve._spherical_signed_area(_cell_loop(order, nested[0], step)) > 0
    if not ccw:
        rings = [r[::-1] for r in rings]
    for ring in rings:
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            t = b - a
            elen = float(np.linalg.norm(t))
            m = a + b
            m /= np.linalg.norm(m)
            left = np.cross(m, t)
            left /= np.linalg.norm(left)
            for s, want in ((1.0, True), (-1.0, False)):
                p = m + s * 0.25 * elen * left
                p /= np.linalg.norm(p)
                lat = float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0))))
                lon = float(np.degrees(np.arctan2(p[1], p[0])))
                w = int(mortie.geo2mort(
                    np.asarray([lat]), np.asarray([lon]), order)[0])
                assert (w in covered) == want, (name, step, s, lon, lat)
