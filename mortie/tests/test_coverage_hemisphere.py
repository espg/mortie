"""Hemisphere-plus / complement / #11 coverage checks vs ``cdshealpix``.

Phase 4 of issue #22.  These tests validate the single robust spherical
point-in-polygon backend at the cases that the old gnomonic / cap-axis path
could not handle:

* a **hemisphere-plus** interior (a polygon whose interior is larger than a
  hemisphere — "everything except a small cap"),
* a **complement** flood (the cap-cull complement guard, issue #22 Phase 2),
* the **issue #11** meridian-box case (a polygon edge lying exactly on a
  base-cell-centre meridian, where the orientation determinant hits exact zero
  at HEALPix cell centres and used to trigger an over-coverage flood).

The oracle is :func:`cdshealpix.nested.polygon_search`, which returns HEALPix
NESTED cell ids at a fixed order for the cells whose centre is inside the
polygon.  mortie's cover is a **superset** of the polygon (it keeps every
boundary cell plus every interior cell), so the contract checked here is
``oracle_interior ⊆ mortie_cover`` — mortie must never *miss* a cell the oracle
calls inside.  We additionally bound the over-coverage to boundary cells so a
runaway flood (the #11 bug) would fail the test.

Both ``cdshealpix`` and the compiled ``mortie._rustie`` extension are optional
in CI; the module skips cleanly when either is unavailable (the extension is
not built without ``maturin``).
"""

import numpy as np
import pytest

import mortie
from mortie.tools import mort2healpix

# The robust coverage path needs the compiled Rust extension; skip the whole
# module if it isn't built (e.g. maturin not installed).
pytest.importorskip("mortie._rustie", reason="compiled mortie._rustie not built")
# cdshealpix is the polygon_search oracle; skip if not installed.
cdshealpix = pytest.importorskip("cdshealpix")

import astropy.units as u  # noqa: E402  (only needed once cdshealpix is present)
from cdshealpix.nested import polygon_search  # noqa: E402


def _mortie_nested(lats, lons, order):
    """mortie cover as a set of HEALPix NESTED ipix at ``order``."""
    morton = mortie.morton_coverage(lats, lons, order=order)
    cells, got_order = mort2healpix(np.asarray(morton))
    assert got_order == order, (got_order, order)
    return set(int(c) for c in np.atleast_1d(cells))


def _oracle_nested(lats, lons, order):
    """cdshealpix interior cells (centre-inside) as NESTED ipix at ``order``."""
    lon = np.asarray(lons, dtype=float) * u.deg
    lat = np.asarray(lats, dtype=float) * u.deg
    ipix = polygon_search(lon, lat, depth=order, flat=True)
    return set(int(c) for c in np.atleast_1d(np.asarray(ipix)))


def _assert_superset_bounded(lats, lons, order, max_extra_ratio=0.6):
    """mortie ⊇ oracle, and over-coverage stays bounded (no #11-style flood).

    ``max_extra_ratio`` caps ``|mortie \\ oracle| / |oracle|`` — boundary cells
    are a thin shell around the interior, so a small ratio is expected and a
    flood (covering the wrong half of the sphere) blows well past it.
    """
    cover = _mortie_nested(lats, lons, order)
    oracle = _oracle_nested(lats, lons, order)
    assert oracle, "oracle returned no interior cells — bad test polygon"
    missed = oracle - cover
    assert not missed, (
        f"mortie missed {len(missed)} oracle-interior cells "
        f"(e.g. {sorted(missed)[:5]})"
    )
    extra = cover - oracle
    assert len(extra) <= max_extra_ratio * len(oracle), (
        f"over-coverage flood: {len(extra)} extra cells vs {len(oracle)} "
        f"oracle cells (ratio {len(extra) / len(oracle):.2f})"
    )


# ---------------------------------------------------------------------------
# #11 — meridian-box (edge exactly on a base-cell-centre meridian)
# ---------------------------------------------------------------------------

def test_issue11_meridian_box_no_flood():
    # Box whose left edge lies exactly on lon 45 — base cell 0's centre
    # (lat 41.81°, lon 45°) sits on that meridian's great circle, the exact
    # degeneracy that flooded coverage before the robust SoS PIP.  CCW winding.
    lats = [40.0, 40.0, 42.0, 42.0]
    lons = [45.0, 47.0, 47.0, 45.0]
    _assert_superset_bounded(lats, lons, order=6)


def test_issue11_meridian_box_lon90():
    # Same degeneracy on the lon-90 base-cell-centre meridian family.
    lats = [40.0, 40.0, 42.0, 42.0]
    lons = [90.0, 92.0, 92.0, 90.0]
    _assert_superset_bounded(lats, lons, order=6)


# ---------------------------------------------------------------------------
# Hemisphere-spanning interior
# ---------------------------------------------------------------------------

def test_hemisphere_spanning_band():
    # A wide CCW box spanning ~150° of longitude and a broad latitude band — its
    # interior is large but still a single sub-hemisphere-vertex polygon, so it
    # exercises the robust winding fill on a big region without a flood.
    lats = [-30.0, -30.0, 30.0, 30.0]
    lons = [-75.0, 75.0, 75.0, -75.0]
    _assert_superset_bounded(lats, lons, order=5)


def test_large_cap_polygon():
    # A near-polar cap polygon (many vertices around lat 70°), CCW so the small
    # north cap is the interior.  Checks the robust path on a polar interior.
    n = 24
    lons = [k * (360.0 / n) for k in range(n)]
    lats = [70.0] * n
    _assert_superset_bounded(lats, lons, order=5)


# ---------------------------------------------------------------------------
# Complement / "everything except a small cap" via world-minus-hole
# ---------------------------------------------------------------------------

def test_complement_world_minus_cap():
    # The hemisphere-plus / complement case in its natural GeoJSON spelling: a
    # whole-world outer ring with a small Antarctic-style hole.  The interior is
    # "everything except the hole" — far larger than a hemisphere — which only
    # the robust backend (+ the Phase-2 complement guard) covers correctly.
    world_lat = [-85.0, -85.0, 85.0, 85.0]
    world_lon = [-179.9, 179.9, 179.9, -179.9]
    # Small hole near the south pole (wound CW relative to the outer ring;
    # ingest normalizes sub-hemisphere rings, so either winding carves the hole).
    hole_lat = [-80.0, -80.0, -75.0, -75.0]
    hole_lon = [-10.0, 10.0, 10.0, -10.0]
    lats = [world_lat, hole_lat]
    lons = [world_lon, hole_lon]
    order = 4
    cover = _mortie_nested(lats, lons, order=order)
    oracle = _oracle_nested(lats, lons, order=order)
    if oracle:
        missed = oracle - cover
        assert not missed, f"mortie missed {len(missed)} oracle-interior cells"
    # Independent of the oracle: the cover is the complement of the hole, so the
    # cell at the north pole (far from the hole) must be covered, and the cell at
    # the centre of the hole must NOT be.
    from cdshealpix.nested import lonlat_to_healpix

    north_ipix = int(
        np.asarray(lonlat_to_healpix(0.0 * u.deg, 89.5 * u.deg, depth=order))
    )
    hole_ipix = int(
        np.asarray(lonlat_to_healpix(0.0 * u.deg, -77.5 * u.deg, depth=order))
    )
    assert north_ipix in cover, "north pole cell must be inside the complement"
    assert hole_ipix not in cover, "hole-centre cell must be excluded (carved out)"
