"""Hemisphere-plus / complement / #11 coverage checks vs ``cdshealpix``.

Phase 4 of issue #22.  These tests validate the single robust spherical
point-in-polygon backend at the cases that the old gnomonic / cap-axis path
could not handle:

* a **hemisphere-spanning** interior (a wide polygon whose interior is large),
* the **complement** case (the cap-cull complement guard, issue #22 Phase 2),
  expressed as a whole-world ring with a hole,
* the **issue #11** meridian-box case (a polygon edge lying exactly on a
  base-cell-centre meridian, where the orientation determinant hits exact zero
  at HEALPix cell centres and used to trigger an over-coverage flood).

**Oracle semantics.**  :func:`cdshealpix.nested.polygon_search` returns, at a
fixed depth, every NESTED cell that **overlaps** the polygon (a BMOC-style
cover), as a ``(ipix, depth, fully_covered)`` tuple.  mortie's flat
``morton_coverage`` at the same order is *also* an overlap cover — every cell it
emits (boundary or interior) touches the polygon — so the soundness contract
checked here is ``mortie_cover ⊆ oracle_overlap`` (up to a small boundary
tolerance for the two libraries' differing great-circle-vs-cell edge tests).
This directly catches the #11 over-coverage *flood*: flooded cells lie far from
the polygon, do not overlap it, and so are absent from the oracle.  We pair it
with explicit point-probe checks (``lonlat_to_healpix``) for interior/exterior
points, which are unambiguous and do not depend on the boundary tolerance.

Both ``cdshealpix`` (and its ``astropy`` dependency) and the compiled
``mortie._rustie`` extension are optional; the module skips cleanly when either
is unavailable (the extension is not built without ``maturin``).
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
from cdshealpix.nested import lonlat_to_healpix, polygon_search  # noqa: E402


def _mortie_nested(lats, lons, order):
    """mortie cover as a set of HEALPix NESTED ipix at ``order``."""
    morton = mortie.morton_coverage(lats, lons, order=order)
    cells, got_order = mort2healpix(np.asarray(morton))
    assert got_order == order, (got_order, order)
    return set(int(c) for c in np.atleast_1d(cells))


def _oracle_overlap(lats, lons, order):
    """cdshealpix overlap cover of a **single** polygon, as NESTED ipix.

    ``polygon_search(..., flat=True)`` returns ``(ipix, depth, fully_covered)``;
    we keep the ``ipix`` array.  Single-ring only — ``polygon_search`` has no
    multipart/hole support, so callers must pass one ring.
    """
    lon = np.asarray(lons, dtype=float) * u.deg
    lat = np.asarray(lats, dtype=float) * u.deg
    ipix, _depth, _fully = polygon_search(lon, lat, depth=order, flat=True)
    return set(int(c) for c in np.atleast_1d(np.asarray(ipix)))


def _cell_at(lat, lon, order):
    """NESTED ipix of the cell containing ``(lat, lon)`` at ``order``.

    cdshealpix 0.8 returns a 1-element 1-D array for scalar input (0.7 returned
    a 0-d array); ``atleast_1d(...)[0]`` reads the scalar either way.
    """
    ipix = lonlat_to_healpix(lon * u.deg, lat * u.deg, depth=order)
    return int(np.atleast_1d(np.asarray(ipix))[0])


def _assert_overlap_sound(lats, lons, order, tol=8):
    """mortie_cover ⊆ oracle_overlap (up to ``tol`` boundary cells).

    Every mortie cell must touch the polygon, so it must appear in the oracle's
    overlap cover.  A handful of boundary cells may differ between the two
    libraries' great-circle-vs-cell edge tests; ``tol`` absorbs that.  A #11
    flood adds *hundreds* of far-away cells, so it blows past ``tol``.
    """
    cover = _mortie_nested(lats, lons, order)
    oracle = _oracle_overlap(lats, lons, order)
    assert oracle, "oracle returned no cells — bad test polygon"
    spurious = cover - oracle
    assert len(spurious) <= tol, (
        f"mortie emitted {len(spurious)} cells that do not overlap the polygon "
        f"(tol {tol}; a #11 flood looks like this) — e.g. {sorted(spurious)[:5]}"
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
    _assert_overlap_sound(lats, lons, order=6)
    # The half west of the lon-45 meridian must NOT be covered (the flood half).
    assert _cell_at(41.0, 44.0, 6) not in _mortie_nested(lats, lons, 6)
    # A point genuinely inside must be covered.
    assert _cell_at(41.0, 46.0, 6) in _mortie_nested(lats, lons, 6)


def test_issue11_meridian_box_lon90():
    # Same degeneracy on the lon-90 base-cell-centre meridian family.
    lats = [40.0, 40.0, 42.0, 42.0]
    lons = [90.0, 92.0, 92.0, 90.0]
    _assert_overlap_sound(lats, lons, order=6)
    assert _cell_at(41.0, 89.0, 6) not in _mortie_nested(lats, lons, 6)
    assert _cell_at(41.0, 91.0, 6) in _mortie_nested(lats, lons, 6)


# ---------------------------------------------------------------------------
# Hemisphere-spanning interior
# ---------------------------------------------------------------------------

def test_hemisphere_spanning_band():
    # A wide CCW box spanning ~150° of longitude and a broad latitude band — a
    # large single sub-hemisphere-vertex polygon, exercising the robust winding
    # fill on a big region without a flood.
    lats = [-30.0, -30.0, 30.0, 30.0]
    lons = [-75.0, 75.0, 75.0, -75.0]
    _assert_overlap_sound(lats, lons, order=5)
    assert _cell_at(0.0, 0.0, 5) in _mortie_nested(lats, lons, 5)
    assert _cell_at(0.0, 160.0, 5) not in _mortie_nested(lats, lons, 5)


def test_large_cap_polygon():
    # A near-polar cap polygon (many vertices around lat 70°), CCW so the small
    # north cap is the interior.  Checks the robust path on a polar interior.
    n = 24
    lons = [k * (360.0 / n) for k in range(n)]
    lats = [70.0] * n
    _assert_overlap_sound(lats, lons, order=5)
    assert _cell_at(85.0, 0.0, 5) in _mortie_nested(lats, lons, 5)
    assert _cell_at(0.0, 0.0, 5) not in _mortie_nested(lats, lons, 5)


# ---------------------------------------------------------------------------
# Complement / "everything except a small cap" via world-minus-hole
# ---------------------------------------------------------------------------

def test_complement_world_minus_cap():
    # The hemisphere-plus / complement case: an outer ring whose CCW interior is
    # *larger* than a hemisphere, with a small hole carved from that interior.
    # Only the robust winding backend (+ the Phase-2 complement guard,
    # `covers_complement`) covers it correctly.  polygon_search has no hole
    # support, so this case is checked by point probes, not the overlap oracle.
    #
    # The outer ring's *vertices* must genuinely span > 90° (here a big ring over
    # lon −90..90, lat −80..80, whose vertex sum is balanced → hemisphere+, so it
    # is never orientation-normalized): a lone sub-hemisphere-vertex ring is
    # normalized to its *smaller* side by `build_ring` and so cannot express a
    # complement — see the winding contract on `coverage::build_ring`.  Wound so
    # the interior is the lon-0-facing hemisphere+; the hole sits at (0, 0).
    order = 4
    world_lat = [-80.0, -80.0, 80.0, 80.0]
    world_lon = [-90.0, 90.0, 90.0, -90.0]
    hole_lat = [-5.0, -5.0, 5.0, 5.0]
    hole_lon = [-5.0, 5.0, 5.0, -5.0]
    lats = [world_lat, hole_lat]
    lons = [world_lon, hole_lon]
    cover = _mortie_nested(lats, lons, order=order)
    # The cover is the complement of the hole: the north-pole cell (far from the
    # hole) is covered, and the hole-centre cell is carved out.
    assert _cell_at(89.5, 0.0, order) in cover, (
        "north pole cell must be inside the complement"
    )
    assert _cell_at(0.0, 0.0, order) not in cover, (
        "hole-centre cell must be excluded (carved out)"
    )
    # An interior point just outside the hole, on the same (lon-0) side, is
    # covered — the interior really is the large region around the hole.
    assert _cell_at(20.0, 0.0, order) in cover, (
        "interior point next to the hole must be covered"
    )
    # The south pole (also on the hemisphere+ interior) is covered — the
    # interior really wraps past the equator to the far pole.
    assert _cell_at(-89.5, 0.0, order) in cover, (
        "far interior point must be covered (hemisphere-plus interior)"
    )
    # The opposite (antimeridian) side is the exterior, so it is NOT covered —
    # this pins the orientation: the lon-0 side is interior, not the whole world.
    assert _cell_at(0.0, 180.0, order) not in cover, (
        "antimeridian side is exterior of this hemisphere+ ring"
    )
