"""
Closed-set incidence at a shared cell corner (issue #117 item 1, in #107).

The #103 closed-set contract says a cell whose boundary the polygon touches
exactly is always included.  For an *edge-collinear* touch — the systematic
on-grid family — that has held since #103.  For a **point** touch it did not:
a polygon vertex landing on a HEALPix cell corner reaches the descent through
``lat/lon -> Vec3`` while the corner comes from ``cell_corners``, so the two
agree to ~1e-16 and never bit-exactly, and the incidence branch tested the
determinant against a literal zero.  Only the vertex's own leaf-owning cell
was guaranteed; the other cells meeting at that corner were dropped.

Closing it took two halves: an incidence test that recognises a determinant at
its own rounding floor (phase 1), and — because the descent's four-corner quad
is a *chord* that a coarse ancestor can be provably off while its descendants
sit on it — a combinatorial vertex clause that expands a boundary-incident
vertex's leaf to its HEALPix neighbourhood (phase 2,
``coverage::boundary_incident_neighbourhood``).

The exact-arithmetic gates live in Rust
(``coverage::tests::test_vertex_on_shared_corner_includes_every_incident_cell``
and ``…::test_apex_on_shared_corner_survives_the_ancestor_walk``), which read
corners straight from ``cell_corners``.  These tests drive the same property
through the public Python surface.

**Coordinate caveat.**  ``mort2polygon`` is the only way to reach a cell corner
from Python, and cells that meet at one corner report it up to ~2.6e-8 rad
apart (measured; see the PR's "Questions for review").  That is seven orders of
magnitude above the incidence bound, so an apex placed from one cell's
``mort2polygon`` output can land *provably* off the boundary of the leaf it
falls in — which is what the phase-2 expansion is gated on — and the closed set
then correctly declines the neighbours.  The sweep below therefore asserts the
contract only where every incident cell agrees on the corner to within the
resolvable tolerance, and reports how many pairs that leaves.
"""

import numpy as np
import pytest

from mortie import geo2mort, mort2geo, mort2polygon, morton_coverage

# Chord separation within which two cells' reported corners are the same point.
# Chosen inside the descent's own incidence slack (`ORIENT_EPS` = 1e-12 scales
# the segment-span test in `edge_touches_cell_edge_degenerate`): an apex
# further out than that is provably past the cell edge's endpoint, so the
# closed set is right to decline it.  Still three orders of magnitude above
# f64 noise (~1e-16), and far below the ~1.5e-8 rad quantization
# `mort2polygon` shows at some corners.
#
# Measured as a **chord** (`norm(p - q)`), not `arccos(dot(p, q))`.  For unit
# vectors the two agree to O(d^3), but `arccos` near 1 cannot resolve below
# ~sqrt(2 * eps) ≈ 1.5e-8 rad, so an angular form of this comparison would
# degenerate to `== 0.0` and silently admit pairs up to that far apart — the
# very quantization the tolerance is meant to sit under.  The chord resolves to
# ~1e-16, so 1e-13 means what it says.
CORNER_SLACK = 1e-13


def _vec3(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _chord(p, q):
    return float(np.linalg.norm(p - q))


def _incident_cells(vlat, vlon, order, eps=1e-7, n=128):
    """Cells meeting at ``(vlat, vlon)``, found by ringing the point."""
    coslat = max(np.cos(np.radians(vlat)), 1e-9)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    lats = vlat + eps * np.cos(th)
    lons = vlon + eps * np.sin(th) / coslat
    return set(int(x) for x in np.asarray(geo2mort(lats, lons, order=order)))


def _shares_corner(cell, v):
    """Does ``cell`` report a corner at ``v`` within the resolvable slack?"""
    pv = _vec3(*v)
    return min(_chord(pv, _vec3(*c)) for c in mort2polygon(cell)[:-1]) <= CORNER_SLACK


def _apex_triangle(v, owner, pull=0.15, half=0.15):
    """Triangle with its apex exactly on ``v``, body inside cell ``owner``."""
    cen = mort2geo(np.array([owner], dtype=np.uint64))
    clat = float(np.asarray(cen[0]).ravel()[0])
    clon = float(np.asarray(cen[1]).ravel()[0])
    mlat = v[0] + (clat - v[0]) * pull
    mlon = v[1] + (clon - v[1]) * pull
    plat = -(clon - v[1]) * half
    plon = (clat - v[0]) * half
    return (
        np.array([v[0], mlat + plat, mlat - plat]),
        np.array([v[1], mlon + plon, mlon - plon]),
    )


# Mid-latitude equatorial-zone samples, away from the poles where the
# four-corner quad stops tracing the true cell boundary.
SAMPLES = [(32.0, 47.0), (5.0, 12.0), (-24.0, 35.0), (48.0, -73.0)]


def test_apex_on_shared_corner_sweep_holds_everywhere():
    """Sweep the contract across orders, samples and corners: no violations.

    Two halves closed this.  Phase 1 fixed the *incidence test*, so at the
    depth where the touched corner lives every incident cell registers the
    touch (73% of the sweep violated before it, 7% after).  Phase 2 fixed the
    **descent**, which had been pruning those cells' subtrees before that depth
    was reached: `node_straddles`' quad clause tests the four-corner *chord*,
    the apex lies on the true (bulging) cell boundary, and a coarse ancestor is
    provably off a chord its own descendants sit on.  The vertex's leaf and its
    HEALPix neighbourhood settle it combinatorially instead — see
    ``coverage::boundary_incident_neighbourhood``.

    Pinned at zero, with a sample-size floor so the sweep cannot pass by
    degenerating.
    """
    violations = checked = 0
    for order in (4, 5, 6, 8):
        for lat0, lon0 in SAMPLES:
            cell = int(
                np.asarray(geo2mort(np.array([lat0]), np.array([lon0]), order=order))[0]
            )
            for v in mort2polygon(cell)[:-1]:
                incident = _incident_cells(v[0], v[1], order)
                agree = {c for c in incident if _shares_corner(c, v)}
                if len(agree) < 2:
                    continue  # no shared corner resolvable at this tolerance
                for owner in agree:
                    lats, lons = _apex_triangle(v, owner)
                    cover = set(
                        int(x)
                        for x in np.asarray(morton_coverage(lats, lons, order=order))
                    )
                    checked += 1
                    if agree - cover:
                        violations += 1
    # The sweep's *shape* comes from libm-sampled geometry (`_incident_cells`
    # rings a point through `geo2mort`; `_shares_corner` compares through
    # `arccos`), so how many cases qualify moves between platforms — this ran
    # 152 locally and 188 on CI's 3.12 runner at phase 1.  Assert a usable
    # sample size, never an exact count.
    assert checked >= 100, f"sweep degenerated to {checked} cases"
    # 125 of 172 (73%) on `main`, 12 of 172 (7.0%) after phase 1's
    # error-bounded incidence test, 0 of 152 after phase 2's vertex
    # neighbourhood.  Zero is the contract, so zero is the pin.
    assert violations == 0, f"closed-set violations: {violations}/{checked}"


@pytest.mark.parametrize("order", [5, 6])
def test_pinned_reproducer_covers_all_four_cells(order):
    """The issue #117 item 1 reproducer: 2 of 4 before the fix, 4 of 4 after."""
    cell = int(np.asarray(geo2mort(np.array([32.0]), np.array([47.0]), order=order))[0])
    checked = 0
    for v in mort2polygon(cell)[:-1]:
        incident = _incident_cells(v[0], v[1], order)
        agree = {c for c in incident if _shares_corner(c, v)}
        if len(agree) < 4:
            continue  # this corner is not resolvably four-way from Python
        lats, lons = _apex_triangle(v, cell)
        cover = set(
            int(x) for x in np.asarray(morton_coverage(lats, lons, order=order))
        )
        assert agree <= cover, f"corner {v}: {sorted(agree - cover)} not covered"
        checked += 1
    # Without this the `continue` above can skip every corner and the test
    # passes having asserted nothing (measured locally: 2 of 4 corners qualify
    # at order 5, 3 of 4 at order 6).  How many qualify is libm-dependent, so
    # the floor is "at least one", not a count.
    assert checked >= 1, "no corner reached the assertion"


def test_owning_cell_is_never_lost():
    """The widening adds cells; it must never drop the cell the body sits in."""
    order = 6
    cell = int(np.asarray(geo2mort(np.array([32.0]), np.array([47.0]), order=order))[0])
    for v in mort2polygon(cell)[:-1]:
        for owner in _incident_cells(v[0], v[1], order):
            lats, lons = _apex_triangle(v, owner)
            cover = set(
                int(x) for x in np.asarray(morton_coverage(lats, lons, order=order))
            )
            assert owner in cover, f"body cell {owner} missing for corner {v}"
