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

The exact-arithmetic gate lives in Rust
(``coverage::tests::test_vertex_on_shared_corner_includes_every_incident_cell``),
which reads corners straight from ``cell_corners``.  These tests drive the
same property through the public Python surface.

**Coordinate caveat.**  ``mort2polygon`` is the only way to reach a cell corner
from Python, and cells that meet at one corner report it up to ~2.6e-8 rad
apart (measured; see the PR's "Questions for review").  That is seven orders of
magnitude above the incidence bound, so an apex placed from one cell's
``mort2polygon`` output is *provably* off some neighbours' boundaries and the
closed set correctly declines them.  The sweep below therefore asserts the
contract only where every incident cell agrees on the corner to within the
resolvable tolerance, and reports how many pairs that leaves.
"""

import numpy as np
import pytest

from mortie import geo2mort, mort2geo, mort2polygon, morton_coverage

# Angular slack within which two cells' reported corners are the same point.
# Chosen inside the descent's own incidence slack (`ORIENT_EPS` = 1e-12 scales
# the segment-span test in `edge_touches_cell_edge_degenerate`): an apex
# further out than that is provably past the cell edge's endpoint, so the
# closed set is right to decline it.  Still three orders of magnitude above
# f64 noise (~1e-16 rad), and far below the ~1.5e-8 rad quantization
# `mort2polygon` shows at some corners.
CORNER_SLACK_RAD = 1e-13


def _vec3(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _sep(p, q):
    return float(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))


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
    return min(_sep(pv, _vec3(*c)) for c in mort2polygon(cell)[:-1]) <= CORNER_SLACK_RAD


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


def test_apex_on_shared_corner_sweep_pins_the_residual():
    """Sweep the contract and pin how far phase 1 gets it.

    Phase 1 fixed the *incidence test*: at the depth where the touched corner
    lives, every incident cell now registers the touch.  It did not fix the
    **descent**, which prunes a subtree before reaching that depth when a
    coarse ancestor does not register the same touch — `cell_corners` at depth
    `d` and at depth `d+1` place the same geometric corner further apart than
    the incidence bound, so an ancestor can decline a corner its child accepts.
    The owner's ancestors survive on the vertex-leaf clause; the neighbours'
    do not.  See the PR's phase 2.

    Pinned so both directions are caught: a regression raises the count, and
    phase 2 driving it to zero has to update the pin.
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
    # 172 locally and 188 on CI's 3.12 runner.  Assert a usable sample size and
    # a *rate*, never an exact count.
    assert checked >= 100, f"sweep degenerated to {checked} cases"
    # 125 of 172 (73%) before phase 1's error-bounded incidence test, 12 of 172
    # (7.0%) after.  The ceiling catches a regression with headroom for the
    # platform spread; phase 2 driving the residual to zero has to move the
    # floor, which is the point.
    rate = violations / checked
    assert rate <= 0.15, f"closed-set violation rate rose: {violations}/{checked}"
    assert violations >= 1, (
        "the residual is gone — phase 2 landed?  Update this pin and the "
        "module docstring rather than deleting the sweep."
    )


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
