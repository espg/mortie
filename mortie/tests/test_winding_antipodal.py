"""Cover-path regression for the antipodal-lens winding defect (issue #107).

``ring_winding_at`` in ``src_rust/src/sphere.rs`` summed the signed angles each
directed edge subtends at the test point, and that sum is **antisymmetric**
under ``x -> -x``: it reports ``2*pi*[k(x) - k(-x)]``, not the winding
indicator ``k(x)``.  Every point whose antipode is also interior therefore
cancelled to zero and read *outside* -- unreachable for a sub-hemisphere
interior, which cannot contain an antipodal pair, but the normal state of
affairs for the hemisphere-plus rings of issue #22.

The witness pinned here is the "wobbly ring" from PR #112
(https://github.com/espg/mortie/pull/112#issuecomment-4934921227): a ring of
radius 85..110 deg about (45N, 0), whose two complementary interiors are
~55.3% and ~44.7% of the sphere.  Pre-repair, **both windings returned the
identical 45.6%-of-sphere cover** -- a direct signature of the antisymmetry,
since reversing a ring negates the winding everywhere and ``w > pi`` then
selects the same region -- and ~99% of the large side's interior points were
missing from it.  Two invariants make that unrepeatable:

  (a) **complementary** -- the two windings must select *different* regions,
      each a superset of its own analytic interior;
  (b) **superset** -- every sampled interior point of a side, taken clear of
      the boundary, must have its cell in that side's cover.

This exercises the *cover* path (seed classification -> descent -> MOC), not
the bare predicate; the predicate-level twins live in
``src_rust/src/sphere/tests.rs``.
"""

import numpy as np
import pytest

from mortie import geo2mort, moc_to_order, morton_coverage_moc

# No `importorskip` for the extension: the Rust path is the sole runtime path
# and `mortie` fails loudly at import without it, so the `from mortie import`
# above has already raised by the time any guard could run.

ORDER = 4
NCELLS = 12 * 4 ** ORDER

# Ring frame: centre at (45N, 0) with a right-handed tangent basis, so
# increasing azimuth traverses the ring counter-clockwise as seen from the
# centre and the CCW interior is the "inside the wobbly radius" side.
_CENTRE = np.array([np.cos(np.radians(45.0)), 0.0, np.sin(np.radians(45.0))])
_E1 = np.cross([0.0, 0.0, 1.0], _CENTRE)
_E1 /= np.linalg.norm(_E1)
_E2 = np.cross(_CENTRE, _E1)


def _radius_at(azimuth):
    """Ring radius (radians) as a function of azimuth about ``_CENTRE``."""
    return np.radians(97.5 + 12.5 * np.sin(3.0 * azimuth))


def _wobbly_ring():
    """The PR #112 ring as (lats, lons) in degrees, 96 vertices."""
    th = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    r = _radius_at(th)
    v = np.cos(r)[:, None] * _CENTRE + np.sin(r)[:, None] * (
        np.cos(th)[:, None] * _E1 + np.sin(th)[:, None] * _E2
    )
    lats = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    lons = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lats, lons


def _sample_sides(step=2.0, margin=0.03):
    """Grid samples split by analytic side, boundary-adjacent points dropped.

    Returns ``(lats, lons, in_small)`` for the kept points, where ``in_small``
    marks the region inside the wobbly radius -- the CCW interior of the
    as-given vertex order.
    """
    glat, glon = np.meshgrid(
        np.arange(-89.0, 90.0, step), np.arange(0.0, 360.0, step)
    )
    glat, glon = glat.ravel(), glon.ravel()
    la, lo = np.radians(glat), np.radians(glon)
    p = np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1
    )
    dist = np.arccos(np.clip(p @ _CENTRE, -1.0, 1.0))
    radius = _radius_at(np.arctan2(p @ _E2, p @ _E1))
    keep = np.abs(dist - radius) > margin
    return glat[keep], glon[keep], (dist < radius)[keep]


def _cover(lats, lons):
    """Order-``ORDER`` cover of one ring as a set of morton cells."""
    moc = morton_coverage_moc(lats, lons, order=ORDER)
    return set(int(c) for c in np.asarray(moc_to_order(moc, ORDER)))


def test_wobbly_hemisphere_plus_ring_windings_are_complementary():
    lats, lons = _wobbly_ring()
    # Ingest must take the hemisphere+ "trust the winding" path, or the test
    # would be checking orientation normalization instead of the winding.
    la, lo = np.radians(lats), np.radians(lons)
    v = np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1
    )
    axis = v.sum(axis=0)
    axis /= np.linalg.norm(axis)
    assert (v @ axis).min() <= 0.0, "ring must be hemisphere+ at ingest"

    small = _cover(lats, lons)
    large = _cover(lats[::-1].copy(), lons[::-1].copy())

    # (a) The two windings must not agree.  Pre-repair they were byte-identical
    # -- both 45.6% of the sphere -- which is impossible for two complementary
    # interiors of ~55.3% and ~44.7%.
    assert small != large
    f_small, f_large = len(small) / NCELLS, len(large) / NCELLS
    assert f_small > f_large, (f_small, f_large)
    # Each cover contains its interior plus at most a one-cell boundary fringe,
    # so the two overlap only on that fringe and sum to a little over 1.
    assert 1.0 < f_small + f_large < 1.10, (f_small, f_large)
    assert 0.55 < f_small < 0.62, f_small
    assert 0.43 < f_large < 0.50, f_large


def test_wobbly_hemisphere_plus_ring_covers_both_interiors():
    lats, lons = _wobbly_ring()
    slat, slon, in_small = _sample_sides()
    cells = np.asarray(geo2mort(slat, slon, order=ORDER))

    for reverse, want_small in ((False, True), (True, False)):
        ring = (lats[::-1].copy(), lons[::-1].copy()) if reverse else (lats, lons)
        cover = _cover(*ring)
        sel = in_small if want_small else ~in_small
        present = np.fromiter(
            (int(c) in cover for c in cells[sel]), dtype=bool, count=int(sel.sum())
        )
        # (b) superset.  Pre-repair the reversed ring returned the *small*
        # side's cover, so only ~1% of the large side's interior points landed
        # in it.
        assert present.all(), (
            f"reverse={reverse}: {(~present).sum()} of {present.size} interior "
            f"samples missing from the cover"
        )


# ── anchor-construction regressions (issue #107 phase 1, rework) ──────────
#
# The first attempt at the crossing anchor stepped an edge midpoint to its left
# by ``min(half_edge, 1e-6)`` rad and trusted the result.  Two ways that broke
# on *sub-hemisphere* input, where the pre-#107 code was already correct -- so
# each of these is a pure regression against ``main``, not a judgement call:
#
#   (1) ``dot(mid, a).acos()`` returns exactly 0.0 for an edge below ~2e-8 rad,
#       so the step was zero and the anchor sat ON the boundary.  Edge 0 is
#       sampled first, so a single 6 mm edge on an otherwise ordinary 10x10
#       square inverted its cover to the whole sphere (161 -> 49 068 cells).
#   (2) The 1e-6 cap is bounded by the edge's own length and says nothing about
#       how close another strand runs, so any feature thinner than ~6.4 m was
#       stepped clean across.  A 20-degree sliver inverted below exactly
#       1e-6 rad of width (46 -> 49 152 cells).
#
# The rework bounds the step by half the distance to the nearest *other*
# strand and then *proves* the candidate's side before using it.  The
# predicate-level twins live in ``src_rust/src/sphere/tests.rs``.

ORDER6 = 6


def _cells(lats, lons, order=ORDER6):
    """Order-``order`` cover of one ring as a set of morton cells."""
    moc = morton_coverage_moc(np.asarray(lats, float), np.asarray(lons, float),
                              order=order)
    return set(int(c) for c in np.asarray(moc_to_order(moc, order)))


# A plain 10x10 degree square and the same square carrying one extra vertex a
# hair along its first edge.  The extra vertex is geometrically inert, so the
# two covers must be identical.
_SQUARE = ([0.0, 0.0, 10.0, 10.0], [0.0, 10.0, 10.0, 0.0])


@pytest.mark.parametrize("gap_rad", [1e-6, 1e-7, 2e-8, 1e-8, 1e-10])
def test_short_first_edge_does_not_change_the_cover(gap_rad):
    # Sweeps across ~2e-8, the width below which the pre-rework `acos` step
    # underflowed to exactly zero.
    plain = _cells(*_SQUARE)
    assert len(plain) == 161, "baseline square cover (matches main)"
    gap = np.degrees(gap_rad)
    got = _cells([0.0, 0.0, 0.0, 10.0, 10.0], [0.0, gap, 10.0, 10.0, 0.0])
    assert got == plain, (
        f"a {gap_rad:.0e} rad first edge changed the cover: "
        f"{len(got)} cells vs {len(plain)}"
    )


@pytest.mark.parametrize("width_rad", [1e-4, 1e-5, 1e-6, 5e-7, 1e-7, 1e-9])
def test_thin_sliver_does_not_invert(width_rad):
    # Widths straddle the old ANCHOR_OFFSET_MAX = 1e-6 exactly.  The sliver is
    # far thinner than an order-6 cell at every width, so the cover is the run
    # of cells its 20-degree length passes through and does not vary.
    width = np.degrees(width_rad)
    got = _cells([0.0, 0.0, width, width], [0.0, 20.0, 20.0, 0.0])
    assert len(got) == 46, (
        f"sliver of width {width_rad:.0e} rad covered {len(got)} cells; "
        "an inverted cover is ~49 152"
    )


@pytest.mark.parametrize("name,build,order", [
    ("square", lambda: _SQUARE, ORDER6),
    ("short-first-edge",
     lambda: ([0.0, 0.0, 0.0, 10.0, 10.0],
              [0.0, np.degrees(1e-9), 10.0, 10.0, 0.0]), ORDER6),
    ("wobbly-hemisphere-plus", _wobbly_ring, ORDER),
])
def test_cover_is_invariant_under_vertex_rotation(name, build, order):
    # Which edges the anchor search samples depends on where the vertex list
    # starts, so a construction that took the first passable candidate could
    # answer differently per rotation.  The cover must not move.
    lats, lons = (np.asarray(a, float) for a in build())
    base = _cells(lats, lons, order=order)
    for k in range(1, len(lats)):
        rot = _cells(np.roll(lats, -k), np.roll(lons, -k), order=order)
        assert rot == base, f"{name}: rotating by {k} changed the cover"
