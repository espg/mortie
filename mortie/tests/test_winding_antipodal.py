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


def _cover(lats, lons, normalize=True):
    """Order-``ORDER`` cover of one ring as a set of morton cells."""
    moc = morton_coverage_moc(lats, lons, order=ORDER, normalize=normalize)
    return set(int(c) for c in np.asarray(moc_to_order(moc, ORDER)))


def test_wobbly_hemisphere_plus_ring_windings_are_complementary():
    # ``normalize=False`` since issue #144 decision (A): with normalization
    # on, ingest now reverses ANY simple ring whose interior decisively reads
    # as the larger region -- hemisphere-plus included -- so the two windings
    # of this ring agree under the default (pinned by the test below).  The
    # winding-respect contract this test pins lives on the escape hatch.
    lats, lons = _wobbly_ring()
    # The ring must be hemisphere+ so the *predicate* (not ingest) is what
    # disambiguates the two sides via the vertex order.
    la, lo = np.radians(lats), np.radians(lons)
    v = np.stack(
        [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1
    )
    axis = v.sum(axis=0)
    axis /= np.linalg.norm(axis)
    assert (v @ axis).min() <= 0.0, "ring must be hemisphere+ at ingest"

    small = _cover(lats, lons, normalize=False)
    large = _cover(lats[::-1].copy(), lons[::-1].copy(), normalize=False)

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


def test_wobbly_ring_normalize_true_takes_the_smaller_side():
    # Decision (A), issue #144: under the default ``normalize=True`` both
    # windings of this simple hemisphere-plus ring cover the SMALLER region
    # (~44.7% of the sphere plus a boundary fringe) -- the S2
    # ``S2Loop::Normalize`` reading.  This is the intentional semantics
    # change signed off on the #144 thread (0.5645 -> 0.4355); the previous
    # "hemisphere+ is never reordered" behaviour moved to normalize=False.
    lats, lons = _wobbly_ring()
    as_given = _cover(lats, lons)
    reversed_ = _cover(lats[::-1].copy(), lons[::-1].copy())
    assert as_given == reversed_, "both windings must normalize identically"
    frac = len(as_given) / NCELLS
    assert 0.43 < frac < 0.50, frac


def test_wobbly_hemisphere_plus_ring_covers_both_interiors():
    # normalize=False: the winding-respect contract (see the complementary
    # test above for why the default no longer exercises it).
    lats, lons = _wobbly_ring()
    slat, slon, in_small = _sample_sides()
    cells = np.asarray(geo2mort(slat, slon, order=ORDER))

    for reverse, want_small in ((False, True), (True, False)):
        ring = (lats[::-1].copy(), lons[::-1].copy()) if reverse else (lats, lons)
        cover = _cover(*ring, normalize=False)
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


# ── the density-biased lemniscate (issue #107, review round 2) ────────────
#
# The counterexample that defeated the anchor construction outright.  Both
# lobes of a lemniscate have the same arc length, so a density bias is
# invisible to any measurement of the ring -- but a fixed index stride
# (``step_by(n / 8)``) puts *every* sampled edge on the clockwise lobe, where
# each candidate agrees on the wrong side.  Measured on the pre-rework tree at
# order 5: 12 250 of 12 288 cells at rotations 0/100/200/250/300 -- the whole
# sphere -- and 84 at rotation 180, where the stride happened to reach the
# other lobe.
#
# Nothing in the decision samples edges any more.  The orientation comes from
# the turning angle, which is a sum over every vertex and so cannot be steered
# by density or by where the list starts; the point test is a closed form.

def _lemniscate(n_dense=360, n_sparse=40, eps=1e-6):
    """``lat = 6 sin 2t``, ``lon = 12 sin t``, lopsidedly sampled."""
    t = np.concatenate([
        np.linspace(eps, np.pi - eps, n_dense),
        np.linspace(np.pi + eps, 2 * np.pi - eps, n_sparse),
    ])
    return 6.0 * np.sin(2 * t), 12.0 * np.sin(t)


def test_lemniscate_cover_is_rotation_invariant_and_not_the_whole_sphere():
    lats, lons = _lemniscate()
    base = _cells(lats, lons, order=5)
    total = 12 * 4 ** 5
    assert len(base) < total // 8, (
        f"cover inverted to {len(base)} of {total} cells"
    )
    far = int(geo2mort(np.array([0.0]), np.array([60.0]), order=5)[0])
    assert far not in base, "the far exterior must not be covered"
    for k in (1, 50, 100, 180, 200, 250, 300, 399):
        rot = _cells(np.roll(lats, -k), np.roll(lons, -k), order=5)
        assert rot == base, (
            f"rotating by {k} changed the cover: {len(rot)} vs {len(base)}"
        )


def test_lemniscate_cover_does_not_depend_on_vertex_density():
    # Same curve, three samplings: lopsided both ways and uniform.  The covers
    # differ only by how finely the chords approximate the curve, so at order 4
    # they must agree exactly.
    covers = {
        name: _cells(*_lemniscate(*counts), order=4)
        for name, counts in [("360/40", (360, 40)), ("40/360", (40, 360)),
                             ("200/200", (200, 200))]
    }
    ref = covers["200/200"]
    for name, cells in covers.items():
        assert cells == ref, f"{name} differs from the uniform sampling"


def test_reversed_small_ring_selects_the_complement():
    # Documented contract, now honoured at every ring size (issue #107).  main
    # returned neither the square nor its complement but the square's
    # *antipodal image*.  `normalize=False` is required: ingest would otherwise
    # correct the winding before the predicate sees it.
    from mortie import morton_coverage

    lats, lons = np.asarray(_SQUARE[0], float), np.asarray(_SQUARE[1], float)
    order = 4
    total = 12 * 4 ** order
    fwd = set(int(c) for c in morton_coverage([lats], [lons], order=order,
                                              normalize=False))
    rev = set(int(c) for c in morton_coverage([lats[::-1]], [lons[::-1]],
                                              order=order, normalize=False))
    assert 0 < len(fwd) < total // 4, f"forward cover is {len(fwd)} cells"
    # Complementary up to the shared boundary cells, which both covers contain.
    assert fwd | rev == set(range(total)) or len(fwd) + len(rev) >= total, (
        "the two windings must together cover the sphere"
    )
    assert len(rev) > total - 2 * len(fwd), (
        f"reversed cover is {len(rev)} of {total}; expected the complement"
    )
    # The antipodal blob `main` reported is gone: the antipode of an interior
    # point is not in the forward cover.
    anti = int(geo2mort(np.array([-5.0]), np.array([185.0]), order=order)[0])
    assert anti not in fwd, "antipodal image must not be covered"


# ── decision (A) and holes (issue #144) ───────────────────────────────────
#
# (A) rekeys ingest normalization on the turning sign, so a *capless* CW hole
# is now reversed where before it was passed through as-authored.  That is the
# case where the even-odd fill visibly improves: pre-(A) such a hole reported
# its 96.4% side, and the donut came out LARGER than its own outer ring --
# the fill inverted.  Sub-hemisphere holes are unaffected *by (A)* -- they were
# already cap-certified and normalized -- but only under ``normalize=True``,
# which is where that normalization happens; with the escape hatch on, a CW
# sub-hemisphere hole inverts the fill exactly the same way (pinned below).

_DONUT_ORDER = 5
_DONUT_NCELLS = 12 * 4 ** _DONUT_ORDER


def _donut():
    """Outer: a lat-2 deg circle CCW (cap-certified).  Hole: the #144 crescent
    wound CW -- capless (vertex-sum ``min_dot = -0.62``) and hemisphere-plus in
    its as-authored reading, so it is exactly the ring (A) changed."""
    from mortie.tests._normalization_corpus import CORPUS

    outer = (np.full(72, 2.0), np.arange(72) * 5.0)
    hole = CORPUS["crescent_cw"]
    return ([outer[0], hole[0]], [outer[1], hole[1]])


def _cell(lat, lon):
    """The order-``_DONUT_ORDER`` cell containing one probe point."""
    return int(geo2mort(np.array([lat]), np.array([lon]),
                        order=_DONUT_ORDER)[0])


def test_capless_cw_hole_is_carved_not_inverted():
    from mortie import morton_coverage

    lats, lons = _donut()
    cover = set(int(c) for c in morton_coverage(lats, lons,
                                                order=_DONUT_ORDER))
    # A point inside the hole (the lat 5-10 deg crescent band) is NOT covered;
    # one in the annulus (the rest of the north cap) is; the far side is not.
    assert _cell(7.5, 150.0) not in cover, "the hole must be carved out"
    assert _cell(45.0, 100.0) in cover, "the annulus must be covered"
    assert _cell(-45.0, 100.0) not in cover, "outside the outer ring"
    # outer (0.4948) - crescent (0.0526) + boundary fringe = 0.4776.
    frac = len(cover) / _DONUT_NCELLS
    assert 0.47 < frac < 0.485, frac

    # normalize=False reproduces the pre-(A) reading exactly here -- the outer
    # ring's as-given winding was already correct -- and it inverts: the hole
    # contributes its 96.4% side, so the "donut" is bigger than its own outer
    # ring and the two probe points swap.
    raw = set(int(c) for c in morton_coverage(lats, lons, order=_DONUT_ORDER,
                                              normalize=False))
    assert _cell(7.5, 150.0) in raw and _cell(45.0, 100.0) not in raw
    assert 0.57 < len(raw) / _DONUT_NCELLS < 0.59, len(raw) / _DONUT_NCELLS


# ── holes under the `normalize=False` escape hatch (issue #107, phase 5) ───
#
# The winding rule is per-ring and has nothing to do with a ring's role: a ring
# selects the region to its LEFT.  A hole therefore carves only when its own
# small region is on its left -- counter-clockwise, like its exterior.  RFC
# 7946's CW-hole spelling is what ``normalize=True`` ingest *delivers*, not
# what the predicate below ingest wants, so under ``normalize=False`` a CW hole
# selects its 98.9% complement and even-odd parity inverts: the carved annulus
# drops out and the two regions the annulus separates (the hole's disc and
# everything outside the exterior) are covered instead.
#
# Concentric caps about (20N, 100E) at order 5: exterior 30 deg (0.06699 of the
# sphere), hole 12 deg (0.01093), annulus 0.05607.  The carved reading is
# annulus + fringe = 776 cells (6.31%); the inverted reading is 1 - annulus +
# fringe = 11 687 cells (95.11%).


def _circle(lat_c, lon_c, radius_deg, n=64, cw=False):
    """Small circle about ``(lat_c, lon_c)``, CCW from the centre unless *cw*."""
    c = np.array([np.cos(np.radians(lat_c)) * np.cos(np.radians(lon_c)),
                  np.cos(np.radians(lat_c)) * np.sin(np.radians(lon_c)),
                  np.sin(np.radians(lat_c))])
    e1 = np.cross([0.0, 0.0, 1.0], c)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if cw:
        th = th[::-1].copy()
    r = np.radians(radius_deg)
    v = np.cos(r) * c + np.sin(r) * (np.cos(th)[:, None] * e1
                                     + np.sin(th)[:, None] * e2)
    return (np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0))),
            np.degrees(np.arctan2(v[:, 1], v[:, 0])))


def _sub_donut(hole_cw):
    """Concentric sub-hemisphere donut; the hole's winding is the variable."""
    outer = _circle(20.0, 100.0, 30.0)
    hole = _circle(20.0, 100.0, 12.0, cw=hole_cw)
    return [outer[0], hole[0]], [outer[1], hole[1]]


def _carved(cover):
    """Probe triple for the carved reading: hole out, annulus in, far side out."""
    return (_cell(20.0, 100.0) not in cover,
            _cell(20.0, 120.0) in cover,
            _cell(-40.0, 250.0) not in cover)


def test_sub_hemisphere_cw_hole_inverts_only_without_normalize():
    from mortie import morton_coverage

    def cover(hole_cw, normalize):
        lats, lons = _sub_donut(hole_cw)
        # Pinned cell counts predate the authalic default; the winding
        # semantics under test are convention-independent (issue #186).
        return set(int(c) for c in morton_coverage(
            lats, lons, order=_DONUT_ORDER, normalize=normalize,
            latitude="geodetic-spherical"))

    # (1) normalize=True: ingest rewinds the CW hole, so the donut is carved.
    norm = cover(hole_cw=True, normalize=True)
    assert _carved(norm) == (True, True, True), "normalize=True must carve"
    assert len(norm) == 776, len(norm)

    # (2) normalize=False with the same CW hole: nothing is rewound, the hole
    # selects its complement, and the fill inverts -- all three probes flip and
    # the "donut" is 15x its own exterior.
    raw = cover(hole_cw=True, normalize=False)
    assert _carved(raw) == (False, False, False), "CW hole must invert here"
    assert len(raw) == 11687, len(raw)
    assert 0.95 < len(raw) / _DONUT_NCELLS < 0.96, len(raw) / _DONUT_NCELLS

    # (3) normalize=False done right: wind the hole CCW too, so its own small
    # region is on its left.  Byte-identical to (1).
    ccw = cover(hole_cw=False, normalize=False)
    assert ccw == norm, "a CCW hole must carve under the escape hatch"
