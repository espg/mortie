"""
Regression matrix for boundary-degenerate polygon coverage (issue #103).

Polygons whose edges lie exactly on the HEALPix degenerate grid — longitudes
that are multiples of 45 deg (base-cell boundary *and* centre/diagonal
meridians) and the equator — used to mis-fill catastrophically: the belt box
with a west edge on lon 0 leaked around the full longitude range, and a cap
box with an edge on lon 45 escaped to the pole.  The fix (the uniform
symbolic crossing predicate + exact determinant signs + seed hardening) is
pinned here by three invariants, checked for every case:

  (a) **superset** — every sampled interior point's cell is in the cover;
  (b) **no escape** — every covered cell centre stays within a small pad of
      the box (the closed-set contract adds at most a one-cell fringe);
  (c) **flat == MOC** — ``morton_coverage`` equals the densified
      ``morton_coverage_moc``.

The sweep covers all six degenerate meridians x three latitude bands (belt /
belt-cap transition / polar cap) x both edge sides x both hemispheres — the
cap rows exercise lon == 45 (mod 90), the pole-escape family a boundary-only
sweep would miss, and 315 is included explicitly (it behaved asymmetrically
to 45/135/225 before the fix).  Width cases pin the non-monotone
opposite-edge dependence, and the flagship reproducers from the issue run at
their original orders.
"""

import numpy as np
import pytest

from mortie import (
    geo2mort,
    moc_to_order,
    mort2geo,
    morton_coverage,
    morton_coverage_moc,
    order2res,
)

KM_PER_DEG = 111.0


def _box(lat_lo, lat_hi, lon_w, lon_e):
    lats = np.array([lat_lo, lat_lo, lat_hi, lat_hi, lat_lo], dtype=float)
    lons = np.array([lon_w, lon_e, lon_e, lon_w, lon_w], dtype=float)
    return [lats], [lons]


def _centres(keys):
    pts = [mort2geo(int(k)) for k in keys]
    lat = np.array([float(p[0][0]) for p in pts])
    lon = np.array([float(p[1][0]) for p in pts])
    return lat, lon


def _lon_dist(lon, mid):
    """Wrap-aware angular distance in longitude degrees."""
    return np.abs((lon - mid + 180.0) % 360.0 - 180.0)


def _bulge(lat, width):
    """Poleward bulge (degrees) of a great-circle arc between two points at
    ``lat`` separated by ``width`` degrees of longitude: the arc's extreme
    latitude is ``atan(tan(lat) / cos(width / 2))``."""
    la = np.radians(abs(lat))
    peak = np.degrees(np.arctan(np.tan(la) / np.cos(np.radians(width / 2.0))))
    return peak - abs(lat)


def _check_box(lat_lo, lat_hi, lon_w, lon_e, order, pad_cells=3.0):
    """Assert the three #103 invariants for a lat/lon box; return the cover."""
    lats, lons = _box(lat_lo, lat_hi, lon_w, lon_e)
    flat = np.asarray(morton_coverage(lats, lons, order=order))
    assert len(flat) > 0

    # (b) no escape: centres within a few cell widths of the box, allowing
    # for the poleward bulge of the great-circle edges beyond the corner
    # parallels.
    pad = pad_cells * order2res(order) / KM_PER_DEG
    width = lon_e - lon_w
    clat, clon = _centres(flat)
    lo_bound = (lat_lo - pad) - (_bulge(lat_lo, width) if lat_lo < 0 else 0.0)
    hi_bound = (lat_hi + pad) + (_bulge(lat_hi, width) if lat_hi > 0 else 0.0)
    assert clat.min() >= lo_bound, f"lat escape: {clat.min():.3f} < {lo_bound:.3f}"
    assert clat.max() <= hi_bound, f"lat escape: {clat.max():.3f} > {hi_bound:.3f}"
    mid = (lon_w + lon_e) / 2.0
    half = width / 2.0
    cos_floor = max(np.cos(np.radians(max(abs(lat_lo), abs(lat_hi)))), 0.05)
    lon_slack = half + pad / cos_floor
    worst = _lon_dist(clon, mid).max()
    assert worst <= lon_slack, f"lon escape: {worst:.3f} > {lon_slack:.3f}"

    # (a) superset: interior sample points' cells are all covered.  The
    # equatorward great-circle edge bulges poleward *into* the lat/lon
    # rectangle, so shrink the sampled band past that bulge — points below
    # it are genuinely outside the spherical polygon.
    shrink = 0.01
    s_lo, s_hi = lat_lo, lat_hi
    if lat_lo >= 0.0:
        s_lo = lat_lo + _bulge(lat_lo, width) + shrink
    if lat_hi <= 0.0:
        s_hi = lat_hi - _bulge(lat_hi, width) - shrink
    assert s_lo < s_hi, "box too thin to sample its interior"
    glat = np.linspace(s_lo, s_hi, 12)[1:-1]
    glon = np.linspace(lon_w, lon_e, 12)[1:-1]
    mlat, mlon = np.meshgrid(glat, glon)
    truth = np.asarray(geo2mort(mlat.ravel(), mlon.ravel(), order=order))
    missing = set(truth.tolist()) - set(flat.tolist())
    assert not missing, f"under-coverage: {len(missing)} interior cells missing"

    # (c) the flat cover equals the densified MOC cover.
    moc = np.asarray(morton_coverage_moc(lats, lons, order=order))
    dens = np.asarray(moc_to_order(moc, order))
    assert set(dens.tolist()) == set(flat.tolist()), "flat != densified MOC"
    return flat


# The degenerate meridians: base-cell boundaries (0 mod 90) and base-cell
# centre/diagonals (45 mod 90); 315 pinned explicitly (pre-fix asymmetry).
MERIDIANS = [0.0, 45.0, 90.0, 135.0, 180.0, 315.0]
# (lat_lo, lat_hi): equatorial belt, belt-cap transition, polar cap.
BANDS = [(20.0, 25.0), (35.0, 45.0), (60.0, 65.0)]


class TestDegenerateMeridianSweep:
    """Boxes with an edge exactly on every 45-deg meridian, both sides,
    three latitude bands, both hemispheres (issue #103)."""

    @pytest.mark.parametrize("lon0", MERIDIANS)
    @pytest.mark.parametrize("band", BANDS)
    @pytest.mark.parametrize("side", ["west", "east"])
    @pytest.mark.parametrize("south", [False, True])
    def test_edge_on_meridian(self, lon0, band, side, south):
        lat_lo, lat_hi = band
        if south:
            lat_lo, lat_hi = -lat_hi, -lat_lo
        if side == "west":
            lon_w, lon_e = lon0, lon0 + 5.0
        else:
            lon_w, lon_e = lon0 - 5.0, lon0
        _check_box(lat_lo, lat_hi, lon_w, lon_e, order=6)


class TestEquatorEdges:
    """Boxes with an edge exactly on the equator, including through the
    base-cell centre at (0, 0)."""

    @pytest.mark.parametrize(
        "lat_lo,lat_hi,lon_w,lon_e",
        [
            (0.0, 5.0, 100.0, 105.0),  # the issue's bottom-edge reproducer
            (-5.0, 0.0, 100.0, 105.0),  # top edge on the equator
            (0.0, 5.0, -2.5, 2.5),  # edge passes through (0, 0) exactly
            (-5.0, 5.0, 42.5, 47.5),  # straddles equator AND lon 45
        ],
    )
    def test_equator_edge(self, lat_lo, lat_hi, lon_w, lon_e):
        _check_box(lat_lo, lat_hi, lon_w, lon_e, order=6)


class TestOppositeEdgeWidths:
    """East edge pinned on lon 45; the opposite edge co-determines which
    probe legs the descent takes (non-monotone pre-fix: 829/66/231/966/957
    cells for these widths).  All must satisfy the invariants now."""

    @pytest.mark.parametrize("lon_w", [44.0, 40.0, 35.0, 30.0, 0.0])
    def test_width(self, lon_w):
        _check_box(60.0, 65.0, lon_w, 45.0, order=6)


class TestIssueReproducers:
    """The flagship reproducers from the issue body / thread, at their
    original orders, with the pre-fix failure sizes as regression bounds."""

    def test_belt_west_edge_lon0_order9(self):
        # pre-fix: 1687 cells spanning lon 0..360; control box: 1886 cells.
        flat = _check_box(20.0, 25.0, 0.0, 5.0, order=9)
        assert 1600 < len(flat) < 2200

    def test_cap_east_edge_lon45_order6_pole_escape(self):
        # pre-fix: 829 cells reaching lat 89.3; expected ~13.
        flat = _check_box(60.0, 65.0, 44.0, 45.0, order=6)
        assert len(flat) < 40

    def test_cap_mirrors_45_135_225_315(self):
        # pre-fix: 45/135/225 escaped to the pole, 315 only locally.
        sizes = set()
        for lon_e in (45.0, 135.0, 225.0, 315.0):
            flat = _check_box(60.0, 65.0, lon_e - 1.0, lon_e, order=6)
            sizes.add(len(flat))
        assert all(n < 40 for n in sizes)

    def test_southern_mirror(self):
        flat = _check_box(-65.0, -60.0, 44.0, 45.0, order=6)
        assert len(flat) < 40

    def test_polar_ring_sector_order9(self):
        # pre-fix: 775 cells reaching lat -89.91 (the pole); the polygon's
        # own great-circle south edge dips to -87.53, so the in-band answer
        # stays above -87.8.
        lats, lons = _box(-87.0, -86.7, 0.0, 45.0)
        flat = np.asarray(morton_coverage(lats, lons, order=9))
        clat, _ = _centres(flat)
        assert len(flat) < 200, f"sector over-covers: {len(flat)}"
        assert clat.min() > -87.8, f"pole escape: reaches {clat.min():.2f}"
        moc = np.asarray(morton_coverage_moc(lats, lons, order=9))
        dens = np.asarray(moc_to_order(moc, 9))
        assert set(dens.tolist()) == set(flat.tolist())

    def test_winding_direction_invariant(self):
        # The same box wound the other way (normalize=True default) must
        # produce the same cover.
        lats, lons = _box(60.0, 65.0, 44.0, 45.0)
        rev = np.asarray(
            morton_coverage([lats[0][::-1]], [lons[0][::-1]], order=6)
        )
        fwd = np.asarray(morton_coverage(lats, lons, order=6))
        assert set(rev.tolist()) == set(fwd.tolist())
