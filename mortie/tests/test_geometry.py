"""Tests for the WKB/WKT geometry codec adapter (issue #71, phase 1).

These pin :mod:`mortie.geometry`: the lazy backend gate, the WKB/WKT (and
EWKB/EWKT) codec, and the decomposition of polygons / multipolygons / holes /
linestrings into ``(lat, lon)`` ring arrays — the input shape the ingest path
(phase 2) feeds to the existing coverage entry points.  The backend is used
only as a codec; spherical correctness is mortie's own job and not exercised
here.
"""

import numpy as np
import pytest

import mortie
from mortie import geometry

shapely = pytest.importorskip("shapely")


def test_decompose_polygon_with_hole():
    # A unit square with a square hole: exterior + one interior ring.
    g = shapely.from_wkt(
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0),"
        "(0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))"
    )
    kind, rings = geometry.decompose(g)
    assert kind == "polygonal"
    assert len(rings) == 2  # exterior + hole
    ext_lat, ext_lon = rings[0]
    # (x, y) = (lon, lat): the exterior spans lon/lat 0..2.
    assert np.isclose(ext_lon.max(), 2.0) and np.isclose(ext_lat.max(), 2.0)
    hole_lat, hole_lon = rings[1]
    assert np.isclose(hole_lon.min(), 0.5) and np.isclose(hole_lat.min(), 0.5)


def test_decompose_multipolygon_flattens_all_rings():
    g = shapely.from_wkt(
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),"
        "((5 5, 6 5, 6 6, 5 6, 5 5),(5.2 5.2, 5.8 5.2, 5.8 5.8, 5.2 5.8, 5.2 5.2)))"
    )
    kind, rings = geometry.decompose(g)
    assert kind == "polygonal"
    # poly1 (1 ring) + poly2 (exterior + 1 hole) = 3 rings, flattened.
    assert len(rings) == 3


def test_decompose_linestring_and_multilinestring():
    ls = shapely.from_wkt("LINESTRING (0 0, 1 1, 2 0)")
    kind, lines = geometry.decompose(ls)
    assert kind == "linear"
    assert len(lines) == 1
    lat, lon = lines[0]
    assert lat.shape == (3,) and lon.shape == (3,)

    mls = shapely.from_wkt("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3, 4 4))")
    kind, lines = geometry.decompose(mls)
    assert kind == "linear"
    assert [ln[0].size for ln in lines] == [2, 3]


def test_decompose_rejects_points_and_collections():
    with pytest.raises(ValueError, match="unsupported geometry type"):
        geometry.decompose(shapely.from_wkt("POINT (1 2)"))
    with pytest.raises(ValueError, match="unsupported geometry type"):
        geometry.decompose(
            shapely.from_wkt("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))")
        )


def test_decompose_rejects_empty_geometry():
    for wkt in ("POLYGON EMPTY", "LINESTRING EMPTY", "MULTIPOLYGON EMPTY"):
        with pytest.raises(ValueError, match="empty geometry"):
            geometry.decompose(shapely.from_wkt(wkt))


def test_decompose_drops_z_coordinate():
    # A 3-D polygon ingests as its 2-D lon/lat footprint (Z is dropped).
    g = shapely.from_wkt("POLYGON Z ((0 0 5, 1 0 5, 1 1 5, 0 1 5, 0 0 5))")
    kind, rings = geometry.decompose(g)
    assert kind == "polygonal"
    lat, lon = rings[0]
    assert lat.ndim == 1 and lon.ndim == 1  # no third column leaked through


def test_wkb_wkt_codec_roundtrip():
    wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
    g = geometry.geometry_from_wkt(wkt)
    # WKB round-trip preserves the rings.
    wkb = geometry.geometry_to_wkb(g)
    assert isinstance(wkb, (bytes, bytearray))
    g2 = geometry.geometry_from_wkb(wkb)
    k1, r1 = geometry.decompose(g)
    k2, r2 = geometry.decompose(g2)
    assert k1 == k2
    assert np.allclose(r1[0][0], r2[0][0]) and np.allclose(r1[0][1], r2[0][1])
    # WKT round-trip.
    g3 = geometry.geometry_from_wkt(geometry.geometry_to_wkt(g))
    assert int(shapely.get_type_id(g3)) == int(shapely.get_type_id(g))


def test_ewkb_ewkt_srid_optin():
    g = geometry.geometry_from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")

    # EWKT carries the SRID prefix; plain WKT does not.
    ewkt = geometry.geometry_to_wkt(g, srid=4326)
    assert ewkt.startswith("SRID=4326;")
    assert not geometry.geometry_to_wkt(g).startswith("SRID=")
    # The EWKT prefix is tolerated on ingest (advisory; contract is EPSG:4326).
    g_back = geometry.geometry_from_wkt(ewkt)
    assert int(shapely.get_type_id(g_back)) == int(shapely.get_type_id(g))

    # EWKB carries the SRID; from_wkb reads it back.
    ewkb = geometry.geometry_to_wkb(g, srid=4326)
    assert int(shapely.get_srid(geometry.geometry_from_wkb(ewkb))) == 4326
    plain = geometry.geometry_from_wkb(geometry.geometry_to_wkb(g))
    assert int(shapely.get_srid(plain)) == 0


# ── Phase 2: ingest reproduces the array-path coverage ─────────────────────

# A small polygon well away from the poles / antimeridian.
_LATS = [40.0, 50.0, 50.0, 40.0]
_LONS = [-120.0, -120.0, -110.0, -110.0]


def _poly_wkt(lats, lons):
    pts = ", ".join(f"{lo} {la}" for la, lo in zip(lats, lons))
    first = f"{lons[0]} {lats[0]}"
    return f"POLYGON (({pts}, {first}))"


def test_ingest_polygon_matches_array_path():
    want = mortie.morton_coverage(_LATS, _LONS, order=6)
    wkt = _poly_wkt(_LATS, _LONS)
    got_wkt = mortie.from_wkt(wkt, order=6)
    got_wkb = mortie.from_wkb(geometry.geometry_to_wkb(shapely.from_wkt(wkt)), order=6)
    assert np.array_equal(got_wkt, want)
    assert np.array_equal(got_wkb, want)


def test_ingest_polygon_with_hole_matches_array_path():
    outer_lat, outer_lon = _LATS, _LONS
    hole_lat = [43.0, 47.0, 47.0, 43.0]
    hole_lon = [-117.0, -117.0, -113.0, -113.0]
    want = mortie.morton_coverage(
        [outer_lat, hole_lat], [outer_lon, hole_lon], order=6
    )
    wkt = (
        "POLYGON (("
        + ", ".join(f"{lo} {la}" for la, lo in zip(outer_lat, outer_lon))
        + f", {outer_lon[0]} {outer_lat[0]}),("
        + ", ".join(f"{lo} {la}" for la, lo in zip(hole_lat, hole_lon))
        + f", {hole_lon[0]} {hole_lat[0]}))"
    )
    assert np.array_equal(mortie.from_wkt(wkt, order=6), want)


def test_ingest_multipolygon_matches_array_path():
    lats2 = [10.0, 20.0, 20.0, 10.0]
    lons2 = [-80.0, -80.0, -70.0, -70.0]
    want = mortie.morton_coverage([_LATS, lats2], [_LONS, lons2], order=6)
    wkt = (
        "MULTIPOLYGON ((("
        + ", ".join(f"{lo} {la}" for la, lo in zip(_LATS, _LONS))
        + f", {_LONS[0]} {_LATS[0]})),(("
        + ", ".join(f"{lo} {la}" for la, lo in zip(lats2, lons2))
        + f", {lons2[0]} {lats2[0]})))"
    )
    assert np.array_equal(mortie.from_wkt(wkt, order=6), want)


def test_ingest_polygon_moc_matches_array_path():
    want = mortie.morton_coverage_moc(_LATS, _LONS, order=8)
    wkt = _poly_wkt(_LATS, _LONS)
    assert np.array_equal(mortie.from_wkt(wkt, order=8, moc=True), want)


def test_ingest_linestring_matches_array_path():
    lats = [40.0, 50.0, 45.0]
    lons = [-120.0, -110.0, -100.0]
    want = mortie.linestring_coverage(lats, lons, order=6)
    wkt = "LINESTRING (" + ", ".join(f"{lo} {la}" for la, lo in zip(lats, lons)) + ")"
    got = mortie.from_wkt(wkt, order=6)
    assert np.array_equal(got, want)


def test_ingest_multilinestring_matches_array_path():
    lats = [[40.0, 50.0], [10.0, 20.0, 15.0]]
    lons = [[-120.0, -110.0], [-80.0, -70.0, -60.0]]
    want = mortie.linestring_coverage(lats, lons, order=6)
    wkt = (
        "MULTILINESTRING (("
        + ", ".join(f"{lo} {la}" for la, lo in zip(lats[0], lons[0]))
        + "),("
        + ", ".join(f"{lo} {la}" for la, lo in zip(lats[1], lons[1]))
        + "))"
    )
    got = mortie.from_wkt(wkt, order=6)
    assert isinstance(got, list) and len(got) == 2
    assert all(np.array_equal(g, w) for g, w in zip(got, want))


def test_ingest_linear_rejects_polygon_only_args():
    wkt = "LINESTRING (0 0, 1 1, 2 0)"
    with pytest.raises(ValueError, match="only to polygonal"):
        mortie.from_wkt(wkt, order=6, moc=True)


def test_ingest_moc_via_wkb_and_clockwise_spelling():
    # moc ingest works through WKB (not just WKT)...
    want = mortie.morton_coverage_moc(_LATS, _LONS, order=8)
    wkb = geometry.geometry_to_wkb(shapely.from_wkt(_poly_wkt(_LATS, _LONS)))
    assert np.array_equal(mortie.from_wkb(wkb, order=8, moc=True), want)
    # ...and a clockwise ring gives the same sub-hemisphere cover as CCW
    # (normalize=True default makes ordinary polygons orientation-insensitive).
    ccw = _poly_wkt(list(reversed(_LATS)), list(reversed(_LONS)))
    cw = _poly_wkt(_LATS, _LONS)
    assert np.array_equal(
        mortie.from_wkt(cw, order=6), mortie.from_wkt(ccw, order=6)
    )


# ── Phase 3: per-cell emit (dissolve=False) ────────────────────────────────


def test_emit_per_cell_one_polygon_per_cell():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    g = geometry.to_geometry(cov, dissolve=False)
    assert g.geom_type == "MultiPolygon"
    assert shapely.get_num_geometries(g) == cov.size


def test_emit_per_cell_mixed_order_moc():
    moc = mortie.morton_coverage_moc(_LATS, _LONS, order=8)
    g = geometry.to_geometry(moc, dissolve=False)
    # Each MOC cell (any order) emits exactly one quad.
    assert shapely.get_num_geometries(g) == moc.size


def test_emit_wkb_wkt_roundtrip_matches_cell_corners():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    wkb = geometry.to_wkb(cov, dissolve=False)
    back = shapely.from_wkb(wkb)
    assert shapely.get_num_geometries(back) == cov.size
    # The first emitted cell's exterior matches mort2polygon's lon/lat corners.
    # (cov is a single-order flat cover, so emit order tracks cov order here.)
    poly0 = shapely.get_geometry(back, 0)
    ring_lonlat = shapely.get_coordinates(shapely.get_exterior_ring(poly0))
    want = np.array([[lon, lat] for lat, lon in mortie.mort2polygon(int(cov[0]))])
    # Compare as ordered (lon, lat) PAIRS (lexsort on rows keeps the pairing, so
    # a per-vertex lon/lat swap would be caught — column-wise sorting would not).
    got_rows = ring_lonlat[np.lexsort((ring_lonlat[:, 1], ring_lonlat[:, 0]))]
    want_rows = want[np.lexsort((want[:, 1], want[:, 0]))]
    assert got_rows.shape == want_rows.shape
    assert np.allclose(got_rows, want_rows)
    # WKT path parses too.
    assert shapely.from_wkt(geometry.to_wkt(cov, dissolve=False)).geom_type \
        == "MultiPolygon"


def test_emit_single_cell_cover():
    # The grp.size == 1 scalar branch of _per_cell_polygons.
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)[:1]
    g = geometry.to_geometry(cov, dissolve=False)
    assert shapely.get_num_geometries(g) == 1
    assert g.is_valid


def test_emit_step_densifies_edges():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)[:1]
    g1 = geometry.to_geometry(cov, dissolve=False, step=1)
    g8 = geometry.to_geometry(cov, dissolve=False, step=8)
    n1 = shapely.get_coordinates(shapely.get_exterior_ring(
        shapely.get_geometry(g1, 0))).shape[0]
    n8 = shapely.get_coordinates(shapely.get_exterior_ring(
        shapely.get_geometry(g8, 0))).shape[0]
    # step=1 → 4 corners (+closing); step=8 → 32 boundary points (+closing).
    assert n1 == 5 and n8 == 33


def test_emit_antimeridian_and_polar_cells_are_valid():
    # A cover straddling the antimeridian and one over the north pole; per-cell
    # emit (with mort2polygon's antimeridian normalization) must stay valid
    # (no self-intersection) — the plan's emit acceptance criterion.
    am = mortie.morton_coverage(
        [10.0, 20.0, 20.0, 10.0], [179.0, 179.0, -179.0, -179.0], order=5
    )
    polar = mortie.morton_coverage(
        [85.0, 85.0, 89.0, 89.0], [-90.0, 90.0, 90.0, -90.0], order=5
    )
    for cov in (am, polar):
        g = geometry.to_geometry(cov, dissolve=False)
        assert g.geom_type == "MultiPolygon" and shapely.get_num_geometries(g) > 0
        # Every emitted cell quad is a valid (non-self-intersecting) polygon.
        for i in range(shapely.get_num_geometries(g)):
            assert shapely.get_geometry(g, i).is_valid


def test_emit_srid_optin_and_empty_cover():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    assert geometry.to_wkt(cov, dissolve=False, srid=4326).startswith("SRID=4326;")
    assert int(shapely.get_srid(shapely.from_wkb(
        geometry.to_wkb(cov, dissolve=False, srid=4326)))) == 4326
    empty = geometry.to_geometry(np.array([], dtype=np.uint64), dissolve=False)
    assert empty.geom_type == "MultiPolygon" and empty.is_empty


def test_emit_dissolve_default_pending_phase4():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    with pytest.raises(NotImplementedError, match="phase 4"):
        geometry.to_wkb(cov)


def test_backend_gate_message(monkeypatch):
    # With no backend importable, a clear ImportError naming shapely/spherely.
    import mortie.geometry as gm

    monkeypatch.setattr(gm, "_BACKEND", None)
    real_import = __import__

    def _block(name, *args, **kwargs):
        if name in ("shapely", "spherely"):
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _block)
    with pytest.raises(ImportError, match="shapely"):
        gm._require_backend()
    # monkeypatch reverts _BACKEND and __import__; the next call re-resolves.
