"""Tests for the WKB/WKT geometry codec adapter (issue #71, phase 1).

These pin :mod:`mortie.geometry`: the lazy backend gate, the WKB/WKT (and
EWKB/EWKT) codec, and the decomposition of polygons / multipolygons / holes /
linestrings into ``(lat, lon)`` ring arrays — the input shape the ingest path
(phase 2) feeds to the existing coverage entry points.  The backend is used
only as a codec; spherical correctness is mortie's own job and not exercised
here.
"""

import struct

import numpy as np
import pytest

import mortie
from mortie import codec, dissolve, geometry
from mortie.tests._normalization_corpus import CORPUS

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
    g = codec._geometry_from_wkt(wkt)
    # WKB round-trip preserves the rings.
    wkb = codec._geometry_to_wkb(g)
    assert isinstance(wkb, (bytes, bytearray))
    g2 = codec._geometry_from_wkb(wkb)
    k1, r1 = geometry.decompose(g)
    k2, r2 = geometry.decompose(g2)
    assert k1 == k2
    assert np.allclose(r1[0][0], r2[0][0]) and np.allclose(r1[0][1], r2[0][1])
    # WKT round-trip.
    g3 = codec._geometry_from_wkt(codec._geometry_to_wkt(g))
    assert int(shapely.get_type_id(g3)) == int(shapely.get_type_id(g))


def test_ewkb_ewkt_srid_optin():
    g = codec._geometry_from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")

    # EWKT carries the SRID prefix; plain WKT does not.
    ewkt = codec._geometry_to_wkt(g, srid=4326)
    assert ewkt.startswith("SRID=4326;")
    assert not codec._geometry_to_wkt(g).startswith("SRID=")
    # The EWKT prefix is tolerated on ingest (advisory; contract is EPSG:4326).
    g_back = codec._geometry_from_wkt(ewkt)
    assert int(shapely.get_type_id(g_back)) == int(shapely.get_type_id(g))

    # EWKB carries the SRID; from_wkb reads it back.
    ewkb = codec._geometry_to_wkb(g, srid=4326)
    assert int(shapely.get_srid(codec._geometry_from_wkb(ewkb))) == 4326
    plain = codec._geometry_from_wkb(codec._geometry_to_wkb(g))
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
    got_wkb = mortie.from_wkb(codec._geometry_to_wkb(shapely.from_wkt(wkt)), order=6)
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


def test_ingest_linear_rejects_normalize_false():
    # A line has no ring orientation: silently ignoring normalize=False would
    # be a no-op, so the linear branch rejects it (issue #108).
    for wkt in ("LINESTRING (0 0, 1 1, 2 0)",
                "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"):
        with pytest.raises(ValueError, match="only to polygonal"):
            mortie.from_wkt(wkt, order=6, normalize=False)


def test_ingest_moc_via_wkb_and_clockwise_spelling():
    # moc ingest works through WKB (not just WKT)...
    want = mortie.morton_coverage_moc(_LATS, _LONS, order=8)
    wkb = codec._geometry_to_wkb(shapely.from_wkt(_poly_wkt(_LATS, _LONS)))
    assert np.array_equal(mortie.from_wkb(wkb, order=8, moc=True), want)
    # ...and a clockwise ring gives the same sub-hemisphere cover as CCW
    # (normalize=True default makes ordinary polygons orientation-insensitive).
    ccw = _poly_wkt(list(reversed(_LATS)), list(reversed(_LONS)))
    cw = _poly_wkt(_LATS, _LONS)
    assert np.array_equal(
        mortie.from_wkt(cw, order=6), mortie.from_wkt(ccw, order=6)
    )


def test_ingest_moc_honours_normalize_false():
    # Issue #144 decision (A) makes ``normalize=False`` load-bearing: it is the
    # only way a ring expresses a bigger-than-complement interior, so the
    # ``moc=True`` branch must thread the caller's flag rather than hard-coding
    # it.  The PR #112 wobbly ring (hemisphere-plus, capless) at order 5: the
    # normalized side is 5474 cells, the winding-respected side 7060.
    lats, lons = CORPUS["wobbly_as_given"]
    wkt = _poly_wkt(lats, lons)
    # Pinned counts predate the authalic default; the normalize flag
    # threading under test is convention-independent (issue #186).
    dense = {
        norm: set(
            int(c)
            for c in mortie.moc.moc_to_order(
                mortie.from_wkt(wkt, order=5, moc=True, normalize=norm,
                                latitude="geodetic-spherical"), 5
            )
        )
        for norm in (True, False)
    }
    assert len(dense[True]) == 5474
    assert len(dense[False]) == 7060
    # And each densified MOC is exactly the flat cover of the same flag — the
    # two entry points cannot disagree about which side was taken.
    for norm in (True, False):
        flat = mortie.from_wkt(wkt, order=5, normalize=norm,
                               latitude="geodetic-spherical")
        assert dense[norm] == set(int(c) for c in flat)


# ── issue #157: from_wkb is backend-free, and unchanged for every input ────


_WKB_INPUT_CLASSES = {
    "polygon": "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))",
    "polygon_with_hole": (
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
        "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74))"
    ),
    "multipolygon": (
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),"
        "((5 5, 6 5, 6 6, 5 6, 5 5),(5.2 5.2, 5.8 5.2, 5.8 5.8, 5.2 5.8,"
        " 5.2 5.2)))"
    ),
    "antimeridian": (
        "POLYGON ((170 -20, -170 -20, -170 -10, 170 -10, 170 -20))"
    ),
    "pole_adjacent": (
        "POLYGON ((0 -89.5, 90 -89.5, 180 -89.5, -90 -89.5, 0 -89.5))"
    ),
    "linestring": "LINESTRING (10 -75, 40 -75, 40 -71)",
    "multilinestring": "MULTILINESTRING ((10 -75, 40 -75), (40 -71, 10 -71))",
    "polygon_z": (
        "POLYGON Z ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))"
    ),
    # M-only is the dimension spelling the Z/ZM cases do not reach: ISO
    # writes it as type 2003 and EWKB as the 0x40000000 flag, and both must
    # reduce to the same 2-D ring.
    "polygon_m": (
        "POLYGON M ((10 -75 1, 40 -75 1, 40 -71 1, 10 -71 1, 10 -75 1))"
    ),
    # Multipart *and* holed in one geometry — the two multi-ring cases above
    # exercise each half separately, and the even-odd descent sees them
    # together only here.
    "multipolygon_with_hole": (
        "MULTIPOLYGON (((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
        "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74)),"
        "((0 0, 2 0, 2 2, 0 2, 0 0)))"
    ),
}


@pytest.mark.parametrize("name", sorted(_WKB_INPUT_CLASSES))
@pytest.mark.parametrize("moc", [False, True])
@pytest.mark.parametrize("byte_order", [0, 1])
def test_from_wkb_is_unchanged_by_the_rust_reader(name, moc, byte_order):
    # from_wkb now parses in Rust instead of decoding through a backend, so
    # pin it against the path it replaced: the decomposition tail
    # (from_geometry on the backend-decoded geometry) is untouched, and every
    # input class must still land on exactly the same cells.
    wkt = _WKB_INPUT_CLASSES[name]
    geom = shapely.from_wkt(wkt)
    kind, _ = geometry.decompose(geom)
    if moc and kind == "linear":
        pytest.skip("moc applies only to polygonal geometry")
    blob = shapely.to_wkb(geom, byte_order=byte_order)
    want = mortie.from_geometry(geom, order=6, moc=moc)
    got = mortie.from_wkb(blob, order=6, moc=moc)
    if isinstance(want, list):  # MultiLineString: one array per line
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert np.array_equal(g, w)
    else:
        assert np.array_equal(got, want)


def test_from_wkb_reads_an_ewkb_srid_on_a_nested_part():
    # EWKB tags each geometry header independently, so a MultiPolygon can
    # carry an SRID on the outer header *and* on every part.  The reader
    # strips per header; nothing pins that from Python otherwise.
    pts = [(10, -75), (40, -75), (40, -71), (10, -71), (10, -75)]
    part = (
        struct.pack("<BII", 1, 3 | 0x20000000, 4326)
        + struct.pack("<II", 1, len(pts))
        + b"".join(struct.pack("<dd", x, y) for x, y in pts)
    )
    blob = (
        struct.pack("<BII", 1, 6 | 0x20000000, 4326)
        + struct.pack("<I", 1)
        + part
    )
    want = mortie.from_geometry(shapely.from_wkb(blob), order=6, moc=True)
    assert np.array_equal(mortie.from_wkb(blob, order=6, moc=True), want)


def test_from_wkb_ewkb_and_srid_are_unchanged():
    wkt = _WKB_INPUT_CLASSES["polygon_with_hole"]
    geom = shapely.from_wkt(wkt)
    want = mortie.from_geometry(geom, order=6)
    ewkb = shapely.to_wkb(shapely.set_srid(geom, 4326), include_srid=True)
    assert np.array_equal(mortie.from_wkb(ewkb, order=6), want)


@pytest.mark.parametrize(
    "wkt", ["POINT (10 -75)", "GEOMETRYCOLLECTION (POINT (10 -75))",
            "POLYGON EMPTY", "MULTIPOLYGON EMPTY"]
)
def test_from_wkb_still_refuses_what_it_refused_before(wkt):
    # These raised ValueError through the backend path and still do — the
    # reader reproduces `decompose`'s refusals rather than widening them.
    blob = shapely.to_wkb(shapely.from_wkt(wkt))
    with pytest.raises(ValueError):
        mortie.from_wkb(blob, order=6)
    with pytest.raises(ValueError):
        mortie.from_geometry(shapely.from_wkb(blob), order=6)


# ── issue #157: the from_wkb input contract (hex in, int iterables out) ────


def test_from_wkb_accepts_a_hex_string_as_the_backend_path_did():
    # `shapely.from_wkb` takes "the WKB byte object or hexadecimal string", so
    # the path this replaces covered a hex spelling; the Rust reader takes
    # bytes, so `_wkb_bytes` has to restore it.  Parity, not a new capability.
    geom = shapely.from_wkt(_WKB_INPUT_CLASSES["polygon_with_hole"])
    blob = shapely.to_wkb(geom)
    want = mortie.from_geometry(shapely.from_wkb(blob.hex()), order=6)
    for spelling in (blob.hex(), blob.hex().upper()):
        assert np.array_equal(mortie.from_wkb(spelling, order=6), want)
    # A string that is not hex is a parse failure, not a type failure.
    with pytest.raises(ValueError, match="hex"):
        mortie.from_wkb("not a wkb blob", order=6)


@pytest.mark.parametrize(
    "wrap", [bytearray, memoryview, lambda b: np.frombuffer(b, dtype=np.uint8)]
)
def test_from_wkb_accepts_byte_buffers_a_deliberate_widening(wrap):
    # NEWLY ACCEPTED, not preserved: all three raised `TypeError` through the
    # backend (`shapely.from_wkb` takes bytes or str only), asserted below.
    # `_wkb_bytes` accepts any one-byte-item buffer on purpose, for
    # arrow-backed callers that hand over a buffer rather than `bytes`.
    blob = shapely.to_wkb(shapely.from_wkt(_WKB_INPUT_CLASSES["polygon"]))
    assert np.array_equal(
        mortie.from_wkb(wrap(blob), order=6), mortie.from_wkb(blob, order=6)
    )
    with pytest.raises(TypeError):
        mortie.from_geometry(shapely.from_wkb(wrap(blob)), order=6)


@pytest.mark.parametrize(
    "bad",
    [list, tuple, lambda b: np.frombuffer(b, dtype=np.uint8).astype(np.float64),
     lambda b: 3, lambda b: None],
    ids=["list", "tuple", "float64_array", "int", "none"],
)
def test_from_wkb_refuses_non_byte_input_by_name(bad):
    # `bytes(data)` would assemble a blob out of *any* iterable of ints, so
    # `list(blob)` used to decode to a plausible-looking cover — a caller who
    # passed the wrong column got cells instead of an error.  Refused now, and
    # by mortie rather than by CPython's `bytes()`.
    blob = shapely.to_wkb(shapely.from_wkt(_WKB_INPUT_CLASSES["polygon"]))
    with pytest.raises(TypeError, match="WKB input must be"):
        mortie.from_wkb(bad(blob), order=6)


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


def test_emit_dissolve_is_the_default():
    # dissolve=True is the default: to_wkb(cov) with no flag emits the dissolved
    # outline (one ring for a contiguous box), not the per-cell quads.
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    dissolved = shapely.from_wkb(geometry.to_wkb(cov))
    assert dissolved.geom_type == "MultiPolygon"
    assert shapely.get_num_geometries(dissolved) == 1
    # The dissolved outline has far fewer vertices than the per-cell emit.
    per_cell = geometry.to_geometry(cov, dissolve=False)
    assert shapely.get_num_coordinates(dissolved) < \
        shapely.get_num_coordinates(per_cell)


def test_backend_gate_message(monkeypatch):
    # With no backend importable, a clear ImportError naming shapely/spherely.
    import mortie.codec as gm

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


# ── dissolved-outline emit (phase 4) ───────────────────────────────────────


def _union_oracle(cov):
    """shapely.unary_union of the per-cell quads — the independent dissolve
    reference, valid away from the antimeridian / poles (planar union)."""
    polys = [
        shapely.Polygon([(lon, lat) for lat, lon in mortie.mort2polygon(int(w))])
        for w in cov
    ]
    return shapely.unary_union(polys)


def _ring_spherical_area(coords):
    """Signed steradian area of a closed lon/lat ring (backend-independent — the
    one oracle that still works across the antimeridian / poles)."""
    a = np.asarray(coords[:-1], dtype=np.float64)
    rlat = np.radians(a[:, 1])
    rlon = np.radians(a[:, 0])
    v = np.column_stack(
        [np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)]
    )
    p0, b, c = v[0], v[1:-1], v[2:]
    num = np.einsum("j,ij->i", p0, np.cross(b, c))
    den = 1.0 + b @ p0 + np.einsum("ij,ij->i", b, c) + c @ p0
    return float(np.sum(2.0 * np.arctan2(num, den)))


def _cover_area(cov):
    _, depths = mortie.orders._rust_mort2nested(
        np.ascontiguousarray(np.asarray(cov, dtype=np.uint64))
    )
    return float(np.sum(4.0 * np.pi / (12.0 * (4.0 ** depths))))


def test_dissolve_matches_union_oracle():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    mp = geometry.to_geometry(cov)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 1
    # Native edge-cancellation dissolve == shapely's planar union, to precision.
    assert mp.symmetric_difference(_union_oracle(cov)).area < 1e-9


def test_dissolve_polygon_with_hole():
    big = mortie.morton_coverage(
        [30.0, 30.0, 60.0, 60.0], [-130.0, -100.0, -100.0, -130.0], order=5
    )
    inner = mortie.morton_coverage(
        [42.0, 42.0, 48.0, 48.0], [-120.0, -110.0, -110.0, -120.0], order=5
    )
    cov = np.array(sorted(set(big.tolist()) - set(inner.tolist())), dtype=np.uint64)
    mp = geometry.to_geometry(cov)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 1
    # The carved-out interior is emitted as exactly one hole.
    assert shapely.get_num_interior_rings(shapely.get_geometry(mp, 0)) == 1
    assert mp.symmetric_difference(_union_oracle(cov)).area < 1e-9


def test_dissolve_disjoint_components():
    a = mortie.morton_coverage(
        [40.0, 40.0, 45.0, 45.0], [-120.0, -115.0, -115.0, -120.0], order=6
    )
    b = mortie.morton_coverage(
        [40.0, 40.0, 45.0, 45.0], [-100.0, -95.0, -95.0, -100.0], order=6
    )
    cov = np.unique(np.concatenate([a, b]))
    mp = geometry.to_geometry(cov)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 2
    assert mp.symmetric_difference(_union_oracle(cov)).area < 1e-9


def test_dissolve_step_densifies_and_matches():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    g1 = geometry.to_geometry(cov, step=1)
    g4 = geometry.to_geometry(cov, step=4)
    # step traces the curved HEALPix edge: more boundary vertices, still valid.
    # (The curved outline differs from the straight-chord union by the chord
    # error, so area conservation — not planar symdiff — is the right oracle.)
    assert g4.is_valid
    assert shapely.get_num_coordinates(g4) > shapely.get_num_coordinates(g1)
    ring = list(shapely.get_coordinates(
        shapely.get_exterior_ring(shapely.get_geometry(g4, 0))))
    assert abs(_ring_spherical_area(ring) - _cover_area(cov)) < 1e-3


def test_dissolve_mixed_order_moc():
    moc = mortie.morton_coverage_moc(_LATS, _LONS, order=8)
    mp = geometry.to_geometry(moc)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 1
    # The MOC is densified to its finest order, so the dissolved outline encloses
    # the same area as the MOC's cells (independent of the planar union, which a
    # coarse-vs-fine chord mismatch would perturb).
    assert abs(_ring_spherical_area(
        list(shapely.get_coordinates(shapely.get_exterior_ring(
            shapely.get_geometry(mp, 0))))
    ) - _cover_area(mortie.moc.moc_to_order(moc, 8))) < 1e-3


def test_dissolve_antimeridian_split():
    # A box straddling +/-180: the outline crosses the antimeridian (an even
    # number of times, no pole) and must split into valid pieces.
    cov = mortie.morton_coverage(
        [10.0, 10.0, 20.0, 20.0], [170.0, -170.0, -170.0, 170.0], order=5
    )
    mp = geometry.to_geometry(cov)
    assert mp.is_valid and shapely.get_num_geometries(mp) >= 2
    # No emitted piece spans more than a hemisphere of longitude (the hallmark of
    # a correctly split antimeridian polygon).
    out_area = 0.0
    for i in range(shapely.get_num_geometries(mp)):
        ring = list(shapely.get_coordinates(
            shapely.get_exterior_ring(shapely.get_geometry(mp, i))))
        lons = np.asarray(ring)[:, 0]
        assert lons.max() - lons.min() <= 180.0 + 1e-9
        out_area += _ring_spherical_area(ring)
    # The split conserves the covered area exactly (no sliver lost at the seam).
    assert abs(out_area - _cover_area(cov)) < 1e-3


def _polar_cap(latlo, lathi, order=4):
    """A ring of cells encircling a pole (the project's polar-data case)."""
    lats, lons = [], []
    for lo in range(-180, 180, 20):
        lats += [latlo, latlo, lathi, lathi]
        lons += [lo, lo + 20, lo + 20, lo]
    return np.unique(mortie.morton_coverage(lats, lons, order=order))


@pytest.mark.parametrize("latlo,lathi,pole", [(82.0, 89.9, 90.0),
                                              (-89.9, -82.0, -90.0)])
def test_dissolve_polar_cap(latlo, lathi, pole):
    # A pole-enclosing cap dissolves to the GeoJSON convention: a single polygon
    # with explicit +/-90 pole vertices stitched down the antimeridian.
    cap = _polar_cap(latlo, lathi)
    mp = geometry.to_geometry(cap)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 1
    coords = np.asarray(shapely.get_coordinates(
        shapely.get_exterior_ring(shapely.get_geometry(mp, 0))))
    # The frozen representation carries an explicit pole vertex at +/-90 and the
    # seam runs down +/-180.
    assert np.any(np.isclose(coords[:, 1], pole))
    assert np.any(np.isclose(np.abs(coords[:, 0]), 180.0))
    # Spherical-area conservation (the planar union oracle breaks at the pole).
    assert abs(_ring_spherical_area(list(map(tuple, coords))) -
               _cover_area(cap)) < 1e-2


def test_dissolve_polar_cap_matches_spherely():
    # Independent sphere-truth cross-check: spherely (s2geometry) is geodesic and
    # has no pole singularity, so the union of the per-cell quads gives the exact
    # spherical area of the dissolved cap.  (Test-only oracle; runtime is
    # numpy-only and never imports spherely.)
    spherely = pytest.importorskip("spherely")
    cap = _polar_cap(-89.9, -82.0)
    quads = [
        spherely.create_polygon(
            shell=[(lon, lat) for lat, lon in mortie.mort2polygon(int(w))])
        for w in cap
    ]
    acc = quads[0]
    for q in quads[1:]:
        acc = spherely.union(acc, q)
    oracle = spherely.area(acc) / (spherely.EARTH_RADIUS_METERS ** 2)
    mp = geometry.to_geometry(cap)
    ring = list(map(tuple, shapely.get_coordinates(
        shapely.get_exterior_ring(shapely.get_geometry(mp, 0)))))
    assert mp.is_valid
    assert abs(abs(_ring_spherical_area(ring)) - oracle) < 1e-9


def test_dissolve_polar_annulus():
    # A band around the south pole: a pole-enclosing exterior AND a pole-enclosing
    # hole.  Their net longitude windings cancel, so the FILLED region does NOT
    # enclose the pole — the splitter must reconnect ext-to-hole along the seam
    # (no pole vertex) into one valid polygon, not route each through the pole.
    big = _polar_cap(-89.9, -75.0, order=4)
    inner = _polar_cap(-89.9, -85.0, order=4)
    ann = np.array(sorted(set(big.tolist()) - set(inner.tolist())), dtype=np.uint64)
    mp = geometry.to_geometry(ann)
    assert mp.is_valid and shapely.get_num_geometries(mp) >= 1
    # The pole seam opens the band's inner hole into a concavity, so the piece is
    # a hole-free C-shape whose signed exterior area already nets out the cavity.
    out = sum(
        _ring_spherical_area(list(map(tuple, shapely.get_coordinates(
            shapely.get_exterior_ring(shapely.get_geometry(mp, i))))))
        for i in range(shapely.get_num_geometries(mp)))
    assert abs(abs(out) - _cover_area(ann)) < 1e-2


def test_dissolve_antimeridian_multi_crossing():
    # Two disjoint antimeridian-straddling boxes: the exterior set crosses +/-180
    # four times.  Each closes on its own side into a valid hemisphere piece.
    a = mortie.morton_coverage(
        [10.0, 10.0, 20.0, 20.0], [170.0, -170.0, -170.0, 170.0], order=5)
    b = mortie.morton_coverage(
        [40.0, 40.0, 50.0, 50.0], [170.0, -170.0, -170.0, 170.0], order=5)
    cov = np.unique(np.concatenate([a, b]))
    mp = geometry.to_geometry(cov)
    assert mp.is_valid
    out = 0.0
    for i in range(shapely.get_num_geometries(mp)):
        ring = list(map(tuple, shapely.get_coordinates(
            shapely.get_exterior_ring(shapely.get_geometry(mp, i)))))
        assert np.ptp(np.asarray(ring)[:, 0]) <= 180.0 + 1e-9
        out += _ring_spherical_area(ring)
    assert abs(out - _cover_area(cov)) < 1e-2


def test_dissolve_antimeridian_crossing_hole():
    # An antimeridian-straddling box with an inner antimeridian-straddling box
    # removed: the hole itself crosses +/-180 (no pole).  Splitting an annulus at
    # the antimeridian opens the hole into a seam-side concavity, so each half is
    # a valid C-shaped piece (the GeoJSON convention) — area must still conserve.
    big = mortie.morton_coverage(
        [10.0, 10.0, 40.0, 40.0], [160.0, -160.0, -160.0, 160.0], order=4)
    inner = mortie.morton_coverage(
        [20.0, 20.0, 30.0, 30.0], [170.0, -170.0, -170.0, 170.0], order=4)
    cov = np.array(sorted(set(big.tolist()) - set(inner.tolist())), dtype=np.uint64)
    mp = geometry.to_geometry(cov)
    assert mp.is_valid
    out = 0.0
    for i in range(shapely.get_num_geometries(mp)):
        g = shapely.get_geometry(mp, i)
        ring = np.asarray(shapely.get_coordinates(shapely.get_exterior_ring(g)))
        assert np.ptp(ring[:, 0]) <= 180.0 + 1e-9  # no piece spans a hemisphere+
        # The seam opens the hole into a concavity, so each piece is hole-free
        # and its signed exterior area already accounts for the carved interior.
        out += _ring_spherical_area(list(map(tuple, ring)))
    assert abs(out - _cover_area(cov)) < 1e-2


def test_dissolve_wkb_wkt_srid_roundtrip():
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    assert shapely.from_wkb(geometry.to_wkb(cov)).geom_type == "MultiPolygon"
    assert shapely.from_wkt(geometry.to_wkt(cov)).geom_type == "MultiPolygon"
    assert int(shapely.get_srid(
        shapely.from_wkb(geometry.to_wkb(cov, srid=4326)))) == 4326
    assert geometry.to_wkt(cov, srid=4326).startswith("SRID=4326;")


def test_dissolve_empty_cover():
    mp = geometry.to_geometry(np.array([], dtype=np.uint64))
    assert mp.geom_type == "MultiPolygon" and mp.is_empty


def test_dissolve_hemisphere_cover_dissolves():
    # Issue #147: the winding-free classifier dissolves hemisphere+ covers
    # instead of raising (the PR #111 guards are retired).  24 order-1 cells
    # (base cells 0-5) tile exactly half the sphere; the polar cap reaching
    # past the equator is a polar-scale cover beyond it.  Both engines agree
    # and the emitted outline conserves the covered area.
    # (Emitted-ring fan areas are no oracle out here — the anchor fan wraps
    # on seam-closed hemisphere+ rings — so validate by interior/exterior
    # probes; the exhaustive centre-sampled validation lives in
    # test_dissolve_hemisphere.py and the Rust corpus tests.)
    hemi = mortie.norm2mort(np.tile(np.arange(4), 6), np.repeat(np.arange(6), 4), 1)
    over = _polar_cap(-2.0, 89.9, order=3)
    for cov, inside, outside in (
        (hemi, (0.0, 85.0), (0.0, -85.0)),
        (over, (120.0, 45.0), (0.0, -45.0)),
    ):
        mp = geometry.to_geometry(cov)
        assert mp.is_valid
        assert mp.covers(shapely.Point(*inside))
        assert not mp.covers(shapely.Point(*outside))
        ext_py, holes_py = dissolve._dissolved_rings_py(cov, 1)
        mp_py = shapely.MultiPolygon(
            dissolve._nest_and_build(shapely, ext_py, holes_py))
        assert mp.symmetric_difference(mp_py).area < 1e-9


def test_dissolve_hemisphere_enclosing_ring_dissolves():
    # A thin equatorial band (~1.3 sr): both boundary rings enclose more than
    # a hemisphere, which wrapped the old mod-4π fan sum (rejected by PR #111,
    # then a stitcher panic before that).  The planar classifier stitches the
    # two circles into one seam-closed shell (issue #147); both engines agree.
    lats, lons = np.meshgrid(np.arange(-3.5, 3.6, 0.5),
                             np.arange(-180.0, 180.0, 1.0))
    band = np.unique(mortie.geo2mort(lats.ravel(), lons.ravel(), order=4))
    mp = geometry.to_geometry(band)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 1
    assert mp.covers(shapely.Point(0.0, 0.0))
    assert mp.covers(shapely.Point(179.9, 0.0))
    assert not mp.covers(shapely.Point(0.0, 30.0))
    ext_py, holes_py = dissolve._dissolved_rings_py(band, 1)
    mp_py = shapely.MultiPolygon(
        dissolve._nest_and_build(shapely, ext_py, holes_py))
    assert mp.symmetric_difference(mp_py).area < 1e-9


def _interleave(x, y, order):
    h = 0
    for i in range(order):
        h |= ((x >> i) & 1) << (2 * i)
        h |= ((y >> i) & 1) << (2 * i + 1)
    return h


def test_dissolve_corner_touch_yields_simple_rings():
    # Two cells in one base face that touch ONLY at a corner (non-manifold
    # boundary vertex).  Angular ring-chaining must keep them as two separate
    # valid polygons rather than crossing into a self-touching bowtie.
    from mortie import _rustie

    order, face, side = 4, 4, 2 ** 4
    nested = np.array(
        [face * side * side + _interleave(5, 5, order),
         face * side * side + _interleave(6, 6, order)],
        dtype=np.uint64,
    )
    cov = np.unique(np.asarray(
        _rustie.rust_mi_from_nested(nested, order), dtype=np.uint64))
    mp = geometry.to_geometry(cov)
    assert mp.is_valid and shapely.get_num_geometries(mp) == 2
    for i in range(2):
        assert shapely.get_geometry(mp, i).is_valid


def test_dissolve_step_cancels_seams_no_spurious_holes():
    # With step>1 the shared sub-edge points between neighbours must still cancel,
    # or failed cancellation would leave sliver interior rings.  A contiguous box
    # must dissolve to a single hole-free outline at every step.
    cov = mortie.morton_coverage(_LATS, _LONS, order=6)
    for step in (2, 4, 8):
        mp = geometry.to_geometry(cov, step=step)
        assert mp.is_valid and shapely.get_num_geometries(mp) == 1
        assert shapely.get_num_interior_rings(shapely.get_geometry(mp, 0)) == 0


# ── Phase 6: Rust dissolve fast path == the Python reference engine ─────────


def _structure(mp):
    """(#polygons, sorted #interior-rings, total #coords) of a MultiPolygon — a
    rotation-invariant structural fingerprint (the ring vertex order/start may
    differ between two correct engines, but the topology must match)."""
    n = shapely.get_num_geometries(mp)
    holes = sorted(
        shapely.get_num_interior_rings(shapely.get_geometry(mp, i)) for i in range(n))
    return n, holes, int(shapely.get_num_coordinates(mp))


@pytest.mark.parametrize("step", [1, 4])
def test_dissolve_rust_matches_python_reference(step):
    # The runtime dissolve is Rust (dissolve._dissolved_polygons -> rust_dissolve);
    # _dissolved_rings_py is the exact-verified Python reference oracle.  They must
    # agree to machine precision across contiguous, holed, antimeridian, and
    # polar-cap covers.
    box = mortie.morton_coverage(_LATS, _LONS, order=6)
    big = mortie.morton_coverage(
        [30.0, 30.0, 60.0, 60.0], [-130.0, -100.0, -100.0, -130.0], order=5)
    inner = mortie.morton_coverage(
        [42.0, 42.0, 48.0, 48.0], [-120.0, -110.0, -110.0, -120.0], order=5)
    holed = np.array(sorted(set(big.tolist()) - set(inner.tolist())), dtype=np.uint64)
    am = mortie.morton_coverage(
        [10.0, 10.0, 20.0, 20.0], [170.0, -170.0, -170.0, 170.0], order=5)
    cap = _polar_cap(-89.9, -82.0)
    for cov in (box, holed, am, cap):
        ext_py, holes_py = dissolve._dissolved_rings_py(cov, step)
        mp_py = shapely.MultiPolygon(
            dissolve._nest_and_build(shapely, ext_py, holes_py))
        mp_rust = geometry.to_geometry(cov, step=step)
        assert mp_rust.is_valid
        # Identical geometry to a machine-precision symmetric difference, and the
        # same topology (polygon / hole / vertex counts) — not just equal area.
        assert mp_rust.symmetric_difference(mp_py).area < 1e-9
        assert _structure(mp_rust) == _structure(mp_py)
