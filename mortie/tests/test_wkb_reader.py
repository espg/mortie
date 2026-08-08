"""Tests for the backend-free Rust WKB reader (issue #157, phase 1).

These pin ``_rustie.rust_wkb_rings`` — the parser that replaces the geometry
backend on the *ingest* side.  The contract is the one
:func:`mortie.geometry.decompose` documents: exterior **and** interior rings of
**every** part, flattened into arrow list layout, in ``(lat, lon)`` degrees.
shapely is used here purely as an oracle (it produced the reference rings the
old path returned); the reader itself never imports it.
"""

import struct

import numpy as np
import pytest

from mortie import _rustie

shapely = pytest.importorskip("shapely")

# An asymmetric footprint: latitudes in [-75, -71], longitudes in [10, 40].
# The two ranges cannot be confused, so an axis swap anywhere in the reader
# fails loudly instead of plausibly.
ASYM_WKT = (
    "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))"
)


def rings(data):
    """Parse *data*, unpacking the ragged result into per-ring arrays.

    Parameters
    ----------
    data : bytes
        WKB or EWKB bytes.

    Returns
    -------
    tuple
        ``(kind, parts)`` — the same shape
        :func:`mortie.geometry.decompose` returns.
    """
    kind, lats, lons, offsets = _rustie.rust_wkb_rings(data)
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    offsets = np.asarray(offsets)
    assert offsets[0] == 0
    assert offsets[-1] == lats.size == lons.size
    parts = [
        (lats[offsets[i]:offsets[i + 1]], lons[offsets[i]:offsets[i + 1]])
        for i in range(offsets.size - 1)
    ]
    return kind, parts


def assert_matches_shapely(wkt, geom_wkb=None):
    """Assert the reader's rings equal the shapely-backed decomposition.

    Parameters
    ----------
    wkt : str
        The geometry, as WKT, that the oracle decomposes.
    geom_wkb : bytes, optional
        The blob to parse; defaults to shapely's own WKB encoding of *wkt*.
        Pass a dialect variant (big-endian, EWKB, Z/M) to assert it reduces
        to the same 2-D rings.
    """
    from mortie import geometry

    g = shapely.from_wkt(wkt)
    exp_kind, exp = geometry.decompose(g)
    got_kind, got = rings(shapely.to_wkb(g) if geom_wkb is None else geom_wkb)
    assert got_kind == exp_kind
    assert len(got) == len(exp)
    for (glat, glon), (elat, elon) in zip(got, exp):
        np.testing.assert_array_equal(glat, elat)
        np.testing.assert_array_equal(glon, elon)


def test_coordinate_order_is_lon_x_lat_y():
    # The swap, pinned in both directions: every latitude lands in the
    # latitude range and every longitude in the longitude range.
    _, parts = rings(shapely.to_wkb(shapely.from_wkt(ASYM_WKT)))
    (lat, lon), = parts
    assert lat.min() == -75.0 and lat.max() == -71.0
    assert lon.min() == 10.0 and lon.max() == 40.0
    # And identically to what the shapely-backed path produced.
    assert_matches_shapely(ASYM_WKT)


def test_both_byte_orders_agree():
    g = shapely.from_wkt(ASYM_WKT)
    little = rings(shapely.to_wkb(g, byte_order=1))
    big = rings(shapely.to_wkb(g, byte_order=0))
    assert little[0] == big[0]
    for (llat, llon), (blat, blon) in zip(little[1], big[1]):
        np.testing.assert_array_equal(llat, blat)
        np.testing.assert_array_equal(llon, blon)
    assert_matches_shapely(ASYM_WKT, shapely.to_wkb(g, byte_order=0))


def test_ewkb_srid_prefix_is_stripped():
    g = shapely.set_srid(shapely.from_wkt(ASYM_WKT), 4326)
    assert_matches_shapely(ASYM_WKT, shapely.to_wkb(g, include_srid=True))


@pytest.mark.parametrize("flavor", ["iso", "extended"])
@pytest.mark.parametrize(
    "wkt",
    [
        "POLYGON Z ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))",
        "POLYGON ZM ((10 -75 7 1, 40 -75 7 1, 40 -71 7 1,"
        " 10 -71 7 1, 10 -75 7 1))",
    ],
)
def test_z_and_m_are_dropped(wkt, flavor):
    # Both dimension spellings (ISO's +1000/+3000 type offsets and EWKB's flag
    # bits) reduce to the same 2-D ring.
    blob = shapely.to_wkb(shapely.from_wkt(wkt), flavor=flavor)
    assert_matches_shapely(ASYM_WKT, blob)


def test_polygon_with_hole_and_multipolygon_flatten_every_ring():
    assert_matches_shapely(
        "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0),"
        "(0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))"
    )
    assert_matches_shapely(
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),"
        "((5 5, 6 5, 6 6, 5 6, 5 5),"
        "(5.2 5.2, 5.8 5.2, 5.8 5.8, 5.2 5.8, 5.2 5.2)))"
    )


def test_linear_geometries():
    assert_matches_shapely("LINESTRING (10 -75, 40 -75, 40 -71)")
    assert_matches_shapely(
        "MULTILINESTRING ((10 -75, 40 -75), (40 -71, 10 -71))"
    )
    # A LinearRing has no WKB type of its own — GEOS writes it as a LineString,
    # which is what the shapely-backed path saw as well.
    ring = shapely.LinearRing([(10, -75), (40, -75), (40, -71), (10, -75)])
    kind, parts = rings(shapely.to_wkb(ring))
    assert kind == "linear"
    assert len(parts) == 1


@pytest.mark.parametrize(
    "wkt", ["POINT (10 -75)", "MULTIPOINT ((10 -75))",
            "GEOMETRYCOLLECTION (POINT (10 -75))"]
)
def test_unsupported_types_raise_value_error(wkt):
    blob = shapely.to_wkb(shapely.from_wkt(wkt))
    with pytest.raises(ValueError, match="unsupported WKB geometry type"):
        rings(blob)


@pytest.mark.parametrize(
    "wkt", ["POLYGON EMPTY", "MULTIPOLYGON EMPTY", "LINESTRING EMPTY"]
)
def test_empty_geometry_raises_value_error(wkt):
    blob = shapely.to_wkb(shapely.from_wkt(wkt))
    with pytest.raises(ValueError, match="empty geometry has no coverage"):
        rings(blob)


def unclosed_polygon_blob():
    """Build a Polygon WKB whose exterior ring is left unclosed.

    Returns
    -------
    bytes
        A little-endian Polygon blob with a 4-vertex, unclosed ring.  shapely
        cannot author this — GEOS refuses to construct it — so it is packed by
        hand.
    """
    pts = [(10, -75), (40, -75), (40, -71), (10, -71)]
    return (
        struct.pack("<BII", 1, 3, 1)
        + struct.pack("<I", len(pts))
        + b"".join(struct.pack("<dd", x, y) for x, y in pts)
    )


def test_unclosed_ring_is_rejected_as_it_is_today():
    # GEOS enforces ring closure at parse time, so this blob raises on the
    # backend-decoded path; the reader must not quietly accept it.
    blob = unclosed_polygon_blob()
    with pytest.raises(Exception, match="closed"):
        shapely.from_wkb(blob)
    with pytest.raises(ValueError, match="polygon ring 0 is not closed"):
        rings(blob)


def test_truncated_and_malformed_blobs_raise_value_error():
    blob = shapely.to_wkb(shapely.from_wkt(ASYM_WKT))
    for cut in range(1, len(blob)):
        with pytest.raises(ValueError, match="truncated WKB"):
            _rustie.rust_wkb_rings(blob[:cut])
    with pytest.raises(ValueError, match="truncated WKB"):
        _rustie.rust_wkb_rings(b"")
    bad = bytearray(blob)
    bad[0] = 7
    with pytest.raises(ValueError, match="invalid WKB byte-order flag"):
        _rustie.rust_wkb_rings(bytes(bad))
