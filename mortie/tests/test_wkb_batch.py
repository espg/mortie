"""Tests for the plural WKB batch ``from_wkbs`` (issue #157, phase 3).

Many blobs in, one ragged MOC array out.  These pin the three things that
make the batch usable in place of a per-blob loop: per-blob byte parity with
the scalar :func:`mortie.from_wkb`, the strict ragged output contract, and
fail-fast naming the **lowest-index** offending blob deterministically —
including offenders straddling the 2048-blob chunk boundary, which is where a
naive implementation stops being deterministic.

Blobs are packed by hand, so the module needs no geometry backend; shapely is
used only where a dialect fixture is easier to author with it.
"""

import struct
import warnings

import numpy as np
import pytest

import mortie
from mortie.geometry import from_wkbs

# The chunk the Rust side copies and covers at a time; the determinism tests
# place offenders either side of it.
CHUNK = 2048


def polygon_blob(rings, byte_order=1):
    """Pack a Polygon WKB from ``(lon, lat)`` rings.

    Parameters
    ----------
    rings : list of list of tuple
        One list of ``(lon, lat)`` degree pairs per ring, exterior first.
    byte_order : int, optional
        1 for little-endian (default), 0 for big-endian.

    Returns
    -------
    bytes
        The WKB blob.
    """
    e = "<" if byte_order else ">"
    out = struct.pack(f"{e}BII", byte_order, 3, len(rings))
    for ring in rings:
        out += struct.pack(f"{e}I", len(ring))
        out += b"".join(struct.pack(f"{e}dd", x, y) for x, y in ring)
    return out


def quad(lon0, lat0, size=1.0):
    """Build a closed square ring as ``(lon, lat)`` degree pairs.

    Parameters
    ----------
    lon0, lat0 : float
        The ring's south-west corner, in degrees.
    size : float, optional
        Side length in degrees.  Default 1.0.

    Returns
    -------
    list of tuple
        The closed ring (last vertex repeats the first).
    """
    return [
        (lon0, lat0),
        (lon0 + size, lat0),
        (lon0 + size, lat0 + size),
        (lon0, lat0 + size),
        (lon0, lat0),
    ]


def corpus(n, seed=157):
    """Build *n* scattered single-ring footprint blobs.

    Parameters
    ----------
    n : int
        How many blobs to build.
    seed : int, optional
        Seed for the placement RNG, so a failure reproduces.

    Returns
    -------
    list of bytes
        One little-endian Polygon WKB blob per footprint.
    """
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-179.0, 179.0, n)
    lat = rng.uniform(-80.0, 80.0, n)
    return [polygon_blob([quad(float(lo), float(la))]) for lo, la in zip(lon, lat)]


def assert_ragged_contract(values, offsets, n):
    """Assert the strict ragged output contract holds.

    Parameters
    ----------
    values : numpy.ndarray
        The concatenated MOC words.
    offsets : numpy.ndarray
        The arrow list offsets into *values*.
    n : int
        The number of input blobs.
    """
    assert offsets.dtype == np.int64
    assert offsets.size == n + 1
    assert offsets[0] == 0
    assert offsets[-1] == values.size
    assert np.all(np.diff(offsets) >= 0)


def test_per_blob_parity_with_the_scalar():
    blobs = corpus(64)
    values, offsets = from_wkbs(blobs, order=8)
    assert_ragged_contract(values, offsets, len(blobs))
    for i, blob in enumerate(blobs):
        np.testing.assert_array_equal(
            values[offsets[i]:offsets[i + 1]],
            mortie.from_wkb(blob, order=8, moc=True),
        )


@pytest.mark.parametrize(
    "wkt",
    [
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))",
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
        "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74))",
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),((5 5, 6 5, 6 6, 5 6, 5 5)))",
        "POLYGON ((170 -20, -170 -20, -170 -10, 170 -10, 170 -20))",
        "POLYGON ((0 -89.5, 90 -89.5, 180 -89.5, -90 -89.5, 0 -89.5))",
        "POLYGON Z ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))",
    ],
)
@pytest.mark.parametrize("byte_order", [0, 1])
def test_dialect_fixtures_match_the_scalar(wkt, byte_order):
    # The classes the ATL03 corpus does not contain: holes, multipart,
    # antimeridian, pole-adjacent, Z, and both endiannesses.
    shapely = pytest.importorskip("shapely")
    blob = shapely.to_wkb(shapely.from_wkt(wkt), byte_order=byte_order)
    values, offsets = from_wkbs([blob], order=7)
    assert_ragged_contract(values, offsets, 1)
    np.testing.assert_array_equal(values, mortie.from_wkb(blob, order=7, moc=True))


def test_ewkb_and_hex_and_buffer_inputs():
    # The batch narrows nothing: the scalar's whole accept list works per item.
    shapely = pytest.importorskip("shapely")
    geom = shapely.from_wkt("POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))")
    plain = shapely.to_wkb(geom)
    ewkb = shapely.to_wkb(shapely.set_srid(geom, 4326), include_srid=True)
    forms = [plain, ewkb, plain.hex(), bytearray(plain), memoryview(plain),
             np.frombuffer(plain, dtype=np.uint8)]
    values, offsets = from_wkbs(forms, order=7)
    want = mortie.from_wkb(plain, order=7, moc=True)
    for i in range(len(forms)):
        np.testing.assert_array_equal(values[offsets[i]:offsets[i + 1]], want)


def test_empty_batch_keeps_the_contract():
    values, offsets = from_wkbs([], order=6)
    assert_ragged_contract(values, offsets, 0)
    assert values.size == 0


def test_shared_stop_criteria_match_the_scalar():
    blobs = corpus(16)
    for kwargs in ({"tolerance": 0.5}, {"max_cells": 32}):
        values, offsets = from_wkbs(blobs, order=10, **kwargs)
        for i, blob in enumerate(blobs):
            np.testing.assert_array_equal(
                values[offsets[i]:offsets[i + 1]],
                mortie.from_wkb(blob, order=10, moc=True, **kwargs),
            )
    with pytest.raises(ValueError, match="at most one"):
        from_wkbs(blobs, order=8, tolerance=0.5, max_cells=32)


def test_budget_raise_warns_once_and_names_the_lowest_index():
    blobs = corpus(8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from_wkbs(blobs, order=8, max_cells=1)
    raised = [w for w in caught if "max_cells=1" in str(w.message)]
    assert len(raised) == 1
    assert "blob 0" in str(raised[0].message)


def test_normalize_is_threaded_through():
    # A ring wound the other way: normalize=False must reach the kernel, or
    # the two spellings would agree.
    blob = polygon_blob([quad(0.0, 0.0)[::-1]])
    for norm in (True, False):
        values, offsets = from_wkbs([blob], order=6, normalize=norm)
        np.testing.assert_array_equal(
            values, mortie.from_wkb(blob, order=6, moc=True, normalize=norm)
        )


@pytest.mark.parametrize("order", [0, 30])
def test_order_range_is_checked(order):
    with pytest.raises(ValueError, match="Order must be between 1 and 29"):
        from_wkbs(corpus(2), order=order)


# ── fail-fast: the lowest-index offender, deterministically ────────────────


def truncated_blob():
    """Build a blob cut off mid-header.

    Returns
    -------
    bytes
        A Polygon blob truncated inside its ring-count field.
    """
    return polygon_blob([quad(0.0, 0.0)])[:9]


def unclosed_blob():
    """Build a Polygon blob whose ring is not closed.

    Returns
    -------
    bytes
        A Polygon blob with an unclosed exterior ring.
    """
    return polygon_blob([quad(0.0, 0.0)[:-1]])


def short_ring_blob():
    """Build a Polygon blob with a closed 2-vertex ring.

    Returns
    -------
    bytes
        A Polygon blob whose only ring has too few vertices.
    """
    return polygon_blob([[(0.0, 0.0), (0.0, 0.0)]])


def nan_blob():
    """Build a Polygon blob with a NaN mid-ring vertex.

    Returns
    -------
    bytes
        A Polygon blob carrying a non-finite coordinate.
    """
    ring = quad(0.0, 0.0)
    ring[1] = (ring[1][0], float("nan"))
    return polygon_blob([ring])


def linestring_blob():
    """Build a LineString blob -- linear geometry.

    Returns
    -------
    bytes
        A LineString blob, which the batch refuses by index.
    """
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]
    return struct.pack("<BII", 1, 2, len(pts)) + b"".join(
        struct.pack("<dd", x, y) for x, y in pts
    )


BAD = {
    "truncated": (truncated_blob, "truncated WKB"),
    "unclosed": (unclosed_blob, "is not closed"),
    "short_ring": (short_ring_blob, "at least 3 vertices"),
    "nan": (nan_blob, "NaN or infinity"),
    "linear": (linestring_blob, "linear geometry"),
    "empty": (lambda: polygon_blob([]), "empty geometry"),
    "point": (lambda: struct.pack("<BIdd", 1, 1, 0.0, 0.0), "unsupported WKB"),
}


@pytest.mark.parametrize("kind", sorted(BAD))
def test_each_failure_class_names_its_blob(kind):
    make, pattern = BAD[kind]
    blobs = corpus(5)
    blobs[3] = make()
    with pytest.raises(ValueError, match=rf"blob 3: .*{pattern}"):
        from_wkbs(blobs, order=6)


@pytest.mark.parametrize("bad_index", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1,
                                       2 * CHUNK + 7, 3 * CHUNK - 1])
def test_lowest_index_is_named_across_chunk_boundaries(bad_index):
    # The chunk loop must not renumber: an offender in chunk 2 is still
    # reported with its global index.
    n = 3 * CHUNK
    blobs = corpus(n)
    blobs[bad_index] = truncated_blob()
    with pytest.raises(ValueError, match=rf"^blob {bad_index}: "):
        from_wkbs(blobs, order=5)


def test_the_lowest_of_several_offenders_wins_every_run():
    # Three offenders spread across three chunks; the lowest must surface
    # every time, whatever order rayon happens to finish them in.
    n = 3 * CHUNK
    blobs = corpus(n)
    for i in (17, CHUNK + 3, 2 * CHUNK + 900):
        blobs[i] = truncated_blob()
    for _ in range(25):
        with pytest.raises(ValueError, match=r"^blob 17: "):
            from_wkbs(blobs, order=5)
    # And within a single chunk, with the offenders adjacent.
    blobs = corpus(64)
    blobs[9] = unclosed_blob()
    blobs[10] = truncated_blob()
    for _ in range(25):
        with pytest.raises(ValueError, match=r"^blob 9: .*is not closed"):
            from_wkbs(blobs, order=6)


def fat_blob(lon0=0.0, lat0=0.0, nvert=65536, radius=0.01):
    """Build a ~1 MiB blob: one tiny ring densified to *nvert* vertices.

    Big in bytes and cheap to cover, which is what a chunk cut by the byte
    budget rather than the 2048-blob count needs exercising with.

    Parameters
    ----------
    lon0, lat0 : float
        Centre of the ring, in degrees.
    nvert : int
        Vertices in the ring; the blob is ``16 * nvert + 13`` bytes.
    radius : float
        Ring radius in degrees -- small, so the cover is a cell or two.

    Returns
    -------
    bytes
        A Polygon blob of ``16 * nvert`` bytes of coordinates.
    """
    t = np.linspace(0.0, 2.0 * np.pi, nvert)
    lon = lon0 + radius * np.cos(t)
    lat = lat0 + radius * np.sin(t)
    lon[-1], lat[-1] = lon[0], lat[0]
    head = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", nvert)
    return head + np.column_stack([lon, lat]).astype("<f8").tobytes()


def test_fat_blobs_cut_the_chunk_by_bytes_not_only_by_count():
    # 70 x ~1 MiB is one chunk by blob count (70 < 2048) but two by bytes (the
    # budget is 64 MiB), so this is the one path where a chunk is not 2048
    # blobs long -- per-blob parity and global numbering both have to survive
    # a variable chunk length.
    blobs = [fat_blob(lon0=0.05 * i) for i in range(70)]
    values, offsets = from_wkbs(blobs, order=5)
    assert_ragged_contract(values, offsets, len(blobs))
    for i in (0, 63, 64, 69):
        want = mortie.from_wkb(blobs[i], order=5, moc=True)
        np.testing.assert_array_equal(values[offsets[i]:offsets[i + 1]], want)
    # And an offender past the byte cut is still named by its global index,
    # not renumbered within its chunk.
    blobs[66] = truncated_blob()
    with pytest.raises(ValueError, match=r"^blob 66: "):
        from_wkbs(blobs, order=5)


def test_input_contract_violations_name_their_index():
    blobs = corpus(4)
    blobs[2] = 12345
    with pytest.raises(TypeError, match=r"^blob 2: WKB input must be"):
        from_wkbs(blobs, order=6)
    blobs[2] = "not hex"
    with pytest.raises(ValueError, match=r"^blob 2: invalid WKB hex string"):
        from_wkbs(blobs, order=6)


def test_a_good_batch_after_a_failure_still_works():
    # The chunk buffers are reused across chunks and across calls; a failed
    # call must not leave the next one holding stale bytes.
    blobs = corpus(CHUNK + 10)
    good = from_wkbs(blobs, order=5)
    blobs[CHUNK + 5] = truncated_blob()
    with pytest.raises(ValueError):
        from_wkbs(blobs, order=5)
    blobs[CHUNK + 5] = corpus(CHUNK + 10)[CHUNK + 5]
    again = from_wkbs(blobs, order=5)
    np.testing.assert_array_equal(again[0], good[0])
    np.testing.assert_array_equal(again[1], good[1])


def test_ragged_pair_composes_with_mocs_to_orders_shape():
    # The output pair is the arrow list layout the rest of the batch surface
    # consumes: variable-length entries, offsets covering values exactly.
    blobs = corpus(40)
    values, offsets = from_wkbs(blobs, order=7)
    assert_ragged_contract(values, offsets, len(blobs))
    lengths = np.diff(offsets)
    assert lengths.min() >= 1 and lengths.max() > lengths.min()
