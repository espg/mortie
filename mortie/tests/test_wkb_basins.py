"""The in-tree Antarctic basins as WKB, through the batch (issue #157, phase 4).

Issue #157's acceptance names these fixtures alongside the ATL03 corpus, and
they cover what that corpus cannot: real high-latitude rings at **fat** blob
sizes (0.65 MiB median, 1.25 MiB max) rather than the ~0.5 KiB of a granule
footprint.  That size class is exactly what the phase-3 byte-cap fold was
added for — a chunk bounded by blob *count* alone would copy the whole
column — so these are the real-fixture case behind that constant.

What the 27 basins are, measured rather than assumed (and pinned below, so
the description cannot go stale either): every basin is Antarctic, spanning
-88.4 to -63.2 degrees latitude, but only **3 of the 27** reach below -85 and
11 never reach -76, so this is not a fixture of uniformly pole-adjacent
rings.  **One** basin crosses the antimeridian (a >180-degree step between
consecutive vertices; two have vertices on both sides of |lon| > 170).  The
pole and antimeridian *classes* proper are pinned by the synthetic
``pole_adjacent`` / ``antimeridian`` entries in ``test_geometry.py``; what
these blobs add on top of those is size and vertex count.

The basins arrive as a lat/lon/basin-id table, not as WKB, so the blobs are
packed here; two of the 27 rings are left open by the fixture and are closed
when packed, since a WKB polygon ring is closed by definition (and mortie's
reader enforces that, matching GEOS).
"""

import struct
from pathlib import Path

import numpy as np
import pytest

import mortie
from mortie.geometry import from_wkbs

COORDS = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
ORDER = 6


def basin_table():
    """Load the Antarctic drainage-basin vertex table.

    Returns
    -------
    numpy.ndarray
        An ``(N, 3)`` array of ``lat, lon, basin_id``.
    """
    if not COORDS.exists():
        pytest.skip("Antarctic polygon data not found")
    return np.loadtxt(COORDS)


def pack_polygon(lats, lons):
    """Pack one closed ring as a little-endian Polygon WKB blob.

    Parameters
    ----------
    lats, lons : numpy.ndarray
        The ring's vertices in degrees, in fixture order.

    Returns
    -------
    bytes
        The WKB blob, with the ring closed if the fixture left it open.
    """
    if lats[0] != lats[-1] or lons[0] != lons[-1]:
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])
    xy = np.empty(lats.size * 2)
    xy[0::2] = lons  # WKB stores (x, y) = (lon, lat)
    xy[1::2] = lats
    return (
        struct.pack("<BIII", 1, 3, 1, lats.size) + xy.astype("<f8").tobytes()
    )


@pytest.fixture(scope="module")
def basins():
    """Pack every basin in the fixture as a WKB blob.

    Returns
    -------
    tuple
        ``(blobs, ids)`` — one little-endian Polygon blob per basin, and the
        basin ids in the same order.
    """
    table = basin_table()
    ids = np.unique(table[:, 2]).astype(int)
    blobs = [
        pack_polygon(table[table[:, 2] == b, 0], table[table[:, 2] == b, 1])
        for b in ids
    ]
    return blobs, ids


def test_the_fixture_is_the_fat_blob_class_it_is_cited_as(basins):
    # The sizes the byte-cap constant and the PR's memory numbers are quoted
    # against; if the fixture ever changed shape, those numbers would go stale
    # silently.
    blobs, ids = basins
    sizes = np.array([len(b) for b in blobs])
    assert len(ids) == 27
    assert 0.6 < np.median(sizes) / 2**20 < 0.7
    assert 1.2 < sizes.max() / 2**20 < 1.3
    assert 18.0 < sizes.sum() / 2**20 < 20.0


def test_how_polar_and_how_antimeridian_the_fixture_actually_is():
    # The module docstring describes this fixture in exact counts rather than
    # "all pole-adjacent, several antimeridian-crossing", which is what it was
    # first cited as.  Pin the counts so the description stays honest.
    table = basin_table()
    ids = np.unique(table[:, 2]).astype(int)
    min_lat = np.array([table[table[:, 2] == b, 0].min() for b in ids])
    crossings = sum(
        np.abs(np.diff(table[table[:, 2] == b, 1])).max() > 180.0 for b in ids
    )
    assert (min_lat < -85.0).sum() == 3
    assert (min_lat > -76.0).sum() == 11
    assert crossings == 1


def test_every_basin_matches_the_scalar(basins):
    # Per-blob byte parity, the batch's core contract, on real high-latitude
    # geometry at fat blob sizes rather than synthetic quads.
    blobs, ids = basins
    values, offsets = from_wkbs(blobs, order=ORDER)
    assert offsets[0] == 0 and offsets[-1] == values.size
    assert offsets.size == len(blobs) + 1
    for i, blob in enumerate(blobs):
        np.testing.assert_array_equal(
            values[offsets[i]:offsets[i + 1]],
            mortie.from_wkb(blob, order=ORDER, moc=True),
            err_msg=f"basin {ids[i]}",
        )


def test_every_basin_matches_the_shapely_backed_path(basins):
    # Issue #157's parity criterion is against the path the Rust reader
    # replaced, not merely against itself: decode with shapely, decompose with
    # shapely, cover through `from_geometry`.
    shapely = pytest.importorskip("shapely")
    blobs, ids = basins
    values, offsets = from_wkbs(blobs, order=ORDER)
    for i, blob in enumerate(blobs):
        want = mortie.from_geometry(shapely.from_wkb(blob), order=ORDER, moc=True)
        np.testing.assert_array_equal(
            values[offsets[i]:offsets[i + 1]], want, err_msg=f"basin {ids[i]}"
        )


def test_basins_survive_a_chunk_boundary_and_keep_their_index(basins):
    # 108 basins is ~76 MiB, so the 64 MiB byte budget cuts this column in two
    # at roughly blob 91 — a cut the 2048-blob count would never have made.
    # Parity must hold either side of it and an offender past it must still be
    # named by its **global** index.
    blobs, _ = basins
    column = (blobs * 4)[:108]
    assert sum(len(b) for b in column) > 64 * 2**20, "column must span two chunks"
    values, offsets = from_wkbs(column, order=ORDER)
    for i in (0, 91, 107):
        np.testing.assert_array_equal(
            values[offsets[i]:offsets[i + 1]],
            mortie.from_wkb(column[i], order=ORDER, moc=True),
        )
    column[100] = column[100][:2048]  # truncate a fat blob in the second chunk
    with pytest.raises(ValueError, match=r"^blob 100: .*truncated WKB"):
        from_wkbs(column, order=ORDER)
