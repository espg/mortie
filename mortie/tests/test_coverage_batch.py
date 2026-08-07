"""Tests for batch polygon coverage (polygons_to_morton_mocs, issue #153).

The batch contract is parity: for every polygon ``i`` in the ragged batch, the
slice ``values[out_offsets[i]:out_offsets[i+1]]`` is byte-identical to the
scalar :func:`mortie.morton_coverage_moc` on that ring alone — for the exact
MOC and for the shared ``tolerance`` / ``max_cells`` variants.  Plus the
layout/error surface (offsets edge cases, lowest-index error naming) and the
GIL-release guarantee.
"""

import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

import mortie

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_basin(basin_id):
    """Load Antarctic basin vertices.  Returns (lats, lons) or skips test."""
    coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
    if not coords_file.exists():
        pytest.skip("Antarctic polygon data not found")
    data = np.loadtxt(coords_file)
    mask = data[:, 2] == basin_id
    return data[mask, 0], data[mask, 1]


def _simplify_vertices(lats, lons, target):
    """Uniformly subsample vertices to approximately *target* count."""
    n = len(lats)
    if n <= target:
        return lats, lons
    step = max(1, n // target)
    idx = np.arange(0, n, step)
    return lats[idx], lons[idx]


def _ragged(rings):
    """Concatenate rings into (lats, lons, offsets) arrow list layout."""
    lats = np.concatenate([np.asarray(la, dtype=np.float64) for la, _ in rings])
    lons = np.concatenate([np.asarray(lo, dtype=np.float64) for _, lo in rings])
    offsets = np.zeros(len(rings) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(la) for la, _ in rings])
    return lats, lons, offsets


def _random_rings(rng, n):
    """*n* random small convex-ish rings scattered over the sphere."""
    rings = []
    for _ in range(n):
        n_verts = rng.integers(3, 9)
        clat = rng.uniform(-85.0, 85.0)
        clon = rng.uniform(-180.0, 180.0)
        radius = rng.uniform(0.5, 6.0)
        angles = np.sort(rng.uniform(0.0, 2 * np.pi, n_verts))
        lats = np.clip(clat + radius * np.cos(angles), -89.9, 89.9)
        lons = (clon + radius * np.sin(angles) + 180.0) % 360.0 - 180.0
        rings.append((lats, lons))
    return rings


def _assert_batch_parity(rings, order, **kwargs):
    """Assert batch output == the scalar MOC per polygon, slice by slice."""
    lats, lons, offsets = _ragged(rings)
    values, out = mortie.polygons_to_morton_mocs(
        lats, lons, offsets, order=order, **kwargs
    )
    assert out.dtype == np.int64 and values.dtype == np.uint64
    assert len(out) == len(rings) + 1
    assert out[0] == 0 and out[-1] == len(values)
    for i, (la, lo) in enumerate(rings):
        expected = mortie.morton_coverage_moc(la, lo, order=order, **kwargs)
        np.testing.assert_array_equal(values[out[i]:out[i + 1]], expected)


# ---------------------------------------------------------------------------
# Parity: batch == scalar per polygon
# ---------------------------------------------------------------------------


def test_randomized_rings_parity():
    """40 random rings across the sphere match the scalar MOC per polygon."""
    rng = np.random.default_rng(42)
    _assert_batch_parity(_random_rings(rng, 40), order=7)


def test_antimeridian_pole_and_basecell_edge_parity():
    """The classic hard rings: antimeridian, pole-adjacent, basecell edges."""
    rings = [
        # Antimeridian-crossing quad.
        (np.array([-10.0, -10.0, 10.0, 10.0]),
         np.array([170.0, -170.0, -170.0, 170.0])),
        # Pole-adjacent (north) triangle.
        (np.array([85.0, 86.0, 87.0]), np.array([-30.0, 60.0, -150.0])),
        # Pole-adjacent (south) quad.
        (np.array([-84.0, -84.0, -88.0, -88.0]),
         np.array([-130.0, -100.0, -100.0, -130.0])),
        # Square aligned on basecell edges (lon multiple of 45, equator).
        (np.array([-5.0, -5.0, 5.0, 5.0]),
         np.array([40.0, 50.0, 50.0, 40.0])),
        # Ring with a vertex exactly on a basecell corner (lat 0, lon 45).
        (np.array([0.0, 10.0, 10.0]), np.array([45.0, 45.0, 55.0])),
    ]
    _assert_batch_parity(rings, order=7)


def test_antarctic_basin_parity():
    """Real pole+antimeridian data: simplified basins batch == scalar."""
    rings = []
    for basin_id in (24, 2):
        la, lo = _load_basin(basin_id)
        rings.append(_simplify_vertices(la, lo, 300))
    _assert_batch_parity(rings, order=6)


def test_closed_ring_and_normalize_false_parity():
    """Duplicate closing vertex and normalize=False funnel like the scalar."""
    open_ring = (np.array([40.0, 50.0, 45.0]), np.array([-120.0, -120.0, -110.0]))
    closed_ring = (
        np.array([40.0, 50.0, 45.0, 40.0]),
        np.array([-120.0, -120.0, -110.0, -120.0]),
    )
    _assert_batch_parity([open_ring, closed_ring], order=7)
    # A CW-wound ring under normalize=False selects the complement; the batch
    # must reproduce the scalar's trust-the-winding behavior, not "fix" it.
    cw_ring = (open_ring[0][::-1].copy(), open_ring[1][::-1].copy())
    _assert_batch_parity([open_ring, cw_ring], order=5, normalize=False)


def test_tolerance_parity():
    rng = np.random.default_rng(7)
    _assert_batch_parity(_random_rings(rng, 10), order=9, tolerance=0.25)


def test_max_cells_parity():
    rng = np.random.default_rng(11)
    _assert_batch_parity(_random_rings(rng, 10), order=9, max_cells=64)


def test_deterministic_across_runs():
    """Two identical calls give identical ragged output (rayon-order-free)."""
    rng = np.random.default_rng(3)
    lats, lons, offsets = _ragged(_random_rings(rng, 25))
    v1, o1 = mortie.polygons_to_morton_mocs(lats, lons, offsets, order=7)
    v2, o2 = mortie.polygons_to_morton_mocs(lats, lons, offsets, order=7)
    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(o1, o2)


# ---------------------------------------------------------------------------
# Layout edge cases
# ---------------------------------------------------------------------------


def test_empty_batch():
    values, out = mortie.polygons_to_morton_mocs([], [], [0], order=7)
    assert values.size == 0 and values.dtype == np.uint64
    np.testing.assert_array_equal(out, [0])


def test_single_polygon():
    la = np.array([40.0, 50.0, 45.0])
    lo = np.array([-120.0, -120.0, -110.0])
    values, out = mortie.polygons_to_morton_mocs(la, lo, [0, 3], order=7)
    expected = mortie.morton_coverage_moc(la, lo, order=7)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(out, [0, len(expected)])


def test_offsets_must_exactly_cover_the_vertices():
    """Strict contract: offsets[0] == 0 and offsets[-1] == len(lats).

    Slice mechanics live in the Arrow skin (which re-bases offsets and slices
    the values), not in the core — see ``test_coverage_batch_arrow.py``.
    """
    la = np.array([0.0, 0.0, 0.0, 40.0, 50.0, 45.0, 1.0, 2.0])
    lo = np.array([0.0, 0.0, 0.0, -120.0, -120.0, -110.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="must start at 0"):
        mortie.polygons_to_morton_mocs(la, lo, [3, 6], order=7)
    with pytest.raises(ValueError, match="must end at the vertex count"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3], order=7)
    # The re-based spelling of that same ring is what the core accepts.
    values, out = mortie.polygons_to_morton_mocs(la[3:6], lo[3:6], [0, 3], order=7)
    expected = mortie.morton_coverage_moc(la[3:6], lo[3:6], order=7)
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(out, [0, len(expected)])


# ---------------------------------------------------------------------------
# Fail-fast errors naming the lowest-index polygon
# ---------------------------------------------------------------------------


def test_short_ring_names_polygon_index():
    la = np.array([40.0, 50.0, 45.0, 10.0, 20.0])
    lo = np.array([-120.0, -120.0, -110.0, -80.0, -80.0])
    with pytest.raises(ValueError, match=r"polygon 1: needs at least 3"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3, 5], order=7)


def test_nan_names_polygon_index():
    la = np.array([40.0, 50.0, 45.0, 10.0, np.nan, 15.0])
    lo = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    with pytest.raises(ValueError, match=r"polygon 1: .*NaN"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3, 6], order=7)


def test_offsets_errors_name_polygon_index():
    la = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    lo = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    with pytest.raises(ValueError, match=r"polygon 1: .*monotonically"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 6, 3], order=7)
    with pytest.raises(ValueError, match=r"polygon 1: .*exceeds"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3, 99], order=7)
    with pytest.raises(ValueError, match="must start at 0"):
        mortie.polygons_to_morton_mocs(la, lo, [-1, 3], order=7)
    with pytest.raises(ValueError, match="at least one element"):
        mortie.polygons_to_morton_mocs(la, lo, [], order=7)


def test_lowest_index_reported_among_multiple_offenders():
    """Two bad polygons (1 and 3): the lowest index is the one named."""
    la = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0, 1.0])
    lo = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0, 1.0])
    with pytest.raises(ValueError, match=r"polygon 1:"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3, 5, 6, 7], order=7)


def test_order_and_exclusivity_validation():
    la = np.array([40.0, 50.0, 45.0])
    lo = np.array([-120.0, -120.0, -110.0])
    with pytest.raises(ValueError, match="Order must be"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3], order=0)
    with pytest.raises(ValueError, match="at most one of"):
        mortie.polygons_to_morton_mocs(la, lo, [0, 3], tolerance=0.1, max_cells=8)
    with pytest.raises(ValueError, match="same length"):
        mortie.polygons_to_morton_mocs(la, lo[:2], [0, 3])


def test_max_cells_floor_warns_once_naming_lowest_index():
    la = np.array([40.0, 50.0, 45.0, 10.0, 20.0, 15.0])
    lo = np.array([-120.0, -120.0, -110.0, -80.0, -80.0, -70.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mortie.polygons_to_morton_mocs(la, lo, [0, 3, 6], order=9, max_cells=1)
    msgs = [str(w.message) for w in caught if "max_cells=1" in str(w.message)]
    assert len(msgs) == 1
    assert "polygon 0" in msgs[0]


# ---------------------------------------------------------------------------
# GIL release
# ---------------------------------------------------------------------------


def test_gil_released_during_batch():
    """A pure-Python counter thread keeps running during a large batch call.

    Same instrument as ``test_threading.test_gil_released_during_rust_compute``:
    with the GIL held for the whole call the counter cannot advance; released,
    it climbs by many thousands.
    """
    rng = np.random.default_rng(5)
    lats, lons, offsets = _ragged(_random_rings(rng, 3000))

    counter = [0]
    stop = threading.Event()
    started = threading.Event()

    def busy():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    b = threading.Thread(target=busy)
    b.start()
    started.wait()
    try:
        pre = counter[0]
        mortie.polygons_to_morton_mocs(lats, lons, offsets, order=8)
        progressed = counter[0] - pre
    finally:
        stop.set()
        b.join()

    assert progressed > 1000, (
        f"Python thread made ~no progress during the batch call ({progressed}); "
        "the GIL was likely held (allow_threads not in effect)"
    )
