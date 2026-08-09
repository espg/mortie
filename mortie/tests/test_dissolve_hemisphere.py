"""Hemisphere+ dissolve: the winding-free classifier (issue #147).

Phase 1 pins the orientation contract the classifier rests on, for the Python
reference oracle (:func:`mortie.dissolve._dissolved_rings_py`) side by side
with the Rust tests in ``src_rust/src/dissolve.rs``:

* every HEALPix cell boundary loop is emitted with a fixed handedness —
  interior on the LEFT at ``step == 1`` (CCW, positive spherical signed area)
  and on the RIGHT at ``step > 1`` (CW, negative), uniformly over every base
  cell and order;
* therefore, after one per-call calibration (reverse every chained ring iff
  the emission handedness is CW), every ring carries the covered region on
  its LEFT.  The property is local — each surviving directed edge bounds
  exactly one covered cell, on the emission side, and chaining preserves edge
  direction — so it holds at any cover scale, including exact-hemisphere and
  over-hemisphere covers the classifier used to reject.

Later phases add the classifier behaviour tests (hemisphere+ covers dissolve
to point-sampled-correct outlines) on both engines.
"""

import numpy as np
import pytest

import mortie
from mortie import _healpix as hp
from mortie import dissolve
from mortie.orders import _rust_mort2nested, _rust_nested2mort


def _cells(bases, order):
    """All nested cells of the given base cells at *order*."""
    n = 4 ** order
    return [b * n + i for b in bases for i in range(n)]


def _to_morton(nested, order):
    """Morton words for nested ids at a single order."""
    arr = np.ascontiguousarray(np.asarray(nested, dtype=np.uint64))
    return _rust_nested2mort(arr, np.full(len(nested), order, dtype=np.uint8))


def _cell_loop(order, nest, step):
    """One cell's boundary loop as an (M, 3) unit-vector array."""
    bnd = hp.boundaries(order, np.asarray([nest], dtype=np.int64), step=step)
    if bnd.ndim == 2:
        bnd = bnd[np.newaxis, ...]
    return np.transpose(bnd, (0, 2, 1))[0]


def _world_minus_one():
    nested = list(range(192))
    del nested[100]
    return nested


_AUDIT_COVERS = [
    ("base 0-3", _cells([0, 1, 2, 3], 1), 1),
    ("hemisphere 0-5", _cells([0, 1, 2, 3, 4, 5], 1), 1),
    ("over-hemisphere 0-6", _cells([0, 1, 2, 3, 4, 5, 6], 1), 1),
    ("exemplar 4/6/7/10", _cells([4, 6, 7, 10], 1), 1),
    ("exemplar 1/2/7", _cells([1, 2, 7], 1), 1),
    ("scattered", [16, 24, 25, 28, 40, 43], 1),
    ("world minus one order-2 cell", _world_minus_one(), 2),
]


# ── phase 1: the orientation contract ──────────────────────────────────────


def test_cell_boundary_emission_orientation_is_uniform():
    # Mirror of the Rust `cell_boundary_emission_orientation_is_uniform`.
    for order in range(6):
        n = 4 ** order
        exact = np.pi / (3.0 * 4.0 ** order)
        for step in (1, 2, 3, 4, 8):
            for base in range(12):
                for nest in (base * n, base * n + n // 2, base * n + n - 1):
                    a = dissolve._spherical_signed_area(
                        _cell_loop(order, nest, step))
                    assert (a > 0) == (step == 1), (order, step, nest, a)
                    assert abs(abs(a) - exact) < 0.35 * exact, \
                        (order, step, nest, a)


@pytest.mark.parametrize("name,nested,order", _AUDIT_COVERS,
                         ids=[c[0] for c in _AUDIT_COVERS])
@pytest.mark.parametrize("step", [1, 3])
def test_oracle_chained_rings_carry_cover_on_left(name, nested, order, step):
    # Mirror of the Rust `chained_rings_carry_cover_on_left`: drives the
    # oracle's `_boundary_rings_xyz` directly (below any guard), calibrates
    # once from one cell's own loop, and probes each ring edge's midpoint a
    # quarter edge-length to the left (must be covered) and right (must not).
    cover = _to_morton(nested, order)
    covered = set(int(w) for w in cover)
    rings = dissolve._boundary_rings_xyz(cover, step)
    assert rings, name
    ccw = dissolve._spherical_signed_area(_cell_loop(order, nested[0], step)) > 0
    if not ccw:
        rings = [r[::-1] for r in rings]
    for ring in rings:
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            t = b - a
            elen = float(np.linalg.norm(t))
            m = a + b
            m /= np.linalg.norm(m)
            left = np.cross(m, t)
            left /= np.linalg.norm(left)
            for s, want in ((1.0, True), (-1.0, False)):
                p = m + s * 0.25 * elen * left
                p /= np.linalg.norm(p)
                lat = float(np.degrees(np.arcsin(np.clip(p[2], -1.0, 1.0))))
                lon = float(np.degrees(np.arctan2(p[1], p[0])))
                w = int(mortie.geo2mort(
                    np.asarray([lat]), np.asarray([lon]), order)[0])
                assert (w in covered) == want, (name, step, s, lon, lat)
