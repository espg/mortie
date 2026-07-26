"""Ring corpus shared by the S2 simplicity-golden generator and its test.

Each entry is ``name -> (lats, lons)`` in degrees, vertex order as-given.
The corpus pairs the decision-(A) normalization rings (all simple, both
windings) with transversal self-intersections chosen to exercise both
relevance culls the bucket descent ANDs: the sub-hemisphere tangles, which
the cap cull resolves, and ``wobbly_tangled``, a hemisphere-plus ring whose
longest edge is ~167 deg, where the cap cull "is useless" and only the
great-circle band test is informative (``sphere/validity.rs`` lines 249-264).

The randomized rings are drawn from splitmix64 rather than
``np.random.default_rng``: NumPy documents "No Compatibility Guarantee" for
``Generator``'s stream, and goldens keyed to a corpus rebuilt at test time
are only meaningful if that rebuild is bit-identical everywhere.  Same
constants and same Fisher-Yates as the Rust fuzz (``validity.rs`` line 755).

:func:`ring_digest` pins that: the generator stores each ring's digest in
the goldens and the test recomputes it, so a coordinate change cannot
masquerade as a ``ring_is_simple`` regression.
"""

import hashlib

import numpy as np

from mortie.tests._normalization_corpus import CORPUS as _NORM_CORPUS

_U64 = (1 << 64) - 1


def _splitmix64(seed):
    """Return a splitmix64 draw function yielding floats in ``[0, 1]``.

    Parameters
    ----------
    seed : int
        Initial 64-bit state.

    Returns
    -------
    callable
        Zero-argument function returning the next value of the stream.
    """
    state = seed & _U64

    def nxt():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & _U64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
        return (z ^ (z >> 31)) / _U64

    return nxt


def _shuffled(n, nxt):
    """Return ``range(n)`` Fisher-Yates shuffled on the given stream.

    Parameters
    ----------
    n : int
        Number of indices.
    nxt : callable
        Draw function from :func:`_splitmix64`.

    Returns
    -------
    numpy.ndarray
        Permutation of ``0..n-1``, dtype ``intp``.
    """
    order = list(range(n))
    for k in range(n - 1, 0, -1):
        j = int(nxt() * (k + 1)) % (k + 1)
        order[k], order[j] = order[j], order[k]
    return np.asarray(order, dtype=np.intp)


def ring_digest(lats, lons):
    """Return the SHA-256 of a ring's coordinates, quantized to nanodegrees.

    The corpus is *rebuilt* wherever it is consumed, and the wobbly family
    goes through ``arcsin``/``arctan2``, whose last-ulp rounding differs
    between platform libm implementations (macOS Accelerate vs glibc broke
    the raw-bytes digest in CI).  Quantizing to 1e-9 degrees (~0.1 mm on
    Earth) erases that jitter while still catching any genuine geometry
    change, which starts at whole degrees in this corpus.  Little-endian
    float64 bytes of the quantized values, latitudes then longitudes.

    Parameters
    ----------
    lats, lons : array_like
        Ring coordinates in degrees.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    h.update(np.round(np.asarray(lats, "<f8"), 9).astype("<f8").tobytes())
    h.update(np.round(np.asarray(lons, "<f8"), 9).astype("<f8").tobytes())
    return h.hexdigest()


def _build():
    rings = {}
    # The decision-(A) corpus: all simple, spanning cap-certified, capless
    # and hemisphere-plus families, both windings.
    for name, (lats, lons) in _NORM_CORPUS.items():
        rings[f"norm_{name}"] = (np.asarray(lats, float), np.asarray(lons, float))

    # Transversal self-intersections.
    rings["bowtie"] = (np.array([0.0, 10.0, 0.0, 10.0]),
                       np.array([0.0, 0.0, 10.0, 10.0]))
    t = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
    rings["lemniscate_73"] = (15.0 * np.sin(2.0 * t), 40.0 * np.cos(t))

    # Shuffled tangles: star vertices connected in a scrambled order —
    # transversally self-intersecting, no shared coordinates.
    nxt = _splitmix64(145)
    for k in range(4):
        n = 12 + 6 * k
        th = np.arange(n) * 2.0 * np.pi / n
        r = np.radians(18.0 + 6.0 * np.array([nxt() for _ in range(n)]))
        clat, clon = 25.0 * (k - 1.5), 60.0 * k
        lats = clat + np.degrees(r) * np.cos(th)
        lons = clon + np.degrees(r) * np.sin(th) * 1.2
        order = _shuffled(n, nxt)
        rings[f"tangle_{k}"] = (lats[order], lons[order])
        rings[f"star_{k}"] = (lats, lons)  # unshuffled twin: simple

    # Hemisphere-plus tangle: the wobbly ring's own vertices in scrambled
    # order.  Its enclosing cap is ~95 deg and its longest edge ~167 deg, so
    # the crossings are found through the band test rather than the cap cull.
    # The unshuffled twin is ``norm_wobbly_as_given``.
    wlats, wlons = _NORM_CORPUS["wobbly_as_given"]
    order = _shuffled(len(wlats), nxt)
    rings["wobbly_tangled"] = (np.asarray(wlats, float)[order],
                               np.asarray(wlons, float)[order])

    return rings


CORPUS = _build()
