"""
Tests for the rank-space (x, y) deinterleave (issue #149).
"""
import numpy as np
import pytest

from mortie import rank_to_xy, xy_to_rank
from mortie.tools import MAX_ORDER

try:
    import healpy as hp
    HAVE_HEALPY = True
except ImportError:
    HAVE_HEALPY = False

# Golden (rank, x, y) triples, healpy-independent. Provenance: generated
# with healpy 1.20.0 via `x, y, f = healpy.pix2xyf(2**depth, rank,
# nest=True)` (face f == 0 for every entry), rng seed 149 -- the normative
# healpy/HEALPix C++ `pix2xyf` convention pinned by docs/specification.md
# §8. Depth 6 = the 64x64 inner chunk, depth 8 = the 256x256 shard.
GOLDEN_DEPTH6 = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 0, 1),
    (3, 1, 1),
    (335, 27, 3),
    (907, 17, 27),
    (1069, 35, 6),
    (1265, 45, 12),
    (1873, 61, 16),
    (2048, 0, 32),
    (2135, 15, 33),
    (2511, 27, 43),
    (3024, 28, 56),
    (3369, 49, 38),
    (3592, 32, 50),
    (4095, 63, 63),
]
GOLDEN_DEPTH8 = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 0, 1),
    (3, 1, 1),
    (3149, 43, 34),
    (13006, 74, 91),
    (30094, 242, 75),
    (32094, 254, 99),
    (32768, 0, 128),
    (35193, 29, 166),
    (36422, 42, 177),
    (45487, 83, 207),
    (58565, 171, 200),
    (61371, 181, 255),
    (65052, 230, 242),
    (65535, 255, 255),
]


class TestGoldenVectors:
    """Committed healpy-provenance triples; no healpy needed to run."""

    @pytest.mark.parametrize("depth,golden", [(6, GOLDEN_DEPTH6),
                                              (8, GOLDEN_DEPTH8)])
    def test_rank_to_xy_golden(self, depth, golden):
        ranks, xs, ys = (np.array(col, dtype=np.uint64)
                         for col in zip(*golden))
        x, y = rank_to_xy(ranks, depth)
        np.testing.assert_array_equal(x, xs)
        np.testing.assert_array_equal(y, ys)

    @pytest.mark.parametrize("depth,golden", [(6, GOLDEN_DEPTH6),
                                              (8, GOLDEN_DEPTH8)])
    def test_xy_to_rank_golden(self, depth, golden):
        ranks, xs, ys = (np.array(col, dtype=np.uint64)
                         for col in zip(*golden))
        np.testing.assert_array_equal(xy_to_rank(xs, ys, depth), ranks)


class TestRoundTrip:
    """Exact round-trip at every depth the packed words reach."""

    @pytest.mark.parametrize("depth", range(1, MAX_ORDER + 1))
    def test_rank_round_trip(self, depth):
        rng = np.random.default_rng(depth)
        n = int(min(4**depth, 2048))
        ranks = rng.integers(0, 4**depth, size=n, dtype=np.uint64)
        x, y = rank_to_xy(ranks, depth)
        assert x.max() < 2**depth and y.max() < 2**depth
        np.testing.assert_array_equal(xy_to_rank(x, y, depth), ranks)

    @pytest.mark.parametrize("depth", range(1, MAX_ORDER + 1))
    def test_xy_round_trip(self, depth):
        rng = np.random.default_rng(1000 + depth)
        n = int(min(4**depth, 2048))
        x = rng.integers(0, 2**depth, size=n, dtype=np.uint64)
        y = rng.integers(0, 2**depth, size=n, dtype=np.uint64)
        x2, y2 = rank_to_xy(xy_to_rank(x, y, depth), depth)
        np.testing.assert_array_equal(x2, x)
        np.testing.assert_array_equal(y2, y)


@pytest.mark.skipif(not HAVE_HEALPY, reason="healpy not installed")
class TestHealpyEquivalence:
    """Pin the convention to healpy pix2xyf / xyf2pix (face 0)."""

    @pytest.mark.parametrize("depth", [1, 2, 6, 8, 13, 20, MAX_ORDER])
    def test_matches_pix2xyf(self, depth):
        rng = np.random.default_rng(depth)
        n = int(min(4**depth, 1024))
        ranks = rng.integers(0, 4**depth, size=n, dtype=np.uint64)
        hx, hy, hf = hp.pix2xyf(2**depth, ranks.astype(np.int64), nest=True)
        assert np.all(hf == 0)
        x, y = rank_to_xy(ranks, depth)
        np.testing.assert_array_equal(x, hx.astype(np.uint64))
        np.testing.assert_array_equal(y, hy.astype(np.uint64))

    @pytest.mark.parametrize("depth", [1, 2, 6, 8, 13, 20, MAX_ORDER])
    def test_matches_xyf2pix(self, depth):
        rng = np.random.default_rng(2000 + depth)
        n = int(min(4**depth, 1024))
        x = rng.integers(0, 2**depth, size=n, dtype=np.uint64)
        y = rng.integers(0, 2**depth, size=n, dtype=np.uint64)
        hpix = hp.xyf2pix(2**depth, x.astype(np.int64),
                          y.astype(np.int64), 0, nest=True)
        np.testing.assert_array_equal(xy_to_rank(x, y, depth),
                                      hpix.astype(np.uint64))


class TestEdgeCases:
    """Corners, empties, scalars, dtypes, and validation."""

    def test_rank_zero_and_last(self):
        for depth in (1, 6, 8, MAX_ORDER):
            side = 2**depth
            assert rank_to_xy(0, depth) == (0, 0)
            assert rank_to_xy(4**depth - 1, depth) == (side - 1, side - 1)
            assert xy_to_rank(0, 0, depth) == 0
            assert xy_to_rank(side - 1, side - 1, depth) == 4**depth - 1

    def test_depth_zero(self):
        assert rank_to_xy(0, 0) == (0, 0)
        assert xy_to_rank(0, 0, 0) == 0

    def test_empty_arrays(self):
        empty = np.array([], dtype=np.uint64)
        x, y = rank_to_xy(empty, 8)
        assert x.size == 0 and y.size == 0
        assert x.dtype == np.uint64 and y.dtype == np.uint64
        rank = xy_to_rank(empty, empty, 8)
        assert rank.size == 0 and rank.dtype == np.uint64

    def test_scalar_passthrough(self):
        x, y = rank_to_xy(3, 6)
        assert isinstance(x, int) and isinstance(y, int)
        assert (x, y) == (1, 1)
        rank = xy_to_rank(1, 1, 6)
        assert isinstance(rank, int) and rank == 3

    def test_output_dtype_uint64(self):
        for dtype in (np.int32, np.int64, np.uint32, np.uint64):
            ranks = np.arange(16, dtype=dtype)
            x, y = rank_to_xy(ranks, 6)
            assert x.dtype == np.uint64 and y.dtype == np.uint64
            rank = xy_to_rank(x.astype(dtype), y.astype(dtype), 6)
            assert rank.dtype == np.uint64

    def test_broadcasting(self):
        ranks = xy_to_rank(np.arange(4, dtype=np.uint64), 0, 6)
        np.testing.assert_array_equal(ranks, [0, 1, 4, 5])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            rank_to_xy(4**6, 6)
        with pytest.raises(ValueError):
            rank_to_xy(-1, 6)
        with pytest.raises(ValueError):
            xy_to_rank(2**6, 0, 6)
        with pytest.raises(ValueError):
            xy_to_rank(0, -1, 6)

    def test_bad_depth_raises(self):
        with pytest.raises(ValueError):
            rank_to_xy(0, -1)
        with pytest.raises(ValueError):
            rank_to_xy(0, MAX_ORDER + 1)
        with pytest.raises(TypeError):
            rank_to_xy(0, 6.5)

    def test_float_input_raises(self):
        with pytest.raises(ValueError):
            rank_to_xy(np.array([1.5]), 6)
        with pytest.raises(ValueError):
            xy_to_rank(np.array([1.0]), np.array([1.0]), 6)
