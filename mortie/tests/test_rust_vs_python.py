"""
Golden-output regression tests for the Rust morton encoders.

After the issue #48 packed-u64 flip the bare-``i64`` morton channel carries the
packed ``decimal_morton`` word (bit-reinterpreted; negative for base cells
8-11), not the retired decimal encoding. The pure-Python parity twins for the
encoders were removed (issue #37); these tests pin the Rust output and the
encode/decode round-trips against the packed wire format.
"""

from pathlib import Path

import numpy as np
import pytest


class TestNorm2Mort:
    """norm2mort is the order-29-native forward encoder, inverse of mort2norm."""

    def test_scalar_roundtrip(self):
        from mortie import tools
        m = tools.norm2mort(1000, 2, 18)
        normed, parent, order = tools.mort2norm(m)
        assert (int(normed), int(parent), order) == (1000, 2, 18)

    def test_array_roundtrip(self):
        from mortie import tools
        normed = np.arange(1000, dtype=np.int64)
        parents = np.array([i % 12 for i in range(1000)], dtype=np.int64)
        # Encode every (normed, parent) at a fixed order, then decode back.
        words = np.array(
            [int(tools.norm2mort(int(n), int(p), 18)) for n, p in zip(normed, parents)],
            dtype=np.int64,
        )
        # Southern base cells (8-11) set the i64 sign bit -> negative word.
        assert np.any(words < 0)
        out_n, out_p, order = tools.mort2norm(words)
        assert order == 18
        np.testing.assert_array_equal(out_n, normed)
        np.testing.assert_array_equal(out_p, parents)

    @pytest.mark.parametrize("order", [6, 10, 14, 18, 25, 29])
    def test_different_orders_roundtrip(self, order):
        from mortie import tools
        m = tools.norm2mort(100, 2, order)
        normed, parent, o = tools.mort2norm(m)
        assert (int(normed), int(parent), o) == (100, 2, order)


class TestGeo2Mort:
    """Golden output for geo2mort (all-in-Rust healpix crate path)."""

    # A spread of latitudes/longitudes including poles and the antimeridian.
    LATS = [0.0, 45.0, -45.0, 89.0, -89.0, 12.34, -77.7, 30.0]
    LONS = [0.0, 90.0, -90.0, 179.0, -179.0, 56.78, 123.4, -60.0]

    def test_order18_roundtrips_to_nested(self):
        """Each packed word decodes back to the cell the healpix crate hashed."""
        from mortie import _healpix as hp
        from mortie import tools
        words = np.asarray(
            tools.geo2mort(np.array(self.LATS), np.array(self.LONS), order=18),
            dtype=np.int64,
        )
        cell_ids, order = tools.mort2healpix(words)
        assert order == 18
        expected = hp.ang2pix(18, np.array(self.LONS), np.array(self.LATS))
        np.testing.assert_array_equal(cell_ids, expected)

    def test_order10_roundtrips_to_nested(self):
        from mortie import _healpix as hp
        from mortie import tools
        words = np.asarray(
            tools.geo2mort(np.array(self.LATS), np.array(self.LONS), order=10),
            dtype=np.int64,
        )
        cell_ids, order = tools.mort2healpix(words)
        assert order == 10
        expected = hp.ang2pix(10, np.array(self.LONS), np.array(self.LATS))
        np.testing.assert_array_equal(cell_ids, expected)

    def test_order29_native(self):
        """geo2mort reaches order 29 now (the packed kernel's MAX_ORDER)."""
        from mortie import tools
        words = np.asarray(
            tools.geo2mort(np.array([45.0, -80.0]), np.array([-120.0, 33.0]), order=29),
            dtype=np.int64,
        )
        _, order = tools.mort2healpix(words)
        assert order == 29

    @pytest.mark.slow
    def test_antarctic_polygon_stable(self):
        """geo2mort is deterministic on the 1.2M-point Antarctic dataset.

        Pins a content hash of the order-18 packed output so a future Rust
        change that shifts the encoding is caught.
        """
        coords_file = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
        if not coords_file.exists():
            pytest.skip("Antarctic polygon data not found")

        from mortie import tools
        data = np.loadtxt(coords_file)
        lats, lons = data[:, 0], data[:, 1]
        result = np.asarray(tools.geo2mort(lats, lons, order=18), dtype=np.int64)
        # Every packed word decodes to the same cell the healpix crate hashes.
        from mortie import _healpix as hp
        cell_ids, order = tools.mort2healpix(result)
        assert order == 18
        np.testing.assert_array_equal(cell_ids, hp.ang2pix(18, lons, lats))


class TestMort2Norm:
    """mort2norm decodes the packed word via the kernel (depth-keyed order)."""

    def test_order1_all_parents(self):
        """Order 1 covers every parent (0-11) and both hemispheres."""
        from mortie import tools
        normed_in = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        parent_in = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
        mortons = np.array(
            [int(tools.norm2mort(int(n), int(p), 1)) for n, p in zip(normed_in, parent_in)],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 1
        np.testing.assert_array_equal(normed, normed_in)
        np.testing.assert_array_equal(parent, parent_in)

    def test_order6_spread(self):
        from mortie import tools
        normed_in = np.array([1822, 2204, 2120, 1406, 3875, 1511, 2694, 1533], dtype=np.int64)
        parent_in = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        mortons = np.array(
            [int(tools.norm2mort(int(n), int(p), 6)) for n, p in zip(normed_in, parent_in)],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 6
        np.testing.assert_array_equal(normed, normed_in)
        np.testing.assert_array_equal(parent, parent_in)

    def test_order29_max(self):
        """Order 29 is the maximum the packed encoding supports."""
        from mortie import tools
        normed_in = np.array([12345, 4**29 - 1], dtype=np.int64)
        parent_in = np.array([2, 11], dtype=np.int64)
        mortons = np.array(
            [int(tools.norm2mort(int(n), int(p), 29)) for n, p in zip(normed_in, parent_in)],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 29
        np.testing.assert_array_equal(normed, normed_in)
        np.testing.assert_array_equal(parent, parent_in)

    def test_scalar_north(self):
        from mortie import tools
        m = tools.norm2mort(2120, 2, 6)
        assert tools.mort2norm(m) == (2120, 2, 6)

    def test_scalar_south(self):
        from mortie import tools
        m = tools.norm2mort(1533, 7, 6)
        assert tools.mort2norm(m) == (1533, 7, 6)

    def test_roundtrip_against_norm2mort(self):
        """mort2norm inverts norm2mort for a random order-14 spread."""
        from mortie import tools
        rng = np.random.default_rng(14)
        normed_in = rng.integers(0, 4**14, size=32, dtype=np.int64)
        parents_in = (np.arange(32) % 12).astype(np.int64)
        mortons = np.array(
            [int(tools.norm2mort(int(n), int(p), 14))
             for n, p in zip(normed_in, parents_in)],
            dtype=np.int64,
        )
        normed, parent, order = tools.mort2norm(mortons)
        assert order == 14
        np.testing.assert_array_equal(normed, normed_in)
        np.testing.assert_array_equal(parent, parents_in)
