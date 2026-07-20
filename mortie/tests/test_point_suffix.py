"""Decimal kind suffix ``p`` for point ids (issue #120; spec sections 2/4).

Golden vectors for the espg-ruled grammar extension: point words render with
a terminal ``p`` (lossless round-trip), unmarked strings keep the area
tie-break, the suffix is illegal below order 29, and paths never carry it.
"""

import numpy as np
import pytest

pytest.importorskip("pandas")

from mortie import MortonIndexArray  # noqa: E402
from mortie.morton_index import _decimal_to_word  # noqa: E402


def _pair():
    """A (point, area) pair at the same location."""
    lat, lon = np.array([45.0]), np.array([45.0])
    point = MortonIndexArray.from_latlon(lat, lon, points=True)
    area = MortonIndexArray.from_latlon(lat, lon)
    return point, area


class TestRender:
    def test_point_renders_with_p(self):
        point, area = _pair()
        s = point.decimal_repr()[0]
        assert s.endswith("p")
        # Area words render unchanged -- no suffix at any order.
        assert not area.decimal_repr()[0].endswith("p")

    def test_scalar_str_carries_suffix(self):
        point, _ = _pair()
        assert str(point[0]).endswith("p")

    def test_to_decimal_is_u32_and_untruncated(self):
        point, _ = _pair()
        out = point.to_decimal()
        assert out.dtype == np.dtype("<U32")
        assert out[0].endswith("p")
        assert out[0] == point.decimal_repr()[0]


class TestParse:
    def test_marked_roundtrip_is_point_word(self):
        point, _ = _pair()
        word = int(np.asarray(point._data, dtype=np.uint64)[0])
        s = point.decimal_repr()[0]
        parsed = _decimal_to_word(s)
        assert parsed == word
        assert parsed & 0x3F >= 48  # point suffix region (spec section 1)

    def test_unmarked_keeps_area_tiebreak(self):
        point, area = _pair()
        s = point.decimal_repr()[0][:-1]  # strip the mark
        assert _decimal_to_word(s) == int(np.asarray(area._data, dtype=np.uint64)[0])

    def test_suffix_illegal_below_order_29(self):
        for bad in ("1231p", "-6p", "3p"):
            with pytest.raises(ValueError, match="legal only"):
                _decimal_to_word(bad)

    def test_bare_or_doubled_suffix_malformed(self):
        for bad in ("p", "-p", "31111pp"):
            with pytest.raises(ValueError):
                _decimal_to_word(bad)


class TestPaths:
    def test_hive_path_refuses_point_words(self):
        point, _ = _pair()
        with pytest.raises(ValueError, match="points do not live in paths"):
            point.hive_path()

    def test_from_hive_path_rejects_marked_leaf(self):
        point, _ = _pair()
        stem = point.decimal_repr()[0]
        with pytest.raises(ValueError, match="points do not live in paths"):
            MortonIndexArray.from_hive_path(f"root/{stem}.zarr")
