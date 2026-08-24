"""Family-wide strict input validation (issue #194).

One posture everywhere, per the ruling on issue #194: float-typed word and
offset arrays are refused rather than truncated, out-of-range values are
refused before any narrowing cast rather than wrapped, and the refusal names
the parameter and the offending value.  The two historical bug classes stay
pinned by name: the issue #185 uncatchable-panic arc (a bad value crossing
into Rust unchecked) and PR #192's silent uint64 wrap.

Valid paths are pinned byte-identical against
``data/strict_validation_goldens.json``, captured at ``4900a7e`` -- the
commit *before* the validators were adopted -- by
``generate_strict_goldens.py``.
"""

import json
import pathlib

import numpy as np
import pytest

from mortie._validate import _as_offsets, _as_u64

GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / "data" /
     "strict_validation_goldens.json").read_text())


def _u64(key):
    """Load a golden entry as a uint64 array.

    Parameters
    ----------
    key : str
        Golden entry name.

    Returns
    -------
    numpy.ndarray
        The pinned words as ``uint64``.
    """
    return np.asarray(GOLDENS[key], dtype=np.uint64)


WORDS = _u64("words")
WORDS_B = _u64("words_b")


class TestValidators:
    """The shared validators themselves (hoisted from the toc module)."""

    def test_u64_refuses_floats(self):
        with pytest.raises(ValueError, match="w must be integer-typed"):
            _as_u64(np.asarray([1.5]), "w")
        # Integral-valued floats are still float-typed: refused, not trusted.
        with pytest.raises(ValueError, match="w must be integer-typed"):
            _as_u64(np.asarray([2.0]), "w")

    def test_u64_refuses_negative_naming_value(self):
        with pytest.raises(ValueError, match=r"w must be non-negative, got -7"):
            _as_u64(np.asarray([3, -7, -2], dtype=np.int64), "w")

    def test_u64_passes_top_bit_words(self):
        # Base cells 7-11 set bit 63 (spec section 1): large uint64 words are
        # valid and must survive unchanged.
        big = np.asarray([2**63 + 5], dtype=np.uint64)
        assert _as_u64(big, "w")[0] == np.uint64(2**63 + 5)

    def test_u64_accepts_untyped_empty(self):
        # The Toc-source ruling: an untyped empty container is not numeric,
        # it is empty.
        out = _as_u64([], "w")
        assert out.size == 0 and out.dtype == np.uint64

    def test_offsets_refuse_floats(self):
        with pytest.raises(ValueError, match="offsets must be integer-typed"):
            _as_offsets(np.asarray([0.0, 2.9]))

    def test_offsets_refuse_uint64_wrap_naming_value(self):
        # The PR #192 wrap class: >= 2**63 would wrap negative through the
        # int64 cast and the kernel would then describe the wrapped copy.
        bad = np.asarray([0, 2**63 + 5], dtype=np.uint64)
        with pytest.raises(
                ValueError,
                match=r"offsets must fit in int64, got 9223372036854775813"):
            _as_offsets(bad)

    def test_offsets_valid_passthrough(self):
        out = _as_offsets([0, 2, 4])
        assert out.dtype == np.int64 and out.tolist() == [0, 2, 4]
