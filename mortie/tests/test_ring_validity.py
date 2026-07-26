"""The public ring-validity surface (issue #145 phase 3)."""

import numpy as np
import pytest

from mortie import _rustie, ring_is_simple


def test_bowtie_is_flagged():
    assert ring_is_simple([0, 10, 0, 10], [0, 0, 10, 10]) is False


def test_box_is_simple_open_and_closed():
    assert ring_is_simple([0, 10, 10, 0], [0, 0, 10, 10]) is True
    assert ring_is_simple([0, 10, 10, 0, 0], [0, 0, 10, 10, 0]) is True


def test_closing_vertex_drop_preserves_edge_numbering():
    # The box above passes with or without the closing-vertex drop (the
    # duplicated edge is degenerate-skipped anyway), so it cannot pin the
    # drop.  The reported crossing pair can: the pair is numbered in the
    # ring's own vertex indices, so a closed bowtie only reports the open
    # bowtie's pair if the closing vertex was dropped before numbering.
    open_pair = _rustie.rust_ring_is_simple(
        np.array([0.0, 10.0, 0.0, 10.0]), np.array([0.0, 0.0, 10.0, 10.0])
    )
    closed_pair = _rustie.rust_ring_is_simple(
        np.array([0.0, 10.0, 0.0, 10.0, 0.0]), np.array([0.0, 0.0, 10.0, 10.0, 0.0])
    )
    assert list(open_pair) == [1, 3]
    assert list(closed_pair) == list(open_pair)


def test_corpus_families_match_expectations():
    # The decision-(A) corpus rings are all simple; the lemniscate is not.
    from mortie.tests._normalization_corpus import CORPUS

    for name, (lats, lons) in CORPUS.items():
        assert ring_is_simple(lats, lons) is True, name
    # 73 samples, not 72: with 72 the sampling is commensurate with the
    # figure-eight's period and vertices 18 and 54 land *on* the crossing
    # point (~1e-16 rad apart), so the flag comes from the pinch/SoS regime
    # rather than a transversal crossing.  73 puts the crossing strictly
    # inside both edges.
    t = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
    lats, lons = 15.0 * np.sin(2.0 * t), 40.0 * np.cos(t)
    assert ring_is_simple(lats, lons) is False
    i, j = _rustie.rust_ring_is_simple(lats, lons)
    sep = np.hypot(lats[i] - lats[j], lons[i] - lons[j])
    assert sep > 1e-6, "flagged pair should be transversal, not a near-pinch"


def test_validation_errors():
    with pytest.raises(ValueError, match="same length"):
        ring_is_simple([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        ring_is_simple([0, np.nan, 2], [0, 1, 2])
    # The same < 3 gate every other coverage entry point applies, with the
    # sibling's own message, so the three sites stay greppable.
    for lats, lons in (([], []), ([0], [0]), ([0, 1], [0, 1])):
        with pytest.raises(ValueError, match="at least 3 vertices"):
            ring_is_simple(lats, lons)
