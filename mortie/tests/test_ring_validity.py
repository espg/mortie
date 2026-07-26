"""The public ring-validity surface (issue #145 phase 3)."""

import numpy as np
import pytest

from mortie import ring_is_simple


def test_bowtie_is_flagged():
    assert ring_is_simple([0, 10, 0, 10], [0, 0, 10, 10]) is False


def test_box_is_simple_open_and_closed():
    assert ring_is_simple([0, 10, 10, 0], [0, 0, 10, 10]) is True
    # A duplicated closing vertex is dropped exactly as coverage ingest does.
    assert ring_is_simple([0, 10, 10, 0, 0], [0, 0, 10, 10, 0]) is True


def test_corpus_families_match_expectations():
    # The decision-(A) corpus rings are all simple; the lemniscate is not.
    from mortie.tests._normalization_corpus import CORPUS

    for name, (lats, lons) in CORPUS.items():
        assert ring_is_simple(lats, lons) is True, name
    t = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    assert ring_is_simple(15.0 * np.sin(2.0 * t), 40.0 * np.cos(t)) is False


def test_validation_errors():
    with pytest.raises(ValueError, match="same length"):
        ring_is_simple([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="NaN"):
        ring_is_simple([0, np.nan, 2], [0, 1, 2])
