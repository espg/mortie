"""Differential test of ``ring_is_simple`` against the C++ S2 goldens.

Issue #145 phase 4.  The goldens (``data/s2_simplicity_goldens.json``) record
whether s2geometry's loop validation — which *is* self-intersection detection
(``S2Loop::FindValidationError``) — accepts each corpus ring, captured
offline through the public spherely API (see
``generate_s2_simplicity_goldens.py``).  This test never imports spherely.

The corpus stays in the convention-free domain (transversal crossings only),
where the identity-keyed and coordinate-keyed perturbation conventions
provably agree; exact degeneracies are pinned by mortie-convention tests in
``test_ring_validity.py`` and the Rust suite instead.
"""

import json
import os

import numpy as np
import pytest

from mortie import ring_is_simple
from mortie.tests.generate_s2_simplicity_goldens import _corpus

_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data",
                            "s2_simplicity_goldens.json")
with open(_GOLDEN_PATH) as _f:
    _DATA = json.load(_f)
GOLDENS = _DATA["goldens"]


def test_corpus_and_goldens_in_sync():
    assert set(GOLDENS) == set(_corpus()), (
        "corpus changed without regenerating the goldens"
    )


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_ring_is_simple_matches_s2(name):
    lats, lons = _corpus()[name]
    ours = ring_is_simple(np.asarray(lats, float), np.asarray(lons, float))
    golden = GOLDENS[name]["s2_simple"]
    assert ours == golden, (
        f"{name}: mortie says simple={ours}, S2 says {golden}"
        + (f" ({GOLDENS[name]['s2_error']})" if not golden else "")
    )
