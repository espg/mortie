"""Differential test of decision-(A) normalization against the S2 goldens.

Issue #144: ``normalize=True`` adopts S2's ``S2Loop::Normalize`` convention —
any simple ring whose right-hand-rule interior is decisively the larger
region is reversed, so the cover is the smaller side.  The goldens
(``data/s2_normalization_goldens.json``) record which side the C++ reference
selects, generated offline from the pinned spherely wheel (see
``generate_s2_normalization_goldens.py``); this test never imports spherely.

A cover is the interior plus a boundary fringe, so the comparison is
one-sided-with-slack: the cover fraction must not undershoot the golden
interior fraction, and must exceed it only by a fringe allowance.  That
allowance is per-ring rather than flat, because it has to stay **below the gap
to the complementary side** for the soundness property to hold — a
normalization landing on the wrong side must not pass.  The test asserts that
separation for every ring instead of assuming it.
"""

import json
import os

import numpy as np
import pytest

from mortie import morton_coverage
from mortie.tests._normalization_corpus import CORPUS

ORDER = 5
NCELLS = 12 * 4**ORDER
_GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data",
                            "s2_normalization_goldens.json")
with open(_GOLDEN_PATH) as _f:
    GOLDENS = json.load(_f)["goldens"]

# Boundary-fringe allowance at order 5.  Measured (fringe = cover − golden;
# gap = the distance from the golden side to its complement):
#
#     ring       fringe    gap to the wrong side
#     box        0.0013    0.9966
#     crescent   0.0165    0.9278
#     band       0.0089    0.1741
#     wobbly     0.0100    0.1290
#     basin      0.0133    0.0281
#
# The basin pair is a near-hemisphere ring whose two sides differ by under 3%,
# so the default allowance — sized for the wiggly hemisphere-scale rings, whose
# boundaries run to ~5% of cells — would swallow its complement (covering the
# as-given 51.4% side reads 0.5265, inside a flat-0.07 window).  It gets its
# own, and the test below checks the separation rather than trusting the table.
FRINGE = 0.07
RING_FRINGE = {"basin_as_given": 0.02, "basin_reversed": 0.02}


def _window(name):
    """``(golden, lo, hi)``: S2's fraction and the accept window around it."""
    golden = GOLDENS[name]["s2_normalized_fraction"]
    return golden, golden - 0.005, golden + RING_FRINGE.get(name, FRINGE)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_normalize_true_matches_s2_side(name):
    golden, lo, hi = _window(name)
    # Soundness of the window itself: a cover of the *complementary* interior
    # is at least ``1 − golden`` of the sphere (its own fringe only pushes it
    # higher), so the window must end below that.  Without this the basin pair
    # could not tell the two sides apart.
    assert 1.0 - golden > hi, (
        f"{name}: accept window ends at {hi:.4f}, which admits the "
        f"complementary side at {1.0 - golden:.4f}"
    )
    lats, lons = CORPUS[name]
    cover = morton_coverage(np.asarray(lats, float), np.asarray(lons, float),
                            order=ORDER, normalize=True)
    frac = len(cover) / NCELLS
    assert lo <= frac <= hi, (
        f"{name}: cover fraction {frac:.4f} vs S2 normalized side "
        f"{golden:.4f}"
    )


@pytest.mark.parametrize(
    "pair",
    [
        ("crescent_ccw", "crescent_cw"),
        ("band_inc", "band_dec"),
        ("wobbly_as_given", "wobbly_reversed"),
        ("box_ccw", "box_cw"),
        ("basin_as_given", "basin_reversed"),
    ],
)
def test_normalize_true_is_winding_invariant(pair):
    covers = []
    for name in pair:
        lats, lons = CORPUS[name]
        covers.append(
            set(
                int(c)
                for c in morton_coverage(
                    np.asarray(lats, float), np.asarray(lons, float),
                    order=ORDER, normalize=True,
                )
            )
        )
    assert covers[0] == covers[1], f"windings disagree under normalize=True: {pair}"
