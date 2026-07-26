"""Differential test of decision-(A) normalization against the S2 goldens.

Issue #144: ``normalize=True`` adopts S2's ``S2Loop::Normalize`` convention —
any simple ring whose right-hand-rule interior is decisively the larger
region is reversed, so the cover is the smaller side.  The goldens
(``data/s2_normalization_goldens.json``) record which side the C++ reference
selects, generated offline from the pinned spherely wheel (see
``generate_s2_normalization_goldens.py``); this test never imports spherely.

A cover is the interior plus a boundary fringe, so the comparison is
one-sided-with-slack: the cover fraction must not undershoot the golden
interior fraction, and must exceed it only by a fringe allowance — far below
the gap to the complementary side, so a normalization landing on the wrong
side cannot pass.
"""

import json
import os
import sys

import numpy as np
import pytest

from mortie import morton_coverage

sys.path.insert(0, os.path.dirname(__file__))
from _normalization_corpus import CORPUS  # noqa: E402

ORDER = 5
NCELLS = 12 * 4**ORDER
GOLDENS = json.load(
    open(os.path.join(os.path.dirname(__file__), "data",
                      "s2_normalization_goldens.json"))
)["goldens"]

# Boundary-fringe allowance at order 5: the wiggly hemisphere-scale rings
# carry the longest boundaries (~5% of cells); every gap to the wrong side
# is > 12%.
FRINGE = 0.07


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_normalize_true_matches_s2_side(name):
    lats, lons = CORPUS[name]
    cover = morton_coverage(np.asarray(lats, float), np.asarray(lons, float),
                            order=ORDER, normalize=True)
    frac = len(cover) / NCELLS
    golden = GOLDENS[name]["s2_normalized_fraction"]
    assert golden - 0.005 <= frac <= golden + FRINGE, (
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
