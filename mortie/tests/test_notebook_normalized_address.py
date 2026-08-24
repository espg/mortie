"""Pin the normalized-address identity ExampleUsage.ipynb relies on (issue #142).

Cell 15 of ``examples/ExampleUsage.ipynb`` used to hand a **UNIQ** value to the
``heal_norm`` arithmetic, which expects a **NESTED** index.  Since
``uniq = 4 * 4**order + nested``, every normalized address overshot by exactly
``4**(order + 1)``, and because ``norm2mort`` composes ``(parent << 2*order) |
normed`` that excess landed in the *base cell* field.  The damage was therefore
data-dependent: base cells 0-3 came back silently wrong, 4-7 were correct only
because the stray bit was already set, and 8-11 raised.  The cell now calls
``geo2mort`` directly, so these tests pin the identity that collapse depends on
across **all twelve** base cells -- notably the ones that used to pass by luck.
"""

from pathlib import Path

import numpy as np
import pytest

import mortie as mt

ORDER = 9
COORDS_FILE = Path(__file__).parent / "Ant_Grounded_DrainageSystem_Polygons.txt"


@pytest.fixture(scope="module")
def sweep():
    """A 5-degree global lat/lon sweep, which reaches every base cell."""
    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(-177.5, 180, 5.0)
    lons, lats = np.meshgrid(lon, lat)
    return lats.ravel(), lons.ravel()


def test_sweep_reaches_every_base_cell(sweep):
    # Guards the tests below: a sweep missing base cells 0-3 or 8-11 is exactly
    # the spot check that let this bug survive in the first place.
    lats, lons = sweep
    parents = mt.unique2parent(mt.geo2uniq(lats, lons, ORDER))
    assert sorted(np.unique(parents)) == list(range(12))


def test_geo2mort_matches_the_normalized_chain(sweep):
    lats, lons = sweep
    uniq = mt.geo2uniq(lats, lons, ORDER)
    parents = mt.unique2parent(uniq)
    # uniq = 4 * 4**order + nested, so the UNIQ marker comes off before the
    # base-cell offset does.
    normed = uniq - 4 * 4**ORDER - parents * 4**ORDER

    assert ((normed >= 0) & (normed < 4**ORDER)).all()
    np.testing.assert_array_equal(
        mt.norm2mort(normed.ravel(), parents.ravel(), ORDER),
        mt.geo2mort(lats, lons, order=ORDER),
    )


def test_unnormalized_uniq_corrupts_the_base_cell(sweep):
    # The bug's signature: dropping the UNIQ marker overshoots by 4**(order+1),
    # which is 2**(2*order+2) -- two bits above the address field.
    lats, lons = sweep
    uniq = mt.geo2uniq(lats, lons, ORDER)
    parents = mt.unique2parent(uniq)
    overshoot = (uniq - parents * 4**ORDER) - (
        uniq - 4 * 4**ORDER - parents * 4**ORDER
    )
    np.testing.assert_array_equal(overshoot, 4 ** (ORDER + 1))

    # Base cells 0-3 are the silent ones: no raise, wrong answer.
    low = parents < 4
    wrong = mt.norm2mort(
        (uniq[low] - parents[low] * 4**ORDER).ravel(), parents[low].ravel(), ORDER
    )
    assert (wrong != mt.geo2mort(lats[low], lons[low], order=ORDER)).all()


@pytest.mark.skipif(not COORDS_FILE.exists(), reason=f"{COORDS_FILE.name} missing")
def test_notebook_data_lands_in_a_raising_base_cell():
    # Basin 4 -- the notebook's own subset -- sits entirely in base cell 11, so
    # the old cell could not have produced its committed outputs at all.
    data = np.loadtxt(COORDS_FILE)
    b4 = data[data[:, 2].astype(np.int32) == 4]
    lats, lons = b4[:, 0], b4[:, 1]

    uniq = mt.geo2uniq(lats, lons, ORDER)
    parents = mt.unique2parent(uniq)
    assert np.unique(parents).tolist() == [11]

    with pytest.raises(ValueError, match="too large for depth"):
        mt.norm2mort((uniq - parents * 4**ORDER).ravel(), parents.ravel(), ORDER)
