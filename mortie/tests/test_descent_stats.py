"""
Smoke test for the descent-stats instrumentation binding (issue #90).

Skipped unless the extension was built with the ``descent-stats`` cargo
feature (``maturin develop --release --features descent-stats``); the default
CI build carries no instrumentation and no binding.
"""

import numpy as np
import pytest

from mortie import _rustie, morton_coverage

pytestmark = pytest.mark.skipif(
    not hasattr(_rustie, "rust_descent_stats_take"),
    reason="extension built without the descent-stats feature",
)

CAUSES = ["vertex_leaf", "quad_cross", "quad_touch", "corner_parity",
          "near_pole_bulge"]


def test_take_returns_cause_tagged_straddle_leaves():
    lats = np.array([20.0, 20.0, 30.0, 30.0])
    lons = np.array([-125.0, -115.0, -115.0, -125.0])
    _rustie.rust_descent_stats_take()  # clear
    flat = np.asarray(morton_coverage(lats, lons, order=6))
    stats = _rustie.rust_descent_stats_take()

    assert stats["causes"] == CAUSES
    n = int(np.sum(stats["leaf_counts"]))
    assert n == len(stats["morton"]) > 0
    # Every straddle leaf is a member of the flat cover, at the target order.
    assert set(stats["morton"].tolist()) <= set(flat.tolist())
    assert np.all(stats["depth"] == 6)
    assert np.all(stats["circ"] > 0)
    r2 = stats["cx"] ** 2 + stats["cy"] ** 2 + stats["cz"] ** 2
    assert np.allclose(r2, 1.0)

    # Take-and-reset: a second take is empty.
    drained = _rustie.rust_descent_stats_take()
    assert len(drained["morton"]) == 0
    assert int(np.sum(drained["leaf_counts"])) == 0
