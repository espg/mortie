"""Ring corpus shared by the S2 normalization-golden generator and its test.

Each entry is ``name -> (lats, lons)`` in degrees, vertex order as-given.
The corpus deliberately spans the decision-(A) families (issue #144): rings
the vertex-sum cap certifies, the crescent family it misses, and genuinely
hemisphere-plus rings — in both windings where the winding is the point.
All rings are simple and free of exact degeneracies, keeping the oracle
inside its convention-free domain (see the #145 notes on the ~29% SoS
divergence at exact degeneracies).
"""

import numpy as np


def _crescent():
    lons = np.concatenate([np.linspace(0, 300, 40), np.linspace(300, 0, 40)])
    lats = np.concatenate([np.full(40, 5.0), np.full(40, 10.0)])
    return lats, lons


def _band():
    lats = np.full(36, -10.0)
    lons = np.arange(36) * 10.0
    return lats, lons


def _wobbly():
    centre = np.array([np.cos(np.radians(45.0)), 0.0, np.sin(np.radians(45.0))])
    e1 = np.cross([0.0, 0.0, 1.0], centre)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    th = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    r = np.radians(97.5 + 12.5 * np.sin(3.0 * th))
    v = np.cos(r)[:, None] * centre + np.sin(r)[:, None] * (
        np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    )
    lats = np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    lons = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    return lats, lons


def _box():
    return np.array([40.0, 40.0, 50.0, 50.0]), np.array(
        [-125.0, -115.0, -115.0, -125.0]
    )


def _basin():
    # The issue #107 reproducer: simple, hemisphere-plus, as-given interior
    # is the larger region (52%).
    return np.array([10.0, 50.0, -10.0, -70.0, -10.0]), np.array(
        [45.0, 45.0, 170.0, 225.0, 280.0]
    )


def _rev(ring):
    lats, lons = ring
    return lats[::-1].copy(), lons[::-1].copy()


CORPUS = {
    "crescent_ccw": _crescent(),
    "crescent_cw": _rev(_crescent()),
    "band_inc": _band(),
    "band_dec": _rev(_band()),
    "wobbly_as_given": _wobbly(),
    "wobbly_reversed": _rev(_wobbly()),
    "box_ccw": _box(),
    "box_cw": _rev(_box()),
    "basin_as_given": _basin(),
    "basin_reversed": _rev(_basin()),
}
