"""Capture healpix-geo cross-validation goldens (issue #186 phase 2).

Writes ``mortie/tests/data/healpix_geo_goldens.json``: the cell ids that
`healpix-geo <https://github.com/keewis/healpix-geo>`_ assigns to the golden
points under its ``ellipsoid="WGS84"`` (authalic) convention, plus the cell
centers and vertices it decodes back, for a subset of cells spanning the
latitude range.

This is the acceptance test issue #186 names: healpix-geo is an independent
authalic implementation (cdshealpix + the ``geodesy`` crate), so agreeing
with it on **point -> cell** and **cell -> boundary** is the cross-validation
that closes the issue.  ``test_authalic.py`` compares mortie against the
committed values; CI never imports ``healpix_geo``, exactly as
``generate_s2_simplicity_goldens.py`` treats spherely.

The inputs are the same golden points ``test_authalic.py`` pins against
``data/authalic_goldens.json`` and ``data/legacy_geodetic_goldens.json``, and
they are written into the fixture so the test can check that its module
constants still match what these values were captured from.

Regenerating (offline, from the repo root)::

    pip install healpix-geo==0.2.1
    python mortie/tests/generate_healpix_geo_goldens.py
"""

import json
import os
import subprocess
from importlib.metadata import version

import numpy as np
from healpix_geo import nested

EXPECTED_HEALPIX_GEO = "0.2.1"

#: Nested depths for the point -> cell comparison (coarse / mid / fine).
POINT_DEPTHS = (6, 12, 18)

#: Nested depth for the cell -> boundary comparison.
BOUNDARY_DEPTH = 10

#: Indices into the golden points whose cells get a boundary golden; chosen
#: to span the latitude range (0, 45, -45, -80, 89.9999, -35.26).
BOUNDARY_POINTS = (0, 1, 2, 4, 5, 8)

# The golden point set spelled out in test_authalic.py (GOLDEN_PTS).
GOLDEN_PTS = [(0.0, 0.0), (45.0, -122.0), (-45.0, 10.0), (30.0, 150.0),
              (-80.0, 120.0), (89.9999, 0.0), (-89.9999, 45.0),
              (22.5, -60.0), (-35.264389, 170.0), (60.0, -45.0)]

OUT_PATH = os.path.join(os.path.dirname(__file__), "data",
                        "healpix_geo_goldens.json")


def healpix_geo_commit():
    """Return the git sha healpix-geo was built from, or ``"unknown"``.

    Returns
    -------
    str
        Short commit sha of the healpix-geo checkout backing the import, or
        ``"unknown"`` when it was installed from a wheel rather than a
        working tree.
    """
    import healpix_geo

    root = os.path.dirname(os.path.dirname(healpix_geo.__file__))
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main():
    """Write the goldens JSON, overwriting any existing file."""
    found = version("healpix-geo")
    if found != EXPECTED_HEALPIX_GEO:
        raise SystemExit(
            f"expected healpix-geo {EXPECTED_HEALPIX_GEO}, found {found}; "
            "bump EXPECTED_HEALPIX_GEO here and in test_authalic.py together"
        )

    lats = np.array([p[0] for p in GOLDEN_PTS])
    lons = np.array([p[1] for p in GOLDEN_PTS])

    cells = {
        f"depth_{d}": nested.lonlat_to_healpix(
            lons, lats, d, ellipsoid="WGS84").tolist()
        for d in POINT_DEPTHS
    }

    ipix = nested.lonlat_to_healpix(lons, lats, BOUNDARY_DEPTH,
                                    ellipsoid="WGS84")
    picked = np.array([ipix[i] for i in BOUNDARY_POINTS], dtype=np.uint64)
    clon, clat = nested.healpix_to_lonlat(picked, BOUNDARY_DEPTH,
                                          ellipsoid="WGS84")
    vlon, vlat = nested.vertices(picked, BOUNDARY_DEPTH, ellipsoid="WGS84")

    data = {
        "comment": (
            "healpix-geo cross-validation goldens (issue #186 phase 2), "
            "captured offline by generate_healpix_geo_goldens.py from an "
            "independent authalic implementation (cdshealpix + geodesy). "
            "Point -> cell is exact; cell centers/vertices carry a spherical-"
            "kernel difference documented in test_authalic.py."
        ),
        "healpix_geo_version": found,
        "healpix_geo_commit": healpix_geo_commit(),
        "geodesy_version": "0.15.0",
        "ellipsoid": "WGS84",
        "generator": "mortie/tests/generate_healpix_geo_goldens.py",
        "inputs": {"lats": lats.tolist(), "lons": lons.tolist()},
        "point_to_cell": cells,
        "boundary_depth": BOUNDARY_DEPTH,
        "boundary": [
            {
                "point_index": int(i),
                "ipix": int(picked[k]),
                "center_lon": float(clon[k]),
                "center_lat": float(clat[k]),
                "vertices_lon": [float(x) for x in vlon[k]],
                "vertices_lat": [float(x) for x in vlat[k]],
            }
            for k, i in enumerate(BOUNDARY_POINTS)
        ],
    }

    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
