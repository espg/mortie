"""Generate the S2 normalization goldens (issue #144 decision (A)).

Writes ``mortie/tests/data/s2_normalization_goldens.json``: for every ring in
:mod:`_normalization_corpus`, the fraction of the sphere S2 selects as the
interior under its own normalization (``spherely.create_polygon`` with
``oriented=False``, which calls ``S2Loop::Normalize`` — the convention
mortie's ``normalize=True`` adopts under decision (A)), plus the
winding-respected fraction (``oriented=True``) for reference.

``test_normalization_s2_goldens.py`` compares mortie's ``normalize=True``
cover fraction against the committed golden, so the claim "mortie normalizes
the way S2 does" is a differential test against the C++ reference, not
prose.  CI never imports spherely: only regeneration needs it.

Regenerating (offline, from the repo root)::

    pip install spherely==0.1.1
    python mortie/tests/generate_s2_normalization_goldens.py

The script refuses to run against an unpinned spherely version.
"""

import json
import math
import os
import sys

EXPECTED_SPHERELY = "0.1.1"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data",
                        "s2_normalization_goldens.json")


def main():
    import spherely

    assert spherely.__version__ == EXPECTED_SPHERELY, (
        f"goldens are pinned to spherely=={EXPECTED_SPHERELY}, found "
        f"{spherely.__version__}; bump deliberately or install the pin"
    )
    sys.path.insert(0, os.path.dirname(__file__))
    from _normalization_corpus import CORPUS

    sphere_area = 4.0 * math.pi
    goldens = {}
    for name, (lats, lons) in CORPUS.items():
        ring = [(float(lo), float(la)) for la, lo in zip(lats, lons)]
        normalized = spherely.create_polygon(ring)  # oriented=False default
        oriented = spherely.create_polygon(ring, oriented=True)
        goldens[name] = {
            "s2_normalized_fraction":
                spherely.area(normalized, radius=1.0) / sphere_area,
            "s2_oriented_fraction":
                spherely.area(oriented, radius=1.0) / sphere_area,
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {"spherely_version": EXPECTED_SPHERELY, "goldens": goldens},
            f, indent=2, sort_keys=True,
        )
        f.write("\n")
    print(f"wrote {OUT_PATH} ({len(goldens)} rings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
