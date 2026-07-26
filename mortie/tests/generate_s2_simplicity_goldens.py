"""Generate the S2 simplicity goldens (issue #145 phase 4).

Writes ``mortie/tests/data/s2_simplicity_goldens.json``: for a corpus of
rings, whether the C++ s2geometry reference accepts the ring as a valid
(simple) loop.  The oracle is the public spherely API: ``create_polygon``
runs ``S2Loop::IsValid`` / ``FindValidationError`` on every ring — exactly
self-intersection detection — and surfaces the verdict as a ``ValueError``
whose text names the crossing edge pair.  No ctypes, no private symbols.

``test_s2_simplicity_goldens.py`` compares ``mortie.ring_is_simple`` against
the committed goldens; CI never imports spherely.

The corpus stays inside the convention-free domain (transversal crossings,
no bit-exact shared coordinates, no consecutive duplicates): at exact
degeneracies S2's coordinate-keyed and mortie's identity-keyed perturbations
legitimately disagree (the ~29% divergence measured on the #145 thread), so
those inputs are pinned by mortie-convention tests, not by this oracle.

Regenerating (offline, from the repo root)::

    pip install spherely==0.1.1
    python mortie/tests/generate_s2_simplicity_goldens.py
"""

import json
import os
import sys

import numpy as np

EXPECTED_SPHERELY = "0.1.1"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data",
                        "s2_simplicity_goldens.json")


def _corpus():
    from mortie.tests._normalization_corpus import CORPUS

    rings = {}
    # The decision-(A) corpus: all simple, spanning cap-certified, capless
    # and hemisphere-plus families, both windings.
    for name, (lats, lons) in CORPUS.items():
        rings[f"norm_{name}"] = (np.asarray(lats, float), np.asarray(lons, float))

    # Transversal self-intersections.
    rings["bowtie"] = (np.array([0.0, 10.0, 0.0, 10.0]),
                       np.array([0.0, 0.0, 10.0, 10.0]))
    t = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
    rings["lemniscate_73"] = (15.0 * np.sin(2.0 * t), 40.0 * np.cos(t))

    # Shuffled tangles: star vertices connected in a scrambled order —
    # transversally self-intersecting, no shared coordinates.
    rng = np.random.default_rng(145)
    for k in range(4):
        n = 12 + 6 * k
        th = np.arange(n) * 2.0 * np.pi / n
        r = np.radians(18.0 + 6.0 * rng.random(n))
        clat, clon = 25.0 * (k - 1.5), 60.0 * k
        lats = clat + np.degrees(r) * np.cos(th)
        lons = clon + np.degrees(r) * np.sin(th) * 1.2
        order = rng.permutation(n)
        rings[f"tangle_{k}"] = (lats[order], lons[order])
        rings[f"star_{k}"] = (lats, lons)  # unshuffled twin: simple

    return rings


def main():
    import spherely

    assert spherely.__version__ == EXPECTED_SPHERELY, (
        f"goldens are pinned to spherely=={EXPECTED_SPHERELY}, found "
        f"{spherely.__version__}"
    )
    goldens = {}
    for name, (lats, lons) in _corpus().items():
        ring = [(float(lo), float(la)) for la, lo in zip(lats, lons)]
        try:
            spherely.create_polygon(ring)
            goldens[name] = {"s2_simple": True, "s2_error": None}
        except ValueError as e:
            msg = str(e)
            assert "crosses" in msg or "duplicate" in msg, (
                f"{name}: unexpected oracle rejection: {msg}"
            )
            # The corpus must stay transversal: a duplicate-vertex rejection
            # means the generator produced a degenerate ring — fix it rather
            # than record a convention-domain verdict.
            assert "crosses" in msg, f"{name}: degenerate corpus ring: {msg}"
            goldens[name] = {"s2_simple": False, "s2_error": msg}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(
            {"spherely_version": EXPECTED_SPHERELY, "goldens": goldens},
            f, indent=2, sort_keys=True,
        )
        f.write("\n")
    n_simple = sum(1 for g in goldens.values() if g["s2_simple"])
    print(f"wrote {OUT_PATH}: {len(goldens)} rings, {n_simple} simple")
    return 0


if __name__ == "__main__":
    sys.exit(main())
