"""Generate the S2 simplicity goldens (issue #145 phase 4).

Writes ``mortie/tests/data/s2_simplicity_goldens.json``: for every ring in
:mod:`_simplicity_corpus`, whether the C++ s2geometry reference accepts the
ring as a valid (simple) loop, plus a digest of the ring's coordinates.  The
oracle is the public spherely API: ``create_polygon`` runs ``S2Loop::IsValid``
/ ``FindValidationError`` on every ring — exactly self-intersection detection
— and surfaces the verdict as a ``ValueError`` whose text names the crossing
edge pair.  No ctypes, no private symbols.

``test_s2_simplicity_goldens.py`` compares ``mortie.ring_is_simple`` against
the committed goldens; CI never imports spherely.

The corpus stays inside the convention-free domain: crossings are
transversal and no ring carries bit-exact shared coordinates (asserted
below) or consecutive duplicates (the oracle rejects those loudly).  At
exact degeneracies S2's coordinate-keyed and mortie's identity-keyed
perturbations legitimately disagree (the ~29% divergence measured on the
#145 thread), so those inputs are pinned by mortie-convention tests, not by
this oracle.  "Convention-free" is a claim about the *decisions the
predicate makes*, not about the vertices in isolation: ``norm_basin_*``
does contain a near-collinear triple — ``(10,45)``, ``(50,45)``, ``(-70,225)``
on the lon 45/225 meridian, exact determinant ``3.29e-18`` — which is
non-zero, so no symbolic tie-break is ever consulted for it, and both
engines' verdicts are stable under perturbation (measured in the phase-4
review).  The transversal margins are review-verified, not asserted here.

Regenerating (offline, from the repo root)::

    pip install spherely==0.1.1
    python mortie/tests/generate_s2_simplicity_goldens.py
"""

import json
import os
import sys

import numpy as np

from mortie.tests._simplicity_corpus import CORPUS, ring_digest

EXPECTED_SPHERELY = "0.1.1"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data",
                        "s2_simplicity_goldens.json")


def _min_vertex_separation(lats, lons):
    """Return the smallest angle in radians between two ring vertices.

    Computed from the chord rather than the dot product, which is what keeps
    the answer meaningful in the regime the assertion cares about (``arccos``
    of a dot returns exactly 0 for separations under ~1e-8 rad).
    """
    la, lo = np.radians(lats), np.radians(lons)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                  np.sin(la)], axis=1)
    chord = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    np.fill_diagonal(chord, np.inf)
    return float(2.0 * np.arcsin(np.clip(chord.min() / 2.0, 0.0, 1.0)))


def main():
    import spherely

    assert spherely.__version__ == EXPECTED_SPHERELY, (
        f"goldens are pinned to spherely=={EXPECTED_SPHERELY}, found "
        f"{spherely.__version__}"
    )
    goldens = {}
    for name, (lats, lons) in CORPUS.items():
        # The docstring's shared-coordinate claim, enforced: bit-exact ties
        # are where the two perturbation conventions may legitimately part.
        sep = _min_vertex_separation(lats, lons)
        assert sep > 1e-9, f"{name}: near-coincident vertices ({sep:.3e} rad)"
        ring = [(float(lo), float(la)) for la, lo in zip(lats, lons)]
        entry = {"coords_sha256": ring_digest(lats, lons)}
        try:
            spherely.create_polygon(ring)
            entry.update(s2_simple=True, s2_error=None)
        except ValueError as e:
            msg = str(e)
            # The corpus must stay transversal: a duplicate-vertex rejection
            # means the generator produced a degenerate ring — fix it rather
            # than record a convention-domain verdict.
            assert "duplicate" not in msg, (
                f"{name}: degenerate corpus ring: {msg}"
            )
            assert "crosses" in msg, (
                f"{name}: unexpected oracle rejection: {msg}"
            )
            entry.update(s2_simple=False, s2_error=msg)
        goldens[name] = entry

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
