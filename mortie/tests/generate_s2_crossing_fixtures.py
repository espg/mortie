"""Generate the S2 crossing-predicate differential fixture (issue #107 phase 2).

Writes ``src_rust/src/sphere/s2_crossing_fixture.tsv``: a table of minor-arc
pair configurations with the verdicts of **both** implementations recorded
side by side —

* the C++ s2geometry reference (``S2::CrossingSign``, ``S2::VertexCrossing``,
  ``S2::EdgeOrVertexCrossing``), reached through the shared library that the
  ``spherely`` wheel bundles (fidelity option (iii) agreed on issue #107: the
  oracle is the *C++ reference*, not a port of a port), and
* mortie's ``arcs_cross_sos`` (via the ``mortie._rustie.rust_arcs_cross_sos``
  test-support wrapper), under the canonical id assignment recorded per row.

The Rust test ``sphere::tests::test_s2_crossing_fixture`` re-computes the
mortie column and asserts:

* every row still gets the recorded mortie verdict (convention-drift pin), and
* on ``generic`` and ``near_coplanar`` rows — four distinct points whose
  orientation determinants are non-zero, where both libraries decide with
  exact arithmetic on the same coordinates — mortie's verdict equals
  ``CrossingSign == +1`` exactly.

``coplanar`` and ``shared_vertex`` rows are the *documented divergence
domain*: there the two Simulation-of-Simplicity conventions legitimately
differ (S2 perturbs coordinates, mortie perturbs identities), so the S2
columns are recorded for reference and no equality is asserted.  See the
"S2 correspondence" section in ``src_rust/src/sphere.rs``.

Regenerating
------------

Run offline from the repo root, with the Rust extension built from the same
tree and ``spherely`` (the ``test`` extra) installed::

    maturin develop --release
    python mortie/tests/generate_s2_crossing_fixtures.py

The script asserts every symbol resolves before writing anything, and refuses
to run against an unexpected spherely version, so a regeneration that would
silently change oracle semantics fails loudly instead.  Coordinates are
written as hex-encoded IEEE-754 bit patterns: the fixture round-trips
bit-exactly, which the degenerate rows require.

The symbol route: the spherely wheel vendors ``libs2`` (``spherely.libs/`` on
Linux, ``spherely.dylibs/`` on macOS) and the Itanium C++ ABI mangles these
free functions identically under gcc and clang, so ``ctypes`` can call the
reference directly.  Wheel-layout details are an implementation detail of the
wheel build — which is why the *fixture* is committed and CI never imports
spherely: only regeneration needs this route, and it asserts its ground truth
before emitting.
"""

import ctypes
import glob
import math
import os
import random
import struct
import sys

EXPECTED_SPHERELY = "0.1.1"
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src_rust", "src", "sphere",
    "s2_crossing_fixture.tsv",
)

# Mangled names of the S2 free functions (Itanium ABI; identical under gcc
# and clang).  ctypes prepends the Mach-O leading underscore on macOS itself.
SYM_CROSSING_SIGN = "_ZN2S212CrossingSignERK7S2PointS2_S2_S2_"
SYM_VERTEX_CROSSING = "_ZN2S214VertexCrossingERK7S2PointS2_S2_S2_"
SYM_EDGE_OR_VERTEX = "_ZN2S220EdgeOrVertexCrossingERK7S2PointS2_S2_S2_"

P3 = ctypes.c_double * 3


def load_s2():
    """Load the libs2 bundled in the installed spherely wheel."""
    import spherely

    assert spherely.__version__ == EXPECTED_SPHERELY, (
        f"fixture oracle is pinned to spherely=={EXPECTED_SPHERELY}, "
        f"found {spherely.__version__}; bump the pin deliberately or install "
        f"the pinned version"
    )
    site = os.path.dirname(os.path.dirname(spherely.__file__))
    candidates = []
    for libdir, pattern, deps in [
        ("spherely.dylibs", "libs2.*.dylib", []),
        # Linux (auditwheel): OpenSSL deps must be loaded RTLD_GLOBAL first.
        ("spherely.libs", "libs2-*.so*", ["libcrypto-*.so*", "libssl-*.so*"]),
    ]:
        d = os.path.join(site, "site-packages", libdir)
        if not os.path.isdir(d):
            d = os.path.join(os.path.dirname(spherely.__file__), "..", libdir)
        hits = glob.glob(os.path.join(d, pattern))
        if hits:
            candidates.append((hits[0], d, deps))
    assert candidates, "no bundled libs2 found next to spherely"
    path, d, deps = candidates[0]
    for dep in deps:
        for hit in glob.glob(os.path.join(d, dep)):
            ctypes.CDLL(hit, mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(path)

    def bind(name, restype):
        fn = getattr(lib, name)  # raises AttributeError if unresolved
        fn.argtypes = [ctypes.POINTER(P3)] * 4
        fn.restype = restype
        return fn

    cs = bind(SYM_CROSSING_SIGN, ctypes.c_int)
    vc = bind(SYM_VERTEX_CROSSING, ctypes.c_bool)
    eov = bind(SYM_EDGE_OR_VERTEX, ctypes.c_bool)

    # Ground-truth sanity before emitting anything: a known transversal
    # crossing, a known disjoint pair, and a shared vertex.
    a, b = _ll(0, -10), _ll(0, 10)
    assert cs(a, b, _ll(-10, 0), _ll(10, 0)) == 1
    assert cs(a, b, _ll(40, 40), _ll(50, 50)) == -1
    s = _ll(0, 0)
    assert cs(s, _ll(0, 10), s, _ll(10, 5)) == 0
    return cs, vc, eov, path


def _ll(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return P3(math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo),
              math.sin(la))


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def rand_unit(rng):
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        if 1e-3 < math.sqrt(sum(x * x for x in v)):
            return unit(v)


def minor_pair(rng, gen_point):
    """Two endpoints forming a minor (< ~177 deg) non-degenerate arc."""
    while True:
        u, v = gen_point(rng), gen_point(rng)
        d = dot(u, v)
        if -0.999 < d < 1.0 - 1e-12:
            return u, v


def hexf(x):
    return format(struct.unpack("<Q", struct.pack("<d", x))[0], "016x")


def emit(rows, kind, quad, ids, cs, vc, eov, mortie_cross):
    pa, pb, pc, pd = (P3(*p) for p in quad)
    s2_cs = cs(pa, pb, pc, pd)
    # VertexCrossing's contract is CrossingSign == 0 (a shared vertex); S2
    # logs an error and returns false otherwise, so only consult it there.
    s2_vc = int(vc(pa, pb, pc, pd)) if s2_cs == 0 else 0
    s2_eov = int(eov(pa, pb, pc, pd))
    m = int(mortie_cross(*quad, *ids))
    coords = "\t".join(hexf(x) for p in quad for x in p)
    idcol = "\t".join(str(i) for i in ids)
    rows.append(f"{kind}\t{coords}\t{idcol}\t{s2_cs}\t{s2_vc}\t{s2_eov}\t{m}")


def main():
    cs, vc, eov, libpath = load_s2()
    # mortie's verdict comes through the same ctypes route as S2's: the
    # C-ABI export `mortie_arcs_cross_sos_ffi` on the built extension module.
    import mortie._rustie as _rustie

    rlib = ctypes.CDLL(_rustie.__file__)
    fficross = rlib.mortie_arcs_cross_sos_ffi
    fficross.argtypes = [ctypes.POINTER(P3)] * 4 + [ctypes.c_uint64] * 4
    fficross.restype = ctypes.c_uint8

    def mortie_cross(a, b, c, d, ia, ib, ic, id_):
        return fficross(P3(*a), P3(*b), P3(*c), P3(*d), ia, ib, ic, id_)

    rng = random.Random(107)
    rows = []
    ids = (10, 11, 12, 13)

    # A: generic quadruples — the agreement domain.
    for _ in range(256):
        a, b = minor_pair(rng, rand_unit)
        c, d = minor_pair(rng, rand_unit)
        emit(rows, "generic", (a, b, c, d), ids, cs, vc, eov, mortie_cross)

    # A2: near-coplanar quadruples — the issue #107 residue family.  All four
    # points within ~1e-13 of one great circle: determinants are 1-ulp
    # residues, but non-zero, so both libraries decide them with exact
    # arithmetic on the same coordinates and must still agree exactly.
    def near_equator(rng):
        lon = rng.uniform(0, 2 * math.pi)
        z = rng.uniform(-1e-13, 1e-13)
        c = math.sqrt(1.0 - z * z)
        return [c * math.cos(lon), c * math.sin(lon), z]

    for _ in range(128):
        a, b = minor_pair(rng, near_equator)
        c, d = minor_pair(rng, near_equator)
        emit(rows, "near_coplanar", (a, b, c, d), ids, cs, vc, eov,
             mortie_cross)

    # B1: bit-exactly coplanar quadruples (equator, z = 0.0): the exact
    # determinants are zero, so each library's symbolic perturbation — S2's
    # coordinate-keyed, mortie's identity-keyed — decides.  Divergence here is
    # convention, not defect; recorded, not asserted equal.
    def equator(rng):
        lon = rng.uniform(0, 2 * math.pi)
        return [math.cos(lon), math.sin(lon), 0.0]

    for _ in range(64):
        a, b = minor_pair(rng, equator)
        c, d = minor_pair(rng, equator)
        emit(rows, "coplanar", (a, b, c, d), ids, cs, vc, eov, mortie_cross)

    # B2: shared-vertex pairs (c is bitwise b): S2 reports CrossingSign == 0
    # and defers to VertexCrossing; mortie's two ids for the shared coordinate
    # perturb it into two distinct points.
    for _ in range(64):
        a, b = minor_pair(rng, rand_unit)
        while True:
            d = rand_unit(rng)
            dd = dot(b, d)
            if -0.999 < dd < 1.0 - 1e-12:
                break
        emit(rows, "shared_vertex", (a, b, list(b), d), ids, cs, vc, eov,
             mortie_cross)

    header = [
        "# S2 crossing-predicate differential fixture (issue #107 phase 2).",
        "# Generated by mortie/tests/generate_s2_crossing_fixtures.py --",
        "# regeneration instructions in that script's docstring.",
        f"# oracle: C++ s2geometry via spherely=={EXPECTED_SPHERELY} bundled "
        f"lib ({os.path.basename(libpath)})",
        "# columns: kind, a.xyz b.xyz c.xyz d.xyz (hex f64 bits), "
        "ia ib ic id, s2_crossing_sign, s2_vertex_crossing, "
        "s2_edge_or_vertex_crossing, mortie_arcs_cross_sos",
    ]
    out = os.path.normpath(OUT_PATH)
    with open(out, "w") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(rows) + "\n")
    counts = {}
    for r in rows:
        counts[r.split("\t")[0]] = counts.get(r.split("\t")[0], 0) + 1
    agree = sum(
        1 for r in rows
        if r.split("\t")[0] in ("generic", "near_coplanar")
        and (r.split("\t")[-4] == "1") == (r.split("\t")[-1] == "1")
    )
    total_a = counts.get("generic", 0) + counts.get("near_coplanar", 0)
    print(f"wrote {out}: {counts}, agreement domain {agree}/{total_a}")
    if agree != total_a:
        print("ERROR: disagreement inside the agreement domain", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
