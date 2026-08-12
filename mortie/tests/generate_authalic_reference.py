"""Derive the authalic-latitude series and reference values (issue #186).

Offline generator — **not** run by the test suite and **not** a runtime
dependency.  Requires ``mpmath`` and ``sympy`` (~10 s)::

    python mortie/tests/generate_authalic_reference.py

It produces two artifacts, both committed:

1. ``mortie/tests/data/authalic_reference.json`` — high-precision reference
   conversions (60 decimal digits internally, printed to 25 significant
   digits) computed **independently of any series**: the forward direction
   uses the closed-form authalic construction (Snyder 1987, eqs. 3-11/3-12),

       q(phi) = (1 - e^2) * [ sin(phi) / (1 - e^2 sin^2(phi))
                              + atanh(e * sin(phi)) / e ]
       beta(phi) = asin(q(phi) / q(pi/2))

   evaluated with mpmath, and the inverse direction root-finds
   ``beta(phi) = beta_target`` to the same precision.  The tests
   (``test_authalic.py`` and the unit tests in ``src_rust/src/authalic.rs``)
   compare the shipped series against these values.

2. The series-coefficient table printed to stdout — the trigonometric series

       beta = phi + sum_{k=1..5} FWD_S2K[k-1] * sin(2k * phi)
       phi = beta + sum_{k=1..5} INV_S2K[k-1] * sin(2k * beta)

   with coefficients derived **symbolically**, as exact rationals in powers
   of e^2 through e^10:

   * forward — perturbation expansion of ``asin(q/qp)`` order by order in
     e^2 (solve ``sin(phi + delta) = q/qp`` for ``delta``);
   * inverse — series reversion of the forward series by fixed-point
     iteration with order bookkeeping.

   These are the expressions hard-coded in ``src_rust/src/authalic.rs``.
   Truncating after e^10 leaves an error of O(e^12) ~ 1e-13 rad; the script
   measures the achieved bound of the series against the closed form on a
   0.01-degree grid and prints it (``series truncation bound`` lines below
   the table).

The e^2/e^4/e^6 leading terms are asserted against their published values
(Snyder 1987, eq. 3-18 and its inverse), so a sympy regression cannot
silently corrupt the table.

WGS84 constants are pinned: a = 6378137 (exact), 1/f = 298.257223563 (exact
decimal).  Everything else derives from them.
"""

import json
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import sympy as sp

mp.mp.dps = 60

# --- pinned WGS84 constants -------------------------------------------------
INV_F = mp.mpf("298.257223563")
F = 1 / INV_F
E2 = F * (2 - F)
E = mp.sqrt(E2)

N_TERMS = 5
ORDER = 2 * N_TERMS + 2  # expand through e^10; error term is O(e^12)


def q_exact(phi):
    """Snyder's q at geodetic latitude ``phi`` (radians, mpmath).

    Parameters
    ----------
    phi : mpmath.mpf
        Geodetic latitude in radians.

    Returns
    -------
    mpmath.mpf
        The authalic function q(phi).
    """
    s = mp.sin(phi)
    return (1 - E2) * (s / (1 - E2 * s * s) + mp.atanh(E * s) / E)


QP = q_exact(mp.pi / 2)


def authalic_exact(phi):
    """Closed-form geodetic -> authalic latitude (radians, mpmath).

    Parameters
    ----------
    phi : mpmath.mpf
        Geodetic latitude in radians.

    Returns
    -------
    mpmath.mpf
        Authalic latitude in radians.
    """
    x = q_exact(phi) / QP
    x = mp.mpf(1) if x > 1 else (mp.mpf(-1) if x < -1 else x)
    return mp.asin(x)


def geodetic_exact(beta):
    """Closed-form authalic -> geodetic latitude by root-finding (mpmath).

    Parameters
    ----------
    beta : mpmath.mpf
        Authalic latitude in radians.

    Returns
    -------
    mpmath.mpf
        Geodetic latitude in radians.
    """
    if abs(abs(beta) - mp.pi / 2) < mp.mpf("1e-50"):
        return beta
    return mp.findroot(lambda p: authalic_exact(p) - beta, beta)


# --- symbolic series derivation ---------------------------------------------


def _linearize(expr):
    """Collapse trig products/powers into single sin/cos of multiples.

    Parameters
    ----------
    expr : sympy expression
        A trigonometric polynomial.

    Returns
    -------
    sympy expression
        The linearized (harmonic) form.
    """
    return sp.expand_trig(expr).rewrite(sp.exp).expand().rewrite(sp.cos).expand()


def _harmonics(expr, var, e):
    """Collect ``expr`` into coefficients of sin(2k*var), k = 1..N_TERMS.

    Parameters
    ----------
    expr : sympy expression
        An odd trigonometric polynomial in ``var`` whose coefficients are
        polynomials in ``e``.
    var : sympy.Symbol
        The latitude symbol.
    e : sympy.Symbol
        The expansion symbol (the eccentricity).

    Returns
    -------
    list of dict
        Entry k-1 maps power-of-e^2 (int, 1..N_TERMS) to the exact rational
        coefficient of ``sin(2k*var)`` at that power.
    """
    expanded = _linearize(expr)
    out = []
    for k in range(1, N_TERMS + 1):
        poly = sp.Poly(sp.expand(expanded.coeff(sp.sin(2 * k * var))), e)
        table = {}
        for (p,), c in poly.terms():
            assert p % 2 == 0, f"odd power of e in sin({2 * k}x) coefficient"
            table[p // 2] = Fraction(sp.Rational(c).p, sp.Rational(c).q)
        out.append(table)
    residual = expanded - sum(
        expanded.coeff(sp.sin(2 * k * var)) * sp.sin(2 * k * var)
        for k in range(1, N_TERMS + 1)
    )
    assert sp.simplify(residual) == 0, "non-harmonic residual in series"
    return out


def derive_forward():
    """Perturbation-expand the geodetic -> authalic direction through e^10.

    Returns
    -------
    list of dict
        The forward coefficient tables (see :func:`_harmonics`).
    """
    e, phi = sp.symbols("e phi", positive=True)
    s = sp.sin(phi)
    q = (1 - e**2) * (s / (1 - e**2 * s**2) + sp.atanh(e * s) / e)
    qp = q.subs(phi, sp.pi / 2)
    sin_beta = sp.series(q / qp, e, 0, ORDER).removeO().expand()

    # beta = phi + delta, delta = sum a_j e^(2j); solve sin(phi + delta) =
    # sin_beta one order at a time (each order's equation is linear in a_j).
    delta = sp.Integer(0)
    for j in range(1, N_TERMS + 1):
        aj = sp.Function(f"A{j}")(phi)
        trial = delta + aj * e ** (2 * j)
        lhs = sp.sin(phi + trial).series(e, 0, 2 * j + 1).removeO().expand()
        eq = sp.expand(lhs - sin_beta).coeff(e, 2 * j)
        delta += sp.simplify(sp.solve(sp.Eq(eq, 0), aj)[0]) * e ** (2 * j)
    return _harmonics(delta, phi, e)


def derive_inverse(fwd):
    """Revert the forward series: authalic -> geodetic through e^10.

    Parameters
    ----------
    fwd : list of dict
        Forward coefficient tables from :func:`derive_forward`.

    Returns
    -------
    list of dict
        The inverse coefficient tables (see :func:`_harmonics`).
    """
    e, beta, t = sp.symbols("e beta t", positive=True)
    a = [sp.Symbol(f"A{k}") for k in range(1, N_TERMS + 1)]

    def f(z):
        # The forward correction with A_k order-weighted by t^k (A_k = O(e^2k)).
        return sum(a[k - 1] * t**k * sp.sin(2 * k * z) for k in range(1, N_TERMS + 1))

    # Fixed-point iteration z <- beta - f(z) converges to order t^N_TERMS.
    z = beta
    for _ in range(N_TERMS):
        z = sp.series(sp.expand(beta - f(z)), t, 0, N_TERMS + 1).removeO()
    g = _linearize(sp.expand(z - beta)).subs(t, 1)

    # Substitute the rational forward coefficients (as series in y = e^2).
    y = sp.Symbol("y")
    subs = {
        a[j - 1]: sum(
            sp.Rational(c.numerator, c.denominator) * y**p for p, c in fwd[j - 1].items()
        )
        for j in range(1, N_TERMS + 1)
    }
    out = []
    for k in range(1, N_TERMS + 1):
        poly = sp.Poly(sp.expand(g.coeff(sp.sin(2 * k * beta)).subs(subs)), y)
        table = {}
        for (p,), c in poly.terms():
            if p <= N_TERMS:  # drop orders beyond the working precision
                table[p] = Fraction(sp.Rational(c).p, sp.Rational(c).q)
        out.append(table)
    return out


def check_published(fwd, inv):
    """Assert the leading terms match Snyder (1987) eq. 3-18 / its inverse.

    Parameters
    ----------
    fwd, inv : list of dict
        The derived coefficient tables.
    """
    fr = Fraction
    assert fwd[0][1] == fr(-1, 3), fwd[0]
    assert fwd[0][2] == fr(-31, 180), fwd[0]
    assert fwd[0][3] == fr(-59, 560), fwd[0]
    assert fwd[1][2] == fr(17, 360), fwd[1]
    assert fwd[1][3] == fr(61, 1260), fwd[1]
    assert fwd[2][3] == fr(-383, 45360), fwd[2]
    assert inv[0][1] == fr(1, 3), inv[0]
    assert inv[0][2] == fr(31, 180), inv[0]
    assert inv[0][3] == fr(517, 5040), inv[0]
    assert inv[1][2] == fr(23, 360), inv[1]
    assert inv[1][3] == fr(251, 3780), inv[1]
    assert inv[2][3] == fr(761, 45360), inv[2]


def rust_table(name, table):
    """Format one harmonic's coefficient dict as a Rust constant.

    Parameters
    ----------
    name : str
        Rust constant name.
    table : dict
        Power-of-e^2 -> Fraction.

    Returns
    -------
    str
        A ``const`` definition in powers of ``E2`` (``E4`` = e^4, ...).
    """
    pow_name = {1: "E2", 2: "E4", 3: "E6", 4: "E8", 5: "E10"}
    terms = [
        f"({table[p].numerator}.0 / {table[p].denominator}.0) * {pow_name[p]}"
        for p in sorted(table)
    ]
    joined = "\n    + ".join(terms)
    return f"const {name}: f64 = {joined};"


def emit(fwd, inv):
    """Print the Rust coefficient block and measure the achieved bound.

    Parameters
    ----------
    fwd, inv : list of dict
        The derived coefficient tables.
    """
    print("// --- paste into src_rust/src/authalic.rs " + "-" * 30)
    for k, table in enumerate(fwd, start=1):
        print(rust_table(f"FWD_S{2 * k}", table))
    for k, table in enumerate(inv, start=1):
        print(rust_table(f"INV_S{2 * k}", table))

    def coeffs(tables):
        return [
            sum(mp.mpf(c.numerator) / mp.mpf(c.denominator) * E2**p
                for p, c in table.items())
            for table in tables
        ]

    fwd_c, inv_c = coeffs(fwd), coeffs(inv)
    worst_f = worst_i = mp.mpf(0)
    for i in range(0, 9001):
        phi = mp.radians(mp.mpf(i) / 100)
        beta = authalic_exact(phi)
        s_f = phi + sum(c * mp.sin(2 * (k + 1) * phi) for k, c in enumerate(fwd_c))
        s_i = beta + sum(c * mp.sin(2 * (k + 1) * beta) for k, c in enumerate(inv_c))
        worst_f = max(worst_f, abs(s_f - beta))
        worst_i = max(worst_i, abs(s_i - phi))
    print(f"// series truncation bound, forward:  {mp.nstr(worst_f, 3)} rad")
    print(f"// series truncation bound, inverse:  {mp.nstr(worst_i, 3)} rad")
    print("// " + "-" * 70)


def reference_json():
    """Build the reference-value table for the committed JSON artifact.

    Returns
    -------
    dict
        The JSON payload.
    """
    special = [
        "0", "1e-7", "0.5", "5", "15", "30", "35.26438968275", "44.9999999",
        "45", "45.0000001", "60", "75", "89", "89.9999999", "90",
    ]
    grid = [str(v) for v in range(-90, 91, 5)]
    lats = sorted(set(special + grid + ["-" + s for s in special if s != "0"]),
                  key=lambda text: mp.mpf(text))

    forward = []
    inverse = []
    for text in lats:
        phi = mp.radians(mp.mpf(text))
        forward.append([text, mp.nstr(mp.degrees(authalic_exact(phi)), 25)])
        # Inverse table: the same numeric value read as an authalic latitude.
        b = mp.radians(mp.mpf(text))
        inverse.append([text, mp.nstr(mp.degrees(geodetic_exact(b)), 25)])
    return {
        "comment": "authalic <-> geodetic latitude reference (degrees), "
                   "mpmath 60 dps; see generate_authalic_reference.py",
        "wgs84": {"a": "6378137", "inv_f": "298.257223563",
                  "e2": mp.nstr(E2, 25), "qp": mp.nstr(QP, 25)},
        "forward_deg": forward,
        "inverse_deg": inverse,
    }


def main():
    """Derive, check, emit, and write the JSON artifact."""
    fwd = derive_forward()
    inv = derive_inverse(fwd)
    check_published(fwd, inv)
    emit(fwd, inv)
    out = Path(__file__).parent / "data" / "authalic_reference.json"
    out.write_text(json.dumps(reference_json(), indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
