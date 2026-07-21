"""Drift pin for docs/specification.md (issue #62).

The spec page's resolution table is regenerated here from the unified
HEALPix-sphere formulas the doc cites (issue #62: the table cannot drift).
**Both** columns derive from the exact HEALPix sphere at mean radius
``EARTH_RADIUS_KM``: every order-k cell has identical area
``4*pi*R**2 / (12 * 4**k)``, and the cell scale is the square root of that
area (RMS cell spacing). This is the sphere-derived normative value, not the
historical ``order2res`` constant kept in ``mortie.tools`` for behavioral
compatibility (spec §3 code-vs-page note; unification tracked as a mortie
follow-up issue). This test compares every regenerated row literally against
the rows between the ``table:order2res`` markers; to refresh the doc after a
deliberate formula change, paste the rows this module's ``table_rows()``
produces.
"""

import math
from pathlib import Path

from mortie.tools import MAX_ORDER

SPEC_PAGE = Path(__file__).resolve().parents[2] / "docs" / "specification.md"
BEGIN = "<!-- table:order2res:begin -->"
END = "<!-- table:order2res:end -->"
# Exact HEALPix sphere, mean Earth radius (km); drives BOTH table columns.
EARTH_RADIUS_KM = 6371.0088


def format_row(order):
    """One markdown table row for ``order``, from the unified-sphere formulas."""
    # Every order-k HEALPix cell has identical area; the cell scale is the
    # square root of that area (RMS cell spacing on the exact sphere).
    area = 4 * math.pi * EARTH_RADIUS_KM**2 / (12 * 4**order)  # km2
    km = math.sqrt(area)  # cell scale (km)
    if km >= 1.0:
        scale = f"{km:,.3f} km"
    elif km >= 1e-3:
        scale = f"{km * 1e3:,.3f} m"
    else:
        scale = f"{km * 1e5:,.3f} cm"
    area_s = f"{area:,.6g} km2" if area >= 1.0 else f"{area * 1e6:,.6g} m2"
    return f"| {order} | {2**order} | {scale} | {area_s} |"


def table_rows():
    """Every data row of the resolution table, orders 0..MAX_ORDER."""
    return [format_row(order) for order in range(MAX_ORDER + 1)]


class TestSpecPageResolutionTable:
    def _doc_rows(self):
        text = SPEC_PAGE.read_text()
        assert BEGIN in text and END in text, "table markers missing from spec page"
        block = text.split(BEGIN, 1)[1].split(END, 1)[0]
        # Drop the header and separator rows; keep data rows only.
        rows = [ln.strip() for ln in block.strip().splitlines()]
        assert rows[0].startswith("| order |"), "table header changed"
        return rows[2:]

    def test_table_matches_order2res(self):
        assert self._doc_rows() == table_rows()

    def test_table_covers_full_order_range(self):
        assert len(self._doc_rows()) == MAX_ORDER + 1


class TestDecimalParseTieBreak:
    """Golden pin for the spec page §4 tie-break (espg-ratified 2026-07-21).

    Kind is encoding-carried: suffix ``0..=47`` = area, ``48..=63`` = point.
    The one cross-kind ambiguity is the order-29 decimal string, and parsing
    it always yields the AREA word — pinned here at the bit level so the
    parser and the page cannot drift apart.
    """

    def test_order29_string_parses_to_area_word(self):
        import numpy as np

        from mortie import MortonIndexArray
        from mortie.morton_index import _decimal_to_word

        arr = MortonIndexArray.from_latlon(
            np.array([45.0]), np.array([45.0]), points=True
        )
        word = int(np.asarray(arr._data, dtype=np.uint64)[0])
        # decimal_repr renders point words with the terminal 'p' kind suffix
        # (spec §2, issue #120); the §4 tie-break concerns the *unmarked*
        # order-29 string, so strip the marker to build the ambiguous probe.
        marked = arr.decimal_repr()[0]
        assert marked.endswith("p")
        dec = marked.rstrip("p")
        t28, t29 = int(dec[-2]) - 1, int(dec[-1]) - 1

        # The point word sits in the point suffix region, at the documented
        # preorder slot 48 + t28*4 + t29 (§1).
        assert word & 0x3F == 48 + t28 * 4 + t29

        # Tie-break: the unmarked parse yields the AREA word — same prefix+body
        # (same path), area suffix 28 + t28*5 + t29 + 1 (§4).
        parsed = int(_decimal_to_word(dec))
        assert parsed & 0x3F == 28 + t28 * 5 + t29 + 1
        assert parsed >> 6 == word >> 6
        assert parsed != word

        # The 'p'-marked string is unambiguous and round-trips to the POINT
        # word (spec §2/§4, issue #120) — the marker is what disambiguates.
        assert int(_decimal_to_word(marked)) == word

        # Non-injectivity: the area word renders the same *unmarked* string,
        # so point-ness cannot round-trip through the unmarked decimal repr.
        area = MortonIndexArray.from_words(np.asarray([parsed], dtype=np.uint64))
        assert area.decimal_repr()[0] == dec

    def test_sub29_strings_are_unambiguous(self):
        import numpy as np

        from mortie import MortonIndexArray
        from mortie.morton_index import _decimal_to_word

        # Points exist only at order 29: any shorter string denotes exactly
        # one word (an area cell), which round-trips bit-identically. Build an
        # explicit sub-29 cell (order 10) so this genuinely exercises a shorter
        # string rather than the order-29 area word from_latlon defaults to.
        arr = MortonIndexArray.from_latlon(
            np.array([45.0]), np.array([45.0]), order=10
        )
        word = int(np.asarray(arr._data, dtype=np.uint64)[0])
        dec = arr.decimal_repr()[0]
        assert len(dec) < 29  # a genuinely shorter string
        assert word & 0x3F <= 27  # sub-29 area region (below the order-29 28..47 band)
        assert int(_decimal_to_word(dec)) == word
