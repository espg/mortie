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
