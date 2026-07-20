"""Drift pin for docs/specification.md (issue #62).

The spec page's resolution table is *generated from* ``order2res`` (the
acceptance criterion on issue #62: the table cannot drift from the code).
This test regenerates every table row from the same formulas the doc cites
and compares them literally against the rows between the
``table:order2res`` markers. To refresh the doc after a deliberate formula
change, paste the rows this module's ``table_rows()`` produces.
"""

import math
from pathlib import Path

from mortie.tools import MAX_ORDER, order2res

SPEC_PAGE = Path(__file__).resolve().parents[2] / "docs" / "specification.md"
BEGIN = "<!-- table:order2res:begin -->"
END = "<!-- table:order2res:end -->"
# Mean Earth radius (km) used for the exact HEALPix cell-area column.
EARTH_RADIUS_KM = 6371.0088


def format_row(order):
    """One markdown table row for ``order``, from the documented formulas."""
    km = order2res(order)
    if km >= 1.0:
        scale = f"{km:,.3f} km"
    elif km >= 1e-3:
        scale = f"{km * 1e3:,.3f} m"
    else:
        scale = f"{km * 1e5:,.3f} cm"
    area = 4 * math.pi * EARTH_RADIUS_KM**2 / (12 * 4**order)
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
