"""Drift pin for the spec page's toc word grammar (issue #193).

The spec page's §10.8 conformance-vector table is regenerated here from the
live kernels — every row's word, decoded bounds, and UTC rendering come from
``time2toc`` / ``span2toc`` / ``toc_merge`` / ``toc2time`` /
``to_datetime64`` and are compared literally against the rows between the
``table:toc_vectors`` markers, so spec<->code drift fails the suite on
whichever side moved.  The classes below the table pin the section's other
quantitative claims: the quoted constants, the 2018 leap-offset derivation,
the exact valid-domain characterization (both directions), the unreachable
all-zero word, and the §10.5 sort tie-breaks.  The algebraic laws (§10.6)
are pinned at volume by the cargo tests in ``src_rust/src/toc.rs``.
"""

from pathlib import Path

import numpy as np
import pytest

from mortie import toc

SPEC_PAGE = Path(__file__).resolve().parents[2] / "docs" / "specification.md"
BEGIN = "<!-- table:toc_vectors:begin -->"
END = "<!-- table:toc_vectors:end -->"


def format_row(label, call, word):
    """One markdown table row for ``word``, kernel-derived end to end."""
    start, end = toc.toc2time(word)
    utc = str(toc.to_datetime64(start))
    return (f"| {label} | `{call}` | `0x{word:016X}` | {word} "
            f"| {start} | {end} | {utc} |")


def vector_rows():
    """Every data row of the §10.8 conformance table, from the live kernels."""
    w_epoch = toc.time2toc(0)
    w_gps = toc.time2toc(4_102_790_400_000_000_000)
    w_2018 = toc.time2toc(5_301_590_418_000_000_000)
    w_last = toc.time2toc(toc.TOC_MAX_NS - 1)
    w_straddle = toc.span2toc(12_384_901_888, 13_384_901_888)
    w_ongrid = toc.span2toc(29_064_771_072, 30_064_771_072)
    return [
        format_row("timestamp: epoch", "time2toc(0)", w_epoch),
        format_row("timestamp: GPS epoch",
                   "time2toc(4102790400000000000)", w_gps),
        format_row("timestamp: 2018-01-01 UTC",
                   "time2toc(5301590418000000000)", w_2018),
        format_row("timestamp: last valid instant",
                   "time2toc(9223372032559808511)", w_last),
        format_row("range: straddles the 2^32 grid",
                   "span2toc(12384901888, 13384901888)", w_straddle),
        format_row("range: end exactly on the 2^32 grid",
                   "span2toc(29064771072, 30064771072)", w_ongrid),
        format_row("merge: the two ranges above",
                   f"toc_merge({w_straddle}, {w_ongrid})",
                   toc.toc_merge(w_straddle, w_ongrid)),
        format_row("merge: the two epoch timestamps above",
                   f"toc_merge({w_epoch}, {w_gps})",
                   toc.toc_merge(w_epoch, w_gps)),
    ]


class TestSpecPageTocVectors:
    def _doc_rows(self):
        text = SPEC_PAGE.read_text()
        assert BEGIN in text and END in text, "toc table markers missing"
        block = text.split(BEGIN, 1)[1].split(END, 1)[0]
        rows = [ln.strip() for ln in block.strip().splitlines()]
        assert rows[0].startswith("| value |"), "table header changed"
        return rows[2:]

    def test_table_matches_kernels(self):
        assert self._doc_rows() == vector_rows()

    def test_table_row_count(self):
        assert len(self._doc_rows()) == len(vector_rows())


class TestQuotedConstants:
    """The constants §10.1 quotes by value."""

    def test_quanta(self):
        assert toc.Q_START_NS == 2**31
        assert toc.Q_END_NS == 2**32

    def test_span_ceiling(self):
        assert toc.TOC_MAX_NS == 2**63 - 2**32 == 9_223_372_032_559_808_512

    def test_gps_epoch(self):
        assert toc.GPS_EPOCH_NS == 47_486 * 86_400 * 10**9
        assert toc.GPS_EPOCH_NS == 4_102_790_400_000_000_000

    def test_2018_leap_offset(self):
        # 2018-01-01T00:00:00 UTC -> internal ns, +18 s (TAI-UTC-19 = 37-19).
        t = toc.from_datetime64(np.datetime64("2018-01-01T00:00:00"))
        assert t == 5_301_590_418_000_000_000

    def test_1972_step_back(self):
        # §10.1: the mapping steps back 9 s across the 1972-01-01 boundary.
        before = toc.from_datetime64(np.datetime64("1971-12-31T23:59:59"))
        after = toc.from_datetime64(np.datetime64("1972-01-01T00:00:00"))
        assert after == before - 9_000_000_000 + 1_000_000_000


class TestValidDomain:
    """§10.4: the valid-domain characterization is exact, both directions."""

    def test_timestamp_high_field_bound(self):
        # Valid timestamps fill high fields 0 .. 2^32 - 3, inclusive.
        assert toc.time2toc(toc.TOC_MAX_NS - 1) >> 32 == 2**32 - 3
        with pytest.raises(ValueError):
            toc.time2toc(toc.TOC_MAX_NS)

    def test_range_words_satisfy_s_le_2e_minus_1(self):
        rng = np.random.default_rng(0x193)
        ends = rng.integers(0, toc.TOC_MAX_NS, 2000, dtype=np.uint64)
        starts = (ends * rng.random(2000)).astype(np.uint64)
        words = toc.span2toc(starts, ends)
        s, e = words >> np.uint64(32), words & np.uint64(2**31 - 1)
        assert np.all(e >= 1)
        assert np.all(s <= 2 * e - 1)

    def test_every_valid_range_word_is_encoder_reachable(self):
        # For any (s, e) with e >= 1 and s <= 2e - 1, the §10.4 witness
        # interval [s * 2^31, max(s * 2^31, (e - 1) * 2^32)] encodes to
        # exactly that word — including the s = 2e - 1 edge.
        rng = np.random.default_rng(0x193)
        e = rng.integers(1, 2**31, 2000, dtype=np.uint64)
        s = (rng.random(2000) * (2 * e - 1)).astype(np.uint64)
        s[:4] = (2 * e[:4] - 1)  # pin the tight edge explicitly
        start = s << np.uint64(31)
        end = np.maximum(start, (e - np.uint64(1)) << np.uint64(32))
        assert np.array_equal(toc.span2toc(start, end),
                              (s << np.uint64(32)) | e)


class TestZeroWordUnreachable:
    """§10.3: no encoder output is the all-zero word."""

    def test_epoch_word_and_minimal_range_word(self):
        assert toc.time2toc(0) == 0x8000_0000
        assert toc.span2toc(0, 0) == 1  # the smallest range word


class TestSortTieBreaks:
    """§10.5: within a tied start quantum — ranges first, shorter first."""

    def test_tied_quantum_order(self):
        base = 1_000_000 * toc.Q_START_NS
        ts_lo = toc.time2toc(base + 7)
        ts_hi = toc.time2toc(base + toc.Q_START_NS - 1)
        rng_short = toc.span2toc(base + 100, base + 200)
        rng_long = toc.span2toc(base + 100, base + 10 * toc.Q_END_NS)
        assert rng_short < rng_long < ts_lo < ts_hi
