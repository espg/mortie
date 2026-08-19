# mortie toc kernel

**The word grammar is normative in the
[specification](../specification.md#10-the-packed-64-bit-toc-word)** (§10,
frozen for the 1.x series — bit layout, epoch and timescale, encode/decode
laws, sort order, merge law, conformance vectors); this page documents the
API surface over it.

The toc word — temporal order coverage (issue #175): one `uint64` packing
either an exact nanosecond timestamp or a conservative time range, sortable
as a plain unsigned integer and closed under a semilattice merge. Times are
ns since 1850-01-01 on a continuous, leap-free, GPS-aligned scale; the
`datetime64` / GPS converters are the only place leap seconds exist. Not an
IVOA T-MOC. These flat-array elementwise ops are the type's scalar surface
(the same relationship [the MOC kernel](moc.md) has to its ops over one
cover), plus one ragged operator — `tocs_reduce`, the segmented sibling of
`toc_reduce` (issue #177), kept here because it folds the word type itself
rather than operating over covers. `toc_normalize` and `toc_and` are the
set-algebra entries the issue #177 call-site audit ruled in: the canonical
cover form and the one set operation over it. The many-*cover* plurals still
land in [mortie.batch](batch.md). The names stay flat on the package
(`mortie.time2toc`, ...).

These are the **kernel layer**: words in, words out, no wrapping cost, and
nothing here is deprecated. The **object layer** over them is
[mortie.Toc](toc_object.md), where every public method is a single delegation
to a function on this page.

!!! warning "`mortie.toc` is no longer a module (issue #198)"

    The implementation moved to `mortie/_toc.py` so that `mortie.toc` could
    become the `Toc` constructor — the same move issue #196 made for
    `mortie.moc`. `import mortie.toc` and `from mortie.toc import …`
    **break**; the flat package names (`mortie.time2toc`,
    `mortie.toc_merge`, …, and now `mortie.Q_START_NS`, `mortie.Q_END_NS`,
    `mortie.TOC_MAX_NS`, `mortie.GPS_EPOCH_NS`) are unchanged and are the
    supported spelling. `mortie.toc.toc_merge`-style attribute access still
    resolves for one minor version, with a `DeprecationWarning`.

Worked example:
[examples/toc_temporal_coverage.ipynb](https://github.com/espg/mortie/blob/HEAD/examples/toc_temporal_coverage.ipynb)
walks the type end-to-end on synthetic data — encoding, the conservative
merge, sorting without a comparator, the window predicates at a quantum
boundary, and the UTC/GPS round-trip
([run it on Binder](https://mybinder.org/v2/gh/espg/mortie/HEAD?labpath=examples%2Ftoc_temporal_coverage.ipynb)).

::: mortie._toc
    options:
      members:
        - time2toc
        - span2toc
        - toc2time
        - toc_merge
        - toc_normalize
        - toc_and
        - toc_reduce
        - tocs_reduce
        - toc_is_range
        - toc_overlaps
        - toc_contains
        - from_datetime64
        - to_datetime64
        - from_gps_ns
        - to_gps_ns
        - Q_START_NS
        - Q_END_NS
        - TOC_MAX_NS
        - GPS_EPOCH_NS
