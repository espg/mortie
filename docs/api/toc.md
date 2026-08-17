# mortie.toc

The toc word — temporal order coverage (issue #175): one `uint64` packing
either an exact nanosecond timestamp or a conservative time range, sortable
as a plain unsigned integer and closed under a semilattice merge. Times are
ns since 1850-01-01 on a continuous, leap-free, GPS-aligned scale; the
`datetime64` / GPS converters are the only place leap seconds exist. Not an
IVOA T-MOC. These flat-array elementwise ops are the type's scalar surface
(the same relationship [mortie.moc](moc.md) has to its ops over one cover),
plus one ragged operator — `tocs_reduce`, the segmented sibling of
`toc_reduce` (issue #177), kept here because it folds the word type itself
rather than operating over covers. The many-*cover* plurals still land in
[mortie.batch](batch.md), and wait on the interval-set algebra, which stays
deferred for want of a consumer. The names stay flat on the package
(`mortie.time2toc`, ...).

Worked example:
[examples/toc_temporal_coverage.ipynb](https://github.com/espg/mortie/blob/HEAD/examples/toc_temporal_coverage.ipynb)
walks the type end-to-end on synthetic data — encoding, the conservative
merge, sorting without a comparator, the window predicates at a quantum
boundary, and the UTC/GPS round-trip
([run it on Binder](https://mybinder.org/v2/gh/espg/mortie/HEAD?labpath=examples%2Ftoc_temporal_coverage.ipynb)).

::: mortie.toc
    options:
      members:
        - time2toc
        - span2toc
        - toc2time
        - toc_merge
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
