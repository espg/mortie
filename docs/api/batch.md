# mortie.batch

Batch kernels over morton sets, MOCs and geometry columns: one call carries a
whole ragged column across the Python/Rust boundary, and element `i` of the
result is bit-identical to the single-item form applied to element `i` alone.
Consolidated by **arity** (issue #170) — a different axis from the domain
split the rest of the package is organised on.

Since issue #187 the public surface is polymorphic — one function per
operation, the batch form selected by input shape or a keyword-only
`offsets=` — so the plural batch names this module used to export
(`mocs_to_orders`, `mocs_and`, `mocs_intersect`, `common_ancestors`,
`children_of`, `from_wkbs`) are retired, and their kernels live on here as
private functions behind the surviving entry points (`mortie.moc_to_order`,
`mortie.moc_and`, `mortie.moc_intersects`, `mortie.common_ancestor`,
`mortie.generate_morton_children`, `mortie.from_wkb`). The one public name
left is the batch-native coverer below, whose ragged signature has no scalar
shape to collapse into; the pyarrow skins (`mortie.arrow.from_wkb`,
`mortie.arrow.polygons_to_morton_mocs`) stay in [mortie.arrow](arrow.md).

::: mortie.batch
    options:
      members:
        - polygons_to_morton_mocs
