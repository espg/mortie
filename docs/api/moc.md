# mortie.moc

The MOC (multi-order coverage) algebra over morton sets: compaction, densify,
boolean set ops, and the ancestry reductions. Split out of `mortie.coverage` by
domain (issue #156); the ragged batch twins — `mocs_to_orders`,
`common_ancestors`, and the 1×N set-op broadcast `mocs_and` /
`mocs_intersect` — live in [mortie.batch](batch.md) (issues #170, #173). The names
stay flat on the package (`mortie.moc_to_order`, `mortie.mocs_to_orders`).

::: mortie.moc
    options:
      members:
        - compress_moc
        - moc_to_order
        - moc_or
        - moc_and
        - moc_intersects
        - moc_minus
        - moc_xor
        - moc_not
        - moc_min
        - common_ancestor
        - split_base_cells
