# mortie.moc

The MOC (multi-order coverage) algebra over morton sets: compaction, densify,
boolean set ops, and the ancestry reductions. Split out of `mortie.coverage` by
domain (issue #156); the ragged batch twins, `mocs_to_orders` and
`common_ancestors`, live in [mortie.batch](batch.md) (issue #170). The names
stay flat on the package (`mortie.moc_to_order`, `mortie.mocs_to_orders`).

::: mortie.moc
    options:
      members:
        - compress_moc
        - moc_to_order
        - moc_or
        - moc_and
        - moc_minus
        - moc_xor
        - moc_not
        - moc_min
        - common_ancestor
        - split_base_cells
