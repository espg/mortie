# mortie.moc

The MOC (multi-order coverage) algebra over morton sets: compaction, densify,
boolean set ops, and the ancestry reductions. Split out of `mortie.coverage` by
domain (issue #156) so each scalar op sits beside its plural batch twin; the
names stay flat on the package (`mortie.moc_to_order`, `mortie.mocs_to_orders`).

::: mortie.moc
    options:
      members:
        - compress_moc
        - moc_to_order
        - mocs_to_orders
        - moc_or
        - moc_and
        - moc_minus
        - moc_xor
        - moc_not
        - moc_min
        - common_ancestor
        - split_base_cells
