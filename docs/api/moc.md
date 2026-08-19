# mortie MOC kernel

The MOC (multi-order coverage) algebra over morton sets: compaction, densify,
boolean set ops, and the ancestry reductions. Split out of `mortie.coverage` by
domain (issue #156); the ragged batch kernels behind the vectorized operators
— reached through each operator's keyword-only `offsets=` form since the
polymorphic consolidation (issue #187) — live in
[mortie.batch](batch.md) as private functions (issues #170, #173). The names
stay flat on the package (`mortie.moc_to_order`).

These are the **kernel layer**: words in, words out, no wrapping cost, and
nothing here is deprecated. The **object layer** over them is
[mortie.Moc](moc_object.md), where every method is a single delegation to a
function on this page.

!!! warning "`mortie.moc` is no longer a module (issue #196)"

    The implementation moved to `mortie/_moc.py` so that `mortie.moc` could
    become the `Moc` constructor. `import mortie.moc` and
    `from mortie.moc import …` **break**; the flat package names
    (`mortie.moc_to_order`, `mortie.compress_moc`, …) are unchanged and are the
    supported spelling. `mortie.moc.moc_and`-style attribute access still
    resolves for one minor version, with a `DeprecationWarning`.

::: mortie._moc
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
