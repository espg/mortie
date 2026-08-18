# mortie.batch

Bulk (plural) operators over morton sets, MOCs and geometry columns. Every
function here is the batch twin of a scalar that lives elsewhere in the
package: one call carries a whole ragged column across the Python/Rust
boundary, and element `i` of the result is bit-identical to the scalar applied
to element `i` alone. Consolidated by **arity** (issue #170) — a different
axis from the domain split the rest of the package is organised on — with a
`See Also` on each side of every scalar/plural pair. The pyarrow skins
(`mortie.arrow.from_wkbs`, `mortie.arrow.polygons_to_morton_mocs`) stay in
[mortie.arrow](arrow.md); the names stay flat on the package
(`mortie.from_wkbs`, `mortie.children_of`).

::: mortie.batch
    options:
      members:
        - polygons_to_morton_mocs
        - from_wkbs
        - mocs_to_orders
        - mocs_and
        - mocs_intersect
        - common_ancestors
        - children_of
