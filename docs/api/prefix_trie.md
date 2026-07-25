# mortie.prefix_trie

Compacted-trie polygon builders over morton words, and the trie node type
they return.

`MortonChild` is documented as a **return type**: obtain nodes from the
builders below, never by constructing one. Its read surface — `characteristic`,
`len`, `children`, `nchildren`, `mantissa_array`, `cell_area` — is the frozen
contract; the constructor is internal and its signature is not (espg-ratified
on [PR #130](https://github.com/espg/mortie/pull/130)).

::: mortie.prefix_trie
    options:
      members:
        - morton_polygon
        - morton_polygon_from_array
        - geo_morton_polygon
        - split_children
        - split_children_geo
        - MortonChild
