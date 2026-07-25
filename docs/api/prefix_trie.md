# mortie.prefix_trie

Compacted-trie polygon builders over morton words.

`MortonChild` — the trie node class these builders return — is deliberately not
documented here while its keep / privatize / reshape verdict is open on
[PR #130](https://github.com/espg/mortie/pull/130).

::: mortie.prefix_trie
    options:
      members:
        - morton_polygon
        - morton_polygon_from_array
        - geo_morton_polygon
        - split_children
        - split_children_geo
