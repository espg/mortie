# mortie.morton_index

The numpy-only surface: the packed-word scalar and the decimal parse functions.
Nothing here imports pandas.

The pandas `MortonIndexDtype` / `MortonIndexArray` pair is re-exported from this
module (`mortie.morton_index.MortonIndexArray` resolves), but it is *defined* in
[`mortie.pandas`](pandas.md) and documented there.

`MortonIndexScalar` constructs from either form of a cell id (issue #152),
disambiguated by type: an `int` / `numpy.uint64` is the packed word, a `str` is
the decimal Morton label (`MortonIndexScalar("-31123")`, point-suffix grammar
included) parsed eagerly through `decimal_to_word` — an invalid label raises
`ValueError` at the boundary, never silently constructs, and `bytes` is refused
with a pointed `TypeError` rather than reinterpreted. The `.decimal` /
`.order` accessors read the label string and the HEALPix order back off the
word; display stays lazy/never-raise (`"<NA>"` / `"<invalid 0x...>"`).

::: mortie.morton_index
    options:
      members:
        - decimal_to_word
        - MortonIndexScalar
