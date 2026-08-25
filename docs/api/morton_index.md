# mortie.morton_index

The numpy-only surface: the packed-word scalar and the decimal parse functions.
Nothing here imports pandas.

The pandas `MortonIndexDtype` / `MortonIndexArray` pair is re-exported from this
module (`mortie.morton_index.MortonIndexArray` resolves), but it is *defined* in
[`mortie.pandas`](pandas.md) and documented there.

`MortonWord` constructs from either form of a cell id (issue #152),
disambiguated by type: a `str` (or a 0-d `"U"` array of one) is the decimal
Morton label (`MortonWord("-31123")`, point-suffix grammar included),
parsed eagerly through `decimal_to_word` — an invalid label raises
`ValueError` at the boundary and never silently constructs. Anything else is
the packed word, handed to `numpy.uint64` and taking its semantics whole
(`int`, `numpy.uint64`, and by numpy parity `bool` and a truncating `float`);
bytes-like input is the one deliberate divergence, refused with a pointed
`TypeError` rather than read as numpy would read it. The `.decimal` /
`.order` / `.base_cell` accessors read the label string, the HEALPix order,
and the base cell back off the word — and they are **strict**: a word that
decodes to no legal cell (the empty sentinel included) raises a pointed
`ValueError` naming the word, rather than propagating a sentinel string
onward. Only the display dunders stay lazy/never-raise (`"<NA>"` for the
empty word, `"<invalid 0x...>"` for one with an invalid prefix). The type is
exported flat as `mortie.MortonWord`.

::: mortie.morton_index
    options:
      members:
        - decimal_to_word
        - MortonWord
