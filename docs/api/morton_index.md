# mortie.morton_index

The numpy-only surface: the packed-word scalar and the decimal parse functions.
Nothing here imports pandas.

The pandas `MortonIndexDtype` / `MortonIndexArray` pair is re-exported from this
module (`mortie.morton_index.MortonIndexArray` resolves), but it is *defined* in
[`mortie.pandas`](pandas.md) and documented there.

`MortonIndexScalar` constructs from either form of a cell id (issue #152),
disambiguated by type: a `str` (or a 0-d `"U"` array of one) is the decimal
Morton label (`MortonIndexScalar("-31123")`, point-suffix grammar included),
parsed eagerly through `decimal_to_word` — an invalid label raises
`ValueError` at the boundary and never silently constructs. Anything else is
the packed word, handed to `numpy.uint64` and taking its semantics whole
(`int`, `numpy.uint64`, and by numpy parity `bool` and a truncating `float`);
bytes-like input is the one deliberate divergence, refused with a pointed
`TypeError` rather than read as numpy would read it. The `.decimal` /
`.order` accessors read the label string and the HEALPix order back off the
word. `.decimal` is exactly the `str` rendering, so the display's
lazy/never-raise sentinels pass straight through it: `"<NA>"` for the empty
word, `"<invalid 0x...>"` for one with an invalid prefix.

::: mortie.morton_index
    options:
      members:
        - decimal_to_word
        - MortonIndexScalar
