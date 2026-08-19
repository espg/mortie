# mortie.morton_index

The numpy-only surface: the packed-word scalar and the decimal parse functions.
Nothing here imports pandas.

The pandas `MortonIndexDtype` / `MortonIndexArray` pair is re-exported from this
module (`mortie.morton_index.MortonIndexArray` resolves), but it is *defined* in
[`mortie.pandas`](pandas.md) and documented there.

::: mortie.morton_index
    options:
      members:
        - decimal_to_word
        - MortonIndexScalar
