# mortie.morton_index

The decimal parse surface. The pandas `MortonIndexDtype` / `MortonIndexArray`
pair is built lazily behind a module `__getattr__` (so a numpy-only install can
still import `mortie`), which puts it out of reach of mkdocstrings' static
analysis — it is documented narratively in
[Morton index datatype](../morton_index_datatype.md) instead.

::: mortie.morton_index
    options:
      members:
        - decimal_to_word
        - decimals_to_words
        - MortonIndexScalar
