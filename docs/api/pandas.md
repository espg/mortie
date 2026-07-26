# mortie.pandas

mortie's pandas **extension** — not pandas itself. This module holds the
`morton_index` pandas `ExtensionDtype` / `ExtensionArray` pair over the packed
64-bit decimal-Morton words. Importing it is what pulls pandas in, which is how
`import mortie` stays numpy-only (see
[Optional dependencies](../morton_index_datatype.md#optional-dependencies-numpy-stays-the-only-runtime-dep)).

!!! note "Import paths"

    The members below are documented under `mortie.pandas` because that is
    where they are defined. The canonical user-facing import is the short one:

    ```python
    from mortie import MortonIndexArray, MortonIndexDtype
    ```

    `from mortie.pandas import MortonIndexArray` and
    `from mortie.morton_index import MortonIndexArray` resolve to the very same
    class objects — all three paths work, and none is deprecated.

For narrative usage — construction, the decimal repr, sorting, the domain
accessors and the parquet round-trip — see
[The `morton_index` datatype](../morton_index_datatype.md).

::: mortie.pandas
    options:
      members:
        - MortonIndexArray
        - MortonIndexDtype
