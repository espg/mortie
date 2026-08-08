# mortie.arrow

Arrow interop: the `morton_index` pyarrow `ExtensionType` and the
library-agnostic Arrow C Data Interface surface. `MortonIndexType` and
`MortonIndexExtArray` are built lazily behind a module `__getattr__` (pyarrow is
optional), so they are documented narratively in
[Arrow interchange](../arrow_interchange.md) rather than here.

::: mortie.arrow
    options:
      members:
        - morton_index_type
        - polygons_to_morton_mocs
        - from_wkbs
        - from_morton_index
        - to_morton_index
        - export_c_array
        - export_c_schema
        - import_c_array
