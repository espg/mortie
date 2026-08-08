# mortie.tools

Encoding, decoding, inspection, and buffering of packed morton words.

::: mortie.tools
    options:
      members:
        - geo2mort
        - mort2geo
        - mort2bbox
        - mort2polygon
        - mort2healpix
        - mort2norm
        - norm2mort
        - infer_order_from_morton
        - orders_of
        - orders_of_uniq
        - is_point
        - validate_morton
        - clip2order
        - generate_morton_children
        - children_of
        - morton_buffer
        - morton_buffer_meters
        - order2res
        - res2display

!!! note "Not yet documented here"

    The UNIQ helpers (`geo2uniq`, `norm2uniq`, `uniq2geo`, `unique2parent`) are
    omitted while their signatures are in flux — see
    [issue #136](https://github.com/espg/mortie/issues/136). `heal_norm` is
    omitted because it is being removed under
    [PR #130](https://github.com/espg/mortie/pull/130).
