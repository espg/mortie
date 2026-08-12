# mortie.convert

Address-space conversions between geographic coordinates, packed morton words,
UNIQ cell numbers and HEALPix NESTED ids — plus `mort2bbox` / `mort2polygon`,
which turn a word into a bounding box or a ring. Split out of `mortie.tools` by
domain (issue #159) so the Python surface mirrors the Rust tree
(`geo2mort.rs`, `morton.rs`, `cell_geom.rs`); the names stay flat on the package
(`mortie.geo2mort`, `mortie.mort2polygon`).

Every geographic entry point here takes a keyword-only `latitude=` argument.
Its default, `"authalic"`, maps WGS84 geodetic latitude to authalic latitude
on the way into the spherical kernel and back on the way out, so cells are
equal-area on the ellipsoid; `latitude="geodetic-spherical"` is the pre-0.10
escape. The two conventions are non-corresponding partitions — see
[specification.md §9](../specification.md#latitude-convention). The
`geodetic_to_authalic` / `authalic_to_geodetic` pair below exposes that
latitude→latitude mapping on its own.

::: mortie.convert
    options:
      members:
        - geo2mort
        - mort2geo
        - mort2bbox
        - mort2polygon
        - mort2healpix
        - mort2norm
        - norm2mort
        - geodetic_to_authalic
        - authalic_to_geodetic

!!! note "Not yet documented here"

    The UNIQ helpers (`geo2uniq`, `norm2uniq`, `uniq2geo`, `unique2parent`) are
    omitted while their signatures are in flux — see
    [issue #136](https://github.com/espg/mortie/issues/136). `heal_norm` is
    omitted because it is being removed under
    [PR #130](https://github.com/espg/mortie/pull/130).
