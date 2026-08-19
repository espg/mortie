# mortie.coverage

Polygon-to-morton coverage. The MOC (multi-order coverage) set algebra over
the covers it produces lives in [the MOC kernel](moc.md); the MOC coverer's
public entry point, `polygons_to_morton_mocs`, lives in
[mortie.batch](batch.md) (issues #170, #187 — the scalar
`morton_coverage_moc` retired with the plural batch names; the one-ring-set
multipart form is reached through `from_geometry` / `from_wkb` / `from_wkt`
with `moc=True`, or `mortie.Moc`).

::: mortie.coverage
    options:
      members:
        - morton_coverage
