# mortie.geometry

Lazy WKB/WKT geometry codec. The geometry backend (`shapely>=2` preferred,
`spherely` accepted) is imported on first use, so `numpy` stays the only runtime
dependency.

::: mortie.geometry
    options:
      members:
        - from_wkb
        - from_wkt
        - from_geometry
        - to_wkb
        - to_wkt
        - to_geometry
