# mortie.geometry

Lazy WKB/WKT geometry codec. The geometry backend (`shapely>=2` preferred,
`spherely` accepted) is imported on first use, so `numpy` stays the only runtime
dependency.

`from_wkb` needs no backend at all (issue #157): mortie parses WKB itself, in
Rust, and covers the rings directly. A backend is still required for WKT ingest
(there is no Rust WKT parser) and for the whole emit direction, which hands back
a backend geometry object by definition.

::: mortie.geometry
    options:
      members:
        - from_wkb
        - from_wkbs
        - from_wkt
        - from_geometry
        - to_wkb
        - to_wkt
        - to_geometry
